"""
gRPC server for the Crucible inference worker.

Implements InferenceService (proto/inference.proto). Import resolution relies on
PYTHONPATH=/app/gen set in the Dockerfile, so the generated stubs and this module
both use 'from proto import ...' without any sys.path manipulation.
"""

import logging
import signal
import threading
import time
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
from proto import inference_pb2 as pb2
from proto import inference_pb2_grpc as pb2_grpc

from config import GRPC_PORT, MAX_WORKERS
from model_runner import InferenceResult, ModelRunner

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("worker.server")

_NOT_READY_STATUSES: tuple[int, ...] = (
    pb2.SERVING_STATUS_UNKNOWN,
    pb2.SERVING_STATUS_DOWN,
)


class InferenceServicer(pb2_grpc.InferenceServiceServicer):
    """gRPC servicer for InferenceService.

    ServingStatus lifecycle::

        UNKNOWN   -> servicer created, serve() not yet called
        DOWN      -> set_loading() called; model is downloading/loading
        OK        -> set_ready() called; serving normally
        DEGRADED  -> _consecutive_errors >= _ERROR_THRESHOLD; auto-recovers to OK on next success
    """

    _ERROR_THRESHOLD: int = 3

    def __init__(self) -> None:
        """Initialise the servicer in UNKNOWN state with no model loaded."""
        self._lock = threading.Lock()
        self._status: int = pb2.SERVING_STATUS_UNKNOWN  # ServingStatus proto enum (int constant)
        self.runner: ModelRunner | None = None
        self._consecutive_errors: int = 0

    # ------------------------------------------------------------------
    # State transitions (called from serve())
    # ------------------------------------------------------------------

    def set_loading(self) -> None:
        """Transition to DOWN, signaling that model loading has started."""
        with self._lock:
            self._status = pb2.SERVING_STATUS_DOWN

    def set_ready(self, runner: ModelRunner) -> None:
        """Transition to OK once the model has finished loading.

        Args:
            runner (ModelRunner): The initialised model runner to use for inference.
        """
        with self._lock:
            self.runner = runner
            self._consecutive_errors = 0
            self._status = pb2.SERVING_STATUS_OK

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_inference_success(self) -> None:
        """Reset error count and restore OK status after a successful inference."""
        with self._lock:
            self._consecutive_errors = 0
            self._status = pb2.SERVING_STATUS_OK

    def _on_inference_error(self) -> None:
        """Increment error count and transition to DEGRADED if threshold is reached."""
        with self._lock:
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._ERROR_THRESHOLD:
                self._status = pb2.SERVING_STATUS_DEGRADED

    # ------------------------------------------------------------------
    # RPC handlers
    # ------------------------------------------------------------------

    def RunInference(
        self,
        request: pb2.InferenceRequest,
        context: grpc.ServicerContext,
    ) -> pb2.InferenceResponse:
        """Handle a single-item inference request by delegating to BatchInference.

        Wraps the single request in a BatchRequest so all inference logic
        remains in one place. The gateway batcher always sends BatchRequest;
        this RPC exists primarily for direct testing via grpcurl.

        Args:
            request (pb2.InferenceRequest): The single inference request.
            context (grpc.ServicerContext): gRPC context for setting status codes.

        Returns:
            pb2.InferenceResponse: The inference result, or an empty response if the
                batch call fails (gRPC error code is already set on context in that case).
        """
        batch_req = pb2.BatchRequest(requests=[request])
        batch_resp = self.BatchInference(batch_req, context)
        if not batch_resp.responses:
            return pb2.InferenceResponse()
        return batch_resp.responses[0]

    def BatchInference(
        self,
        request: pb2.BatchRequest,
        context: grpc.ServicerContext,
    ) -> pb2.BatchResponse:
        """Run inference on a batch of inputs and return per-request results.

        Args:
            request (pb2.BatchRequest): A batch of InferenceRequest messages.
            context (grpc.ServicerContext): gRPC context for setting status codes.

        Returns:
            pb2.BatchResponse: A BatchResponse containing one InferenceResponse per
                input, with ``output`` formatted as ``"LABEL (score)"``.
                Returns an empty BatchResponse with a gRPC error code set on
                context if the model is not ready or inference fails.
        """
        # Early exit if model isn't ready
        if self._status in _NOT_READY_STATUSES:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Model is not ready.")
            return pb2.BatchResponse()

        if not request.requests:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("BatchRequest must contain at least one request.")
            return pb2.BatchResponse()

        # Invariant: runner is set in set_ready() before status flips to OK,
        # so this guard should never trigger under normal operation.
        if self.runner is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Runner is unexpectedly uninitialized.")
            return pb2.BatchResponse()

        # Extract inputs and run inference, handling any errors
        inputs: list[str] = [r.input for r in request.requests]
        t0 = time.perf_counter()
        try:
            results: list[InferenceResult] = self.runner.predict(inputs)
        except Exception as exc:
            self._on_inference_error()
            log.error("batch_inference failed", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(exc))
            return pb2.BatchResponse()

        # Handle successful inference and package results into response messages
        self._on_inference_success()
        latency_ms = (time.perf_counter() - t0) * 1000
        log.info("batch_inference batch_size=%d latency_ms=%.1f", len(inputs), latency_ms)

        responses: list[pb2.InferenceResponse] = [
            pb2.InferenceResponse(
                request_id=req.request_id,
                output=f"{res['label']} ({res['score']:.4f})",
            )
            for req, res in zip(request.requests, results)
        ]
        return pb2.BatchResponse(responses=responses)

    def HealthCheck(
        self,
        request: pb2.HealthRequest,
        context: grpc.ServicerContext,
    ) -> pb2.HealthResponse:
        """Return the current serving status.

        Args:
            request (pb2.HealthRequest): Empty health check request.
            context (grpc.ServicerContext): gRPC context (unused).

        Returns:
            pb2.HealthResponse: Response containing the current ServingStatus.
        """
        return pb2.HealthResponse(status=self._status)


def serve() -> None:
    """Start the gRPC server and block until terminated.

    Starts the server before loading the model so that health checks return
    SERVING_STATUS_DOWN (rather than connection refused) during the ~30-45s
    model load window.

    Raises:
        Exception: Re-raises any exception thrown during model loading after
            stopping the server and logging at CRITICAL level. The container
            will exit and Docker will restart it.
    """
    # Start gRPC server & register servicer with thread pool executor for concurrent handling of requests.
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=MAX_WORKERS))
    servicer = InferenceServicer()
    pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")

    service_names = (
        pb2.DESCRIPTOR.services_by_name["InferenceService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    server.start()
    log.info("gRPC server started on port %d", GRPC_PORT)

    # Load model after starting server, raising exception to restart container
    log.info("Loading model...")
    servicer.set_loading()
    t0 = time.perf_counter()
    try:
        runner = ModelRunner()
    except Exception:
        log.critical("Model load failed — shutting down", exc_info=True)
        server.stop(grace=0)
        raise

    servicer.set_ready(runner)
    log.info(
        "Model ready (device=%s, load_time=%.1fs)",
        runner.device,
        time.perf_counter() - t0,
    )
    # Catch termination signals (SIGTERM/SIGINT) to gracefully shut down the server, 
    # allowing 5 seconds for in-flight requests to finish.
    def _handle_shutdown(signum: int, _frame: object) -> None:
        log.info("Shutdown signal received (sig=%d), stopping server (grace=5s)...", signum)
        server.stop(grace=5)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
