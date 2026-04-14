"""
gRPC servicer for the Crucible inference worker.

Implements InferenceService (proto/inference.proto). Import resolution relies on
PYTHONPATH=/app/gen set in the Dockerfile, so the generated stubs and this module
both use 'from proto import ...' without any sys.path manipulation.
"""

import threading
import time
from typing import TYPE_CHECKING

import grpc
import structlog
from grpc_health.v1 import health_pb2 as health_pb2
from proto import inference_pb2 as pb2
from proto import inference_pb2_grpc as pb2_grpc

from model_runner import InferenceResult, ModelRunner

if TYPE_CHECKING:
    from grpc_health.v1 import health as grpc_health

log = structlog.get_logger()

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

    def __init__(
        self,
        health_servicer: "grpc_health.HealthServicer | None" = None,
    ) -> None:
        """Initialise the servicer in UNKNOWN state with no model loaded."""
        self._lock = threading.Lock()
        self._status: int = pb2.SERVING_STATUS_UNKNOWN  # ServingStatus proto enum (int constant)
        self.runner: ModelRunner | None = None
        self._consecutive_errors: int = 0
        self._health_servicer = health_servicer

    # ------------------------------------------------------------------
    # State transitions (called from serve())
    # ------------------------------------------------------------------

    def set_loading(self) -> None:
        """Transition to DOWN, signaling that model loading has started."""
        with self._lock:
            self._status = pb2.SERVING_STATUS_DOWN
            self._sync_health()

    def set_ready(self, runner: ModelRunner) -> None:
        """Transition to OK once the model has finished loading.

        Args:
            runner (ModelRunner): The initialised model runner to use for inference.
        """
        with self._lock:
            self.runner = runner
            self._consecutive_errors = 0
            self._status = pb2.SERVING_STATUS_OK
            self._sync_health()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _on_inference_success(self) -> None:
        """Reset error count and restore OK status after a successful inference."""
        with self._lock:
            self._consecutive_errors = 0
            self._status = pb2.SERVING_STATUS_OK
            self._sync_health()

    def _on_inference_error(self) -> None:
        """Increment error count and transition to DEGRADED if threshold is reached."""
        with self._lock:
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._ERROR_THRESHOLD:
                self._status = pb2.SERVING_STATUS_DEGRADED
                self._sync_health()

    def _sync_health(self) -> None:
        """Sync grpc.health.v1 status with current ServingStatus. Must hold self._lock."""
        if self._health_servicer is None:
            return
        if self._status in (pb2.SERVING_STATUS_OK, pb2.SERVING_STATUS_DEGRADED):
            status = health_pb2.HealthCheckResponse.SERVING
        else:
            status = health_pb2.HealthCheckResponse.NOT_SERVING
        self._health_servicer.set("", status)
        self._health_servicer.set("inference.InferenceService", status)

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
        # Snapshot shared state under lock so reads are synchronised.
        # Holding the lock through predict() would serialise all inference
        # calls, so we copy to locals and release before doing any work.
        # This is safe: runner is never unset once assigned, and status is
        # only used as a gate check before we proceed (don't need to check
        # if runner valid in mid-inference).
        with self._lock:
            status = self._status
            runner = self.runner

        # Early exit if model isn't ready
        if status in _NOT_READY_STATUSES:
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Model is not ready.")
            return pb2.BatchResponse()

        if not request.requests:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("BatchRequest must contain at least one request.")
            return pb2.BatchResponse()

        # Invariant: runner is set in set_ready() before status flips to OK,
        # so this guard should never trigger under normal operation.
        if runner is None:
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Runner is unexpectedly uninitialized.")
            return pb2.BatchResponse()

        # Extract inputs and run inference, handling any errors
        inputs: list[str] = [r.input for r in request.requests]
        request_ids: list[str] = [r.request_id for r in request.requests]
        bound_log = log.bind(request_ids=request_ids)
        t0 = time.perf_counter()
        try:
            results: list[InferenceResult] = runner.predict(inputs)
        except Exception:
            self._on_inference_error()
            bound_log.error("batch_inference_failed", exc_info=True)
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details("Inference failed.")
            return pb2.BatchResponse()

        # Handle successful inference and package results into response messages
        self._on_inference_success()
        latency_ms = (time.perf_counter() - t0) * 1000
        bound_log.info("batch_inference", batch_size=len(inputs), latency_ms=round(latency_ms, 1))

        responses: list[pb2.InferenceResponse] = [
            pb2.InferenceResponse(
                request_id=req.request_id,
                output=f"{res['label']} ({res['score']:.4f})",
            )
            for req, res in zip(request.requests, results)
        ]
        return pb2.BatchResponse(responses=responses)

    def GetWorkerStatus(
        self,
        request: pb2.WorkerStatusRequest,
        context: grpc.ServicerContext,
    ) -> pb2.WorkerStatusResponse:
        """Return the current serving status.

        Args:
            request (pb2.WorkerStatusRequest): Empty worker status request.
            context (grpc.ServicerContext): gRPC context (unused).

        Returns:
            pb2.WorkerStatusResponse: Response containing the current ServingStatus.
        """
        with self._lock:
            status = self._status
        return pb2.WorkerStatusResponse(status=status)
