"""
gRPC server entry point for the Crucible inference worker.

Starts the server, loads the model, and handles graceful shutdown.
The InferenceServicer implementation lives in servicer.py.
"""

import logging
import signal
import time
from concurrent import futures

import grpc
from grpc_reflection.v1alpha import reflection
from proto import inference_pb2 as pb2
from proto import inference_pb2_grpc as pb2_grpc

from config import GRPC_PORT, MAX_WORKERS
from model_runner import ModelRunner
from servicer import InferenceServicer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
log = logging.getLogger("worker.server")


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
