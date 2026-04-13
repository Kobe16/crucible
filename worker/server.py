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
import structlog
from grpc_health.v1 import health as grpc_health
from grpc_health.v1 import health_pb2 as health_pb2
from grpc_health.v1 import health_pb2_grpc as health_pb2_grpc
from grpc_reflection.v1alpha import reflection
from proto import inference_pb2 as pb2
from proto import inference_pb2_grpc as pb2_grpc

from config import GRPC_PORT, LOG_LEVEL, MAX_WORKERS
from model_runner import ModelRunner
from servicer import InferenceServicer


def _configure_logging() -> None:
    level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    # Keep stdlib basicConfig so gRPC's internal logs still work.
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    structlog.configure(
        processors=[
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


_configure_logging()
log = structlog.get_logger()


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

    # Standard grpc.health.v1 health servicer — starts as NOT_SERVING until model loads.
    health_servicer = grpc_health.HealthServicer()
    health_servicer.set("", health_pb2.HealthCheckResponse.NOT_SERVING)
    health_servicer.set(
        "inference.InferenceService",
        health_pb2.HealthCheckResponse.NOT_SERVING,
    )
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    servicer = InferenceServicer(health_servicer=health_servicer)
    pb2_grpc.add_InferenceServiceServicer_to_server(servicer, server)
    server.add_insecure_port(f"[::]:{GRPC_PORT}")

    service_names = (
        pb2.DESCRIPTOR.services_by_name["InferenceService"].full_name,
        "grpc.health.v1.Health",
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(service_names, server)

    server.start()
    log.info("grpc_server_started", port=GRPC_PORT)

    # Load model after starting server, raising exception to restart container
    log.info("model_loading")
    servicer.set_loading()
    t0 = time.perf_counter()
    try:
        runner = ModelRunner()
    except Exception:
        log.critical("model_load_failed", exc_info=True)
        server.stop(grace=0)
        raise

    servicer.set_ready(runner)
    log.info("model_ready", device=runner.device, load_time_s=round(time.perf_counter() - t0, 1))
    # Catch termination signals (SIGTERM/SIGINT) to gracefully shut down the server,
    # allowing 5 seconds for in-flight requests to finish.
    def _handle_shutdown(signum: int, _frame: object) -> None:
        log.info("shutdown_signal", sig=signum)
        server.stop(grace=5)

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    server.wait_for_termination()


if __name__ == "__main__":
    serve()
