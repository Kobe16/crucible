from unittest.mock import MagicMock

import grpc
from proto import inference_pb2 as pb2

from servicer import InferenceServicer


def test_batch_inference_success(servicer, grpc_context):
    req = pb2.BatchRequest(
        requests=[pb2.InferenceRequest(request_id="r1", input="great movie")]
    )
    resp = servicer.BatchInference(req, grpc_context)

    assert len(resp.responses) == 1
    assert resp.responses[0].request_id == "r1"
    assert resp.responses[0].output == "POSITIVE (0.9500)"


def test_batch_inference_multiple_inputs(servicer, mock_runner, grpc_context):
    mock_runner.predict.return_value = [
        {"label": "POSITIVE", "score": 0.95},
        {"label": "NEGATIVE", "score": 0.87},
        {"label": "POSITIVE", "score": 0.62},
    ]
    req = pb2.BatchRequest(
        requests=[
            pb2.InferenceRequest(request_id="r1", input="good"),
            pb2.InferenceRequest(request_id="r2", input="bad"),
            pb2.InferenceRequest(request_id="r3", input="ok"),
        ]
    )
    resp = servicer.BatchInference(req, grpc_context)

    assert len(resp.responses) == 3
    assert resp.responses[0].output == "POSITIVE (0.9500)"
    assert resp.responses[1].output == "NEGATIVE (0.8700)"
    assert resp.responses[2].output == "POSITIVE (0.6200)"


def test_batch_inference_empty_request(servicer, grpc_context):
    req = pb2.BatchRequest(requests=[])
    servicer.BatchInference(req, grpc_context)

    grpc_context.set_code.assert_called_once_with(grpc.StatusCode.INVALID_ARGUMENT)


def test_batch_inference_model_not_ready(grpc_context):
    svc = InferenceServicer(health_servicer=None)
    req = pb2.BatchRequest(
        requests=[pb2.InferenceRequest(request_id="r1", input="test")]
    )
    svc.BatchInference(req, grpc_context)

    grpc_context.set_code.assert_called_once_with(grpc.StatusCode.UNAVAILABLE)


def test_batch_inference_model_loading(grpc_context):
    svc = InferenceServicer(health_servicer=None)
    svc.set_loading()
    req = pb2.BatchRequest(
        requests=[pb2.InferenceRequest(request_id="r1", input="test")]
    )
    svc.BatchInference(req, grpc_context)

    grpc_context.set_code.assert_called_once_with(grpc.StatusCode.UNAVAILABLE)


def test_run_inference_delegates_to_batch(servicer, grpc_context):
    req = pb2.InferenceRequest(request_id="r1", input="hello")
    resp = servicer.RunInference(req, grpc_context)

    assert resp.request_id == "r1"
    assert resp.output == "POSITIVE (0.9500)"


def test_get_worker_status_ok(servicer, grpc_context):
    resp = servicer.GetWorkerStatus(pb2.WorkerStatusRequest(), grpc_context)
    assert resp.status == pb2.SERVING_STATUS_OK


def test_get_worker_status_unknown(grpc_context):
    svc = InferenceServicer(health_servicer=None)
    resp = svc.GetWorkerStatus(pb2.WorkerStatusRequest(), grpc_context)
    assert resp.status == pb2.SERVING_STATUS_UNKNOWN


def test_error_tracking_degrades_after_threshold(grpc_context):
    svc = InferenceServicer(health_servicer=None)
    runner = MagicMock()
    runner.predict.side_effect = RuntimeError("boom")
    svc.set_ready(runner)

    req = pb2.BatchRequest(
        requests=[pb2.InferenceRequest(request_id="r1", input="test")]
    )
    for _ in range(3):
        svc.BatchInference(req, grpc_context)

    resp = svc.GetWorkerStatus(pb2.WorkerStatusRequest(), grpc_context)
    assert resp.status == pb2.SERVING_STATUS_DEGRADED


def test_error_tracking_recovers_on_success(mock_runner, grpc_context):
    svc = InferenceServicer(health_servicer=None)
    failing_runner = MagicMock()
    failing_runner.predict.side_effect = RuntimeError("boom")
    svc.set_ready(failing_runner)

    req = pb2.BatchRequest(
        requests=[pb2.InferenceRequest(request_id="r1", input="test")]
    )
    for _ in range(3):
        svc.BatchInference(req, grpc_context)

    # Now swap in a working runner and verify recovery
    svc.runner = mock_runner
    svc.BatchInference(req, grpc_context)

    resp = svc.GetWorkerStatus(pb2.WorkerStatusRequest(), grpc_context)
    assert resp.status == pb2.SERVING_STATUS_OK


def test_error_tracking_resets_before_threshold(mock_runner, grpc_context):
    svc = InferenceServicer(health_servicer=None)
    failing_runner = MagicMock()
    failing_runner.predict.side_effect = RuntimeError("boom")
    svc.set_ready(failing_runner)

    req = pb2.BatchRequest(
        requests=[pb2.InferenceRequest(request_id="r1", input="test")]
    )

    # 2 errors, then 1 success, then 2 more errors — never hits 3 consecutive
    svc.BatchInference(req, grpc_context)
    svc.BatchInference(req, grpc_context)
    svc.runner = mock_runner
    svc.BatchInference(req, grpc_context)
    svc.runner = failing_runner
    svc.BatchInference(req, grpc_context)
    svc.BatchInference(req, grpc_context)

    resp = svc.GetWorkerStatus(pb2.WorkerStatusRequest(), grpc_context)
    assert resp.status == pb2.SERVING_STATUS_OK
