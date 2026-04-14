from unittest.mock import MagicMock

import grpc
import pytest

from model_runner import ModelRunner
from servicer import InferenceServicer


@pytest.fixture()
def mock_runner():
    runner = MagicMock(spec=ModelRunner)
    runner.predict.return_value = [{"label": "POSITIVE", "score": 0.9500}]
    return runner


@pytest.fixture()
def servicer(mock_runner):
    svc = InferenceServicer(health_servicer=None)
    svc.set_ready(mock_runner)
    return svc


@pytest.fixture()
def grpc_context():
    ctx = MagicMock(spec=grpc.ServicerContext)
    return ctx
