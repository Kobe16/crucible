from unittest.mock import MagicMock, patch

import torch

from model_runner import ModelRunner


class _FakeBatchEncoding(dict):
    """Dict subclass with .to() to mimic transformers BatchEncoding."""

    def to(self, device):
        return self


def _make_runner(logits):
    """Create a ModelRunner with mocked model and tokenizer, using real torch tensors."""
    mock_tokenizer = MagicMock()
    # Return a dict-like object with .to() to mimic BatchEncoding
    mock_tokenizer.return_value = _FakeBatchEncoding({
        "input_ids": torch.ones(logits.shape[0], 5, dtype=torch.long),
        "attention_mask": torch.ones(logits.shape[0], 5, dtype=torch.long),
    })

    mock_model = MagicMock()
    mock_model.return_value = MagicMock(logits=logits)
    mock_model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model

    with (
        patch("model_runner.AutoTokenizer.from_pretrained", return_value=mock_tokenizer),
        patch("model_runner.AutoModelForSequenceClassification.from_pretrained", return_value=mock_model),
        patch("model_runner.torch.cuda") as mock_cuda,
        patch("model_runner.torch.backends") as mock_backends,
    ):
        mock_cuda.is_available.return_value = False
        mock_backends.mps.is_available.return_value = False
        runner = ModelRunner()

    return runner


def test_predict_single():
    # logits: strongly POSITIVE (index 1)
    logits = torch.tensor([[-2.0, 3.0]])
    runner = _make_runner(logits)

    results = runner.predict(["great movie"])

    assert len(results) == 1
    assert results[0]["label"] == "POSITIVE"
    assert results[0]["score"] > 0.9


def test_predict_batch():
    # first input: POSITIVE, second input: NEGATIVE
    logits = torch.tensor([[-2.0, 3.0], [3.0, -2.0]])
    runner = _make_runner(logits)

    results = runner.predict(["great", "terrible"])

    assert len(results) == 2
    assert results[0]["label"] == "POSITIVE"
    assert results[1]["label"] == "NEGATIVE"
    assert results[0]["score"] > 0.9
    assert results[1]["score"] > 0.9
