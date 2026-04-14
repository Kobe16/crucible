from typing import TypedDict

import structlog
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import MODEL_NAME, USE_CUSTOM_KERNEL

log = structlog.get_logger()


class InferenceResult(TypedDict):
    """A single classification result returned by :meth:`ModelRunner.predict`."""

    label: str
    score: float


class ModelRunner:
    """
    Class to load model and its tokenizer, run inference on batched
    inputs, and return outputs. The predict() method is designed to
    be called by the worker's gRPC server, which will handle
    batching and concurrency.

    Note: This model runner uses a sequence classification model
    for sentiment analysis.
    """
    def __init__(self):
        """Load the model and tokenizer, and set the device (GPU or CPU).

        This is called once when the worker starts up to avoid loading
        model on every inference request. There is option to swap in a
        custom GPU kernel for the softmax operation.
        """
        # Device priority: CUDA > Apple Silicon MPS > CPU.
        # Note: requirements.txt uses torch+cpu (Linux/Docker). MPS activates only
        # when running locally on Apple Silicon with a standard (non-+cpu) PyTorch install.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load model, tokenizer, and label mapping (maps indices to human-readable labels)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = (
            AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            .to(self.device)
            .eval()
        )
        self.id2label = self.model.config.id2label

        # TODO: replace F.softmax with custom kernel for better performance on GPU.
        if USE_CUSTOM_KERNEL:
            if not torch.cuda.is_available():
                log.warning("custom_kernel_unavailable", reason="CUDA not available, falling back to F.softmax")
            else:
                raise RuntimeError(
                    "USE_CUSTOM_KERNEL=true but no custom kernel is installed yet. "
                    "Implement cuda_kernels/ in Sprint 5."
                )
        self._softmax = F.softmax

    def predict(self, inputs: list[str]) -> list[InferenceResult]:
        """Run inference on a batch of text inputs and return predicted labels and scores.

        Args:
            inputs (list[str]): List of input strings to classify.

        Returns:
            list[InferenceResult]: One result per input, each with:
                - ``label`` (str): Predicted class label (e.g. ``"POSITIVE"``).
                - ``score`` (float): Confidence score in ``[0.0, 1.0]``.
        """
        encoded = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=512,
        ).to(self.device)

        # Run in inference mode instead of no_grad() to skip gradient
        # calculation, view tracking, & version counting
        with torch.inference_mode():
            logits = self.model(**encoded).logits

        # Convert logits -> probabilities -> top scores & corresponding label indices for each input
        probs = self._softmax(logits, dim=-1)
        scores, indices = torch.max(probs, dim=-1)
        scores, indices = scores.tolist(), indices.tolist()

        return [
            {"label": self.id2label[idx], "score": score}
            for idx, score in zip(indices, scores)
        ]
