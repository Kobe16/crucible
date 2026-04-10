import os

GRPC_PORT = int(os.getenv("GRPC_PORT", "50051"))
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "4"))
MODEL_NAME = os.getenv("MODEL_NAME", "distilbert-base-uncased-finetuned-sst-2-english")
USE_CUSTOM_KERNEL = os.getenv("USE_CUSTOM_KERNEL", "false").lower() == "true"
