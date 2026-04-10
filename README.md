# crucible

Distributed GPU Inference Engine — a Go API gateway that forwards inference requests to a Python/PyTorch worker via gRPC.

## Architecture

```
Client
  │
  │  HTTP POST /predict
  ▼
┌─────────────────────┐
│   Gateway (Go)      │  :8080
│   net/http + zerolog│
└─────────┬───────────┘
          │  gRPC BatchInference RPC
          ▼
┌─────────────────────┐
│   Worker (Python)   │  :50051
│   PyTorch + gRPC    │
│   DistilBERT SST-2  │
└─────────────────────┘
```

## Prerequisites

- [Docker](https://docs.docker.com/get-docker/) + Docker Compose v2
- [Make](https://www.gnu.org/software/make/)
- *(optional, for local dev)* Go 1.22+, Python 3.11+

## Quickstart

```bash
# 1. Generate gRPC stubs (requires Docker)
make proto

# 2. Build images and start the stack
make up

# 3. Check health
curl http://localhost:8080/health

# 4. Run inference
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"input": "I love this!"}'
```

## Development

```bash
make logs    # tail all service logs
make down    # stop containers
make clean   # remove generated stubs + containers + images
make proto   # regenerate proto stubs
```

**GPU migration**: see comments in `worker/requirements.txt` for the steps required to switch from CPU to CUDA.

## Testing the Worker Directly

The worker has [gRPC server reflection](https://grpc.github.io/grpc/core/md_doc_server_reflection_tutorial.html) enabled, so you can use `grpcurl` without a local proto file:

```bash
# Install grpcurl (macOS)
brew install grpcurl

# Health check — returns DOWN during model load (~30-45s), then OK
grpcurl -plaintext localhost:50051 inference.InferenceService/HealthCheck

# Single inference
grpcurl -plaintext \
  -d '{"request_id": "r1", "input": "great movie"}' \
  localhost:50051 inference.InferenceService/RunInference
# → {"requestId": "r1", "output": "POSITIVE (0.9987)"}

# Batch inference
grpcurl -plaintext \
  -d '{"requests": [{"request_id": "r1", "input": "great"}, {"request_id": "r2", "input": "terrible"}]}' \
  localhost:50051 inference.InferenceService/BatchInference

# List all available RPCs
grpcurl -plaintext localhost:50051 list inference.InferenceService
```

## Configuration

Both services are configured via environment variables, which can be overridden in `docker-compose.yml` or passed directly when running containers locally.

| Variable | Service | Default | Description |
|---|---|---|---|
| `HTTP_PORT` | gateway | `8080` | Gateway listen port |
| `WORKER_ADDR` | gateway | `worker:50051` | Worker gRPC address |
| `LOG_LEVEL` | gateway | `info` | Zerolog level (debug/info/warn/error) |
| `GRPC_PORT` | worker | `50051` | Worker gRPC listen port |
| `MAX_WORKERS` | worker | `4` | gRPC thread pool size |
| `MODEL_NAME` | worker | `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace model to load |
| `USE_CUSTOM_KERNEL` | worker | `false` | Enable custom CUDA/Triton softmax kernel (Sprint 5) |

## Project Structure

```
crucible/
├── proto/                    # gRPC service definition (inference.proto)
├── gateway/                  # Go HTTP→gRPC gateway
│   ├── go.mod
│   └── cmd/gateway/
│       └── main.go
├── worker/                   # Python gRPC inference server
│   ├── server.py             # Entry point (serve(), signal handling, model loading)
│   ├── servicer.py           # gRPC servicer (RunInference, BatchInference, HealthCheck)
│   ├── model_runner.py       # DistilBERT model loading and batch inference
│   ├── config.py             # Env var parsing
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── Makefile
```
