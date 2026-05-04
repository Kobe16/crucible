# crucible

[![CI](https://github.com/Kobe16/crucible/actions/workflows/ci.yml/badge.svg)](https://github.com/Kobe16/crucible/actions/workflows/ci.yml)

Distributed GPU Inference Engine — a Go API gateway that forwards inference requests to a Python/PyTorch worker via gRPC.

## Architecture

```
Client
  │
  │  HTTP POST /predict
  ▼
┌──────────────────────────────────┐
│   Gateway (Go)             :8080 │
│   ┌─────────┐    ┌─────────────┐ │
│   │ Handler │ ─▶ │   Batcher   │ │  groups concurrent requests
│   └─────────┘    └──────┬──────┘ │  by size or timeout
└──────────────────────────┼───────┘
                           │  gRPC BatchInference
                           ▼
                  ┌─────────────────────┐
                  │   Worker (Python)   │  :50051
                  │   PyTorch + gRPC    │
                  │   DistilBERT SST-2  │
                  └─────────────────────┘
```

The gateway accepts one HTTP request per client, but groups concurrent requests
into a single gRPC `BatchInference` call. A batch fires as soon as either
`MAX_BATCH_SIZE` requests have accumulated or `BATCH_TIMEOUT_MS` has elapsed
since the first request of the current batch arrived.

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

# 4. Check status
curl http://localhost:8080/status

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

### Running tests and linters

**Go** (from `gateway/`):

```bash
cd gateway
go test -v -race ./...    # run tests with race detector
go vet ./...              # static analysis
test -z "$(gofmt -l .)"  # formatting check
```

**Python** (from `worker/`):

```bash
cd worker
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r ../requirements-dev.txt
pytest -v tests/          # run tests
ruff check .              # linter
```

**GPU migration**: see comments in `worker/requirements.txt` for the steps required to switch from CPU to CUDA.

## Testing the Worker Directly

The worker has [gRPC server reflection](https://grpc.github.io/grpc/core/md_doc_server_reflection_tutorial.html) enabled, so you can use `grpcurl` without a local proto file:

```bash
# Install grpcurl (macOS)
brew install grpcurl

# Standard health check (grpc.health.v1) — NOT_SERVING during model load, then SERVING
grpcurl -plaintext localhost:50051 grpc.health.v1.Health/Check

# Application-level worker status
grpcurl -plaintext localhost:50051 inference.InferenceService/GetWorkerStatus

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

## First-run note

On first startup, the worker downloads the model weights
(`distilbert-base-uncased-finetuned-sst-2-english`, ~250MB) from
HuggingFace and caches them in a Docker volume (`model-cache`).

Observed startup times on an M2 MacBook Air:
- **First run** (download + load): ~15.6s for the worker container to become healthy
- **Subsequent runs** (cached): ~5.6s for the worker container to become healthy

The `start_period: 60s` healthcheck setting leaves generous headroom for
slower connections on first run.

To clear the cache and force a re-download:
​```bash
docker volume rm crucible_model-cache
​```

## Configuration

Both services are configured via environment variables, which can be overridden in `docker-compose.yml` or passed directly when running containers locally.

| Variable | Service | Default | Description |
|---|---|---|---|
| `HTTP_PORT` | gateway | `8080` | Gateway listen port |
| `WORKER_ADDR` | gateway | `worker:50051` | Worker gRPC address |
| `LOG_LEVEL` | gateway | `info` | Log level (debug/info/warn/error) |
| `MAX_BATCH_SIZE` | gateway | `8` | Max requests per batch before flushing |
| `BATCH_TIMEOUT_MS` | gateway | `50` | Max time (ms) to wait for a batch to fill before flushing |
| `QUEUE_DEPTH` | gateway | `1000` | Maximum buffered requests in the batcher queue |
| `GRPC_PORT` | worker | `50051` | Worker gRPC listen port |
| `MAX_WORKERS` | worker | `4` | gRPC thread pool size |
| `MODEL_NAME` | worker | `distilbert-base-uncased-finetuned-sst-2-english` | HuggingFace model to load |
| `USE_CUSTOM_KERNEL` | worker | `false` | Enable custom CUDA/Triton softmax kernel (Sprint 5) |

## Project Structure

```
crucible/
├── proto/                       # gRPC service definition (inference.proto)
├── gateway/                     # Go HTTP→gRPC gateway
│   ├── go.mod
│   ├── cmd/gateway/
│   │   └── main.go              # Entry point: wires config → inference.Client → batcher → handler → HTTP server
│   └── internal/
│       ├── config/config.go     # Env var loading (incl. MAX_BATCH_SIZE, BATCH_TIMEOUT_MS, QUEUE_DEPTH)
│       ├── inference/client.go  # gRPC connection wrapper (BatchInference, CheckHealth, GetWorkerStatus)
│       ├── batcher/             # Dynamic request batcher: queue + goroutine that flushes by size or timeout
│       └── handler/handler.go   # HTTP handlers (POST /predict, GET /health, GET /status)
├── worker/                   # Python gRPC inference server
│   ├── server.py             # Entry point (serve(), signal handling, model loading)
│   ├── servicer.py           # gRPC servicer (RunInference, BatchInference, GetWorkerStatus)
│   ├── model_runner.py       # DistilBERT model loading and batch inference
│   ├── config.py             # Env var parsing
│   ├── requirements.txt
│   └── Dockerfile
├── docker-compose.yml
└── Makefile
```
