FROM python:3.11-slim

WORKDIR /app

# gen/ is on PYTHONPATH so 'from proto import ...' works in server.py and
# the generated stubs resolve their own internal imports without sys.path hacks.
ENV PYTHONPATH=/app/gen

# Install dependencies before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install grpc_health_probe for container health checks
ARG TARGETARCH
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates \
    && wget -qO /bin/grpc_health_probe \
       https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.25/grpc_health_probe-linux-${TARGETARCH} \
    && chmod +x /bin/grpc_health_probe \
    && apt-get purge -y wget \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy generated gRPC stubs and application source
COPY gen/ ./gen/
COPY *.py .

EXPOSE 50051

CMD ["python", "server.py"]
