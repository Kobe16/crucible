FROM python:3.11-slim

WORKDIR /app

# gen/ is on PYTHONPATH so 'from proto import ...' works in server.py and
# the generated stubs resolve their own internal imports without sys.path hacks.
ENV PYTHONPATH=/app/gen

# Install dependencies before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy generated gRPC stubs and application source
COPY gen/ ./gen/
COPY config.py .
COPY model_runner.py .
COPY server.py .

EXPOSE 50051

CMD ["python", "server.py"]
