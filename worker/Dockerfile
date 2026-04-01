FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying source so this layer is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy generated gRPC stubs and application source
COPY gen/ ./gen/
COPY server.py .

EXPOSE 50051

CMD ["python", "server.py"]
