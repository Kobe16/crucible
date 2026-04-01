FROM golang:1.22-alpine AS builder

WORKDIR /app

COPY go.mod go.sum ./
RUN go mod download

# Copy generated gRPC stubs and application source
COPY gen/ ./gen/
COPY cmd/ ./cmd/

RUN CGO_ENABLED=0 GOOS=linux go build -o /gateway ./cmd/gateway

FROM alpine:3.20

RUN apk add --no-cache ca-certificates wget

WORKDIR /app

COPY --from=builder /gateway .

EXPOSE 8080

CMD ["./gateway"]
