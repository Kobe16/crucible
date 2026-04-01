.PHONY: up down proto build logs clean

PROTO_IMAGE := namely/protoc-all:1.51_4
PWD         := $(shell pwd)

## up: generate proto stubs, build images, and start the stack
up: proto build
	docker compose up -d

## down: stop and remove containers
down:
	docker compose down

## proto: generate Go and Python gRPC stubs from proto/inference.proto
proto:
	@mkdir -p gateway/gen worker/gen
	docker run --rm \
	  -v "$(PWD):/defs" \
	  $(PROTO_IMAGE) \
	  -f proto/inference.proto \
	  -l go \
	  -o gateway/gen \
	  --go-module-prefix github.com/Kobe16/crucible/gateway
	docker run --rm \
	  -v "$(PWD):/defs" \
	  $(PROTO_IMAGE) \
	  -f proto/inference.proto \
	  -l python \
	  -o worker/gen

## build: build Docker images without starting containers
build:
	docker compose build

## logs: tail logs from all services
logs:
	docker compose logs -f

## clean: remove generated stubs and tear down containers + images
clean:
	rm -rf gateway/gen worker/gen
	docker compose down --rmi local --volumes
