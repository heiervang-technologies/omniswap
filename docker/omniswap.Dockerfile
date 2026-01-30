# Stage 1: Build omniswap binary
FROM golang:1.25 AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Create placeholder for embedded UI (tests/build require it)
RUN mkdir -p proxy/ui_dist && touch proxy/ui_dist/placeholder.txt

ARG GIT_HASH=unknown
ARG BUILD_DATE=unknown
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-X main.commit=${GIT_HASH} -X main.version=docker_${GIT_HASH} -X main.date=${BUILD_DATE}" \
    -o /omniswap

# Stage 2: Runtime — vllm-omni base with omniswap added
ARG VLLM_OMNI_IMAGE=vllm/vllm-openai
ARG VLLM_OMNI_TAG=v0.14.0
FROM ${VLLM_OMNI_IMAGE}:${VLLM_OMNI_TAG}

WORKDIR /app

COPY --from=builder /omniswap /app/omniswap
COPY docker/omniswap-config.example.yaml /app/config.yaml

ENV PATH="/app:${PATH}"

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

ENTRYPOINT ["/app/omniswap", "-config", "/app/config.yaml"]
