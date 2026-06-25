# llm-pool.Dockerfile — thin omniswap router image for the cluster "monolith".
#
# Builds the omniswap binary from source (no GGUFs, no llama.cpp, no GPU) and
# ships it on distroless-static. This is the image k8s deploys as default/llm-pool;
# it supersedes the old stock-mostlygeek/llama-swap binary so the cluster gets
# the <model>@<node> addressing grammar and the dynamic `any` resolver.
#
# Build (from the omniswap repo root):
#   docker build -f docker/llm-pool.Dockerfile \
#     --build-arg GIT_HASH=$(git rev-parse --short HEAD) \
#     -t zot.ht.local/llm-pool:<tag> .
FROM golang:1.25 AS builder
WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
# Placeholder so the go:embed in proxy/ui_embed.go is satisfied without a full
# UI build — llm-pool serves the API surface, not the bundled web UI.
RUN mkdir -p proxy/ui_dist && touch proxy/ui_dist/placeholder.txt
ARG GIT_HASH=unknown
ARG BUILD_DATE=unknown
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 go build \
    -ldflags="-X main.commit=${GIT_HASH} -X main.version=llmpool_${GIT_HASH} -X main.date=${BUILD_DATE}" \
    -o /omniswap

# Runtime: distroless static. The binary is CGO_ENABLED=0 (fully static), the
# pool runs peer-mode only (no child llama-server, so no shell needed), and both
# deployments use httpGet probes (no exec/curl), so debian + curl bought nothing.
# distroless/static is ~2MB (vs debian-slim ~74MB + apt layers) → ~70MB smaller
# image, which matters: this is pushed to the gem zot over the flaky WiFi diode,
# where fewer/smaller blobs = faster, more reliable skopeo copies. ca-certificates
# are bundled in distroless/static (parity for any HTTPS peer). Runs as nonroot.
FROM gcr.io/distroless/static-debian12:nonroot
# Keep the binary named llama-swap so the existing Deployment CMD is unchanged.
COPY --from=builder /omniswap /usr/local/bin/llama-swap
EXPOSE 8080
# Config mounted at /config/llamaswap.yaml via ConfigMap.
ENTRYPOINT ["/usr/local/bin/llama-swap"]
CMD ["--config", "/config/llamaswap.yaml", "--listen", "0.0.0.0:8080"]
