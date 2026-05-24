# Peer proxy: configurable ResponseHeaderTimeout

**Filed:** 2026-05-24 by snoop-kube (cloud-side maintainer)
**Status:** Spec — not yet implemented
**Related:** PR https://github.com/heiervang-technologies/cloud/pull/118 (lithium ht-fork Vulkan deployment) flagged this as a follow-up

## Problem

`proxy/peerproxy.go:45` hardcodes `ResponseHeaderTimeout: 60 * time.Second` on the peer transport. For non-streaming `/v1/chat/completions` requests where the peer doesn't write response headers until generation completes, any single-turn inference exceeding 60s through the pool yields a `502 Bad Gateway` from the reverse proxy and `net/http: timeout awaiting response headers` in the pool log.

The 60s ceiling is too low for our slowest peer (lithium-llm — AMD Radeon 780M iGPU via Vulkan, ~12-14 t/s decode). Concrete repro:

```
$ curl -X POST -H 'Host: llm.ht.local' http://192.168.8.170/v1/chat/completions \
    -d '{"model":"gemma-4-26b-128k","messages":[...long prompt...],
         "max_tokens":2500,"stream":false}'
# returns 502 after exactly 60.027s

# llm-pool log:
[WARN] peer lithium: proxy error: net/http: timeout awaiting response headers
[INFO] status=502, path=/v1/chat/completions, 1m0.027652298s
```

Same request to lithium directly (bypassing the pool) completes in ~165s with a valid response.

Streaming clients are unaffected — peers write headers as soon as the first token streams.

The existing comment at `proxy/peerproxy.go:38` literally says "these can be tuned with feedback later". This is the feedback.

## Proposed change

Add a top-level `peerResponseHeaderTimeout` config field with sensible default. Wire it into the transport.

### Config schema

`proxy/config/config.go`:

```go
type Config struct {
    HealthCheckTimeout          int `yaml:"healthCheckTimeout"`
    PeerResponseHeaderTimeout   int `yaml:"peerResponseHeaderTimeout"` // NEW
    // ...existing fields...
}
```

Defaults in `LoadConfigFromReader`:

```go
config := Config{
    HealthCheckTimeout:        120,
    PeerResponseHeaderTimeout: 600,   // NEW — 10 min, generous ceiling for slow Vulkan peers
    // ...
}
```

Validation:

```go
if config.PeerResponseHeaderTimeout < 15 {
    config.PeerResponseHeaderTimeout = 15
}
```

### Signature change

`proxy/peerproxy.go`:

```go
func NewPeerProxy(
    peers config.PeerDictionaryConfig,
    responseHeaderTimeout int,           // NEW
    proxyLogger *LogMonitor,
) (*PeerProxy, error) {
    // ...
    peerTransport := &http.Transport{
        DialContext: (&net.Dialer{...}).DialContext,
        TLSHandshakeTimeout:   10 * time.Second,
        ResponseHeaderTimeout: time.Duration(responseHeaderTimeout) * time.Second,  // CHANGED
        // ...
    }
}
```

Production caller at `proxy/proxymanager.go:140` updates to pass `proxyConfig.PeerResponseHeaderTimeout`. Four test callsites in `proxy/peerproxy_test.go` pass a sane test value (e.g., 60) explicitly.

## Test plan

- `TestProxyManager_PeerResponseHeaderTimeoutDefault` — config without the field gets 600s
- `TestProxyManager_PeerResponseHeaderTimeoutOverride` — config sets it to 30, transport reflects 30
- `TestProxyManager_PeerResponseHeaderTimeoutMinimum` — config sets it to 5, gets clamped to 15

Existing peer proxy tests stay green (pass a default value).

## Why config not hardcode bump

A configurable value beats a bigger hardcode for two reasons:

1) **Operator visibility.** A non-default value in `llamaswap.yaml` documents that this peer is slow, vs. the secret-knowledge magic constant in code.
2) **Tightening later is cheap.** If a future deployment wants 60s back (e.g., to fail fast on a dead peer), they set it without a code change.

## Open questions

None that should block landing the change.

## Cluster-side follow-up

After this lands and ships in a new omniswap image, the cluster-side ConfigMap (`k8s/ai/llm-pool.yaml` in `heiervang-technologies/cloud`) sets `peerResponseHeaderTimeout: 600` explicitly to make the slow-peer reality visible at the config layer.
