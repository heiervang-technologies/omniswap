# Aggregated /v1/models: enrich peer entries with live status

**Filed:** 2026-05-24 by snoop-kube
**Status:** Spec — not yet implemented
**Related to:** [[2026-05-24-dynamic-any-alias.md]] — both need per-peer loaded-state polling infra

## Problem

`proxy/proxymanager.go:445-525` `listModelsHandler` builds the aggregated `/v1/models` response by:
- iterating LOCAL models with their real per-process state via `getModelStatus()`, AND
- iterating each peer's STATIC `peer.Models` config slice as bare records with only `id`, `name`, `object`, `created`, `owned_by`, `meta.llamaswap.peerID`.

No HTTP call to the peer's `/v1/models` is made. So peer entries in the aggregated response carry **no `status` field at all** — clients can't tell which peer-served model is hot vs. cold from the aggregator alone.

Confirmed:

```
$ curl -H 'Host: llm.ht.local' http://192.168.8.170/v1/models | jq '.data[] | select(.id=="gemma-4-26b")'
{
  "created": 1779641861,
  "id": "gemma-4-26b",
  "meta": {"llamaswap": {"peerID": "titan"}},
  "name": "titan: gemma-4-26b",
  "object": "model",
  "owned_by": "llama-swap"
}
# no `status` field
```

Whereas direct per-peer hit returns the full status object with `status.value` ∈ `{loaded, unloaded, loading, ...}` and `status.args` + `status.preset`.

## Impact

heierchat (and other multi-peer clients) currently work around this by hitting each peer's `/v1/models` individually on launch — 3s timeout × N peers serialised, observable launch lag. The picker UX depends on per-peer hot-vs-cold dots that should come from one aggregator call.

## Proposed change

Enrich peer entries with live status fetched at request-time, behind a short-TTL cache.

### Sketch

`proxy/peerproxy.go` — add a thin cached peer-models fetcher:

```go
type peerModelsCache struct {
    mu      sync.RWMutex
    entries map[string]peerModelsCacheEntry  // keyed by peerID
}
type peerModelsCacheEntry struct {
    data      []gin.H   // raw /v1/models[i] objects from peer
    fetchedAt time.Time
}

func (p *PeerProxy) GetPeerModels(peerID string, ttl time.Duration) ([]gin.H, error) {
    // RLock, check TTL, return cached
    // else Lock, fetch peer.ProxyURL/v1/models with the existing peerTransport
    // (so it inherits ResponseHeaderTimeout per the other ai-plan),
    // parse data array, store, return
}
```

`proxy/proxymanager.go:listModelsHandler` — replace the static peer-models block with:

```go
if pm.peerProxy != nil {
    for peerID, peer := range pm.peerProxy.ListPeers() {
        peerModels, err := pm.peerProxy.GetPeerModels(peerID, 5 * time.Second)
        if err != nil {
            // Peer unreachable — fall back to static config slice with no status,
            // log a Warn. UX degrades gracefully (clients see model exists but
            // can't tell loaded state) rather than failing the whole request.
            for _, modelID := range peer.Models {
                record := newRecord(modelID, config.ModelConfig{
                    Name:     fmt.Sprintf("%s: %s", peerID, modelID),
                    Metadata: map[string]any{"peerID": peerID, "peerStatus": "unreachable"},
                })
                data = append(data, record)
            }
            continue
        }
        for _, m := range peerModels {
            // Merge: take fields from peer response, attach peerID
            record := gin.H{}
            for k, v := range m { record[k] = v }
            if meta, ok := record["meta"].(gin.H); ok {
                meta["llamaswap"] = gin.H{"peerID": peerID}
            } else {
                record["meta"] = gin.H{"llamaswap": gin.H{"peerID": peerID}}
            }
            data = append(data, record)
        }
    }
}
```

### Cache TTL choice

5 seconds is the sweet spot: short enough that model swaps surface within one polling round in a UI, long enough that bursts of `/v1/models` calls (heierchat's launch + every NodeSwitch open) don't flood peers.

### Failure mode

If a peer is unreachable, return the static config list for that peer with `meta.llamaswap.peerStatus: "unreachable"`. The client UI can dim that peer's entries rather than dropping them entirely (better than the current "always show, can't tell why"). This also unblocks the offline-lithium case that surfaces during travel laptop sleep.

## Test plan

- `TestProxyManager_AggregatedModels_PeerStatusEnriched` — set up a mock peer responding with a model in `status.value: loaded`, verify aggregator response includes that field on the matching entry.
- `TestProxyManager_AggregatedModels_PeerUnreachable` — peer returns 502 / DNS fail, verify aggregator returns the static models with `peerStatus: "unreachable"` rather than 500ing.
- `TestProxyManager_AggregatedModels_Cache` — two consecutive `/v1/models` requests within TTL = one peer fetch. After TTL = two peer fetches. Use httptest.Server with a counter.
- `TestProxyManager_AggregatedModels_NoPeers` — empty peers config = behaviour unchanged (regression check).

## Open questions

1. **Cache invalidation on swap.** Probably fine to ignore — 5s TTL is short enough that user-perceived staleness is invisible. If a peer swap completes mid-second, the next aggregator hit is fresh. Worst case 5s staleness on the dot.
2. **Concurrent peer fetches.** Should fan out in parallel (errgroup or sync.WaitGroup) so a slow peer doesn't serialize the whole response. With 3-4 peers and a 1-2s per-peer cap, parallel = ~2s worst-case latency vs ~8s serial.

## Cluster-side follow-up

Once this lands and ships in a new omniswap image, the heierchat client at `heiervang-technologies/heierchat` can drop its per-peer launch-time probing of `/v1/models` and rely on the aggregated endpoint alone — eliminates 3s × N peer-probe latency on launch. Pings will go to %7 (heierchat agent).

## Why not push status onto the existing static slice without fetching?

The whole point is that `status.value` is dynamic per-process state, not config. The aggregator already knows the model exists from the static config slice; what it doesn't know is whether the model is currently resident in VRAM on that peer. Only the peer knows.
