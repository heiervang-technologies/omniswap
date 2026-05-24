# Dynamic `any` Model Alias

## Overview

Add a synthetic model id `any` that omniswap resolves at request time to whatever peer currently has any model loaded. Today this routing decision lives in the heierchat client (see snoop-kube + heierchat coord-thread, 2026-05-24): the picker scans `nodesStore.runtime.loaded` and rewrites the model id locally before dispatch. That works for one client but means non-heierchat callers (curl, OpenAI SDKs hitting `llm.ht.local`) get nothing, and the routing decision is invisible from the cluster's perspective.

Moving the alias into omniswap gives every client the behaviour and centralises the policy where the peer-state truth lives.

## Current State

- `proxy/peerproxy.go:24` — `proxyMap[modelID]*peerProxyMember` is built once at startup from the static `peers:` config block. Duplicate model ids across peers get warned + skipped (first wins).
- `proxy/peerproxy.go:127` — `ProxyRequest(model_id, ...)` is a literal-string lookup in that map. No notion of per-peer loaded state at the omniswap layer — loaded state lives only on each peer's `/v1/models`.

## Design Requirements

### 1. Loaded-state cache

Omniswap needs a live view of each peer's loaded set. Trade-offs:

- **Per-request `/v1/models` poll**: simplest, but adds ~50–200 ms across LAN × N peers to the first byte of every `any` request. Not acceptable.
- **Short-TTL cache** (5–10 s): right call. Staleness bounded to one missed model swap on the heaviest peer; in practice swaps take longer than 10 s anyway.

The cache should:

- Refresh in the background on a fixed interval (proposed: `loaded_state_refresh_seconds`, default 5).
- Refresh out-of-band on peer health-change events if peerproxy is wired into the monitor system.
- Be readable from `ProxyRequest` without contention (`sync.RWMutex` + map snapshot).
- Tolerate peer failures: a peer that times out during refresh keeps its previous snapshot for one cycle, then drops to "no loaded models" on the next miss.

### 2. Virtual-alias resolution in `ProxyRequest`

When `model_id == "any"`:

1. Read the loaded-state cache.
2. Walk peers in priority order (see §3).
3. Pick the first peer whose loaded set is non-empty.
4. Pick the first loaded model on that peer (deterministic ordering — by model id alphabetical, or by config-declared order).
5. Rewrite `model_id` to the concrete loaded id before continuing the existing dispatch logic.

The rewrite has to happen *before* the body is forwarded — `llama-server` validates the `model` field in chat-completion bodies and 400s on `"any"`. The proxy should also rewrite the JSON request body's `model` field to match.

When the cache reports zero loaded peers, return `503` with a body explaining the alias couldn't resolve. Don't fall back to "first peer that hosts anything" — that would silently trigger a load round-trip, which is what `any` is supposed to avoid.

### 3. Priority + pinning policy

Today heierchat hard-codes the rank as `titan > centurion > other > lithium` (favouring discrete-GPU peers over the iGPU laptop). That doesn't belong in the client.

Add config:

```yaml
peers:
  titan:
    proxy: http://192.168.8.158:30184
    priority: 0   # lower = preferred (default: 100)
  centurion:
    proxy: http://192.168.8.170:30192
    priority: 1
  lithium:
    proxy: http://192.168.8.119:30187
    priority: 100  # last resort
```

The `any` resolver sorts by `priority` ascending, breaking ties by config-declared order.

For per-request user pinning, accept an `X-Omniswap-Peer-Hint: <peer-name>` header. When present and the named peer has loaded models, jump it to the front of the priority list for that request only. Unknown peer name → ignore the hint (don't error).

### 4. Config surface

Add a top-level block:

```yaml
virtual_aliases:
  any:
    enabled: true
    refresh_seconds: 5
    require_priority_config: true   # if false, fall back to config order
```

Default `enabled: false` to keep upgrades surprise-free. The schema (`config-schema.json`) needs matching additions.

## Testing

- `TestPeerProxy_ResolveAny_NoLoadedPeers` — empty cache → 503.
- `TestPeerProxy_ResolveAny_SingleLoadedPeer` — exactly one peer loaded → rewritten id matches.
- `TestPeerProxy_ResolveAny_PriorityOrder` — two peers loaded, lower priority wins.
- `TestPeerProxy_ResolveAny_HeaderHint` — `X-Omniswap-Peer-Hint` overrides priority.
- `TestPeerProxy_ResolveAny_HeaderHintColdPeer` — hint points to a peer with nothing loaded → falls through to priority.
- `TestPeerProxy_ResolveAny_BodyRewrite` — incoming body `{"model":"any",...}` is rewritten before forwarding.
- `TestPeerProxy_LoadedCache_Refresh` — cache picks up a newly-loaded model within `refresh_seconds + ε`.
- `TestPeerProxy_LoadedCache_PeerOffline` — peer that fails to respond keeps last snapshot, then empties on next failure.

## Non-Goals

- Cross-peer fan-out (replying with the fastest of N peers) — different feature, separate proposal.
- Load-balancing across multiple loaded peers — first-by-priority is enough; users who want explicit selection use the normal pin.
- Persisting the loaded-state cache across omniswap restarts — peer `/v1/models` is the source of truth, recompute on startup.

## Out of Scope for v1

- Multiple virtual aliases (`any-vision`, `any-coder`, etc). The framework should be extensible (a `virtual_aliases` map keyed by alias name) but v1 only ships `any`.

## Client Migration

Heierchat keeps its client-side "Any (loaded)" picker shortcut for now — that resolves to a concrete model id at click time, which is a different UX (the conversation records the resolved id, not `"any"`). Once dynamic-any lands cluster-side, heierchat adds a *sticky-any* option: store `"any"` as the conversation's model literal and let omniswap resolve per-turn. The two UX entries can coexist.

## References

- `proxy/peerproxy.go:24` — `proxyMap` build.
- `proxy/peerproxy.go:127` — `ProxyRequest` entry.
- heierchat `webui/src/lib/stores/nodes.svelte.ts:541` — current client-side picker logic for reference (priority order encoded here).
- snoop-kube ↔ heierchat coordination thread, 2026-05-24 (heiervang-technologies team chat).
