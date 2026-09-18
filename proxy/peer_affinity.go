package proxy

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"net/http"
	"strings"
	"sync/atomic"

	"github.com/tidwall/gjson"
)

// Prompt-cache affinity for peer routing.
//
// pickPeerForModel ranks peers by `inFlight*2 + bias`, where bias is MODEL
// residency (warm = the model is loaded here). That is the right instinct for a
// cold model load, which happens once. It is the wrong instinct for prompt
// prefill, which happens on every request: when every peer already has the model
// resident every bias is 0, the rank collapses to pure least-in-flight, and each
// request is dispatched to the peer LEAST likely to hold its prefix.
//
// Measured on the gems fleet: the same 14,422-token prompt sent twice to one
// worker costs 59.82s then 0.30s (cached_tokens 0 then 14,421) — 198x. Sent to
// two different workers it costs 59.82s twice. A user whose turns round-robin
// across four gems never pays less than full prefill, and if their client times
// out first, the retry lands on a third gem and pays it again.
//
// So: hash a stable conversation seed to one peer and send every turn of that
// conversation there. Rendezvous (HRW) hashing is used rather than modulo so
// that losing a peer only remaps the conversations bound to the peer that left,
// instead of reshuffling all of them — a modulo ring would flush every warm
// prefix in the fleet each time a gem dropped out.
//
// Every pool in the fleet runs the same peer list and the same hash, so all of
// them independently map a conversation to the SAME peer. That is what makes
// this work despite the tunnel choosing a different ENTRY gem per request.
//
// The preference is a DISCOUNT on the existing rank, not a separate code path
// and not an absolute threshold. A peer is chosen for affinity only when its
// ordinary rank is within affinityBonus of the best available rank, which bounds
// how much load imbalance affinity can create — one hot conversation cannot park
// itself on a GPU while the others idle.

// defaultAffinityBonus is the rank discount granted to a conversation's peer.
//
// The ordinary rank is inFlight*2 + bias, so a bonus of 4 lets the affine peer
// win while it holds up to two more in-flight requests than the best
// alternative. That is deliberately sticky: reusing a 37k-token prefix beats
// prefilling it at ANY throughput, so queueing behind one or two requests on the
// warm peer is still far cheaper than a cold start elsewhere. Past that the
// discount is exhausted and the load-aware pick takes over.
const defaultAffinityBonus = 4

// affinityConfig is the resolved runtime setting. Zero value = disabled, which
// keeps the request path byte-identical to before (same land-dark convention as
// peer admission and the credit gate).
type affinityConfig struct {
	enabled bool
	bonus   int64
	// headers are request headers, in priority order, whose value is used as the
	// conversation key when present. An operator escape hatch for clients that do
	// carry a session id; empty (the default) means body-derived keys only.
	headers []string
}

// affinityStats counts what affinity actually did, because the difference
// between "working" and "thrashing" is invisible otherwise: a key that rehashes
// every turn (a system prompt with an injected timestamp, say) produces a
// perfectly healthy-looking stream of hits while never reusing a single prefix.
// Hits rising with prompt-eval time NOT falling is the thrash signature.
type affinityStats struct {
	hits   atomic.Int64 // affine peer chosen
	spills atomic.Int64 // affine peer known but too loaded / unreachable / capped
	noKey  atomic.Int64 // no stable key in the request
}

// AffinityStats returns (hits, spills, noKey) for operator visibility.
func (p *PeerProxy) AffinityStats() (int64, int64, int64) {
	return p.affinityStats.hits.Load(), p.affinityStats.spills.Load(), p.affinityStats.noKey.Load()
}

// affinityKeyFromRequest prefers an explicit session header when the operator has
// configured one, falling back to the body-derived conversation seed.
func (a affinityConfig) affinityKeyFromRequest(header http.Header, body []byte) string {
	for _, h := range a.headers {
		if v := strings.TrimSpace(header.Get(h)); v != "" {
			sum := sha256.Sum256(append([]byte("h:"), v...))
			return string(sum[:])
		}
	}
	return affinityKeyFromBody(body)
}

// affinityKeyFromBody derives a conversation key from an inference request body.
//
// It keys on the FIRST USER MESSAGE ONLY, in full. Three things pushed it there:
//
//   - It must be stable as the conversation grows, so it cannot include the
//     latest turn.
//   - It must not be the system prompt. System prompts are routinely rebuilt per
//     request with injected volatile context (timestamp, cwd, open files), and a
//     key that rehashes every turn scatters a conversation across the fleet —
//     strictly worse than no affinity, because it also looks like it is working.
//   - It must not be TRUNCATED anywhere a caller controls the prefix. Capping the
//     seed and appending the system message first means any client with a system
//     prompt longer than the cap contributes zero bytes of anything
//     conversation-specific, so every one of that client's conversations hashes
//     identically and pins to a single gem. Hashing 150 KB costs ~150µs against a
//     60s prefill; the cap bought nothing and risked exactly the failure this
//     feature exists to prevent. (Caught in review by astra, 2026-09-18.)
//
// The tradeoff accepted: two conversations that open with the same words share a
// peer. That is harmless — they merely queue together — whereas volatility and
// truncation are not.
//
// Returns "" when there is nothing stable to key on, which disables affinity for
// that request rather than inventing a key: an empty or unparseable body must
// fall through to the load-aware pick, not collide with every other keyless one.
func affinityKeyFromBody(body []byte) string {
	if len(body) == 0 {
		return ""
	}

	var tag string
	var seed string
	if messages := gjson.GetBytes(body, "messages"); messages.IsArray() {
		messages.ForEach(func(_, msg gjson.Result) bool {
			if msg.Get("role").String() != "user" {
				return true
			}
			// .Raw, not .String(), so multimodal content (an array of parts) keys
			// as well as plain text.
			seed, tag = msg.Get("content").Raw, "u:"
			return false
		})
	}
	if seed == "" {
		// /v1/completions and friends: no messages, one prompt.
		seed, tag = gjson.GetBytes(body, "prompt").Raw, "p:"
	}
	if seed == "" {
		return ""
	}

	h := sha256.New()
	h.Write([]byte(tag))
	// Length-prefixed so no concatenation of parts can be forged into another;
	// cheap insurance if a future key ever spans more than one field.
	var n [8]byte
	binary.BigEndian.PutUint64(n[:], uint64(len(seed)))
	h.Write(n[:])
	h.Write([]byte(seed))
	return string(h.Sum(nil))
}

// affinityScore is the rendezvous weight of one peer for one key. sha256 keeps
// the mapping identical across processes, restarts and hosts — every pool in the
// fleet must compute the same peer for the same conversation, or affinity buys
// nothing once the tunnel picks a different entry gem.
func affinityScore(key, peerID string) uint64 {
	h := sha256.New()
	h.Write([]byte(key))
	h.Write([]byte{0})
	h.Write([]byte(peerID))
	return binary.BigEndian.Uint64(h.Sum(nil)[:8])
}

// affinePeer returns the peer a key is bound to, among candidates, or nil when
// the key is empty. Ties (astronomically unlikely, but defined anyway) go to the
// lower peerID so the result never depends on slice order — two pools that list
// their peers in different orders must still agree.
func affinePeer(key string, candidates []*peerProxyMember) *peerProxyMember {
	if key == "" || len(candidates) == 0 {
		return nil
	}
	var best *peerProxyMember
	var bestScore uint64
	for _, pp := range candidates {
		score := affinityScore(key, pp.peerID)
		if best == nil || score > bestScore || (score == bestScore && pp.peerID < best.peerID) {
			best, bestScore = pp, score
		}
	}
	return best
}

// affinityKeyPrefix renders a key for logs. The key is raw binary; never log it
// whole, and never log the seed it came from — that is user prompt text.
func affinityKeyPrefix(key string) string {
	if key == "" {
		return "-"
	}
	if len(key) > 4 {
		key = key[:4]
	}
	return hex.EncodeToString([]byte(key))
}
