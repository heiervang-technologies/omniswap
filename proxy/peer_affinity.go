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

// defaultAffinityMinBodyBytes is the request size below which affinity is not
// attempted.
//
// The claim "reusing a prefix beats prefilling it at any throughput" is only
// true when the queue wait is shorter than the prefill it saves. That holds
// comfortably for a 37k-token prefix and not at all for a 400-token prompt with
// a long decode, which has nothing worth saving and would pay the rank
// distortion anyway. A body floor separates the two cheaply. A growing
// conversation crosses it once and is bound from then on. (astra, PR #38 pass 2.)
const defaultAffinityMinBodyBytes = 2048

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
	// minBodyBytes is the request size below which affinity is not attempted.
	// A small request has little prefix to reuse, so binding it to a peer pays
	// the rank distortion for nothing. See defaultAffinityMinBodyBytes.
	minBodyBytes int
	// keyCompletions extends affinity to single-shot /v1/completions. Off by
	// default: a FIM code-completion prompt changes on every keystroke, so each
	// request gets a fresh key and affinity degenerates from least-loaded into
	// hash-random placement — measured 108/106/86 over 300 single-shot picks
	// across three peers. Bounded, but pure loss. (astra, PR #38 pass 2.)
	keyCompletions bool
}

// affinityStats counts what affinity DECIDED. The three spill reasons are kept
// apart because they are entirely different operational signals: cold means the
// fleet is not holding the model where conversations want it, capped means
// admission is the binding constraint, busy means the discount is mistuned.
//
// What these counters CANNOT tell you is whether affinity is working. A stable
// key whose prefix keeps changing underneath it — a volatile system prompt, RAG
// context injected above the user turn — reads as 100% hits while every request
// still prefills from scratch. Only the upstream's own
// usage.prompt_tokens_details.cached_tokens can confirm reuse; these counters
// tell you where requests WENT, not what they found when they got there.
// (Distinction drawn by astra, PR #38 review pass 2.)
type affinityStats struct {
	hits         atomic.Int64 // affine peer chosen
	spillsCold   atomic.Int64 // affine peer does not hold the model (bias != 0)
	spillsCapped atomic.Int64 // affine peer at its admission cap, another has room
	spillsBusy   atomic.Int64 // discount exhausted against a less loaded peer
	noKey        atomic.Int64 // no stable key in the request
}

// AffinityStats returns (hits, spillsCold, spillsCapped, spillsBusy, noKey) for
// operator visibility.
func (p *PeerProxy) AffinityStats() (int64, int64, int64, int64, int64) {
	return p.affinityStats.hits.Load(),
		p.affinityStats.spillsCold.Load(),
		p.affinityStats.spillsCapped.Load(),
		p.affinityStats.spillsBusy.Load(),
		p.affinityStats.noKey.Load()
}

// affinityKeyFromRequest prefers an explicit session header when the operator has
// configured one, falling back to the body-derived conversation seed.
func (a affinityConfig) affinityKeyFromRequest(header http.Header, body []byte) string {
	// An explicit session id is authoritative and is honoured at any size: the
	// client has told us these requests belong together.
	for _, h := range a.headers {
		if v := strings.TrimSpace(header.Get(h)); v != "" {
			sum := sha256.Sum256(append([]byte("h:"), v...))
			return string(sum[:])
		}
	}
	if len(body) < a.minBodyBytes {
		return ""
	}
	return affinityKeyFromBody(body, a.keyCompletions)
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
//
// The deeper reason the first user message is the right choice, rather than
// merely a low-collision one: the key is a literal SUBSTRING OF THE CACHED
// PREFIX. So key stability and prefix stability are the same property. When RAG
// context is injected into the first user turn, or a sliding window trims it,
// the key rehashes — and that is correct, because the prefix genuinely changed
// and there was no cache left to keep. The key tracks the thing it is standing
// in for. (astra, PR #38 pass 2.)
//   - It must not be TRUNCATED anywhere a caller controls the prefix. Capping the
//     seed and appending the system message first means any client with a system
//     prompt longer than the cap contributes zero bytes of anything
//     conversation-specific, so every one of that client's conversations hashes
//     identically and pins to a single gem. Hashing 150 KB costs ~150µs against a
//     60s prefill; the cap bought nothing and risked exactly the failure this
//     feature exists to prevent. (Caught in review by astra, 2026-09-18.)
//
// Collisions do occur in practice — canned starter-prompt chips in a chat UI,
// title/tag-generation sub-requests with a fixed opener — and co-locating them
// is arguably the right answer rather than a tolerated cost, since those
// requests really do share a prefix.
//
// Returns "" when there is nothing stable to key on, which disables affinity for
// that request rather than inventing a key: an empty or unparseable body must
// fall through to the load-aware pick, not collide with every other keyless one.
func affinityKeyFromBody(body []byte, keyCompletions bool) string {
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
	if seed == "" && keyCompletions {
		// /v1/completions and friends: no messages, one prompt. Opt-in — see
		// affinityConfig.keyCompletions for why this is off by default.
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
