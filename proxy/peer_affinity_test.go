package proxy

import (
	"encoding/hex"
	"fmt"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// turn builds a chat body whose FIRST USER message is fixed, with `extra`
// further turns appended. Affinity keys on that first user message, so every
// turn of one conversation has to produce the same key.
func turn(system, firstUser string, extra ...string) []byte {
	body := `{"model":"m","messages":[{"role":"system","content":"` + system +
		`"},{"role":"user","content":"` + firstUser + `"}`
	for i, e := range extra {
		role := "assistant"
		if i%2 == 1 {
			role = "user"
		}
		body += `,{"role":"` + role + `","content":"` + e + `"}`
	}
	return []byte(body + `]}`)
}

func TestAffinityKeyFromBody(t *testing.T) {
	t.Run("stable as the conversation grows", func(t *testing.T) {
		k1 := affinityKeyFromBody(turn("you are helpful", "explain rendezvous hashing"))
		k2 := affinityKeyFromBody(turn("you are helpful", "explain rendezvous hashing",
			"It maps keys to nodes.", "and what about modulo?"))
		k3 := affinityKeyFromBody(turn("you are helpful", "explain rendezvous hashing",
			"It maps keys to nodes.", "and what about modulo?",
			"Modulo reshuffles everything.", "thanks"))
		require.NotEmpty(t, k1)
		assert.Equal(t, k1, k2, "turn 2 must hash to the conversation's key, not a new one")
		assert.Equal(t, k1, k3, "turn 3 likewise")
	})

	t.Run("different opener is a different conversation", func(t *testing.T) {
		assert.NotEqual(t,
			affinityKeyFromBody(turn("you are helpful", "question A")),
			affinityKeyFromBody(turn("you are helpful", "question B")),
			"a shared system prompt must not collapse every conversation onto one peer")
	})

	// The system prompt is deliberately NOT part of the key. Clients rebuild it
	// per request with injected volatile context (timestamp, cwd, open files); a
	// key that moves every turn scatters the conversation across the fleet while
	// still producing a healthy-looking stream of affinity hits. Keying on the
	// first user message trades a rare harmless collision for immunity to that.
	t.Run("volatile system prompt does not move the conversation", func(t *testing.T) {
		k1 := affinityKeyFromBody(turn("today is 2026-09-18T19:04:11Z cwd=/a", "start work"))
		k2 := affinityKeyFromBody(turn("today is 2026-09-18T19:07:52Z cwd=/b", "start work"))
		assert.Equal(t, k1, k2,
			"a timestamped system prompt must not rehash the conversation every turn")
	})

	// Regression test for the bug this nearly shipped with: the seed was capped at
	// 4096 bytes and the system message was appended FIRST, so any client with a
	// system prompt at or over the cap contributed zero bytes of anything
	// conversation-specific — every conversation from that client hashed
	// identically and pinned to one gem. (Caught in review by astra, 2026-09-18.)
	t.Run("a huge system prompt cannot collapse distinct conversations", func(t *testing.T) {
		huge := strings.Repeat("x", 64*1024)
		seen := map[string]bool{}
		for i := 0; i < 20; i++ {
			seen[affinityKeyFromBody(turn(huge, fmt.Sprintf("conversation %d", i)))] = true
		}
		assert.Len(t, seen, 20, "distinct conversations collapsed to %d keys", len(seen))
	})

	t.Run("a huge first user message is not truncated into a collision", func(t *testing.T) {
		big := strings.Repeat("y", 200*1024)
		a := affinityKeyFromBody(turn("sys", big+"ending A"))
		b := affinityKeyFromBody(turn("sys", big+"ending B"))
		assert.NotEqual(t, a, b, "the distinguishing bytes must survive, wherever they sit")
	})

	t.Run("no stable seed yields no key", func(t *testing.T) {
		// Each of these must fall through to the load-aware pick rather than share
		// one key — a shared fallback key would pin ALL of them to one peer.
		for name, body := range map[string][]byte{
			"empty":            {},
			"not json":         []byte("<html>"),
			"no messages":      []byte(`{"model":"m"}`),
			"empty messages":   []byte(`{"model":"m","messages":[]}`),
			"system only":      []byte(`{"model":"m","messages":[{"role":"system","content":"s"}]}`),
			"assistant only":   []byte(`{"model":"m","messages":[{"role":"assistant","content":"hi"}]}`),
			"messages not arr": []byte(`{"model":"m","messages":"nope"}`),
		} {
			assert.Empty(t, affinityKeyFromBody(body), "case %q", name)
		}
	})

	t.Run("completions prompt is a usable seed", func(t *testing.T) {
		k := affinityKeyFromBody([]byte(`{"model":"m","prompt":"once upon a time"}`))
		assert.NotEmpty(t, k)
		assert.NotEqual(t, k, affinityKeyFromBody([]byte(`{"model":"m","prompt":"different"}`)))
	})

	t.Run("multimodal content keys without being a string", func(t *testing.T) {
		body := []byte(`{"model":"m","messages":[{"role":"user","content":[{"type":"text","text":"describe"},{"type":"image_url","image_url":{"url":"data:x"}}]}]}`)
		assert.NotEmpty(t, affinityKeyFromBody(body),
			"content as an array must still produce a seed")
	})
}

func TestAffinityKeyFromRequest(t *testing.T) {
	body := turn("sys", "hello")
	hdr := http.Header{}
	hdr.Set("X-Session-Id", "sess-42")

	t.Run("header overrides the body when configured", func(t *testing.T) {
		a := affinityConfig{headers: []string{"X-Session-Id"}}
		assert.NotEqual(t, affinityKeyFromBody(body), a.affinityKeyFromRequest(hdr, body))
	})

	t.Run("two requests with the same session id share a key", func(t *testing.T) {
		a := affinityConfig{headers: []string{"X-Session-Id"}}
		other := turn("sys", "a completely different opener")
		assert.Equal(t, a.affinityKeyFromRequest(hdr, body), a.affinityKeyFromRequest(hdr, other))
	})

	t.Run("falls back to the body when the header is absent or blank", func(t *testing.T) {
		a := affinityConfig{headers: []string{"X-Session-Id"}}
		blank := http.Header{}
		blank.Set("X-Session-Id", "   ")
		assert.Equal(t, affinityKeyFromBody(body), a.affinityKeyFromRequest(blank, body))
		assert.Equal(t, affinityKeyFromBody(body), a.affinityKeyFromRequest(http.Header{}, body))
	})

	t.Run("unconfigured headers are ignored", func(t *testing.T) {
		assert.Equal(t, affinityKeyFromBody(body),
			affinityConfig{}.affinityKeyFromRequest(hdr, body))
	})
}

// TestAffinityHashIsPinned nails the hash to a constant. Every pool in the fleet
// must map a conversation to the SAME peer — that agreement is the entire reason
// affinity survives a tunnel that picks a different entry pool per request. A
// change to the seed construction or the digest silently desyncs a mixed-version
// fleet, turning every rolling deploy into a cache flush with nothing failing
// visibly. So the expectation is hard-coded, not recomputed from the code under
// test: a self-inverting assertion would pass against any implementation.
func TestAffinityHashIsPinned(t *testing.T) {
	key := affinityKeyFromBody(turn("you are helpful", "explain rendezvous hashing"))
	assert.Equal(t, "f97f9b94900dc5dec6c03bf38ed083f4e89917af3136ef887c65541422ee0297", hex.EncodeToString([]byte(key)),
		"conversation-seed hashing changed: a mixed-version fleet will disagree about "+
			"which peer owns a conversation, flushing every prompt cache")
	assert.Equal(t, uint64(0xc4c92c2079babe61), affinityScore(key, "opal"),
		"rendezvous weighting changed: same consequence as above")
}

func affinityTestProxy(t *testing.T, bonus int) (*PeerProxy, map[string]*peerProxyMember) {
	t.Helper()
	now := time.Now()
	warm := peerLoadedSet{all: map[string]bool{"m": true}, order: []string{"m"},
		served: map[string]bool{"m": true}, fetchedAt: now}
	a := &peerProxyMember{peerID: "a"}
	b := &peerProxyMember{peerID: "b"}
	c := &peerProxyMember{peerID: "c"}
	p := &PeerProxy{
		peers:       config.PeerDictionaryConfig{"a": {}, "b": {}, "c": {}},
		modelPeers:  map[string][]*peerProxyMember{"m": {a, b, c}},
		loadedCache: map[string]peerLoadedSet{"a": warm, "b": warm, "c": warm},
		loadedTTL:   time.Hour,
		logger:      testLogger,
		admission:   newPeerAdmission(0, 0, []string{"a", "b", "c"}),
	}
	p.setAffinity(true, bonus, nil)
	return p, map[string]*peerProxyMember{"a": a, "b": b, "c": c}
}

func TestAffinityRouting(t *testing.T) {
	// Pinned, not derived: "conv-one" belongs to peer c. Deriving the
	// expectation from affinePeer would make these tests agree with any
	// implementation, including a broken one.
	const keyToC = "conv-one"

	t.Run("pinned key ownership", func(t *testing.T) {
		p, m := affinityTestProxy(t, 4)
		assert.Equal(t, m["c"], affinePeer(keyToC, p.modelPeers["m"]))
	})

	t.Run("affinity beats an idler", func(t *testing.T) {
		// All three warm and equal, so the load-aware pick would take "a" (first on
		// a tie). Affinity must send this conversation elsewhere.
		p, m := affinityTestProxy(t, 4)
		assert.Equal(t, m["c"], p.pickPeerForModelWithAffinity("m", keyToC))
	})

	t.Run("every turn of a conversation lands on the same peer", func(t *testing.T) {
		p, m := affinityTestProxy(t, 4)
		for i := 0; i < 5; i++ {
			got := p.pickPeerForModelWithAffinity("m", keyToC)
			require.Equal(t, m["c"], got, "turn %d", i)
			atomic.AddInt64(&got.inFlight, -1) // request completed
		}
	})

	t.Run("holds the affine peer while the discount covers the gap", func(t *testing.T) {
		// bonus 4 = tolerate two extra in-flight. Reusing a 37k prefix beats
		// prefilling it at any throughput, so queueing behind one or two requests
		// on the warm peer is still the cheaper answer.
		p, m := affinityTestProxy(t, 4)
		for _, n := range []int64{1, 2} {
			atomic.StoreInt64(&m["c"].inFlight, n)
			assert.Equal(t, m["c"], p.pickPeerForModelWithAffinity("m", keyToC),
				"at %d in-flight the discount still covers the gap", n)
			atomic.AddInt64(&m["c"].inFlight, -1) // undo the reservation
		}
	})

	t.Run("yields once the discount is exhausted", func(t *testing.T) {
		p, m := affinityTestProxy(t, 4)
		atomic.StoreInt64(&m["c"].inFlight, 3) // rank 6, best 0, 6-4 > 0
		assert.Equal(t, m["a"], p.pickPeerForModelWithAffinity("m", keyToC),
			"must fall back rather than pile onto a saturated GPU")
	})

	t.Run("never routes to an unreachable peer for cache locality", func(t *testing.T) {
		p, m := affinityTestProxy(t, 4)
		p.loadedCache["c"] = peerLoadedSet{served: map[string]bool{}, fetchedAt: time.Now()}
		assert.NotEqual(t, m["c"], p.pickPeerForModelWithAffinity("m", keyToC),
			"a slow answer beats no answer")
	})

	// Admission is live on the gems (maxInflightPerPeer 4, queueTimeout 10s), so
	// affinity that feeds a peer at its admission cap converts a slow answer into
	// a 429 while another gem sits idle. (Raised in review by astra, 2026-09-18.)
	t.Run("never feeds a peer that admission would reject", func(t *testing.T) {
		p, m := affinityTestProxy(t, 40) // bonus large enough to win on rank alone
		p.setAdmission(2, time.Second)
		atomic.StoreInt64(&m["c"].inFlight, 2) // at the admission cap
		got := p.pickPeerForModelWithAffinity("m", keyToC)
		assert.NotEqual(t, m["c"], got, "would 429 while another peer has room")
	})

	t.Run("still uses the affine peer when every peer is capped", func(t *testing.T) {
		p, m := affinityTestProxy(t, 40)
		p.setAdmission(2, time.Second)
		for _, id := range []string{"a", "b", "c"} {
			atomic.StoreInt64(&m[id].inFlight, 2)
		}
		assert.Equal(t, m["c"], p.pickPeerForModelWithAffinity("m", keyToC),
			"when nothing has room, affinity is no worse than the alternative")
	})

	t.Run("empty key falls through to the load-aware pick", func(t *testing.T) {
		p, m := affinityTestProxy(t, 4)
		assert.Equal(t, m["a"], p.pickPeerForModelWithAffinity("m", ""))
	})

	t.Run("disabled is identical to the load-aware pick", func(t *testing.T) {
		p, _ := affinityTestProxy(t, 4)
		p.setAffinity(false, 0, nil)
		var got []string
		for i := 0; i < 6; i++ {
			got = append(got, p.pickPeerForModelWithAffinity("m", keyToC).peerID)
		}
		assert.Equal(t, []string{"a", "b", "c", "a", "b", "c"}, got,
			"with affinity off the reservation fan-out must be untouched")
	})

	t.Run("counters separate a hit from a spill from a keyless request", func(t *testing.T) {
		// Without these an operator cannot tell working affinity from a key that
		// rehashes every turn: both produce a busy, healthy-looking system.
		p, m := affinityTestProxy(t, 4)
		p.pickPeerForModelWithAffinity("m", keyToC) // hit
		atomic.StoreInt64(&m["c"].inFlight, 9)
		p.pickPeerForModelWithAffinity("m", keyToC) // spill
		p.pickPeerForModelWithAffinity("m", "")     // no key
		hits, spills, noKey := p.AffinityStats()
		assert.EqualValues(t, 1, hits)
		assert.EqualValues(t, 1, spills)
		assert.EqualValues(t, 1, noKey)
	})
}

// TestAffinityCannotStarveTheFleet is the guard against the failure this feature
// could plausibly introduce: one hot conversation monopolising a single GPU while
// the rest of the fleet idles. The invariant is that the chosen peer's ORDINARY
// rank never exceeds the best available ordinary rank by more than the bonus —
// which is what expressing affinity as a discount (rather than a threshold) buys.
func TestAffinityCannotStarveTheFleet(t *testing.T) {
	const bonus = 4
	p, m := affinityTestProxy(t, bonus)
	const key = "conv-one"

	var wg sync.WaitGroup
	picks := make([]string, 30)
	for i := range picks {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			picks[i] = p.pickPeerForModelWithAffinity("m", key).peerID
		}(i)
	}
	wg.Wait()

	counts := map[string]int{}
	for _, id := range picks {
		counts[id]++
	}
	for _, id := range []string{"a", "b", "c"} {
		assert.Positive(t, counts[id], "peer %s idled while %v was chosen", id, counts)
	}

	var lo, hi int64 = 1 << 30, 0
	var total int64
	for _, id := range []string{"a", "b", "c"} {
		n := atomic.LoadInt64(&m[id].inFlight)
		total += n
		lo, hi = min(lo, n), max(hi, n)
	}
	assert.EqualValues(t, len(picks), total, "every pick must reserve exactly once")
	// Ranks are inFlight*2 (all peers warm, bias 0), so the rank gap the bonus
	// permits translates to an in-flight spread of bonus/2, plus one for the
	// request that was in flight when the discount was last still covering.
	assert.LessOrEqual(t, hi-lo, int64(bonus/2+1),
		"in-flight spread %d exceeds what the bonus permits (%v)", hi-lo, counts)
}

// TestAffinityDistributesConversations checks the other direction: distinct
// conversations must spread. A hash that bunched them would turn affinity into a
// static one-peer pin.
func TestAffinityDistributesConversations(t *testing.T) {
	p, _ := affinityTestProxy(t, 4)
	candidates := p.modelPeers["m"]
	counts := map[string]int{}
	const n = 600
	for i := 0; i < n; i++ {
		key := affinityKeyFromBody(turn("you are helpful", fmt.Sprintf("conversation %d", i)))
		counts[affinePeer(key, candidates).peerID]++
	}
	for _, id := range []string{"a", "b", "c"} {
		assert.Greater(t, counts[id], n/6,
			"peer %s got %d/%d conversations — distribution is lopsided", id, counts[id], n)
	}
}

// TestAffinityIsStableAcrossPools is the fleet property. Each gem runs its own
// pool process; the tunnel picks which one a turn enters through. Affinity only
// works if independently constructed pools agree on the owner.
func TestAffinityIsStableAcrossPools(t *testing.T) {
	// Deliberately built in two different orders, to prove the result does not
	// depend on slice position.
	sorted := []*peerProxyMember{{peerID: "amber"}, {peerID: "diamond"},
		{peerID: "opal"}, {peerID: "ruby"}}
	shuffled := []*peerProxyMember{{peerID: "opal"}, {peerID: "ruby"},
		{peerID: "diamond"}, {peerID: "amber"}}

	for i := 0; i < 50; i++ {
		key := affinityKeyFromBody(turn("sys", fmt.Sprintf("conversation %d", i)))
		assert.Equal(t, affinePeer(key, sorted).peerID, affinePeer(key, shuffled).peerID,
			"two pools disagreed about who owns conversation %d", i)
	}
}

// TestAffinityRemapsOnlyTheLostPeer is why this is rendezvous hashing and not
// `hash % len(peers)`. Losing one gem must not relocate the conversations bound
// to the other three — that would flush every warm prefix in the fleet.
func TestAffinityRemapsOnlyTheLostPeer(t *testing.T) {
	full := []*peerProxyMember{{peerID: "ruby"}, {peerID: "amber"},
		{peerID: "diamond"}, {peerID: "opal"}}
	reduced := full[:3] // opal drops out

	var moved, stayed, wasOpal int
	for i := 0; i < 400; i++ {
		key := affinityKeyFromBody(turn("sys", fmt.Sprintf("conversation %d", i)))
		before := affinePeer(key, full).peerID
		after := affinePeer(key, reduced).peerID
		switch {
		case before == "opal":
			wasOpal++
		case before != after:
			moved++
		default:
			stayed++
		}
	}
	assert.Positive(t, wasOpal, "test is vacuous unless some conversations were on opal")
	assert.Zero(t, moved, "%d conversations not on the lost peer were remapped anyway", moved)
	assert.Positive(t, stayed)
}
