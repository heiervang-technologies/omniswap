package proxy

import (
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
)

// loaded-state fixtures, matching TestPickPeerForModel.
func schedFixtures(now time.Time) (warm, vacant, other, unreachable peerLoadedSet) {
	warm = peerLoadedSet{all: map[string]bool{"m": true}, order: []string{"m"}, served: map[string]bool{"m": true}, fetchedAt: now}
	vacant = peerLoadedSet{all: map[string]bool{}, order: nil, served: map[string]bool{"m": true}, fetchedAt: now}
	other = peerLoadedSet{all: map[string]bool{"x": true}, order: []string{"x"}, served: map[string]bool{"m": true, "x": true}, fetchedAt: now}
	unreachable = peerLoadedSet{served: map[string]bool{}, fetchedAt: now} // poll failed
	return
}

// newSchedProxy builds an a/b/c PeerProxy the same way NewPeerProxy would (sorted
// peer order, memberByPeer populated) plus its GemsRanker, for parity testing
// against pickPeerForModel.
func newSchedProxy() (*PeerProxy, map[string]*peerProxyMember, *GemsRanker) {
	a := &peerProxyMember{peerID: "a"}
	b := &peerProxyMember{peerID: "b"}
	c := &peerProxyMember{peerID: "c"}
	p := &PeerProxy{
		peers:        config.PeerDictionaryConfig{"a": {}, "b": {}, "c": {}},
		modelPeers:   map[string][]*peerProxyMember{"m": {a, b, c}},
		memberByPeer: map[string]*peerProxyMember{"a": a, "b": b, "c": c},
		loadedCache:  map[string]peerLoadedSet{},
		loadedTTL:    time.Hour, // pre-seeded cache never expires during the test
	}
	return p, map[string]*peerProxyMember{"a": a, "b": b, "c": c}, NewGemsRanker(p)
}

// TestGemsRanker_ParityWithPickPeerForModel is the load-bearing stage-1 claim:
// GemsRanker.Rank's best candidate is EXACTLY pickPeerForModel's pick, across
// every scenario the legacy selection is tested on. Rank is called first (it is
// a pure read); pickPeerForModel then reserves. Same pre-state -> same winner.
func TestGemsRanker_ParityWithPickPeerForModel(t *testing.T) {
	now := time.Now()
	warm, vacant, other, unreachable := schedFixtures(now)

	cases := []struct {
		name      string
		a, b, c   peerLoadedSet
		inflightA int64
	}{
		{name: "warm beats vacant and other", a: vacant, b: warm, c: other},
		{name: "no warm: vacant beats occupied", a: other, b: vacant, c: other},
		{name: "warm but busy spills to vacant", a: warm, b: vacant, c: vacant, inflightA: 1},
		{name: "warm idle beats warm busy", a: warm, b: warm, c: vacant, inflightA: 2},
		{name: "equal rank breaks to first", a: warm, b: warm, c: vacant},
		{name: "unreachable deprioritized below reachable", a: unreachable, b: vacant, c: unreachable},
		{name: "unreachable beaten by warm", a: unreachable, b: unreachable, c: warm},
		{name: "all unreachable still returns one", a: unreachable, b: unreachable, c: unreachable},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			p, m, r := newSchedProxy()
			p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = tc.a, tc.b, tc.c
			atomic.StoreInt64(&m["a"].inFlight, tc.inflightA)

			ranked := r.Rank("m")
			assert.NotEmpty(t, ranked, "Rank must return candidates")
			rankTop := ranked[0].Node.Name

			pick := p.pickPeerForModel("m") // reserves; called AFTER the pure Rank
			assert.Equal(t, pick.peerID, rankTop,
				"GemsRanker.Rank top (%s) must equal pickPeerForModel pick (%s)", rankTop, pick.peerID)
		})
	}
}

// TestGemsRanker_FullOrdering checks the whole slice is legacy-key ordered
// (warm < vacant < occupied), not just the winner.
func TestGemsRanker_FullOrdering(t *testing.T) {
	now := time.Now()
	warm, vacant, other, _ := schedFixtures(now)
	p, _, r := newSchedProxy()
	// a=vacant(1), b=warm(0), c=other(3)  -> ordered b, a, c
	p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = vacant, warm, other

	ranked := r.Rank("m")
	got := []string{ranked[0].Node.Name, ranked[1].Node.Name, ranked[2].Node.Name}
	assert.Equal(t, []string{"b", "a", "c"}, got)
}

// TestGemsRanker_EmptyForUnknownModel: no peers -> nil (nothing can host).
func TestGemsRanker_EmptyForUnknownModel(t *testing.T) {
	_, _, r := newSchedProxy()
	assert.Nil(t, r.Rank("absent"))
}

// TestGemsRanker_RankKeyPopulated: the RankKey fields are filled for the home
// side + observability even though gems order by the legacy key (DECIDE 1).
func TestGemsRanker_RankKeyPopulated(t *testing.T) {
	now := time.Now()
	warm, _, other, unreachable := schedFixtures(now)
	p, m, r := newSchedProxy()
	p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = warm, other, unreachable
	atomic.StoreInt64(&m["a"].inFlight, 3)

	byNode := map[string]Fit{}
	for _, cand := range r.Rank("m") {
		byNode[cand.Node.Name] = cand.Fit
	}

	// a: warm + 3 in-flight
	assert.True(t, byNode["a"].Fits)
	assert.Equal(t, 1, byNode["a"].Score.Warm)
	assert.Equal(t, -3, byNode["a"].Score.Load)
	assert.Equal(t, gemVacantFreeMB, byNode["a"].Score.FreeMB)
	// b: a different model resident -> occupied, lower free proxy, not warm
	assert.Equal(t, 0, byNode["b"].Score.Warm)
	assert.Equal(t, gemOccupiedFreeMB, byNode["b"].Score.FreeMB)
	// c: unreachable -> Fits stays true (reachability is a rank signal), free 0, last
	assert.True(t, byNode["c"].Fits)
	assert.Equal(t, gemDeadFreeMB, byNode["c"].Score.FreeMB)
}

// TestGemsFitter_Fit: reachable peer that serves the model fits; unknown node
// does not.
func TestGemsFitter_Fit(t *testing.T) {
	now := time.Now()
	warm, _, _, _ := schedFixtures(now)
	p, _, _ := newSchedProxy()
	p.loadedCache["a"] = warm
	f := GemsFitter{p: p}

	fit := f.Fit("m", Node{Name: "a"})
	assert.True(t, fit.Fits)
	assert.Equal(t, 1, fit.Score.Warm) // warm

	miss := f.Fit("m", Node{Name: "nope"})
	assert.False(t, miss.Fits) // unknown node never fits
}

// TestRankKey_Cmp: the shared comparator is lexicographic, higher-is-better,
// Warm dominating. (The gems ordering does NOT use this yet — DECIDE 1 — but the
// home side + stage-3 do, so the contract comparator is pinned here.)
func TestRankKey_Cmp(t *testing.T) {
	// Warm dominates even a huge FreeMB gap.
	assert.Positive(t, RankKey{Warm: 1}.Cmp(RankKey{Warm: 0, FreeMB: 99999}))
	// Equal Warm -> more FreeMB wins.
	assert.Positive(t, RankKey{Warm: 1, FreeMB: 8000}.Cmp(RankKey{Warm: 1, FreeMB: 4000}))
	// Equal Warm+FreeMB -> higher Load (fewer in-flight) wins.
	assert.Positive(t, RankKey{FreeMB: 8000, Load: -1}.Cmp(RankKey{FreeMB: 8000, Load: -5}))
	// Full tie -> 0.
	assert.Zero(t, RankKey{Warm: 1, FreeMB: 8000, Load: -2, Local: 1}.Cmp(RankKey{Warm: 1, FreeMB: 8000, Load: -2, Local: 1}))
}

// TestNewPeerProxy_WiresInertRanker: NewPeerProxy constructs the ranker and
// exposes it, but it is not in the selection path (stage 1, dark).
func TestNewPeerProxy_WiresInertRanker(t *testing.T) {
	p, err := NewPeerProxy(config.PeerDictionaryConfig{}, testLogger)
	assert.NoError(t, err)
	assert.NotNil(t, p.SchedulerRanker())
}
