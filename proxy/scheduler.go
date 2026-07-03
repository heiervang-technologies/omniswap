package proxy

import (
	"sort"
	"sync/atomic"
)

// scheduler.go — the serverless eligibility/ranking CONTRACT (cloud
// k8s/SERVERLESS-ELIGIBILITY-DESIGN.md, RFC #176/#177), stage 1.
//
// This lands the Fitter/Ranker interface plus the gems implementation, INERT:
// nothing here is wired into the request path yet. pickPeerForModel is
// unchanged — GemsRanker.Rank reproduces its selection ORDER (parity-tested in
// scheduler_test.go) so a later stage can drive selection through the interface
// without a routing change, behind a canary. snoop-kube's home Fitter/Ranker
// implement the SAME interface for the heterogeneous home cluster; the two
// converge at stage 3.
//
// Two DECIDEs this lift surfaced, deferred to stage 3 so stage 1 stays strictly
// behavior-neutral (see the PR):
//
//  1. COMPARATOR. Legacy pickPeerForModel ranks by a WEIGHTED SUM
//     (inFlight*2 + bias): a busy-warm peer YIELDS to a vacant GPU. The RankKey
//     below is LEXICOGRAPHIC (Warm dominates — a warm peer wins regardless of
//     load). They disagree under load. GemsRanker.Rank therefore orders by the
//     legacy key today (exact parity); RankKey is populated for the home side +
//     observability but is NOT yet the gems ordering key. Unifying on one
//     comparator — and whether warm should strictly dominate now that the T2
//     admission semaphore sheds overload as 429 rather than by pre-emptive
//     spill — is the stage-3 call.
//
//  2. RANKKEY EXPRESSIVENESS. Gems routing has FOUR states
//     (warm / vacant / other-model-resident / unreachable). Warm+Load alone
//     cannot distinguish vacant from other-resident from dead, so FreeMB is used
//     here as a COARSE vacancy proxy — no measured VRAM crosses the gems diode
//     (the documented capacity gap; home uses real DCGM). A dedicated cold-cost
//     field vs FreeMB-as-vacancy is a stage-3 shape decision.

// Node identifies a serving node. GpuIndex distinguishes GPUs on a multi-GPU
// home host (titan 2×3090); always 0 for the single-GPU gems. Matches #353.
type Node struct {
	Name     string
	GpuIndex int
}

// RankKey is an ordered, comparable ranking key. Separate fields (not one packed
// float) so a rank is debuggable, and gems (fills Warm+Load) and home (fills
// all) share ONE comparator via Cmp. NOTE (DECIDE 1): gems currently orders by
// the legacy weighted-sum, not Cmp — see the file header.
type RankKey struct {
	Warm   int // 1 if the model is already resident on the node (prefer a hot copy)
	FreeMB int // free VRAM after the fit (roomiest GPU wins); gems: coarse vacancy proxy
	Load   int // -inFlight (fewer in-flight ranks higher)
	Local  int // locality tiebreak
}

// Cmp orders two keys lexicographically on Warm, FreeMB, Load, Local (each
// higher-is-better): >0 if a ranks ahead of b, <0 if behind, 0 if equal.
func (a RankKey) Cmp(b RankKey) int {
	for _, d := range []int{a.Warm - b.Warm, a.FreeMB - b.FreeMB, a.Load - b.Load, a.Local - b.Local} {
		if d != 0 {
			return d
		}
	}
	return 0
}

// Fit is the auditable answer to "can node N host model M right now?". Weights/
// KV/Headroom are surfaced (not just Fits) so Fits and Score are debuggable, not
// a black box. Gems fill them coarsely; home computes them from GGUF + DCGM.
type Fit struct {
	Fits       bool    // node can host the model right now
	FreeVramMB int     // free VRAM on the node's GPU
	WeightsMB  int     // model weights
	KvMB       int     // SWA-aware KV at the lane's ctx + kv-quant
	HeadroomMB int     // safety margin reserved by the fit
	Score      RankKey // ranking key
}

// Fitter is the pluggable capacity signal — a PURE function (no side effects, no
// control-loop state) so it is trivially testable and prototyped standalone.
// GemsFitter and (home) HomeFitter both implement it.
type Fitter interface {
	Fit(model string, node Node) Fit
}

// Candidate pairs a node with its Fit for a requested model.
type Candidate struct {
	Node Node
	Fit  Fit
}

// Ranker returns the Fits==true candidates for a model, best-first. An empty
// slice means nothing can host it right now without a boot/evict (the stage-3
// model-admission decision, out of scope here).
type Ranker interface {
	Rank(model string) []Candidate
}

// Coarse VRAM constants for the homogeneous 16GB Pascal gems. Real per-gem free
// VRAM does NOT cross the gems diode (no home DCGM there — the documented
// capacity gap), so gems fit is a config fact + a coarse vacancy proxy, never a
// measurement. Home uses measured DCGM instead. These only need to be MONOTONIC
// with the routing preference (vacant > other-resident > dead), not accurate.
const (
	gemVacantFreeMB   = 16000 // free/warm GPU: cold-load lands cleanly (or already warm)
	gemOccupiedFreeMB = 4000  // a different model resident -> an evict would be needed
	gemDeadFreeMB     = 0     // peer unreachable (poll failing / stale)
)

// GemsFitter is the trivial Fitter for the homogeneous gems fleet: every LLM gem
// is an identical 16GB P5200 and the models we serve are pre-sized to fit
// (that's why the 256k lane is q4 KV). So Fit is "peer reachable && serves the
// model" with a coarse free-VRAM vacancy proxy; GpuIndex is always 0. Fits stays
// trivially-true even for an unreachable peer (reachability is a RANK signal,
// FreeMB=0, not a fit signal) so GemsRanker preserves pickPeerForModel's
// "pick a dead peer only if EVERY candidate is dead" behavior.
type GemsFitter struct {
	p *PeerProxy
}

// Fit implements Fitter for the gems fleet.
func (f GemsFitter) Fit(model string, node Node) Fit {
	pp, ok := f.p.memberByPeer[node.Name]
	if !ok {
		return Fit{Fits: false}
	}
	// bias: 0 warm, 1 vacant, 3 other-model resident, 6 unreachable.
	bias := f.p.peerModelBias(model, pp)
	warm, freeMB := 0, gemVacantFreeMB
	switch bias {
	case 0: // resident here -> warm
		warm = 1
	case 1: // nothing loaded -> vacant
		freeMB = gemVacantFreeMB
	case 3: // a different model resident -> an evict is needed
		freeMB = gemOccupiedFreeMB
	default: // unreachable
		freeMB = gemDeadFreeMB
	}
	load := -int(atomic.LoadInt64(&pp.inFlight))
	return Fit{
		Fits:       true, // gems fit is trivially-true for a reachable peer that serves the model
		FreeVramMB: freeMB,
		Score:      RankKey{Warm: warm, FreeMB: freeMB, Load: load},
	}
}

// GemsRanker lifts pickPeerForModel's selection ORDER behind the Ranker
// interface for the homogeneous gems fleet. Rank is a PURE read — no reserve, no
// mutation; the atomic reserve stays in pickPeerForModel. Rank orders by the
// LEGACY key (inFlight*2 + bias, lower wins) so its best candidate equals
// pickPeerForModel's pick (parity-tested); each Fit's RankKey is populated for
// the home side + observability (see DECIDE 1 in the file header).
type GemsRanker struct {
	p      *PeerProxy
	fitter GemsFitter
}

// NewGemsRanker builds the gems Ranker over a PeerProxy.
func NewGemsRanker(p *PeerProxy) *GemsRanker {
	return &GemsRanker{p: p, fitter: GemsFitter{p: p}}
}

// Rank returns the model's candidate peers best-first by the legacy weighted-sum
// key. Deterministic: ties keep the input (sorted-peerID) order, exactly like
// pickPeerForModel's first-wins-on-tie.
func (r *GemsRanker) Rank(model string) []Candidate {
	members := r.p.modelPeers[model]
	if len(members) == 0 {
		return nil
	}
	type scored struct {
		cand Candidate
		key  int64 // legacy weighted-sum; lower is better
	}
	scoredCands := make([]scored, 0, len(members))
	for _, pp := range members {
		bias := r.p.peerModelBias(model, pp)
		key := atomic.LoadInt64(&pp.inFlight)*2 + bias
		node := Node{Name: pp.peerID}
		scoredCands = append(scoredCands, scored{
			cand: Candidate{Node: node, Fit: r.fitter.Fit(model, node)},
			key:  key,
		})
	}
	sort.SliceStable(scoredCands, func(i, j int) bool { return scoredCands[i].key < scoredCands[j].key })
	out := make([]Candidate, len(scoredCands))
	for i, s := range scoredCands {
		out[i] = s.cand
	}
	return out
}

// Ensure the gems types satisfy the contract at compile time.
var (
	_ Fitter = GemsFitter{}
	_ Ranker = (*GemsRanker)(nil)
)
