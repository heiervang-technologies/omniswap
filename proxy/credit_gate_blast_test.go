package proxy

import (
	"io"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// brokenGate is an enforcing gate whose ledger has a nil keys map, so Reserve
// panics (assignment to entry in nil map) — a realistic "gate/ledger bug" fault
// to exercise the blast-radius containment. failOpen selects the fault policy.
func brokenGate(failOpen bool) *creditGate {
	prices := newPriceBook(map[string]ModelRate{
		"m": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: 4096},
	})
	g := newCreditGate(true, &creditLedger{}, prices, nil, nil)
	g.failOpen = failOpen
	return g
}

// --- shape 6: recover boundary contains a gate fault ----------------------------

func TestCreditGate_RecoverGateFault_ContainsAndCounts(t *testing.T) {
	g, _ := gateWithBalance(true, "k", 100)
	faulted, failOpen := g.recoverGateFault("m", func() { panic("boom") })
	assert.True(t, faulted, "panic contained, reported as faulted")
	assert.False(t, failOpen, "default policy is fail-closed")
	assert.Equal(t, int64(1), g.gatePanicCount(), "fault counted")
}

func TestCreditGate_RecoverGateFault_NoPanicPassesThrough(t *testing.T) {
	g, _ := gateWithBalance(true, "k", 100)
	ran := false
	faulted, _ := g.recoverGateFault("m", func() { ran = true })
	assert.False(t, faulted)
	assert.True(t, ran, "fn ran on the normal path")
	assert.Equal(t, int64(0), g.gatePanicCount(), "no fault on the happy path")
}

// --- guardedAdmit: fault -> fail-CLOSED deny by default -------------------------

func TestCreditGate_GuardedAdmit_FaultFailsClosed(t *testing.T) {
	g := brokenGate(false)
	rsv, d := g.guardedAdmit("k", "m", 1000, 500)
	assert.Nil(t, rsv)
	assert.Equal(t, admitFault, d, "gate fault -> fail-closed deny (default)")
	assert.False(t, d.served(), "a faulted admit does NOT serve")
	assert.Equal(t, int64(1), g.gatePanicCount())
}

func TestCreditGate_GuardedAdmit_CanaryFailOpenServes(t *testing.T) {
	g := brokenGate(true) // explicit canary opt-in
	rsv, d := g.guardedAdmit("k", "m", 1000, 500)
	assert.Equal(t, admitOff, d, "canary fail-open -> serve un-metered on fault")
	assert.True(t, d.served())
	assert.NotNil(t, rsv, "served with a no-op reservation")
	assert.Equal(t, int64(1), g.gatePanicCount())
}

func TestCreditGate_GuardedAdmit_HappyPath(t *testing.T) {
	g, _ := gateWithBalance(true, "k", 100)
	rsv, d := g.guardedAdmit("k", "m", 1000, 500)
	assert.Equal(t, admitOK, d, "no fault -> normal admit result")
	assert.NotNil(t, rsv)
	assert.Equal(t, int64(0), g.gatePanicCount())
}

// --- guardedResolve: shape-6 lock 3 — recover AND hold-released both hold --------

func TestCreditGate_GuardedResolve_ReleasesAndContainsPanic(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, d := g.admit("k", "m", 1000, 500)
	require.Equal(t, admitOK, d)
	require.Equal(t, int64(94), ledger.Available("k"))

	rsv.guardedResolve(func() { panic("resolve boom") }) // simulated ledger/gate bug
	assert.Equal(t, int64(100), ledger.Available("k"), "hold RELEASED despite the panic — no leak")
	assert.Equal(t, int64(1), g.gatePanicCount(), "resolve fault counted")
}

func TestCreditGate_GuardedResolve_HappyPath(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, _ := g.admit("k", "m", 1000, 500)
	rsv.guardedResolve(func() { rsv.trueUp(1000, 250) }) // actual 4
	assert.Equal(t, int64(96), ledger.Available("k"), "normal resolve trues up through the guard")
	assert.Equal(t, int64(0), g.gatePanicCount())
}

func TestCreditGate_GuardedResolve_OffReservationNoOp(t *testing.T) {
	g, ledger := gateWithBalance(false, "k", 100) // OFF -> no-op reservation
	rsv, d := g.admit("k", "m", 1000, 500)
	require.Equal(t, admitOff, d)
	assert.NotPanics(t, func() { rsv.guardedResolve(func() { rsv.trueUp(1000, 250) }) })
	assert.Equal(t, int64(100), ledger.Available("k"), "OFF resolve never touches the ledger")
}

// --- failure-domain isolation: a fault doesn't corrupt the gate -----------------

func TestCreditGate_FaultDoesNotCorruptGate(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	// a contained fault...
	g.recoverGateFault("m", func() { panic("boom") })
	// ...then a normal admit + resolve still works (gate healthy, serving path fine)
	rsv, d := g.admit("k", "m", 1000, 500)
	require.Equal(t, admitOK, d, "gate healthy after a contained fault")
	rsv.trueUp(1000, 250)
	assert.Equal(t, int64(96), ledger.Available("k"))
}

// --- observability: a fault WARNs, so a recover-denying gate is VISIBLE ----------

func TestCreditGate_FaultWarns(t *testing.T) {
	prices := newPriceBook(map[string]ModelRate{"m": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: 4096}})
	lg := NewLogMonitorWriter(io.Discard)
	g := newCreditGate(true, newCreditLedger(), prices, nil, lg)

	g.recoverGateFault("m", func() { panic("ledger unreachable") })
	hist := string(lg.GetHistory())
	assert.Contains(t, hist, "FAULT")
	assert.Contains(t, hist, "fail-CLOSED", "the WARN names the contained action")
}
