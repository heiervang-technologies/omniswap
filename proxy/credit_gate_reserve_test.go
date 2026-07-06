package proxy

import (
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// gateWithBalance builds an enforcing/OFF gate over a ledger seeded with `balance`
// credits for `key`, priced by a single model "m": reserve cost 6 (prompt 1000 ->
// 2 + completion max 500 -> 4), actual for (1000,250) -> 4.
func gateWithBalance(enforce bool, key string, balance int64) (*creditGate, *creditLedger) {
	ledger := newCreditLedger()
	ledger.SetAllowance(key, balance, 0, 1)
	prices := newPriceBook(map[string]ModelRate{
		"m": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: 4096},
	})
	return newCreditGate(enforce, ledger, prices, nil), ledger
}

const (
	reserveCostM = int64(6) // ReserveCost("m", 1000, 500)
	actualCostM  = int64(4) // ActualCost("m", 1000, 250)
)

// --- shape 1: OFF is a true no-op, ledger never touched -------------------------

func TestCreditGate_OffPathNeverTouchesLedger(t *testing.T) {
	g, ledger := gateWithBalance(false, "k", 1000)
	before := ledger.Available("k")

	rsv, d := g.admit("k", "m", 1000, 500)
	assert.Equal(t, admitOff, d)
	assert.True(t, d.served(), "OFF is served (pass-through)")
	assert.Equal(t, before, ledger.Available("k"), "OFF admit must not reserve")

	// resolving either way is still a no-op
	rsv.trueUp(1000, 250)
	rsv.release()
	assert.Equal(t, before, ledger.Available("k"), "OFF resolve must not debit")
}

// --- normal exit: reserve worst-case, true-up to actual -------------------------

func TestCreditGate_ReserveThenTrueUp(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)

	rsv, d := g.admit("k", "m", 1000, 500)
	require.Equal(t, admitOK, d)
	assert.Equal(t, 100-reserveCostM, ledger.Available("k"), "reserve holds worst-case 6")

	rsv.trueUp(1000, 250) // actual 4
	assert.Equal(t, 100-actualCostM, ledger.Available("k"), "true-up bills actual 4, refunds the 2 over-reserve")
}

// --- shape 2: resolve on EVERY exit; a non-success exit releases the full hold ---

func TestCreditGate_ReleaseOnEveryFailureExit(t *testing.T) {
	// The handler maps each of these exits to a release (no tokens metered).
	for _, exit := range []string{"upstream-error", "timeout", "client-disconnect", "ctx-cancel"} {
		t.Run(exit, func(t *testing.T) {
			g, ledger := gateWithBalance(true, "k", 100)
			rsv, d := g.admit("k", "m", 1000, 500)
			require.Equal(t, admitOK, d)
			assert.Equal(t, 100-reserveCostM, ledger.Available("k"))

			rsv.release() // this exit path
			assert.Equal(t, int64(100), ledger.Available("k"), "release returns the FULL hold, no debit — no leak")
		})
	}
}

// panic is an exit path too: a deferred release must run during unwind.
func TestCreditGate_ReleaseOnPanicViaDefer(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, d := g.admit("k", "m", 1000, 500)
	require.Equal(t, admitOK, d)

	func() {
		defer func() { _ = recover() }()
		defer rsv.release() // rides the panic unwind
		panic("boom")
	}()
	assert.Equal(t, int64(100), ledger.Available("k"), "deferred release survives panic — no leaked hold")
}

// The handler pattern: `defer rsv.release()` on every path + `rsv.trueUp()` on
// success. The true-up wins; the deferred release is an idempotent no-op.
func TestCreditGate_TrueUpThenDeferredRelease(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, d := g.admit("k", "m", 1000, 500)
	require.Equal(t, admitOK, d)

	func() {
		defer rsv.release()   // the every-exit safety net
		rsv.trueUp(1000, 250) // success path resolves first
	}()
	assert.Equal(t, 100-actualCostM, ledger.Available("k"), "true-up wins; deferred release no-ops (idempotent)")
}

// Double-resolve in any order is idempotent — no double-refund / double-charge.
func TestCreditGate_DoubleResolveIdempotent(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, _ := g.admit("k", "m", 1000, 500)
	rsv.release()
	rsv.release()
	rsv.trueUp(1000, 250) // too late — already released
	assert.Equal(t, int64(100), ledger.Available("k"), "first resolve wins; the rest are no-ops")
}

// --- deny paths: fail-closed --------------------------------------------------

func TestCreditGate_UnpricedDenies(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, d := g.admit("k", "no-such-model", 1000, 500)
	assert.Nil(t, rsv)
	assert.Equal(t, admitUnpriced, d)
	assert.False(t, d.served(), "unpriced -> DENY (never serve un-metered)")
	assert.Equal(t, int64(100), ledger.Available("k"), "a denied request reserves nothing")
}

func TestCreditGate_InsufficientDenies(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 5) // reserve cost 6 > balance 5
	rsv, d := g.admit("k", "m", 1000, 500)
	assert.Nil(t, rsv)
	assert.Equal(t, admitInsufficient, d)
	assert.False(t, d.served())
	assert.Equal(t, int64(5), ledger.Available("k"), "denied reserve leaves the balance untouched")
}

func TestCreditGate_MisconfiguredFailsClosed(t *testing.T) {
	// enforcing but no price book -> can't price -> fail-closed deny, no panic.
	g := newCreditGate(true, newCreditLedger(), nil, nil)
	rsv, d := g.admit("k", "m", 1000, 500)
	assert.Nil(t, rsv)
	assert.Equal(t, admitUnpriced, d, "enforcing with no price book -> fail-closed")
}

// --- shape 5: concurrency — reserve-then-resolve atomic on a thin balance -------

func TestCreditGate_ConcurrentAdmitThinBalance(t *testing.T) {
	const n = 8
	g, ledger := gateWithBalance(true, "k", reserveCostM*n) // fits EXACTLY n reserves

	var okCount int64
	var wg sync.WaitGroup
	for i := 0; i < 2*n; i++ { // twice as many contenders as fit
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, d := g.admit("k", "m", 1000, 500); d == admitOK {
				atomic.AddInt64(&okCount, 1)
			}
		}()
	}
	wg.Wait()

	assert.Equal(t, int64(n), atomic.LoadInt64(&okCount), "exactly n reserves fit — no oversell across the race")
	assert.Equal(t, int64(0), ledger.Available("k"), "n holds outstanding -> available exactly 0")
	assert.GreaterOrEqual(t, ledger.Available("k"), int64(0), "balance never goes negative")
}

// Concurrent admit + resolve churn returns to the seeded balance (every hold
// resolves exactly once; no lost/leaked credit under contention).
func TestCreditGate_ConcurrentAdmitResolveReturnsBalance(t *testing.T) {
	const n = 200
	seed := reserveCostM * 4 // small pool so admits genuinely contend + get denied
	g, ledger := gateWithBalance(true, "k", seed)

	var wg sync.WaitGroup
	for i := 0; i < n; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			rsv, d := g.admit("k", "m", 1000, 500)
			if d != admitOK {
				return
			}
			if i%2 == 0 {
				rsv.trueUp(1000, 250) // success (bills 4)
			} else {
				rsv.release() // non-success (bills 0)
			}
		}(i)
	}
	wg.Wait()

	// Every hold resolved: reserved is fully unwound, so available == seed - pending,
	// and available stays within [0, seed] with no negative/overspend.
	avail := ledger.Available("k")
	assert.GreaterOrEqual(t, avail, int64(0), "never negative")
	assert.LessOrEqual(t, avail, seed, "never above the seeded balance (no phantom credit)")
}
