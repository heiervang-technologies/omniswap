package proxy

import (
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCreditLedger_ReserveWithinBalance_DeniesOver: reserve up to the balance,
// deny anything that would overspend.
func TestCreditLedger_ReserveWithinBalance_DeniesOver(t *testing.T) {
	l := newCreditLedger()
	l.SetAllowance("k", 100, 0, 1)

	r1, ok := l.Reserve("k", 60)
	require.True(t, ok)
	require.NotNil(t, r1)
	assert.Equal(t, int64(40), l.Available("k"))

	_, ok = l.Reserve("k", 50) // 50 > 40 available
	assert.False(t, ok, "overspend denied")
	assert.Equal(t, int64(40), l.Available("k"), "a denied reserve holds nothing")

	_, ok = l.Reserve("k", 40)
	assert.True(t, ok)
	assert.Equal(t, int64(0), l.Available("k"))
}

// TestCreditLedger_UnknownKeyDeniesAll: a never-allowanced key has 0 balance ->
// fail-closed default.
func TestCreditLedger_UnknownKeyDeniesAll(t *testing.T) {
	l := newCreditLedger()
	_, ok := l.Reserve("ghost", 1)
	assert.False(t, ok)
	assert.Equal(t, int64(0), l.Available("ghost"))
}

// TestCreditLedger_ReleaseRestores_Idempotent: Release returns the full hold and
// is a no-op if called twice.
func TestCreditLedger_ReleaseRestores_Idempotent(t *testing.T) {
	l := newCreditLedger()
	l.SetAllowance("k", 100, 0, 1)
	r, _ := l.Reserve("k", 60)
	assert.Equal(t, int64(40), l.Available("k"))

	r.Release()
	assert.Equal(t, int64(100), l.Available("k"), "hold returned")
	r.Release() // idempotent
	assert.Equal(t, int64(100), l.Available("k"), "double release is a no-op")
}

// TestCreditLedger_TrueUp: success path bills the ACTUAL cost (clamped to the
// reservation), releases the rest, and is idempotent.
func TestCreditLedger_TrueUp(t *testing.T) {
	l := newCreditLedger()
	l.SetAllowance("k", 100, 0, 1)

	r, _ := l.Reserve("k", 60) // reserve worst-case 60
	assert.True(t, r.TrueUp(25), "first true-up records the debit")
	assert.Equal(t, int64(75), l.Available("k"), "billed actual 25, released the 35 over-reservation")
	assert.False(t, r.TrueUp(10), "second true-up is a no-op")
	assert.Equal(t, int64(75), l.Available("k"))

	// actual clamps to [0, cost]: over-claim can't bill more than reserved
	r2, _ := l.Reserve("k", 30)
	r2.TrueUp(999)
	assert.Equal(t, int64(45), l.Available("k"), "clamped to the 30 reserved (75-30)")
}

// TestCreditLedger_ReleaseAfterTrueUp_NoDoubleCount: whichever of TrueUp/Release
// fires first wins; the other no-ops (the leak-proof idempotency guarantee).
func TestCreditLedger_ReleaseAfterTrueUp_NoDoubleCount(t *testing.T) {
	l := newCreditLedger()
	l.SetAllowance("k", 100, 0, 1)
	r, _ := l.Reserve("k", 40)
	require.True(t, r.TrueUp(30))
	r.Release() // must NOT return the already-billed hold
	assert.Equal(t, int64(70), l.Available("k"), "release after true-up is a no-op, no double-count")
}

// TestCreditLedger_ConfirmAndAgeOut: a trued-up debit gets a seq (Confirm), then
// a snapshot nets it (SetAllowance ages it out) — available stays consistent.
func TestCreditLedger_ConfirmAndAgeOut(t *testing.T) {
	l := newCreditLedger()
	l.SetAllowance("k", 100, 0, 1)
	r, _ := l.Reserve("k", 60)
	r.TrueUp(25)
	assert.Equal(t, int64(75), l.Available("k")) // pending 25

	l.Confirm("k", 25, 5) // seq 5 assigned; pending -> confirmed
	assert.Equal(t, int64(75), l.Available("k"), "confirm is available-neutral")

	// home nets the debit: new balance 75 (was 100, minus the 25 consumed) @ watermark 5
	l.SetAllowance("k", 75, 5, 2)
	assert.Equal(t, int64(75), l.Available("k"), "age-out is available-neutral: balance dropped by exactly the aged debit")
}

// TestCreditLedger_ConfirmBelowWatermarkDropped: a seq already netted into the
// allowance is not double-counted.
func TestCreditLedger_ConfirmBelowWatermarkDropped(t *testing.T) {
	l := newCreditLedger()
	l.SetAllowance("k", 100, 10, 1) // watermark already at 10
	r, _ := l.Reserve("k", 20)
	r.TrueUp(5)
	assert.Equal(t, int64(95), l.Available("k")) // pending 5

	l.Confirm("k", 5, 8) // seq 8 <= watermark 10 -> already in allowance, drop
	assert.Equal(t, int64(100), l.Available("k"), "debit already netted into the balance is not double-counted")
}

// TestCreditLedger_ConcurrentReserve_ExactlyCapSucceed: the core race safety —
// N goroutines race a balance that fits exactly `cap` reservations; exactly cap
// succeed, none oversell.
func TestCreditLedger_ConcurrentReserve_ExactlyCapSucceed(t *testing.T) {
	l := newCreditLedger()
	const cost, capacity = 10, 7
	l.SetAllowance("k", cost*capacity, 0, 1)

	var ok int64
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, got := l.Reserve("k", cost); got {
				atomic.AddInt64(&ok, 1)
			}
		}()
	}
	wg.Wait()
	assert.Equal(t, int64(capacity), ok, "exactly cap reservations fit the balance — no oversell")
	assert.Equal(t, int64(0), l.Available("k"))
}

// TestCreditLedger_ConcurrentResolveSameReservation: Release and TrueUp racing
// the SAME reservation resolve it exactly once (never double-count).
func TestCreditLedger_ConcurrentResolveSameReservation(t *testing.T) {
	l := newCreditLedger()
	l.SetAllowance("k", 100, 0, 1)
	r, _ := l.Reserve("k", 60)

	var wg sync.WaitGroup
	wg.Add(2)
	go func() { defer wg.Done(); r.Release() }()
	go func() { defer wg.Done(); r.TrueUp(30) }()
	wg.Wait()

	// Exactly one resolution took effect: either release (avail 100) or true-up
	// (avail 70). Never 130 (double release) or 40 (double count).
	avail := l.Available("k")
	assert.Contains(t, []int64{100, 70}, avail, "resolved exactly once, no double-count")
}

// TestCreditLedger_ConfirmIdempotent: the confirm path is at-least-once (cursor
// replay on restart), so a replayed Confirm of the same seq must be a no-op —
// else confirmedSum inflates permanently (phantom credit loss). big-dog #30 fix.
func TestCreditLedger_ConfirmIdempotent(t *testing.T) {
	l := newCreditLedger()
	l.SetAllowance("k", 100, 0, 1)
	r, _ := l.Reserve("k", 60)
	r.TrueUp(25)

	l.Confirm("k", 25, 5)
	assert.Equal(t, int64(75), l.Available("k"))
	l.Confirm("k", 25, 5) // REPLAY — must be a no-op, not a phantom -25
	assert.Equal(t, int64(75), l.Available("k"), "replayed Confirm of the same seq is a no-op")

	// and the dedupe survives age-out (the exact permanent-corruption case)
	l.SetAllowance("k", 75, 5, 2)
	l.Confirm("k", 25, 5) // replay after the debit was aged out
	assert.Equal(t, int64(75), l.Available("k"), "replay after age-out stays consistent, no stuck confirmedSum")
}

// TestCreditLedger_SetAllowance_IgnoresStaleSnapshot: the money core must not let
// a stale/out-of-order snapshot with a HIGHER balance inflate available (which
// would permit over-spend). big-dog #30 fix.
func TestCreditLedger_SetAllowance_IgnoresStaleSnapshot(t *testing.T) {
	l := newCreditLedger()
	l.SetAllowance("k", 40, 0, 5) // current: balance 40 @ ver 5
	assert.Equal(t, int64(40), l.Available("k"))

	l.SetAllowance("k", 100, 0, 3) // STALE (ver 3 < 5) carrying a HIGHER balance
	assert.Equal(t, int64(40), l.Available("k"), "a stale snapshot can't inflate the balance — over-spend guard")

	l.SetAllowance("k", 90, 0, 6) // fresh (ver 6 > 5) applies
	assert.Equal(t, int64(90), l.Available("k"))
}
