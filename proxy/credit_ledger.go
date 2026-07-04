package proxy

import "sync"

// credit_ledger.go — the per-key credit accounting at the heart of the billing
// GATE (cloud project_gems_monetization, stage 2b). PURE in-memory accounting:
// no I/O, no request-path wiring, no pricing. The gate hook (2b.3) prices a
// request and calls Reserve at admission + the returned Reservation's TrueUp
// (success) / Release (any non-success) on exit; the allowance ingester (2b.2)
// calls SetAllowance from the pushed snapshot; the pull/watermark path advances
// the ack watermark via SetAllowance.
//
// Per-key balance identity (in abstract credit units — pricing is the caller's):
//
//	available = allowance - reserved - pending - confirmedSum
//
//	allowance    pushed balance; already NETS debits with seq <= ackWatermark
//	reserved     outstanding reservation holds (admission, not yet resolved)
//	pending      trued-up debits awaiting their log seq (count immediately, so
//	             there is no over-spend window between TrueUp and Confirm)
//	confirmedSum Σ confirmed debits with seq > ackWatermark (awaiting a snapshot
//	             that nets them; kept incrementally so available() is O(1))
//
// Concurrency (big-dog invariant 4): every per-key mutation serializes on the
// single ledger mutex, so two requests racing a near-zero balance can't both
// pass — the second sees the first's reservation and is denied. Cap-enforcement
// and race-safety both fall out of this one serialization point.
//
// Reservation lifecycle (big-dog invariant 1): Reserve returns a Reservation the
// caller MUST resolve on EVERY exit (success TrueUp, or Release on error /
// timeout / client-disconnect / ctx-cancel / panic). Both are idempotent (a
// `done` latch), so a deferred Release + a success TrueUp can't double-count and
// a leaked/double resolution can't corrupt the balance.

type creditLedger struct {
	mu    sync.Mutex
	keys  map[string]*keyCredit
	skips int64 // Confirm no-ops (seq<=confirmedHigh); see Confirm — steady-state nonzero = ordering violation
}

type keyCredit struct {
	allowance     int64
	allowanceVer  uint64           // monotonic snapshot version; a stale/out-of-order push is ignored
	ackWatermark  uint64           // debits with seq <= this are already netted into `allowance`
	confirmed     map[uint64]int64 // seq -> cost, only for seq > ackWatermark
	confirmedSum  int64
	confirmedHigh uint64 // highest seq ever Confirmed; dedupes at-least-once Confirm replays
	pending       int64
	reserved      int64
}

func newCreditLedger() *creditLedger {
	return &creditLedger{keys: map[string]*keyCredit{}}
}

// keyLocked returns key's account, creating a zero one on first use. Caller holds
// l.mu. A never-allowanced key has allowance 0 -> available 0 -> Reserve denies,
// which is the fail-closed default for an unknown key.
func (l *creditLedger) keyLocked(key string) *keyCredit {
	k := l.keys[key]
	if k == nil {
		k = &keyCredit{confirmed: map[uint64]int64{}}
		l.keys[key] = k
	}
	return k
}

func (k *keyCredit) available() int64 {
	return k.allowance - k.reserved - k.pending - k.confirmedSum
}

// Reservation is a hold placed by Reserve; resolve it exactly once via TrueUp or
// Release (both idempotent, defer-safe).
type Reservation struct {
	l    *creditLedger
	key  string
	cost int64
	done bool
}

// Reserve holds `cost` credits against key's available balance. Returns
// (nil, false) when that would overspend (available < cost) — the deny path. A
// non-positive cost reserves nothing but still returns a valid (no-op) hold.
func (l *creditLedger) Reserve(key string, cost int64) (*Reservation, bool) {
	l.mu.Lock()
	defer l.mu.Unlock()
	k := l.keyLocked(key)
	if cost < 0 {
		cost = 0
	}
	if k.available() < cost {
		return nil, false
	}
	k.reserved += cost
	return &Reservation{l: l, key: key, cost: cost}, true
}

// Release returns the full hold with no debit — the non-success exit path.
// Idempotent; safe on a nil receiver (Reserve deny returns nil).
func (r *Reservation) Release() {
	if r == nil {
		return
	}
	r.l.mu.Lock()
	defer r.l.mu.Unlock()
	if r.done {
		return
	}
	r.done = true
	r.l.keyLocked(r.key).reserved -= r.cost
}

// TrueUp resolves the hold on SUCCESS: releases the reservation and records
// `actual` (clamped to [0, cost] — never bill more than was reserved) as a
// pending debit that counts against available immediately (no over-spend window
// before the log seq is known). Pair a later Confirm(key, actual, seq) with the
// same `actual`. Idempotent; returns false if already resolved.
func (r *Reservation) TrueUp(actual int64) bool {
	if r == nil {
		return false
	}
	r.l.mu.Lock()
	defer r.l.mu.Unlock()
	if r.done {
		return false
	}
	r.done = true
	if actual < 0 {
		actual = 0
	}
	if actual > r.cost {
		actual = r.cost
	}
	k := r.l.keyLocked(r.key)
	k.reserved -= r.cost
	k.pending += actual
	return true
}

// Confirm assigns a durable log seq to a previously trued-up debit: moves `cost`
// out of pending and, if the debit is NOT yet netted into the allowance
// (seq > ackWatermark), into the seq-keyed confirmed set so a later snapshot can
// age it out. If seq <= ackWatermark a snapshot already accounts for it, so it is
// only removed from pending (never double-counted). Available-neutral for the
// seq > watermark case.
//
// IDEMPOTENT per seq via confirmedHigh: the confirm path is at-least-once (a
// restart/cursor replay re-reads the log tail). A replay re-reads seqs in
// INCREASING order, so a seq <= confirmedHigh no-ops the whole replayed run in
// O(1) — including replays of already-aged seqs that a watermark-pruned per-seq
// set could not catch. Guards BOTH the pending decrement and the confirmedSum
// add, else a replay would phantom-drop pending and permanently inflate
// confirmedSum.
//
// IN-ORDER CONFIRM IS A CORRECTNESS REQUIREMENT, NOT A PERFORMANCE NICETY. If
// Confirm(6) ran before Confirm(5), confirmedHigh would jump to 6 and Confirm(5)
// would be skipped as a "replay" — but cost5 is a NEW debit that is then never
// moved out of pending, and nothing ever clears it (SetAllowance doesn't touch
// pending), so available stays understated by cost5 PERMANENTLY. So Confirm MUST
// be driven single-threaded in log/append order (the single writer, or one
// single-threaded tailer) — NEVER goroutine-per-event, never a parallel batch.
// The 2b.3 gate wiring asserts this at the call site. `skips` counts the no-ops:
// a nonzero rate OUTSIDE a restart replay flags an ordering violation that would
// silently eat credits.
func (l *creditLedger) Confirm(key string, cost int64, seq uint64) {
	if cost <= 0 {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	k := l.keyLocked(key)
	if seq <= k.confirmedHigh {
		l.skips++
		return // already confirmed this seq (replay) — no-op
	}
	k.confirmedHigh = seq
	if k.pending >= cost {
		k.pending -= cost
	} else {
		k.pending = 0
	}
	if seq > k.ackWatermark {
		k.confirmed[seq] = cost
		k.confirmedSum += cost
	}
}

// SetAllowance applies a pushed snapshot: `balance` (already nets debits with
// seq <= watermark), the ack `watermark`, and the snapshot's monotonic `ver`.
// It ages out confirmed debits at or below the watermark — now reflected in
// `balance` — so they aren't double counted. available stays consistent because
// `balance` dropped by exactly the aged debits' cost.
//
// The ledger is the last line before money moves, so a STALE / out-of-order
// snapshot is IGNORED: `ver` is monotonic per key and a `ver < allowanceVer`
// push is dropped. Without this, an old snapshot carrying a HIGHER balance would
// overstate available and permit over-spend. (A same-or-newer ver re-applies;
// the watermark stays independently monotonic-guarded.)
func (l *creditLedger) SetAllowance(key string, balance int64, watermark uint64, ver uint64) {
	l.mu.Lock()
	defer l.mu.Unlock()
	k := l.keyLocked(key)
	if ver < k.allowanceVer {
		return // stale / out-of-order snapshot — never inflate the money core
	}
	k.allowanceVer = ver
	k.allowance = balance
	if watermark > k.ackWatermark {
		for seq, cost := range k.confirmed {
			if seq <= watermark {
				k.confirmedSum -= cost
				delete(k.confirmed, seq)
			}
		}
		k.ackWatermark = watermark
	}
}

// Available is the current spendable balance for key (locked accessor for the
// gate's staleness/observability decisions). An unknown key reads 0.
func (l *creditLedger) Available(key string) int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.keyLocked(key).available()
}

// confirmSkips is the count of Confirm no-ops (replayed/out-of-order seqs).
// Expected to spike during a restart replay; a nonzero rate in steady state
// signals a Confirm ordering violation (see Confirm) that would eat credits.
func (l *creditLedger) confirmSkips() int64 {
	l.mu.Lock()
	defer l.mu.Unlock()
	return l.skips
}
