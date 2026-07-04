package proxy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sync"
)

// allowance.go — the pushed credit-allowance snapshot: parse, per-key monotonic
// application to the creditLedger, and two-tier staleness (cloud
// project_gems_monetization, stage 2b.2).
//
// HOME computes per-key balances from the finance ledger and pushes this
// snapshot gems-ward (the same diode-safe rail the API keys ride); the gate
// (2b.3) applies each snapshot to the creditLedger and consults its freshness.
// This file is the ingester + policy — it does NOT poll files or touch the
// request path (that is 2b.3). Nothing constructs it yet (dormant).
//
// Fix 2 in 2b.1 moved a trust boundary onto this ingester: the ledger's
// SetAllowance ver-guard TRUSTS a monotonic ver, so the ingester OWNS that
// guarantee. Three requirements it enforces (big-dog 2b.2):
//   1. ver is AUTHORITATIVE + monotonic — it comes from the home snapshot's own
//      version (Snapshot.Ver), never a local clock that could skew or regress.
//   2. FAIL-SAFE: a malformed / partial / truncated / stale / out-of-order push
//      is REJECTED and the last-good state is HELD — it never zeroes or lowers a
//      balance, never closes-wrong-way or over-opens.
//   3. staleness is measured against the snapshot's own timestamp with a skew
//      tolerance; a missing / future-skewed timestamp is treated as EXPIRED
//      (unknown freshness = conservative). CLOSED-default beyond hard-max.

// AllowanceEntry is one key's pushed balance + ack watermark.
type AllowanceEntry struct {
	KeyFingerprint string `json:"key_fingerprint"`
	Balance        int64  `json:"balance"`   // credits; already nets debits <= Watermark
	Watermark      uint64 `json:"watermark"` // highest contiguous ingested debit seq
}

// AllowanceSnapshot is one pushed snapshot from home.
type AllowanceSnapshot struct {
	Ver        uint64           `json:"ver"`     // authoritative monotonic snapshot version (home seq); required, > 0
	TsUnix     int64            `json:"ts_unix"` // snapshot time (home clock) for staleness; 0/absent => treated as expired
	Allowances []AllowanceEntry `json:"allowances"`
}

// parseAllowanceSnapshot decodes + validates a pushed snapshot. It is strict so a
// malformed/partial push FAILS (the caller holds last-good): truncated JSON, a
// missing/zero ver (can't order it), a missing key_fingerprint, or a negative
// balance are all errors. A missing ts is NOT an error here — it parses, and the
// freshness policy treats it as expired (so a ts-less push can't silently look
// fresh). A negative balance is rejected rather than clamped, because it signals
// a corrupt/partial push and clamping could mask an over-open.
func parseAllowanceSnapshot(raw []byte) (AllowanceSnapshot, error) {
	var s AllowanceSnapshot
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	if err := dec.Decode(&s); err != nil {
		return AllowanceSnapshot{}, fmt.Errorf("allowance: parse: %w", err)
	}
	if s.Ver == 0 {
		return AllowanceSnapshot{}, fmt.Errorf("allowance: missing/zero ver (cannot order snapshot)")
	}
	seen := make(map[string]struct{}, len(s.Allowances))
	for i, e := range s.Allowances {
		if e.KeyFingerprint == "" {
			return AllowanceSnapshot{}, fmt.Errorf("allowance: entry %d missing key_fingerprint", i)
		}
		if e.Balance < 0 {
			return AllowanceSnapshot{}, fmt.Errorf("allowance: entry %d (%s) negative balance %d", i, e.KeyFingerprint, e.Balance)
		}
		if _, dup := seen[e.KeyFingerprint]; dup {
			return AllowanceSnapshot{}, fmt.Errorf("allowance: duplicate key_fingerprint %s", e.KeyFingerprint)
		}
		seen[e.KeyFingerprint] = struct{}{}
	}
	return s, nil
}

// Freshness classifies how much a snapshot can be trusted.
type Freshness int

const (
	// Fresh: within the grace window — enforce normally against the balance.
	Fresh Freshness = iota
	// Stale: past grace, within hard-max (a push lag) — the gate fails OPEN on a
	// positive balance (don't block a paying customer over a brief lag).
	Stale
	// Expired: past hard-max, or unknown/future-skewed timestamp — the gate fails
	// CLOSED + pages (the balance can't be trusted; serving risks over-spend).
	Expired
)

func (f Freshness) String() string {
	switch f {
	case Fresh:
		return "fresh"
	case Stale:
		return "stale"
	default:
		return "expired"
	}
}

// snapshotFreshness classifies a snapshot's age against grace + hard-max, with a
// skew tolerance for the home/gems clock difference. tsUnix <= 0 (missing) or a
// timestamp more than skewSec in the FUTURE (an untrustable clock) is Expired —
// unknown freshness is conservative, never optimistic.
func snapshotFreshness(tsUnix, nowUnix, graceSec, hardMaxSec, skewSec int64) Freshness {
	if tsUnix <= 0 {
		return Expired
	}
	age := nowUnix - tsUnix
	if age < -skewSec {
		return Expired // snapshot from the future beyond tolerated skew — don't trust it
	}
	if age <= graceSec {
		return Fresh
	}
	if age <= hardMaxSec {
		return Stale
	}
	return Expired
}

// allowanceIngester applies pushed snapshots to a creditLedger, owning the
// monotonic-ver + fail-safe guarantees the ledger trusts. Thread-safe.
type allowanceIngester struct {
	ledger  *creditLedger
	logger  *LogMonitor
	mu      sync.Mutex
	lastVer uint64 // highest applied snapshot ver (the monotonic gate)
	lastTs  int64  // ts of the last applied snapshot (for freshness)
	rejects int64  // snapshots rejected (malformed/stale) — observability
}

func newAllowanceIngester(ledger *creditLedger, logger *LogMonitor) *allowanceIngester {
	return &allowanceIngester{ledger: ledger, logger: logger}
}

// Apply parses raw and, if it is well-formed AND strictly newer (ver > lastVer),
// applies every entry to the ledger and records its ver/ts. A malformed /
// partial / stale / out-of-order push is REJECTED (counted + logged) and the
// last-good state is HELD — it never zeroes or lowers a balance. Returns whether
// it applied.
func (a *allowanceIngester) Apply(raw []byte) bool {
	s, err := parseAllowanceSnapshot(raw)
	if err != nil {
		a.reject("malformed snapshot: %v", err)
		return false
	}
	a.mu.Lock()
	defer a.mu.Unlock()
	if s.Ver <= a.lastVer {
		a.rejects++
		if a.logger != nil {
			a.logger.Warnf("allowance: holding last-good, rejecting stale/replayed snapshot ver=%d <= applied=%d", s.Ver, a.lastVer)
		}
		return false
	}
	for _, e := range s.Allowances {
		a.ledger.SetAllowance(e.KeyFingerprint, e.Balance, e.Watermark, s.Ver)
	}
	a.lastVer = s.Ver
	a.lastTs = s.TsUnix
	if a.logger != nil {
		a.logger.Infof("allowance: applied snapshot ver=%d (%d keys)", s.Ver, len(s.Allowances))
	}
	return true
}

func (a *allowanceIngester) reject(format string, args ...any) {
	a.mu.Lock()
	a.rejects++
	a.mu.Unlock()
	if a.logger != nil {
		a.logger.Warnf("allowance: holding last-good, "+format, args...)
	}
}

// Freshness of the last-applied snapshot at nowUnix. Never having ingested a
// snapshot is Expired (conservative — nothing trustworthy to enforce against).
func (a *allowanceIngester) Freshness(nowUnix, graceSec, hardMaxSec, skewSec int64) Freshness {
	a.mu.Lock()
	defer a.mu.Unlock()
	if a.lastVer == 0 {
		return Expired
	}
	return snapshotFreshness(a.lastTs, nowUnix, graceSec, hardMaxSec, skewSec)
}

// AppliedVer / rejectCount are observability accessors.
func (a *allowanceIngester) AppliedVer() uint64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.lastVer
}

func (a *allowanceIngester) rejectCount() int64 {
	a.mu.Lock()
	defer a.mu.Unlock()
	return a.rejects
}
