package proxy

// credit_gate.go — the prepaid-credit billing gate object (cloud
// project_gems_monetization, stage 2b.3c piece 1). DORMANT: this is the
// OFF-by-default gate SKELETON. It holds the accounting components (ledger, price
// book, allowance ingester) and the explicit enforce flag, but has NO request-path
// hook yet — reserve@admission and resolve-on-every-exit land in piece 2, behind
// enforcing(). Constructed-but-unwired, so the request path is byte-identical.
//
// enforce is DECOUPLED from component presence (the #352 landmine class): the gate
// enforces ONLY when CreditGateEnforce is explicitly set, never because rates or
// allowances happen to be loaded. OFF => a true no-op (piece 2's hook checks
// enforcing() and returns before any ledger touch).

type creditGate struct {
	enforce   bool
	ledger    *creditLedger
	prices    *priceBook
	allowance *allowanceIngester
}

// newCreditGate builds the gate from its components and the EXPLICIT enforce flag.
// A gate with enforce=false (the default) never touches the ledger — piece 2's
// hook consults enforcing() first.
func newCreditGate(enforce bool, ledger *creditLedger, prices *priceBook, allowance *allowanceIngester) *creditGate {
	return &creditGate{enforce: enforce, ledger: ledger, prices: prices, allowance: allowance}
}

// enforcing reports whether the gate actively meters requests. False (the default)
// = OFF: no reserve, no debit, no ledger touch, request path byte-identical. A nil
// gate is never enforcing — the safe zero value for the unconfigured pool, so the
// hook can be called unconditionally and no-op when billing was never set up.
func (g *creditGate) enforcing() bool {
	return g != nil && g.enforce
}

// admitDecision is the outcome of admit(): whether the request may be served and,
// if not, why (for the caller's status code + logging).
type admitDecision int

const (
	admitOff          admitDecision = iota // gate OFF — served, NOT metered (no-op reservation)
	admitOK                                // priced + reserved — served, metered
	admitUnpriced                          // no price for the model — DENY (fail-closed, never serve un-metered)
	admitInsufficient                      // priced but balance too low — DENY (402-class)
)

// served reports whether the request may proceed (OFF pass-through or a successful
// reserve). The two deny outcomes are false.
func (d admitDecision) served() bool { return d == admitOff || d == admitOK }

// creditReservation is a request's hold, resolved EXACTLY ONCE on exit. The no-op
// reservation (gate OFF) has a nil inner hold, so every method is a safe no-op and
// the request path can `defer rsv.release()` unconditionally with zero ledger
// touch when OFF. inner's TrueUp/Release are idempotent (done latch), so a success
// trueUp followed by the deferred release can't double-count — the release wins
// only when trueUp never ran (a non-success exit).
type creditReservation struct {
	inner  *Reservation // ledger hold; nil = no-op (OFF / not admitted)
	prices *priceBook   // for the true-up ActualCost; nil on a no-op reservation
	model  string
}

// admit prices a request's WORST-CASE cost (prompt + completion upper bound,
// rounded up) and reserves it against key's balance at admission. It rides
// enforcing() the way the T2 semaphore rides enabled():
//
//   - OFF          -> (no-op reservation, admitOff): served, ledger untouched.
//   - unpriced     -> (nil, admitUnpriced): DENY — the gate can't price it, so it
//     can't meter it, so it fail-closes (never serve un-metered / free).
//   - insufficient -> (nil, admitInsufficient): DENY — the hold would overspend.
//   - reserved     -> (reservation, admitOK): served; resolve on EVERY exit.
//
// A missing price book or ledger while enforcing is treated as unpriced (DENY) —
// fail-closed against a misconfigured gate rather than serving un-metered.
func (g *creditGate) admit(key, model string, promptTokens, requestedMax int64) (*creditReservation, admitDecision) {
	if !g.enforcing() {
		return &creditReservation{}, admitOff
	}
	if g.prices == nil || g.ledger == nil {
		return nil, admitUnpriced // misconfigured gate -> fail-closed
	}
	cost, priced := g.prices.ReserveCost(model, promptTokens, requestedMax)
	if !priced {
		return nil, admitUnpriced
	}
	hold, ok := g.ledger.Reserve(key, cost)
	if !ok {
		return nil, admitInsufficient
	}
	return &creditReservation{inner: hold, prices: g.prices, model: model}, admitOK
}

// trueUp resolves the hold on SUCCESS: bills the ACTUAL metered cost (priced from
// the request's real token counts), which the ledger clamps to the reserved hold
// (reserve worst-case >= actual, so the clamp never bites in-bounds). No-op on a
// no-op reservation (gate OFF). Idempotent.
func (rsv *creditReservation) trueUp(promptTokens, completionTokens int64) {
	if rsv == nil || rsv.inner == nil {
		return
	}
	actual, ok := rsv.prices.ActualCost(rsv.model, promptTokens, completionTokens)
	if !ok {
		// Unpriced can't happen after a successful reserve, but never guess a bill:
		// release-equivalent (0 debit) is the safe direction.
		actual = 0
	}
	rsv.inner.TrueUp(actual)
}

// release resolves the hold on ANY non-success exit (upstream error / timeout /
// client-disconnect / ctx-cancel / panic): returns the FULL hold with no debit.
// No-op-safe (nil receiver / no-op reservation) + idempotent, so the handler can
// `defer rsv.release()` on every path and a prior trueUp wins.
func (rsv *creditReservation) release() {
	if rsv == nil || rsv.inner == nil {
		return
	}
	rsv.inner.Release()
}
