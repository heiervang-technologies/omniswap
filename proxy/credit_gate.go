package proxy

import "sync"

// credit_gate.go — the prepaid-credit billing gate object (cloud
// project_gems_monetization, stage 2b.3c pieces 1-2). DORMANT: reserve@admission
// (admit) + resolve-on-every-exit (trueUp/release) are here and unit-tested, but
// NOT wired into the request path yet (the handler hook is the next slice). OFF =>
// a true no-op, so the request path is byte-identical.
//
// enforce is DECOUPLED from component presence (the #352 landmine class): the gate
// enforces ONLY when CreditGateEnforce is explicitly set, never because rates or
// allowances happen to be loaded. OFF => enforcing() is false and admit() returns
// before any ledger touch.

type creditGate struct {
	enforce   bool
	ledger    *creditLedger
	prices    *priceBook
	allowance *allowanceIngester
	logger    *LogMonitor
	// failOpen: on a gate/ledger FAULT (recovered panic), serve un-metered instead
	// of deny. Default false = fail-CLOSED (contain the blast radius). Set from
	// config.CreditGateCanaryFailOpen by the wiring; the zero value is the safe one.
	failOpen bool

	mu         sync.Mutex
	anomalies  map[string]*modelAnomaly // per-model billing-anomaly counts (observability; big-dog ask-2)
	gatePanics int64                    // recovered gate/ledger faults (blast-radius); a rising count = the gate is recover-denying, must be VISIBLE not a silent blanket-deny
}

// modelAnomaly counts the two "safe-but-wrong" true-up cases per model so a
// pricing bug is SEEN in prod rather than silently leaking revenue.
type modelAnomaly struct {
	clamps       int64 // actual cost exceeded the reserved hold -> ledger clamped -> bounded UNDER-bill
	pricingFails int64 // ActualCost failed after a successful reserve (can't-happen) -> billed 0
}

// newCreditGate builds the gate from its components and the EXPLICIT enforce flag.
// A gate with enforce=false (the default) never touches the ledger — admit()
// consults enforcing() first. logger may be nil (WARNs then no-op).
func newCreditGate(enforce bool, ledger *creditLedger, prices *priceBook, allowance *allowanceIngester, logger *LogMonitor) *creditGate {
	return &creditGate{enforce: enforce, ledger: ledger, prices: prices, allowance: allowance, logger: logger}
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
	admitFault                             // gate/ledger FAULT (recovered panic) — DENY (fail-closed) unless canary fail-open
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
	gate         *creditGate  // back-ref for pricing + anomaly reporting; nil on a no-op reservation
	inner        *Reservation // ledger hold; nil = no-op (OFF / not admitted)
	model        string
	reservedCost int64 // the worst-case hold, so true-up can detect an actual-exceeds-reserve clamp
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
	return &creditReservation{gate: g, inner: hold, model: model, reservedCost: cost}, admitOK
}

// trueUp resolves the hold on SUCCESS: bills the ACTUAL metered cost (priced from
// the request's real token counts), which the ledger clamps to the reserved hold
// (reserve worst-case >= actual, so the clamp never bites in-bounds). No-op on a
// no-op reservation (gate OFF). Idempotent.
//
// Both "safe-but-wrong" cases are made OBSERVABLE per-model (big-dog #35/ask-2) —
// never a silent revenue leak:
//   - ActualCost fails after a successful reserve (a pricing-consistency bug that
//     can't happen with a static price book): bill 0 (never overcharge on our own
//     bug) but COUNT + WARN, because a silent free completion is the leak direction.
//   - actual exceeds the reserved hold (backend overshot max_tokens, or the prompt
//     estimate under-shot): the ledger clamps to the hold = a bounded UNDER-bill;
//     COUNT + WARN so a mispriced model's leak is seen, not quietly absorbed.
func (rsv *creditReservation) trueUp(promptTokens, completionTokens int64) {
	if rsv == nil || rsv.inner == nil {
		return
	}
	actual, ok := rsv.gate.prices.ActualCost(rsv.model, promptTokens, completionTokens)
	if !ok {
		rsv.gate.recordPricingFail(rsv.model)
		rsv.inner.TrueUp(0)
		return
	}
	if actual > rsv.reservedCost {
		rsv.gate.recordClamp(rsv.model)
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

// trueUpReserved bills the FULL reserved hold. Used when a request was SERVED (200)
// but returned no usable usage (backend omitted the usage block / empty body): the
// completion — the expensive part — can't be measured, so we charge the reserved
// worst case rather than leak a free completion (big-dog: no-usage -> charge the
// estimate, never 0/free). The reserve was priced from the prompt upper bound +
// completion ceiling, so it is the conservative, no-under-bill figure. No-op-safe +
// idempotent (the ledger's done latch), so it composes with the deferred release.
func (rsv *creditReservation) trueUpReserved() {
	if rsv == nil || rsv.inner == nil {
		return
	}
	rsv.inner.TrueUp(rsv.reservedCost)
}

// anomalyLocked returns model's anomaly counters, creating them on first use.
// Caller holds g.mu.
func (g *creditGate) anomalyLocked(model string) *modelAnomaly {
	if g.anomalies == nil {
		g.anomalies = map[string]*modelAnomaly{}
	}
	a := g.anomalies[model]
	if a == nil {
		a = &modelAnomaly{}
		g.anomalies[model] = a
	}
	return a
}

// recordClamp counts + WARNs a true-up whose actual cost exceeded the reserved
// hold (the ledger clamps it -> bounded under-bill). A rising rate for a model
// means its reserve is mispriced and leaking; keep it visible.
func (g *creditGate) recordClamp(model string) {
	g.mu.Lock()
	a := g.anomalyLocked(model)
	a.clamps++
	n := a.clamps
	g.mu.Unlock()
	if g.logger != nil {
		g.logger.Warnf("credit gate: true-up CLAMP for model %q — actual cost exceeded the reserved hold; billed the reserve (bounded under-bill). This model's reserve is mispriced. clamps=%d", model, n)
	}
}

// recordPricingFail counts + WARNs the can't-happen case where a request priced
// at reserve is unpriced at true-up. Billed 0 (no overcharge on our own bug), but
// LOUD — a silent free completion is the leak direction.
func (g *creditGate) recordPricingFail(model string) {
	g.mu.Lock()
	a := g.anomalyLocked(model)
	a.pricingFails++
	n := a.pricingFails
	g.mu.Unlock()
	if g.logger != nil {
		g.logger.Warnf("credit gate: PRICING ANOMALY for model %q — priced at reserve but ActualCost failed at true-up; billed 0 (no overcharge) but this must never happen. pricingFails=%d", model, n)
	}
}

// clampCount / pricingFailCount are per-model observability accessors (big-dog
// ask-2). Kept at the gate, not the ledger, because the ledger is per-KEY and has
// no model context — the anomaly is a property of a model's pricing.
func (g *creditGate) clampCount(model string) int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	if a := g.anomalies[model]; a != nil {
		return a.clamps
	}
	return 0
}

func (g *creditGate) pricingFailCount(model string) int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	if a := g.anomalies[model]; a != nil {
		return a.pricingFails
	}
	return 0
}

// --- blast-radius containment (stage 2b.3c piece 3, big-dog shape 6) ------------
//
// The gate shares ONE process (+ GPU) with request serving AND the Ranker
// model-admission loop, so a gate/ledger fault must NEVER propagate to crash or
// block those. recoverGateFault is the containment BOUNDARY: it runs a gate call
// under recover so a panic (or a would-be crash from an internal bug) is caught,
// counted, WARNed, and turned into a fail-CLOSED deny by default — or, only when
// the canary fail-open config is explicitly set, a served pass-through. The fault
// stays inside the gate call; serving and the Ranker are untouched.

// recoverGateFault runs fn under panic containment. On a panic it records the
// fault (count + WARN) and reports faulted=true with the gate's fail-open policy;
// on the normal path it returns (false, false). The named returns are set from the
// deferred recover, so a panic in fn never escapes this call.
func (g *creditGate) recoverGateFault(model string, fn func()) (faulted bool, failOpen bool) {
	defer func() {
		if r := recover(); r != nil {
			faulted = true
			failOpen = g.failOpen
			g.recordGateFault(model, r)
		}
	}()
	fn()
	return false, false
}

// guardedAdmit is admit() inside the containment boundary: a gate/ledger fault
// recovers to a fail-CLOSED admitFault (deny) by default, or — only under the
// explicit canary fail-open — a served no-op reservation (admitOff, un-metered).
// The wiring calls THIS, never admit() directly, so a gate bug can't crash serving.
func (g *creditGate) guardedAdmit(key, model string, promptTokens, requestedMax int64) (rsv *creditReservation, d admitDecision) {
	faulted, failOpen := g.recoverGateFault(model, func() {
		rsv, d = g.admit(key, model, promptTokens, requestedMax)
	})
	if faulted {
		if failOpen {
			return &creditReservation{}, admitOff // canary: serve un-metered (deliberate opt-in)
		}
		return nil, admitFault // fail-closed: deny
	}
	return rsv, d
}

// guardedResolve runs a resolve (trueUp or release) inside the containment
// boundary. If it panics, the hold is STILL released (idempotent, so it composes
// with the handler's `defer release()` — big-dog shape-6 lock 3: recover-to-deny
// AND hold-released both hold) and the fault is counted. A no-op reservation (gate
// OFF) has no gate to guard against, so fn (inert) just runs.
//
// The compensating release is ITSELF guarded: it touches the same ledger, so when
// the LEDGER is the fault source, release() panics AGAIN — an unguarded second
// panic would escape and crash serving + the Ranker (the exact blast-radius event
// this piece prevents). A hold stranded by a double-faulting ledger is a bounded,
// observable loss (counted); a crash is not. (big-dog #36)
func (rsv *creditReservation) guardedResolve(fn func()) {
	if rsv == nil {
		return
	}
	if rsv.gate == nil {
		fn() // OFF no-op reservation — fn is inert
		return
	}
	if faulted, _ := rsv.gate.recoverGateFault(rsv.model, fn); faulted {
		rsv.gate.recoverGateFault(rsv.model, func() { rsv.release() })
	}
}

// recordGateFault counts a recovered gate/ledger fault + WARNs. Global (not
// per-model): the danger is a gate silently recover-DENYING everything (e.g. the
// ledger is unreachable), which must look different from a normal fail-closed deny.
func (g *creditGate) recordGateFault(model string, r any) {
	g.mu.Lock()
	g.gatePanics++
	n := g.gatePanics
	g.mu.Unlock()
	if g.logger != nil {
		g.logger.Warnf("credit gate: FAULT recovered (model %q) — %v; contained to a %s (serving + Ranker unaffected). gatePanics=%d", model, r, g.faultAction(), n)
	}
}

func (g *creditGate) faultAction() string {
	if g.failOpen {
		return "canary FAIL-OPEN (served un-metered)"
	}
	return "fail-CLOSED deny"
}

// gatePanicCount is the total recovered gate/ledger faults (blast-radius
// observability). A nonzero-and-climbing value = the gate is faulting, not a
// healthy fail-closed — page on it, don't mistake it for normal denies.
func (g *creditGate) gatePanicCount() int64 {
	g.mu.Lock()
	defer g.mu.Unlock()
	return g.gatePanics
}
