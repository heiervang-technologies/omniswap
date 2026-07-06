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
