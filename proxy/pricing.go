package proxy

// pricing.go — token→credit pricing for the billing gate (cloud
// project_gems_monetization, stage 2b.3a). PURE + dormant: the gate hook prices
// a request's WORST-CASE cost at admission (to Reserve high) and the ACTUAL cost
// at true-up (to bill accurately). Nothing constructs it yet.
//
// Rounding is UP at both reserve and bill — a fractional credit always rounds in
// the house's favour, so a request can never cost less than it's charged
// (revenue-safe, big-dog item 5). Combined with reserve-worst-case /
// true-up-actual, the hold always covers the bill: ReserveCost >= ActualCost for
// any actual completion within the reserved upper bound, so the ledger's
// clamp-to-cost never bites and there is no over-spend.
//
// An UNPRICED model returns ok=false: the gate cannot price it, so it cannot
// safely enforce, so it fail-closes (never serve un-metered / free).

// ModelRate is a model's credit price. MaxOutputTokens is the completion ceiling
// used to reserve worst-case when a request sets no (or an oversized) max_tokens
// — the true upper bound, never an expected-value guess.
type ModelRate struct {
	InputPer1k      int64 // credits per 1000 prompt (input) tokens
	OutputPer1k     int64 // credits per 1000 completion (output) tokens
	MaxOutputTokens int64 // model's output/context ceiling; the worst-case completion size
}

type priceBook struct {
	rates map[string]ModelRate
}

func newPriceBook(rates map[string]ModelRate) *priceBook {
	if rates == nil {
		rates = map[string]ModelRate{}
	}
	return &priceBook{rates: rates}
}

// has reports whether the model is priced (the gate fail-closes when not).
func (pb *priceBook) has(model string) bool {
	_, ok := pb.rates[model]
	return ok
}

// ReserveCost is the WORST-CASE cost to hold at admission (rounded UP), or
// (0,false) if the model is unpriced. The completion upper bound is the request's
// max_tokens when it is set AND below the model ceiling, otherwise the model
// ceiling — so an absent or oversized max_tokens reserves against the true
// ceiling, never an expected guess.
func (pb *priceBook) ReserveCost(model string, promptTokens, requestedMax int64) (int64, bool) {
	r, ok := pb.rates[model]
	if !ok {
		return 0, false
	}
	if promptTokens < 0 {
		promptTokens = 0
	}
	completionUpper := r.MaxOutputTokens
	if requestedMax > 0 && requestedMax < completionUpper {
		completionUpper = requestedMax
	}
	return ceilDiv(promptTokens*r.InputPer1k, 1000) + ceilDiv(completionUpper*r.OutputPer1k, 1000), true
}

// ActualCost is the billed cost from the completed request's real token counts
// (rounded UP), or (0,false) if unpriced. For any actual completion <= the
// reserved upper bound, ActualCost <= the matching ReserveCost.
func (pb *priceBook) ActualCost(model string, promptTokens, completionTokens int64) (int64, bool) {
	r, ok := pb.rates[model]
	if !ok {
		return 0, false
	}
	if promptTokens < 0 {
		promptTokens = 0
	}
	if completionTokens < 0 {
		completionTokens = 0
	}
	return ceilDiv(promptTokens*r.InputPer1k, 1000) + ceilDiv(completionTokens*r.OutputPer1k, 1000), true
}

// ceilDiv returns ceil(a/b) for a>=0, b>0 (0 for a<=0). Integer-only so pricing
// is exact + deterministic (no float rounding surprises on the money path).
func ceilDiv(a, b int64) int64 {
	if a <= 0 {
		return 0
	}
	return (a + b - 1) / b
}
