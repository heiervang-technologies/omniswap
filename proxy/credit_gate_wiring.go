package proxy

import (
	"context"
	"net/http"

	"github.com/tidwall/gjson"
)

// credit_gate_wiring.go — the request-path wiring for the prepaid-credit billing
// gate (cloud project_gems_monetization, stage 2b.3c: the request-path slice). The
// gate object (admit/trueUp/release + blast-radius containment) lives in
// credit_gate.go; this file is the glue that reserves a hold at admission and
// resolves it on EVERY exit, driven from proxyInferenceHandler.
//
// ALL of this is inert when the gate is OFF: the handler guards the whole block on
// creditGate.enforcing() (false by default), so the request path is byte-identical
// until CreditGateEnforce is explicitly set. The estimator + resolve are pure and
// unit-tested so the money math is provable without standing up a live pool.

// creditCapture is a request-scoped slot the metrics monitor fills synchronously
// (via setCreditCapture*) with the served outcome, so the deferred resolve bills
// from THIS request's real numbers with no cross-request correlation. Mirrors the
// servedBy *string slot: installed only when the gate is enforcing, absent (and
// every setter a no-op) otherwise.
type creditCapture struct {
	status int // upstream HTTP status: 200 => served/billable, anything else => refund
	input  int // metered prompt tokens (0 => no usable usage returned)
	output int // metered completion tokens (0 => no usable usage returned)
}

// contextWithCreditCapture installs the request-scoped capture slot. The handler
// calls this only while enforcing; without it every setter/reader below no-ops.
func contextWithCreditCapture(ctx context.Context, cap *creditCapture) context.Context {
	return context.WithValue(ctx, proxyCtxKey("creditCapture"), cap)
}

func creditCaptureFromContext(r *http.Request) *creditCapture {
	if r == nil {
		return nil
	}
	if c, ok := r.Context().Value(proxyCtxKey("creditCapture")).(*creditCapture); ok {
		return c
	}
	return nil
}

// setCreditCaptureStatus records the served HTTP status. Slot-optional + nil-safe:
// a request without the slot (the default, gate OFF) is a silent no-op.
func setCreditCaptureStatus(r *http.Request, status int) {
	if c := creditCaptureFromContext(r); c != nil {
		c.status = status
	}
}

// setCreditCaptureTokens records the metered token counts. Slot-optional + nil-safe.
func setCreditCaptureTokens(r *http.Request, input, output int) {
	if c := creditCaptureFromContext(r); c != nil {
		c.input = input
		c.output = output
	}
}

// creditPromptUpperBound returns a conservative UPPER bound on the request's prompt
// token count from the raw body — used only to size the worst-case reservation at
// admission (true-up corrects it down to the real usage). A token encodes >= 1 byte
// of content, so the byte length of the prompt-bearing fields is a hard upper bound
// on prompt tokens. We deliberately bound HIGH: over-reserving is a refundable hold,
// but under-reserving would let a request slip in under-priced. Cheap — the raw JSON
// span of `messages` + `prompt` already contains every content byte, no per-message
// iteration.
func creditPromptUpperBound(body []byte) int64 {
	var n int64
	if m := gjson.GetBytes(body, "messages"); m.Exists() {
		n += int64(len(m.Raw))
	}
	if p := gjson.GetBytes(body, "prompt"); p.Exists() {
		n += int64(len(p.Raw))
	}
	if n == 0 {
		n = int64(len(body)) // no recognized prompt field -> the whole body is the bound
	}
	return n
}

// creditRequestedMax reads the request's completion cap (max_tokens, or the OpenAI
// max_completion_tokens alias). Returns 0 when absent — ReserveCost treats 0 as
// "use the model's ceiling", so an unset cap reserves against the true worst case.
func creditRequestedMax(body []byte) int64 {
	if v := gjson.GetBytes(body, "max_tokens"); v.Exists() {
		return v.Int()
	}
	if v := gjson.GetBytes(body, "max_completion_tokens"); v.Exists() {
		return v.Int()
	}
	return 0
}

// resolveReservation resolves a held reservation from the captured outcome. Called
// inside guardedResolve (blast-radius contained) on EVERY handler exit:
//
//   - not served (dispatch error / timeout / disconnect / panic) or upstream
//     non-200 -> release: nothing billable reached the client, full refund.
//   - served 200 with usage -> true-up the ACTUAL metered cost.
//   - served 200 but NO usable usage (backend omitted usage / empty body) ->
//     true-up the RESERVED worst case, never 0/free (big-dog). The completion is the
//     unmeasured, expensive part; billing anything less than the reserve here would
//     leak in the completion direction, so we charge the conservative upper bound.
func resolveReservation(rsv *creditReservation, cap *creditCapture) {
	if cap == nil || cap.status != http.StatusOK {
		rsv.release()
		return
	}
	if cap.input <= 0 && cap.output <= 0 {
		rsv.trueUpReserved()
		return
	}
	rsv.trueUp(int64(cap.input), int64(cap.output))
}

// creditDenyStatus maps a non-served admit decision to an HTTP status. Insufficient
// balance and an unpriced model are client-facing 402s (payment required); a gate
// fault is a 503 (our side, retryable). served() decisions never reach here.
func creditDenyStatus(d admitDecision) int {
	switch d {
	case admitFault:
		return http.StatusServiceUnavailable
	default:
		return http.StatusPaymentRequired
	}
}

// creditDenyMessage is the client-facing reason for a denied request.
func creditDenyMessage(d admitDecision) string {
	switch d {
	case admitInsufficient:
		return "insufficient credits for this request"
	case admitUnpriced:
		return "model is not available for prepaid billing"
	case admitFault:
		return "billing temporarily unavailable, please retry"
	default:
		return "request denied by billing"
	}
}
