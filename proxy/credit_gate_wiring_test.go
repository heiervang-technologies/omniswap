package proxy

import (
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- estimator: prompt upper bound is a HARD upper bound, cheap, never under ------

func TestCreditPromptUpperBound_IsUpperBound(t *testing.T) {
	// A token encodes >= 1 byte of content, so the bound must be >= the content's
	// rune count for any prompt-bearing field.
	cases := []struct {
		name    string
		body    string
		content string // the actual prompt text the bound must dominate
	}{
		{"chat messages", `{"model":"m","messages":[{"role":"user","content":"hello world"}]}`, "hello world"},
		{"legacy prompt", `{"model":"m","prompt":"summarize this please"}`, "summarize this please"},
		{"both fields", `{"model":"m","messages":[{"role":"user","content":"abc"}],"prompt":"def"}`, "abcdef"},
		{"multi message", `{"model":"m","messages":[{"role":"system","content":"be terse"},{"role":"user","content":"why?"}]}`, "be tersewhy?"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := creditPromptUpperBound([]byte(tc.body))
			assert.Greater(t, got, int64(0), "a prompt-bearing body must reserve a positive bound")
			assert.GreaterOrEqual(t, got, int64(len([]rune(tc.content))),
				"the bound must dominate the actual content length (never under-reserve)")
		})
	}
}

func TestCreditPromptUpperBound_NoPromptFieldFallsBackToBodyLen(t *testing.T) {
	// A body with no recognized prompt field still gets a positive, body-sized bound
	// — we never reserve 0 for a request we're about to serve.
	body := []byte(`{"model":"m","temperature":0.7}`)
	assert.Equal(t, int64(len(body)), creditPromptUpperBound(body))
}

// --- requestedMax: reads the cap, aliases, absent => 0 (=> ceiling in ReserveCost) -

func TestCreditRequestedMax(t *testing.T) {
	assert.Equal(t, int64(128), creditRequestedMax([]byte(`{"max_tokens":128}`)))
	assert.Equal(t, int64(256), creditRequestedMax([]byte(`{"max_completion_tokens":256}`)), "OpenAI alias honored")
	assert.Equal(t, int64(128), creditRequestedMax([]byte(`{"max_tokens":128,"max_completion_tokens":256}`)), "max_tokens wins when both set")
	assert.Equal(t, int64(0), creditRequestedMax([]byte(`{"model":"m"}`)), "absent => 0 => ReserveCost uses the ceiling")
}

// --- resolve: bill actual / estimate / refund, driven by the captured outcome -----
//
// gateWithBalance builds model "m" at {input 2, output 8/1k, ceiling 4096}; the
// reserve for admit("k","m",1000,500) is 6 and the true-up for (1000,250) is 4
// (see credit_gate_blast_test.go). Balance starts at 100.

func TestResolveReservation_ServedWithUsage_BillsActual(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, d := g.admit("k", "m", 1000, 500)
	require.Equal(t, admitOK, d)
	require.Equal(t, int64(94), ledger.Available("k"), "hold taken")

	resolveReservation(rsv, &creditCapture{status: http.StatusOK, input: 1000, output: 250})
	assert.Equal(t, int64(96), ledger.Available("k"), "served 200 + usage => billed ACTUAL (4), hold released")
}

func TestResolveReservation_Served200NoUsage_BillsReservedEstimate(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, _ := g.admit("k", "m", 1000, 500)
	require.Equal(t, int64(94), ledger.Available("k"))

	// 200 but the backend returned no usable usage (input=output=0): charge the
	// reserved worst case, NEVER free (the completion is the unmeasured, costly part).
	resolveReservation(rsv, &creditCapture{status: http.StatusOK, input: 0, output: 0})
	assert.Equal(t, int64(94), ledger.Available("k"), "no-usage => billed the RESERVED estimate (6), not 0/free")
}

func TestResolveReservation_UpstreamNon200_Refunds(t *testing.T) {
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, _ := g.admit("k", "m", 1000, 500)
	require.Equal(t, int64(94), ledger.Available("k"))

	resolveReservation(rsv, &creditCapture{status: http.StatusBadGateway})
	assert.Equal(t, int64(100), ledger.Available("k"), "upstream non-200 => full refund, nothing billable served")
}

func TestResolveReservation_NilCapture_Refunds(t *testing.T) {
	// A nil capture is the "never reached serving" exit (dispatch error / timeout /
	// disconnect / panic before wrapHandler filled the slot) => full refund.
	g, ledger := gateWithBalance(true, "k", 100)
	rsv, _ := g.admit("k", "m", 1000, 500)
	require.Equal(t, int64(94), ledger.Available("k"))

	resolveReservation(rsv, nil)
	assert.Equal(t, int64(100), ledger.Available("k"), "no capture => treated as not-served => full refund")
}

func TestResolveReservation_OffReservation_NeverTouchesLedger(t *testing.T) {
	// Gate OFF: admit returns a no-op reservation (nil inner). Resolve must be inert
	// regardless of the captured outcome — the byte-identical-when-OFF guarantee.
	g, ledger := gateWithBalance(false, "k", 100)
	rsv, d := g.admit("k", "m", 1000, 500)
	require.Equal(t, admitOff, d)
	for _, cap := range []*creditCapture{
		{status: http.StatusOK, input: 1000, output: 250},
		{status: http.StatusOK, input: 0, output: 0},
		{status: http.StatusBadGateway},
		nil,
	} {
		resolveReservation(rsv, cap)
	}
	assert.Equal(t, int64(100), ledger.Available("k"), "OFF reservation never debits or refunds")
}

// --- capture slot: request-scoped, nil-safe, no-op without the slot ---------------

func TestCreditCapture_SlotOptionalAndNilSafe(t *testing.T) {
	// No slot installed (gate OFF): the setters are silent no-ops and the reader
	// returns nil — a request without billing is untouched.
	req, _ := http.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader("{}"))
	assert.Nil(t, creditCaptureFromContext(req))
	assert.NotPanics(t, func() {
		setCreditCaptureStatus(req, 200)
		setCreditCaptureTokens(req, 10, 20)
	})
	assert.Nil(t, creditCaptureFromContext(req), "still no slot; setters were inert")

	// Slot installed: setters fill it, reader returns it.
	cap := &creditCapture{}
	ctx := contextWithCreditCapture(req.Context(), cap)
	req = req.WithContext(ctx)
	require.Same(t, cap, creditCaptureFromContext(req))
	setCreditCaptureStatus(req, 200)
	setCreditCaptureTokens(req, 10, 20)
	assert.Equal(t, 200, cap.status)
	assert.Equal(t, 10, cap.input)
	assert.Equal(t, 20, cap.output)
}

// --- deny mapping: served() decisions never reach here; each deny is client-legible

func TestCreditDenyStatusAndMessage(t *testing.T) {
	assert.Equal(t, http.StatusPaymentRequired, creditDenyStatus(admitInsufficient))
	assert.Equal(t, http.StatusPaymentRequired, creditDenyStatus(admitUnpriced))
	assert.Equal(t, http.StatusServiceUnavailable, creditDenyStatus(admitFault))
	for _, d := range []admitDecision{admitInsufficient, admitUnpriced, admitFault} {
		assert.NotEmpty(t, creditDenyMessage(d))
	}
}
