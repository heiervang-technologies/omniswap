package proxy

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
)

func reqWithCtx(vals map[proxyCtxKey]any) *http.Request {
	r := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	ctx := r.Context()
	for k, v := range vals {
		ctx = context.WithValue(ctx, k, v)
	}
	return r.WithContext(ctx)
}

func TestKeyFingerprintFromContext(t *testing.T) {
	r := reqWithCtx(map[proxyCtxKey]any{proxyCtxKey("key_fingerprint"): "ab12cd34ef56"})
	assert.Equal(t, "ab12cd34ef56", keyFingerprintFromContext(r))
	assert.Empty(t, keyFingerprintFromContext(reqWithCtx(nil)), "unset reads empty")
	assert.Empty(t, keyFingerprintFromContext(nil), "nil request is safe")
}

// TestServedBy_RoundTrip: the dispatch writes the serving node through the shared
// *string slot; the metrics read it back after the response.
func TestServedBy_RoundTrip(t *testing.T) {
	slot := new(string)
	r := reqWithCtx(map[proxyCtxKey]any{proxyCtxKey("servedBy"): slot})

	assert.Empty(t, nodeFromContext(r), "unwritten slot reads empty")
	setServedBy(r, "amber")
	assert.Equal(t, "amber", nodeFromContext(r))
	assert.Equal(t, "amber", *slot, "writes through the shared pointer the metrics read")
}

// TestServedBy_NoSlotNoPanic: without the slot (debit logging off) setServedBy is
// a silent no-op — it must never affect routing or panic.
func TestServedBy_NoSlotNoPanic(t *testing.T) {
	r := reqWithCtx(nil) // no servedBy slot installed
	assert.NotPanics(t, func() { setServedBy(r, "amber") })
	assert.Empty(t, nodeFromContext(r))

	assert.NotPanics(t, func() { setServedBy(nil, "x") })
	assert.Empty(t, nodeFromContext(nil), "nil request is safe")
}
