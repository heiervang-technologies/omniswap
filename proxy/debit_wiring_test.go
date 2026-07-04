package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestEmitDebit_MapsMetricsToEvent: onRecord's emit maps TokenMetrics field-for-
// field into a DebitEvent, mints a request_id, and the log assigns the Seq.
func TestEmitDebit_MapsMetricsToEvent(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	dl := newDebitLog(path, testLogger)
	em := newDebitEmitter(dl, 16, testLogger)
	pm := &ProxyManager{proxyLogger: testLogger, debitLog: dl, debitEmitter: em}

	pm.emitDebit(TokenMetrics{
		Timestamp:      time.Unix(1_700_000_000, 0).UTC(),
		KeyFingerprint: "ab12cd34ef56",
		Client:         "shay",
		Model:          "gemma-4-12b-256k",
		InputTokens:    1234,
		OutputTokens:   567,
		Node:           "amber",
		Country:        "NO",
	})
	em.Close() // drains the writer
	_ = dl.close()

	got := dl.Since(0)
	require.Len(t, got, 1)
	assert.NotEmpty(t, got[0].RequestID, "request_id minted")
	assert.Equal(t, uint64(1), got[0].Seq, "seq assigned by the log")
	assert.Equal(t, "ab12cd34ef56", got[0].KeyFingerprint)
	assert.Equal(t, "shay", got[0].ClientLabel)
	assert.Equal(t, "gemma-4-12b-256k", got[0].Model)
	assert.Equal(t, int64(1234), got[0].InputTokens)
	assert.Equal(t, int64(567), got[0].OutputTokens)
	assert.Equal(t, "amber", got[0].Node)
	assert.Equal(t, "NO", got[0].Country)
	assert.Equal(t, time.Unix(1_700_000_000, 0).UTC(), got[0].Ts)
}

// TestEmitDebit_DisabledNoOp: with debit logging off (nil emitter), emit is a
// safe no-op — the land-dark default.
func TestEmitDebit_DisabledNoOp(t *testing.T) {
	pm := &ProxyManager{proxyLogger: testLogger} // no debitEmitter
	assert.NotPanics(t, func() { pm.emitDebit(TokenMetrics{Model: "x", InputTokens: 5}) })
}

func pullReq(t *testing.T, pm *ProxyManager, client, query string) (int, map[string]any) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Set("client", client)
	c.Request = httptest.NewRequest("GET", "/debits"+query, nil)
	pm.debitsPullHandler(c)
	var body map[string]any
	if w.Body.Len() > 0 {
		_ = json.Unmarshal(w.Body.Bytes(), &body)
	}
	return w.Code, body
}

// TestDebitsPullHandler covers the billing pull: admin-gated, cursor-filtered,
// reports high_water, and a clean disabled shape.
func TestDebitsPullHandler(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	dl := newDebitLog(path, testLogger)
	em := newDebitEmitter(dl, 16, testLogger)
	pm := &ProxyManager{proxyLogger: testLogger, debitLog: dl, debitEmitter: em, config: config.Config{}}
	pm.emitDebit(TokenMetrics{Model: "m", Client: "shay", InputTokens: 1})
	pm.emitDebit(TokenMetrics{Model: "m", Client: "shay", InputTokens: 2})
	em.Close() // drain; log stays open for reads
	defer dl.close()

	// admin, no cursor -> both events + high_water 2
	code, body := pullReq(t, pm, "admin", "")
	require.Equal(t, http.StatusOK, code)
	assert.Equal(t, true, body["enabled"])
	assert.Equal(t, float64(2), body["high_water"])
	assert.Len(t, body["events"].([]any), 2)

	// admin, cursor past seq 1 -> only seq 2
	_, body = pullReq(t, pm, "admin", "?since=1")
	assert.Len(t, body["events"].([]any), 1)

	// non-admin -> forbidden
	code, _ = pullReq(t, pm, "someone", "")
	assert.Equal(t, http.StatusForbidden, code)
}

// TestDebitsPullHandler_Disabled: reports enabled:false + empty when off.
func TestDebitsPullHandler_Disabled(t *testing.T) {
	pm := &ProxyManager{proxyLogger: testLogger, debitLog: newDebitLog("", testLogger), config: config.Config{}}
	code, body := pullReq(t, pm, "admin", "")
	require.Equal(t, http.StatusOK, code)
	assert.Equal(t, false, body["enabled"])
	assert.Empty(t, body["events"])
}
