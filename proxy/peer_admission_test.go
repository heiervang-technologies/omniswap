package proxy

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
)

func closedDone() <-chan struct{} {
	ch := make(chan struct{})
	close(ch)
	return ch
}

func TestPeerAdmission_DisabledByDefault(t *testing.T) {
	pa := newPeerAdmission(0, 0, []string{"opal", "amber"})
	assert.False(t, pa.enabled())
	// acquire is a no-op pass-through and never blocks/rejects when disabled.
	rel, ok := pa.acquire("opal", nil)
	assert.True(t, ok)
	rel() // must be safe to call
}

func TestPeerAdmission_BoundsInflightNoWait(t *testing.T) {
	pa := newPeerAdmission(2, 0, []string{"opal"}) // no queue wait
	assert.True(t, pa.enabled())

	r1, ok1 := pa.acquire("opal", nil)
	assert.True(t, ok1)
	r2, ok2 := pa.acquire("opal", nil)
	assert.True(t, ok2)

	// 3rd exceeds the cap and there's no queue wait -> immediate reject.
	r3, ok3 := pa.acquire("opal", nil)
	assert.False(t, ok3)
	assert.Nil(t, r3)

	// releasing one frees exactly one slot.
	r1()
	r4, ok4 := pa.acquire("opal", nil)
	assert.True(t, ok4)
	r2()
	r4()
}

func TestPeerAdmission_PerPeerIsolation(t *testing.T) {
	pa := newPeerAdmission(1, 0, []string{"opal", "amber"})
	r1, ok1 := pa.acquire("opal", nil)
	assert.True(t, ok1)
	// opal is full, but amber has its own slot.
	_, okOpal := pa.acquire("opal", nil)
	assert.False(t, okOpal)
	rAmber, okAmber := pa.acquire("amber", nil)
	assert.True(t, okAmber)
	r1()
	rAmber()
}

func TestPeerAdmission_QueueTimeoutRejects(t *testing.T) {
	pa := newPeerAdmission(1, 40*time.Millisecond, []string{"opal"})
	r1, ok1 := pa.acquire("opal", nil)
	assert.True(t, ok1)

	start := time.Now()
	r2, ok2 := pa.acquire("opal", nil) // full; waits ~40ms then rejects
	waited := time.Since(start)
	assert.False(t, ok2)
	assert.Nil(t, r2)
	assert.GreaterOrEqual(t, waited, 35*time.Millisecond)
	r1()
}

func TestPeerAdmission_QueueWaitThenAdmit(t *testing.T) {
	pa := newPeerAdmission(1, 500*time.Millisecond, []string{"opal"})
	r1, _ := pa.acquire("opal", nil)
	// free the slot shortly; the waiter should then be admitted well within the wait.
	go func() { time.Sleep(30 * time.Millisecond); r1() }()
	start := time.Now()
	r2, ok2 := pa.acquire("opal", nil)
	assert.True(t, ok2)
	assert.Less(t, time.Since(start), 400*time.Millisecond)
	r2()
}

func TestPeerAdmission_ContextCancelRejects(t *testing.T) {
	pa := newPeerAdmission(1, time.Second, []string{"opal"})
	r1, _ := pa.acquire("opal", nil)
	defer r1()
	// peer full; a client that's already gone bails immediately, not after 1s.
	start := time.Now()
	r2, ok2 := pa.acquire("opal", closedDone())
	assert.False(t, ok2)
	assert.Nil(t, r2)
	assert.Less(t, time.Since(start), 200*time.Millisecond)
}

func TestPeerAdmission_UnknownPeerPassThrough(t *testing.T) {
	pa := newPeerAdmission(1, 0, []string{"opal"})
	// A peer not in the admission set must not block traffic (bookkeeping gap).
	rel, ok := pa.acquire("ghost", nil)
	assert.True(t, ok)
	rel()
}

func TestWriteTooManyRequests(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	req.Header.Set("Origin", "http://llm.ht.local")
	writeTooManyRequests(rec, req, 3*time.Second)

	assert.Equal(t, http.StatusTooManyRequests, rec.Code)
	assert.Equal(t, "3", rec.Header().Get("Retry-After"))
	// Echoes Origin so a browser can read the 429 (mirrors #23).
	assert.Equal(t, "http://llm.ht.local", rec.Header().Get("Access-Control-Allow-Origin"))
}

func TestWriteTooManyRequests_RetryAfterFloorAndNoOrigin(t *testing.T) {
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	writeTooManyRequests(rec, req, 200*time.Millisecond) // sub-second -> floor to 1
	assert.Equal(t, "1", rec.Header().Get("Retry-After"))
	// No Origin on the request -> no ACAO header emitted.
	assert.Empty(t, rec.Header().Get("Access-Control-Allow-Origin"))
}
