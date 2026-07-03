package proxy

import (
	"net/http"
	"strconv"
	"time"
)

// peer_admission.go — per-peer in-flight admission control for the pool.
//
// DISABLED by default: maxInflight <= 0 -> enabled() is false and acquire() is a
// no-op pass-through, so it CANNOT affect traffic until an operator sets
// maxInflightPerPeer (land-dark). It bounds the number of requests concurrently
// in flight to EACH peer; a request that can't get a slot within queueTimeout is
// rejected so the caller returns 429 + Retry-After instead of piling onto a
// saturated Pascal gem (the baseline 502 + long-tail p99 at saturation).
// queueTimeout <= 0 -> reject immediately when the peer is full (no queue wait).
//
// Per-peer slot channels are built once at construction from the known peer id
// set and never mutated afterward, so acquire() needs no lock on the hot path (a
// buffered channel is itself the counting semaphore).

type peerAdmission struct {
	maxInflight  int
	queueTimeout time.Duration
	slots        map[string]chan struct{}
}

func newPeerAdmission(maxInflight int, queueTimeout time.Duration, peerIDs []string) *peerAdmission {
	pa := &peerAdmission{
		maxInflight:  maxInflight,
		queueTimeout: queueTimeout,
		slots:        make(map[string]chan struct{}, len(peerIDs)),
	}
	if maxInflight > 0 {
		for _, id := range peerIDs {
			pa.slots[id] = make(chan struct{}, maxInflight)
		}
	}
	return pa
}

// enabled reports whether admission control is active (a positive per-peer cap).
func (pa *peerAdmission) enabled() bool {
	return pa != nil && pa.maxInflight > 0
}

// acquire tries to take an in-flight slot for peerID, waiting up to queueTimeout
// and honoring request cancellation via done. It returns a release func + true on
// success, or nil + false if the peer stays saturated past the wait (the caller
// should 429). A no-op pass-through when disabled or the peer is unknown.
func (pa *peerAdmission) acquire(peerID string, done <-chan struct{}) (func(), bool) {
	if !pa.enabled() {
		return func() {}, true
	}
	ch := pa.slots[peerID]
	if ch == nil {
		// Peer not in the admission set (shouldn't happen for a routed peer) —
		// don't block real traffic on a bookkeeping gap.
		return func() {}, true
	}
	// Fast path: a slot is free right now.
	select {
	case ch <- struct{}{}:
		return func() { <-ch }, true
	default:
	}
	if pa.queueTimeout <= 0 {
		return nil, false // full, no queue wait configured -> reject now
	}
	timer := time.NewTimer(pa.queueTimeout)
	defer timer.Stop()
	select {
	case ch <- struct{}{}:
		return func() { <-ch }, true
	case <-timer.C:
		return nil, false
	case <-done:
		// Client went away while queued — treat as not admitted; the caller's
		// 429 write is a harmless no-op on an already-closed connection.
		return nil, false
	}
}

// writeTooManyRequests emits a 429 with a Retry-After hint on a raw
// ResponseWriter (the peer-dispatch path has no gin.Context). It echoes the
// request Origin into Access-Control-Allow-Origin so a browser can actually read
// the 429 body (mirrors #23, which did the same for auth 401/429s).
func writeTooManyRequests(w http.ResponseWriter, r *http.Request, retryAfter time.Duration) {
	if origin := r.Header.Get("Origin"); origin != "" {
		w.Header().Set("Access-Control-Allow-Origin", origin)
		w.Header().Set("Vary", "Origin")
	}
	secs := int(retryAfter / time.Second)
	if secs < 1 {
		secs = 1
	}
	w.Header().Set("Retry-After", strconv.Itoa(secs))
	http.Error(w, "peer at capacity: too many concurrent requests in flight, retry shortly", http.StatusTooManyRequests)
}
