package proxy

import (
	"sync"
	"time"
)

// rate_limiter.go — per-client (per-label) inference rate limiting for the pool.
//
// DISABLED by default: with no default limit and no overrides, enabled() is
// false and the limiter is never consulted, so it CANNOT affect traffic until an
// operator sets rateLimitPerMin (or a per-label override). A fixed 1-minute
// window per client; a non-positive limit means unlimited.

type rlWindow struct {
	start time.Time
	count int
}

type rateLimiter struct {
	mu        sync.Mutex
	perMin    int
	overrides map[string]int
	windows   map[string]*rlWindow
}

func newRateLimiter(perMin int, overrides map[string]int) *rateLimiter {
	return &rateLimiter{perMin: perMin, overrides: overrides, windows: map[string]*rlWindow{}}
}

// enabled reports whether any limit is configured (a default or any override).
func (rl *rateLimiter) enabled() bool {
	return rl != nil && (rl.perMin > 0 || len(rl.overrides) > 0)
}

// limitFor returns the per-minute cap for a client (override beats default).
// 0 / negative = unlimited.
func (rl *rateLimiter) limitFor(client string) int {
	if rl.overrides != nil {
		if v, ok := rl.overrides[client]; ok {
			return v
		}
	}
	return rl.perMin
}

// allow records a request for client and reports whether it is within the limit.
func (rl *rateLimiter) allow(client string) bool {
	lim := rl.limitFor(client)
	if lim <= 0 {
		return true
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()
	now := time.Now()
	w := rl.windows[client]
	if w == nil || now.Sub(w.start) >= time.Minute {
		rl.windows[client] = &rlWindow{start: now, count: 1}
		return true
	}
	if w.count >= lim {
		return false
	}
	w.count++
	return true
}
