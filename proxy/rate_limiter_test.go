package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestRateLimiter_DisabledByDefault(t *testing.T) {
	rl := newRateLimiter(0, nil)
	assert.False(t, rl.enabled(), "no limit + no overrides => disabled")
	for i := 0; i < 1000; i++ {
		assert.True(t, rl.allow("anyone"), "disabled limiter is a no-op")
	}
}

func TestRateLimiter_DefaultLimit(t *testing.T) {
	rl := newRateLimiter(3, nil)
	assert.True(t, rl.enabled())
	assert.True(t, rl.allow("shay"))
	assert.True(t, rl.allow("shay"))
	assert.True(t, rl.allow("shay"))
	assert.False(t, rl.allow("shay"), "4th request in the window is blocked")
	assert.True(t, rl.allow("markus"), "a different client has its own window")
}

func TestRateLimiter_PerLabelOverride(t *testing.T) {
	rl := newRateLimiter(1, map[string]int{"markus": 100})
	// override wins for markus
	for i := 0; i < 50; i++ {
		assert.True(t, rl.allow("markus"), "override 100 not yet hit")
	}
	// default applies to others
	assert.True(t, rl.allow("shay"))
	assert.False(t, rl.allow("shay"), "default limit 1 hit")
}

func TestUsageMeter_Reset(t *testing.T) {
	m := NewUsageMeter("", nil)
	m.Record("shay", "gemma-4-12b", "NO", "1.2.3.4", 10, 5)
	m.Reset()
	snap := m.Snapshot()
	assert.Empty(t, snap["clients"].(map[string]*clientUsage))
	assert.Empty(t, snap["by_ip"].(map[string]*ipUsage))
	assert.Empty(t, snap["by_country"].(map[string]*modelUsage))
}
