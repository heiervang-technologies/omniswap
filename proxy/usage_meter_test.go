package proxy

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUsageMeter_RecordAggregatePersist(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "usage.json")

	m := NewUsageMeter(path, nil)
	m.Record("shay", "gemma-4-12b", "NO", 100, 50)
	m.Record("shay", "gemma-4-12b", "NO", 10, 5)
	m.Record("shay", "qwen3.6-35b-a3b", "NO", 7, 3)
	m.Record("markus", "gemma-4-12b", "US", 1000, 2000)
	m.Record("", "gemma-4-12b", "", 1, 1) // empty -> client "unknown", country "local"

	snap := m.Snapshot()
	clients := snap["clients"].(map[string]*clientUsage)

	// by-country rollup
	byCountry := snap["by_country"].(map[string]*modelUsage)
	assert.EqualValues(t, 3, byCountry["NO"].Requests, "shay's 3 from NO")
	assert.EqualValues(t, 1, byCountry["US"].Requests, "markus from US")
	require.Contains(t, byCountry, "local", "empty country buckets as local")

	// per-client totals
	require.Contains(t, clients, "shay")
	assert.EqualValues(t, 3, clients["shay"].Requests)
	assert.EqualValues(t, 117, clients["shay"].InputTokens)
	assert.EqualValues(t, 58, clients["shay"].OutputTokens)
	// per-model breakdown
	assert.EqualValues(t, 110, clients["shay"].ByModel["gemma-4-12b"].InputTokens)
	assert.EqualValues(t, 7, clients["shay"].ByModel["qwen3.6-35b-a3b"].InputTokens)
	// markus + unknown bucketing
	assert.EqualValues(t, 3000, clients["markus"].InputTokens+clients["markus"].OutputTokens)
	require.Contains(t, clients, "unknown")

	// totals + ordering (busiest first = markus)
	totals := snap["totals"].(map[string]int64)
	assert.EqualValues(t, 5, totals["requests"])
	assert.EqualValues(t, 175+3000+2, totals["total_tokens"]) // shay+markus+unknown
	order := snap["order"].([]string)
	assert.Equal(t, "markus", order[0])

	// persist + reload
	m.flush()
	m2 := NewUsageMeter(path, nil)
	s2 := m2.Snapshot()["clients"].(map[string]*clientUsage)
	assert.EqualValues(t, 117, s2["shay"].InputTokens, "usage should survive a restart")
}

func TestKeyFingerprint_StableNonReversible(t *testing.T) {
	a := keyFingerprint("sk-gems-secret-123")
	b := keyFingerprint("sk-gems-secret-123")
	c := keyFingerprint("sk-gems-other-456")
	assert.Equal(t, a, b, "fingerprint must be stable")
	assert.NotEqual(t, a, c, "different keys -> different fingerprints")
	assert.Len(t, a, 12)
	assert.NotContains(t, "sk-gems-secret-123", a, "fingerprint must not leak the key")
}
