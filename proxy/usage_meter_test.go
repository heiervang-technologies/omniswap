package proxy

import (
	"encoding/json"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUsageMeter_RecordAggregatePersist(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "usage.json")

	m := NewUsageMeter(path, nil)
	m.Record("shay", "gemma-4-12b", "NO", "1.2.3.4", 100, 50)
	m.Record("shay", "gemma-4-12b", "NO", "1.2.3.4", 10, 5)
	m.Record("shay", "qwen3.6-35b-a3b", "NO", "1.2.3.4", 7, 3)
	m.Record("markus", "gemma-4-12b", "US", "8.8.8.8", 1000, 2000)
	m.Record("", "gemma-4-12b", "", "", 1, 1) // empty -> client "unknown", country "local", ip "unknown"

	snap := m.Snapshot()
	clients := snap["clients"].(map[string]*clientUsage)

	// by-country rollup
	byCountry := snap["by_country"].(map[string]*modelUsage)
	assert.EqualValues(t, 3, byCountry["NO"].Requests, "shay's 3 from NO")
	assert.EqualValues(t, 1, byCountry["US"].Requests, "markus from US")
	require.Contains(t, byCountry, "local", "empty country buckets as local")

	// by-ip rollup
	byIP := snap["by_ip"].(map[string]*ipUsage)
	assert.EqualValues(t, 3, byIP["1.2.3.4"].Requests, "shay's 3 from one IP")
	assert.Equal(t, "shay", byIP["1.2.3.4"].Client)
	assert.Equal(t, "NO", byIP["1.2.3.4"].Country)
	require.Contains(t, byIP, "unknown", "empty ip buckets as unknown")

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

// TestUsageMeter_ConcurrentSnapshotMarshal guards the /usage path: the dashboard
// polls Snapshot() and marshals it to JSON while live inference traffic calls
// Record() concurrently. Snapshot() must return data INDEPENDENT of the live
// maps — if it leaks the live maps, json.Marshal reads them while Record writes,
// which is a data race and a fatal "concurrent map read and map write" panic
// that crashes the pool. Run under -race to catch the leak.
func TestUsageMeter_ConcurrentSnapshotMarshal(t *testing.T) {
	m := NewUsageMeter("", nil) // no persistence path -> no flush goroutine

	done := make(chan struct{})
	go func() { // writer: hammer Record like live inference does
		for {
			select {
			case <-done:
				return
			default:
				m.Record("shay", "gemma-4-12b", "NO", "1.2.3.4", 10, 5)
				m.Record("markus", "qwen3.6-35b-a3b", "US", "8.8.8.8", 1, 1)
			}
		}
	}()

	// reader: Snapshot + marshal exactly as usageHandler does, concurrently.
	for i := 0; i < 3000; i++ {
		if _, err := json.Marshal(m.Snapshot()); err != nil {
			close(done)
			t.Fatalf("marshal snapshot: %v", err)
		}
	}
	close(done)
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
