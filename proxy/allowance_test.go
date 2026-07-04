package proxy

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func entry(fp string, bal int64, wm uint64) AllowanceEntry {
	return AllowanceEntry{KeyFingerprint: fp, Balance: bal, Watermark: wm}
}

func snapJSON(t *testing.T, ver uint64, ts int64, entries ...AllowanceEntry) []byte {
	t.Helper()
	b, err := json.Marshal(AllowanceSnapshot{Ver: ver, TsUnix: ts, Allowances: entries})
	require.NoError(t, err)
	return b
}

func TestParseAllowanceSnapshot_Valid(t *testing.T) {
	s, err := parseAllowanceSnapshot(snapJSON(t, 3, 1_700_000_000, entry("ab12", 100, 5), entry("cd34", 50, 0)))
	require.NoError(t, err)
	assert.Equal(t, uint64(3), s.Ver)
	assert.Equal(t, int64(1_700_000_000), s.TsUnix)
	require.Len(t, s.Allowances, 2)
	assert.Equal(t, "ab12", s.Allowances[0].KeyFingerprint)
	assert.Equal(t, int64(100), s.Allowances[0].Balance)
	assert.Equal(t, uint64(5), s.Allowances[0].Watermark)
}

// TestParseAllowanceSnapshot_FailsSafe: every malformed/partial form is rejected
// so the caller holds last-good (never applies a corrupt push).
func TestParseAllowanceSnapshot_FailsSafe(t *testing.T) {
	cases := []struct {
		name string
		raw  []byte
	}{
		{"truncated json", []byte(`{"ver":3,"ts_unix":1,"allowances":[`)},
		{"missing ver", snapJSON(t, 0, 1_700_000_000, entry("ab12", 100, 5))},
		{"missing key_fingerprint", snapJSON(t, 3, 1_700_000_000, entry("", 100, 5))},
		{"negative balance", snapJSON(t, 3, 1_700_000_000, entry("ab12", -1, 5))},
		{"duplicate key", snapJSON(t, 3, 1_700_000_000, entry("ab12", 100, 5), entry("ab12", 50, 0))},
		{"unknown field", []byte(`{"ver":3,"ts_unix":1,"allowances":[],"evil":1}`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseAllowanceSnapshot(tc.raw)
			assert.Error(t, err, "malformed/partial snapshot must be rejected")
		})
	}
}

// TestParseAllowanceSnapshot_MissingTsParsesButExpires: a ts-less snapshot parses
// (balances usable) but is conservatively Expired, never silently "fresh".
func TestParseAllowanceSnapshot_MissingTsParsesButExpires(t *testing.T) {
	s, err := parseAllowanceSnapshot(snapJSON(t, 3, 0, entry("ab12", 100, 5)))
	require.NoError(t, err, "missing ts is not a parse error")
	assert.Equal(t, Expired, snapshotFreshness(s.TsUnix, 1_700_000_000, 60, 600, 30))
}

func TestSnapshotFreshness(t *testing.T) {
	const now, grace, hardMax, skew = 1_000_000, 60, 600, 30
	assert.Equal(t, Fresh, snapshotFreshness(now-10, now, grace, hardMax, skew), "age within grace")
	assert.Equal(t, Fresh, snapshotFreshness(now-60, now, grace, hardMax, skew), "age == grace")
	assert.Equal(t, Stale, snapshotFreshness(now-300, now, grace, hardMax, skew), "grace < age <= hardMax")
	assert.Equal(t, Stale, snapshotFreshness(now-600, now, grace, hardMax, skew), "age == hardMax")
	assert.Equal(t, Expired, snapshotFreshness(now-601, now, grace, hardMax, skew), "age > hardMax")
	assert.Equal(t, Expired, snapshotFreshness(0, now, grace, hardMax, skew), "missing ts")
	assert.Equal(t, Fresh, snapshotFreshness(now+20, now, grace, hardMax, skew), "future within skew tolerated")
	assert.Equal(t, Expired, snapshotFreshness(now+31, now, grace, hardMax, skew), "future beyond skew untrusted")
}

// TestAllowanceIngester_AppliesAndHoldsLastGood: newer applies; stale/malformed
// are rejected and never zero or lower a live balance.
func TestAllowanceIngester_AppliesAndHoldsLastGood(t *testing.T) {
	l := newCreditLedger()
	ing := newAllowanceIngester(l, testLogger)

	require.True(t, ing.Apply(snapJSON(t, 1, 1000, entry("ab12", 100, 0))))
	assert.Equal(t, int64(100), l.Available("ab12"))
	assert.Equal(t, uint64(1), ing.AppliedVer())

	require.True(t, ing.Apply(snapJSON(t, 2, 2000, entry("ab12", 40, 0))), "newer snapshot applies")
	assert.Equal(t, int64(40), l.Available("ab12"))

	assert.False(t, ing.Apply(snapJSON(t, 1, 3000, entry("ab12", 999, 0))), "older ver rejected")
	assert.Equal(t, int64(40), l.Available("ab12"), "a stale snapshot can't inflate/replace the balance")

	assert.False(t, ing.Apply([]byte(`{"ver":9,"ts_unix":9,"allowances":[`)), "malformed rejected")
	assert.Equal(t, int64(40), l.Available("ab12"), "a malformed push never zeroes a balance")
	assert.Equal(t, uint64(2), ing.AppliedVer(), "applied ver held at last-good")
	assert.Positive(t, ing.rejectCount())
}

// TestAllowanceIngester_FreshnessNeverIngested: no snapshot yet = Expired.
func TestAllowanceIngester_FreshnessNeverIngested(t *testing.T) {
	ing := newAllowanceIngester(newCreditLedger(), testLogger)
	assert.Equal(t, Expired, ing.Freshness(1000, 60, 600, 30), "never ingested = conservative Expired")

	require.True(t, ing.Apply(snapJSON(t, 1, 1000, entry("ab12", 100, 0))))
	assert.Equal(t, Fresh, ing.Freshness(1030, 60, 600, 30), "reflects last-applied ts")
	assert.Equal(t, Expired, ing.Freshness(2000, 60, 600, 30), "aged past hard-max")
}
