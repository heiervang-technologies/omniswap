package proxy

import (
	"io"
	"testing"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// capturingLogger returns a LogMonitor that discards to no visible output but
// keeps the formatted history so a test can assert on emitted warnings.
func capturingLogger() *LogMonitor { return NewLogMonitorWriter(io.Discard) }

// TestPriceBookFromConfig_Builds: config rates convert 1:1 into priceBook rates,
// verified through the money math (ReserveCost/ActualCost), and unpriced models
// fail closed.
func TestPriceBookFromConfig_Builds(t *testing.T) {
	pb := priceBookFromConfig(map[string]config.CreditRate{
		"m": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: 4096},
	}, capturingLogger())

	require.True(t, pb.has("m"))
	assert.False(t, pb.has("other"), "an un-configured model is unpriced (gate fails closed)")

	// prompt 1500 -> ceil(3.0)=3 ; completion 500 -> ceil(4.0)=4 -> 7
	c, ok := pb.ActualCost("m", 1500, 500)
	require.True(t, ok)
	assert.Equal(t, int64(7), c)

	// reserve with no max_tokens holds against the ceiling: 4096*8/1000=ceil(32.768)=33 + 3 = 36
	r, ok := pb.ReserveCost("m", 1500, 0)
	require.True(t, ok)
	assert.Equal(t, int64(36), r)
}

// TestPriceBookFromConfig_EmptyFailsClosed: nil/empty rates yield an empty book,
// so every model is unpriced and the gate denies rather than serving free.
func TestPriceBookFromConfig_EmptyFailsClosed(t *testing.T) {
	pb := priceBookFromConfig(nil, capturingLogger())
	assert.False(t, pb.has("anything"))
	_, ok := pb.ReserveCost("anything", 100, 100)
	assert.False(t, ok, "unpriced model must return ok=false (fail-closed)")
}

// TestPriceBookFromConfig_DropsUnusable: a non-positive ceiling or negative rate
// is dropped (defense-in-depth for a rate that bypassed LoadConfig), so the model
// becomes unpriced and the gate fails closed — never served un-metered.
func TestPriceBookFromConfig_DropsUnusable(t *testing.T) {
	lg := capturingLogger()
	pb := priceBookFromConfig(map[string]config.CreditRate{
		"good":     {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: 4096},
		"zeroceil": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: 0},
		"negceil":  {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: -1},
		"negrate":  {InputPer1k: -2, OutputPer1k: 8, MaxOutputTokens: 4096},
	}, lg)

	assert.True(t, pb.has("good"))
	assert.False(t, pb.has("zeroceil"), "non-positive ceiling dropped -> unpriced")
	assert.False(t, pb.has("negceil"), "negative ceiling dropped -> unpriced")
	assert.False(t, pb.has("negrate"), "negative rate dropped -> unpriced")

	hist := string(lg.GetHistory())
	assert.Contains(t, hist, `"zeroceil"`)
	assert.Contains(t, hist, `"negceil"`)
	assert.Contains(t, hist, `"negrate"`)
	assert.Contains(t, hist, "fail-closed")
}

// TestPriceBookFromConfig_WarnsButKeepsSuspicious: a suspiciously-LOW but positive
// ceiling, and a zero output price, are WARNED about but the model is still priced
// (the operator may have meant it; the warn makes it visible).
func TestPriceBookFromConfig_WarnsButKeepsSuspicious(t *testing.T) {
	lg := capturingLogger()
	pb := priceBookFromConfig(map[string]config.CreditRate{
		"lowceil": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: 64}, // < suspiciouslyLowCeiling
		"freeout": {InputPer1k: 2, OutputPer1k: 0, MaxOutputTokens: 4096},
	}, lg)

	assert.True(t, pb.has("lowceil"), "a low-but-positive ceiling is warned, not dropped")
	assert.True(t, pb.has("freeout"), "a zero output price is warned, not dropped")

	hist := string(lg.GetHistory())
	assert.Contains(t, hist, "suspiciously low")
	assert.Contains(t, hist, "TRUE hard output max")
	assert.Contains(t, hist, "bill as FREE")
}

// TestSuspiciouslyLowCeilingBoundary: the warn fires strictly below the floor and
// not at/above it (the floor is a real chat model's plausible low bound).
func TestSuspiciouslyLowCeilingBoundary(t *testing.T) {
	atFloor := capturingLogger()
	priceBookFromConfig(map[string]config.CreditRate{
		"m": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: suspiciouslyLowCeiling},
	}, atFloor)
	assert.NotContains(t, string(atFloor.GetHistory()), "suspiciously low", "at the floor is not flagged")

	below := capturingLogger()
	priceBookFromConfig(map[string]config.CreditRate{
		"m": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: suspiciouslyLowCeiling - 1},
	}, below)
	assert.Contains(t, string(below.GetHistory()), "suspiciously low", "below the floor is flagged")
}

// TestPriceBookFromConfig_NilLoggerSafe: a nil logger is a no-op, not a panic.
func TestPriceBookFromConfig_NilLoggerSafe(t *testing.T) {
	assert.NotPanics(t, func() {
		pb := priceBookFromConfig(map[string]config.CreditRate{
			"m": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: 8}, // triggers the low-ceiling warn path
		}, nil)
		assert.True(t, pb.has("m"))
	})
}
