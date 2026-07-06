package config

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestConfig_CreditRatesValid: a well-formed creditRates block parses into the
// map keyed by model ID.
func TestConfig_CreditRatesValid(t *testing.T) {
	content := `
startPort: 10000
creditRates:
  gemma-4-12b-256k:
    inputPer1k: 2
    outputPer1k: 8
    maxOutputTokens: 262144
  qwen3.6-35b-a3b:
    inputPer1k: 3
    outputPer1k: 12
    maxOutputTokens: 262144

models:
  test:
    cmd: echo hi
    proxy: http://localhost:8080
`
	config, err := LoadConfigFromReader(strings.NewReader(content))
	require.NoError(t, err)
	require.Len(t, config.CreditRates, 2)
	assert.Equal(t, int64(2), config.CreditRates["gemma-4-12b-256k"].InputPer1k)
	assert.Equal(t, int64(8), config.CreditRates["gemma-4-12b-256k"].OutputPer1k)
	assert.Equal(t, int64(262144), config.CreditRates["gemma-4-12b-256k"].MaxOutputTokens)
}

// TestConfig_CreditRatesAbsentIsFine: no creditRates at all is the default (the
// gate fails closed on every model, but that's the gate's job, not a load error).
func TestConfig_CreditRatesAbsentIsFine(t *testing.T) {
	content := `
startPort: 10000
models:
  test:
    cmd: echo hi
    proxy: http://localhost:8080
`
	config, err := LoadConfigFromReader(strings.NewReader(content))
	require.NoError(t, err)
	assert.Empty(t, config.CreditRates)
}

// TestConfig_CreditGateEnforceDefaultsOff: absent creditGateEnforce => false
// (OFF). The landmine-proof default — billing enforcement is never on unless
// explicitly set, decoupled from rate/allowance presence.
func TestConfig_CreditGateEnforceDefaultsOff(t *testing.T) {
	content := `
startPort: 10000
creditRates:
  m:
    inputPer1k: 2
    outputPer1k: 8
    maxOutputTokens: 4096

models:
  test:
    cmd: echo hi
    proxy: http://localhost:8080
`
	config, err := LoadConfigFromReader(strings.NewReader(content))
	require.NoError(t, err)
	assert.False(t, config.CreditGateEnforce, "absent creditGateEnforce defaults OFF even with rates present")
}

// TestConfig_CreditGateEnforceExplicit: the flag parses true/false when set.
func TestConfig_CreditGateEnforceExplicit(t *testing.T) {
	for _, tc := range []struct {
		val  string
		want bool
	}{{"true", true}, {"false", false}} {
		content := "startPort: 10000\ncreditGateEnforce: " + tc.val + `
models:
  test:
    cmd: echo hi
    proxy: http://localhost:8080
`
		config, err := LoadConfigFromReader(strings.NewReader(content))
		require.NoError(t, err)
		assert.Equal(t, tc.want, config.CreditGateEnforce, "creditGateEnforce: %s", tc.val)
	}
}

// TestConfig_CreditRatesReject: structurally-broken rates fail to load so a typo
// can never yield a nonsensical or under-billing price.
func TestConfig_CreditRatesReject(t *testing.T) {
	cases := []struct {
		name    string
		rate    string
		wantSub string
	}{
		{
			name:    "negative input rate",
			rate:    "inputPer1k: -1\n    outputPer1k: 8\n    maxOutputTokens: 4096",
			wantSub: "negative rate",
		},
		{
			name:    "negative output rate",
			rate:    "inputPer1k: 2\n    outputPer1k: -8\n    maxOutputTokens: 4096",
			wantSub: "negative rate",
		},
		{
			name:    "zero ceiling",
			rate:    "inputPer1k: 2\n    outputPer1k: 8\n    maxOutputTokens: 0",
			wantSub: "maxOutputTokens must be > 0",
		},
		{
			name:    "negative ceiling",
			rate:    "inputPer1k: 2\n    outputPer1k: 8\n    maxOutputTokens: -4096",
			wantSub: "maxOutputTokens must be > 0",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			content := `
startPort: 10000
creditRates:
  m:
    ` + tc.rate + `

models:
  test:
    cmd: echo hi
    proxy: http://localhost:8080
`
			_, err := LoadConfigFromReader(strings.NewReader(content))
			require.Error(t, err, "structurally-broken rate must be rejected")
			assert.Contains(t, err.Error(), tc.wantSub)
		})
	}
}
