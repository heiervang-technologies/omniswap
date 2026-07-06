package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// TestCreditGate_OffByDefault: the unconfigured/OFF gate never enforces — the
// landmine-proof default. A nil gate (billing never set up) and an explicit
// enforce=false both report not-enforcing, so piece 2's hook no-ops.
func TestCreditGate_OffByDefault(t *testing.T) {
	var nilGate *creditGate
	assert.False(t, nilGate.enforcing(), "nil gate (unconfigured pool) is OFF")

	g := newCreditGate(false, newCreditLedger(), newPriceBook(nil), nil, nil)
	assert.False(t, g.enforcing(), "enforce=false is OFF")
}

// TestCreditGate_EnforcingReflectsFlag: enforcement follows ONLY the explicit
// flag, not the presence of ledger/prices/allowance components.
func TestCreditGate_EnforcingReflectsFlag(t *testing.T) {
	// components present but flag OFF -> still OFF (decoupled from data presence)
	off := newCreditGate(false, newCreditLedger(), newPriceBook(map[string]ModelRate{
		"m": {InputPer1k: 1, OutputPer1k: 1, MaxOutputTokens: 4096},
	}), nil, nil)
	assert.False(t, off.enforcing(), "rates loaded but flag OFF -> not enforcing")

	on := newCreditGate(true, newCreditLedger(), newPriceBook(nil), nil, nil)
	assert.True(t, on.enforcing(), "explicit enforce=true -> enforcing")
}
