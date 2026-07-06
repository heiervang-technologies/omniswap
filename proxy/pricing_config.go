package proxy

import "github.com/mostlygeek/llama-swap/proxy/config"

// pricing_config.go — bridges validated config credit rates into a priceBook for
// the billing gate (cloud project_gems_monetization, stage 2b.3b). PURE +
// dormant: nothing on the request path calls this yet (the gate hook, a later
// slice, does). It keeps pricing.go dependency-free — pricing.go is the money
// math, this file is the config seam.

// suspiciouslyLowCeiling: a configured MaxOutputTokens below this is almost
// certainly a fat-finger — a dropped digit (8192 typed as 819) or a transposed
// value (2000 as 200), NOT a real model's hard output max. No production
// chat/completions model caps its TRUE hard output below 2048 (even small,
// short-context models allow >2k output), so this floor won't fire on a legit
// model yet catches the dropped-digit / transposed-typo class. It is a heuristic
// backstop, not the rule: the rule is "MaxOutputTokens = the model's TRUE hard
// output max (or slightly above)" so the reserved worst-case always covers the
// actual bill. WARN-only + startup-only, so the asymmetry favours catching more —
// a false warn costs one cleared log line, a MISSED warn is a silent revenue leak
// (big-dog, #33 review: prefer 2048, floor at 1024). config.LoadConfig already
// hard-rejects a non-positive ceiling; a genuinely tiny model is possible, so we
// never refuse to serve on this alone.
const suspiciouslyLowCeiling = 2048

// priceBookFromConfig builds a priceBook from config credit rates, emitting
// startup sanity warnings and dropping any UNUSABLE rate fail-closed (an unpriced
// model makes the gate deny it rather than under-bill). config.LoadConfig has
// already rejected structurally-broken rates, but this guards independently too
// (defense-in-depth on the money path: a rate built programmatically, bypassing
// LoadConfig, is still handled safely). Warnings it surfaces:
//   - a non-positive / negative rate -> DROP (fail-closed, model becomes unpriced)
//   - a ceiling below suspiciouslyLowCeiling -> warn (likely not the true max)
//   - a zero output price on a priced model -> warn (output would bill as free)
//
// Returns a ready priceBook; an empty/nil map yields an empty book, so every
// model is unpriced and the gate fails closed.
func priceBookFromConfig(rates map[string]config.CreditRate, logger *LogMonitor) *priceBook {
	m := make(map[string]ModelRate, len(rates))
	for model, r := range rates {
		if r.InputPer1k < 0 || r.OutputPer1k < 0 {
			warnPricing(logger, "pricing: model %q has a negative rate (inputPer1k=%d outputPer1k=%d) — dropping it (fail-closed: the gate will deny this model rather than mis-bill)", model, r.InputPer1k, r.OutputPer1k)
			continue
		}
		if r.MaxOutputTokens <= 0 {
			warnPricing(logger, "pricing: model %q has maxOutputTokens=%d — cannot reserve a worst-case completion, dropping it (fail-closed: the gate will deny this model rather than under-bill)", model, r.MaxOutputTokens)
			continue
		}
		if r.MaxOutputTokens < suspiciouslyLowCeiling {
			warnPricing(logger, "pricing: model %q maxOutputTokens=%d looks suspiciously low — it MUST be the model's TRUE hard output max (or slightly above); a ceiling below the real max under-bills long completions (actual cost exceeds the reserved hold, then the ledger clamps it). Verify this is not a typo.", model, r.MaxOutputTokens)
		}
		if r.OutputPer1k == 0 {
			warnPricing(logger, "pricing: model %q outputPer1k=0 — completion tokens will bill as FREE; likely a misconfiguration for a paid model", model)
		}
		m[model] = ModelRate{
			InputPer1k:      r.InputPer1k,
			OutputPer1k:     r.OutputPer1k,
			MaxOutputTokens: r.MaxOutputTokens,
		}
	}
	return newPriceBook(m)
}

// warnPricing logs a pricing warning, nil-safe (a nil logger is a no-op so the
// builder is safe to call before/without a logger, e.g. in tests).
func warnPricing(logger *LogMonitor, format string, args ...any) {
	if logger != nil {
		logger.Warnf(format, args...)
	}
}
