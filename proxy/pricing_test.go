package proxy

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func testBook() *priceBook {
	return newPriceBook(map[string]ModelRate{
		// 2 credits / 1k input, 8 credits / 1k output, ceiling 4096 out
		"m": {InputPer1k: 2, OutputPer1k: 8, MaxOutputTokens: 4096},
	})
}

func TestCeilDiv(t *testing.T) {
	assert.Equal(t, int64(0), ceilDiv(0, 1000))
	assert.Equal(t, int64(1), ceilDiv(1, 1000), "1/1000 rounds UP to 1 (never charge less than actual)")
	assert.Equal(t, int64(1), ceilDiv(1000, 1000))
	assert.Equal(t, int64(2), ceilDiv(1001, 1000))
	assert.Equal(t, int64(0), ceilDiv(-5, 1000))
}

func TestPriceBook_ReserveCost(t *testing.T) {
	pb := testBook()

	// requested max below the ceiling: reserve prompt + requested-max output.
	// prompt 1500*2/1000=ceil(3.0)=3 ; out 500*8/1000=ceil(4.0)=4 -> 7
	c, ok := pb.ReserveCost("m", 1500, 500)
	assert.True(t, ok)
	assert.Equal(t, int64(7), c)

	// no max_tokens (0): reserve against the model CEILING, not a guess.
	// out 4096*8/1000 = ceil(32.768) = 33 ; prompt 1500 -> 3 -> 36
	c, _ = pb.ReserveCost("m", 1500, 0)
	assert.Equal(t, int64(36), c)

	// requested max ABOVE the ceiling clamps to the ceiling (same as no-max).
	c, _ = pb.ReserveCost("m", 1500, 99999)
	assert.Equal(t, int64(36), c)

	// unpriced model -> fail-closed signal
	_, ok = pb.ReserveCost("nope", 100, 100)
	assert.False(t, ok)
}

func TestPriceBook_ActualCost(t *testing.T) {
	pb := testBook()
	// prompt 1500 -> 3 ; completion 500 -> 4 -> 7
	c, ok := pb.ActualCost("m", 1500, 500)
	assert.True(t, ok)
	assert.Equal(t, int64(7), c)

	_, ok = pb.ActualCost("nope", 100, 100)
	assert.False(t, ok)
}

// TestPriceBook_ReserveCoversActual: the load-bearing invariant — for any actual
// completion within the reserved upper bound, the reservation covers the bill,
// so the ledger clamp never bites and there's no over-spend.
func TestPriceBook_ReserveCoversActual(t *testing.T) {
	pb := testBook()
	const prompt = 3210
	for _, reqMax := range []int64{0, 100, 1000, 4096, 99999} {
		reserve, _ := pb.ReserveCost("m", prompt, reqMax)
		// actual completion can be anything from 0 up to the true upper bound
		upper := int64(4096)
		if reqMax > 0 && reqMax < upper {
			upper = reqMax
		}
		for _, actualOut := range []int64{0, 1, upper / 2, upper} {
			actual, _ := pb.ActualCost("m", prompt, actualOut)
			assert.GreaterOrEqual(t, reserve, actual,
				"reserve(%d) must cover actual(out=%d) — reqMax=%d", reserve, actualOut, reqMax)
		}
	}
}
