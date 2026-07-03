package proxy

import (
	"path/filepath"
	"regexp"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// blockingAppender gates each Append on release, signalling entered first, so a
// test can park the writer goroutine and fill the buffer deterministically.
type blockingAppender struct {
	entered chan struct{} // buffered; one signal per Append entry
	release chan struct{} // close to let all parked/future Appends proceed
	count   int64
}

func newBlockingAppender() *blockingAppender {
	return &blockingAppender{entered: make(chan struct{}, 64), release: make(chan struct{})}
}

func (b *blockingAppender) Append(ev DebitEvent) (DebitEvent, error) {
	b.entered <- struct{}{}
	<-b.release
	atomic.AddInt64(&b.count, 1)
	return ev, nil
}

// TestDebitEmitter_DrainsToSink: emitted events reach the durable log, in order,
// after a graceful Close.
func TestDebitEmitter_DrainsToSink(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	dl := newDebitLog(path, testLogger)
	defer dl.close()
	e := newDebitEmitter(dl, 64, testLogger)

	for _, id := range []string{"r1", "r2", "r3", "r4", "r5"} {
		e.Emit(sampleEvent(id))
	}
	e.Close() // drains + waits for the writer

	got := dl.Since(0)
	require.Len(t, got, 5)
	assert.Equal(t, []uint64{1, 2, 3, 4, 5}, []uint64{got[0].Seq, got[1].Seq, got[2].Seq, got[3].Seq, got[4].Seq})
	assert.Zero(t, e.droppedCount())
	assert.Equal(t, int64(5), e.appendedCount())
	assert.Zero(t, e.failedCount())
}

// TestDebitEmitter_ConcurrentEmitDuringClose locks in the Emit/Close race
// guarantee: many goroutines Emit while Close runs concurrently — under -race
// this must not panic (send-on-closed) or data-race. Assertion is the clean run
// itself; accepted events (appended + dropped) never exceed what was offered.
func TestDebitEmitter_ConcurrentEmitDuringClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	dl := newDebitLog(path, testLogger)
	defer dl.close()
	e := newDebitEmitter(dl, 128, testLogger)

	const goroutines, per = 8, 50
	var wg sync.WaitGroup
	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for j := 0; j < per; j++ {
				e.Emit(sampleEvent("r"))
			}
		}()
	}
	e.Close() // races the Emit storm
	wg.Wait()

	assert.NotPanics(t, func() { e.Emit(sampleEvent("late")) })
	assert.LessOrEqual(t, e.appendedCount()+e.droppedCount(), int64(goroutines*per))
}

// TestDebitEmitter_OverflowDropsBounded: a full buffer sheds the excess (counted,
// never blocking) instead of stalling the caller.
func TestDebitEmitter_OverflowDropsBounded(t *testing.T) {
	ba := newBlockingAppender()
	e := newDebitEmitter(ba, 2, testLogger) // buffer depth 2

	// #1 is taken by the writer and parks in Append (buffer now empty).
	e.Emit(sampleEvent("r1"))
	<-ba.entered // writer is inside Append on r1, blocked on release

	// Fill the buffer (depth 2), then one more overflows -> dropped.
	e.Emit(sampleEvent("r2"))
	e.Emit(sampleEvent("r3"))
	e.Emit(sampleEvent("r4")) // buffer full -> dropped
	assert.Equal(t, int64(1), e.droppedCount(), "exactly the overflow event is dropped")

	close(ba.release) // let the writer drain r1, r2, r3
	e.Close()
	assert.Equal(t, int64(3), atomic.LoadInt64(&ba.count), "the 3 accepted events all persisted")
	assert.Equal(t, int64(1), e.droppedCount())
}

// TestDebitEmitter_EmitAfterCloseNoPanic: Emit after Close is a safe no-op (no
// send-on-closed-channel panic).
func TestDebitEmitter_EmitAfterCloseNoPanic(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	dl := newDebitLog(path, testLogger)
	defer dl.close()
	e := newDebitEmitter(dl, 8, testLogger)
	e.Emit(sampleEvent("r1"))
	e.Close()

	assert.NotPanics(t, func() { e.Emit(sampleEvent("r2")) })
	e.Close() // idempotent
	assert.Len(t, dl.Since(0), 1, "post-close emit did not persist")
}

// TestMintRequestID_ShapeAndUniqueness: RFC-4122 v4 form + distinct values.
func TestMintRequestID_ShapeAndUniqueness(t *testing.T) {
	v4 := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)
	seen := map[string]bool{}
	for i := 0; i < 200; i++ {
		id, err := mintRequestID()
		require.NoError(t, err)
		assert.Regexp(t, v4, id, "must be a v4 uuid (version nibble 4, variant 8-b)")
		assert.False(t, seen[id], "uuids must be unique")
		seen[id] = true
	}
}
