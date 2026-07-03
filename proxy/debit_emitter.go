package proxy

import (
	"crypto/rand"
	"fmt"
	"sync"
	"sync/atomic"
)

// debit_emitter.go — the OFF-HOT-PATH writer for the debit-event log (cloud
// project_gems_monetization, stage 2a.2).
//
// The billing emit must NEVER block or slow an inference response. The log's
// Append does a synchronous fsync (~ms), so calling it inline on the metrics
// onRecord hook (the request path) would add that latency to every request.
// debitEmitter decouples them: Emit() is a non-blocking push onto a buffered
// channel drained by ONE background writer goroutine that calls Append. A single
// writer keeps appends serialized so Seq stays monotonic.
//
// The writer fsyncs ONCE PER EVENT (Append is per-event) — NOT batched. Real
// drain-batching (one fsync per drained run of events) is a deferred throughput
// lever, not done here on purpose: doing it correctly needs an atomic multi-line
// write with rollback, or a partial write reuses/gaps a Seq (a worse failure than
// the fsync cost it saves). That per-event fsync is the writer's throughput
// ceiling — a go-live sizing check (confirm it clears peak request rate, else add
// atomic drain-batching then).
//
// Overflow is bounded + LOUD, never blocking: if the buffer is full (writer
// can't keep up — a slow disk), Emit drops the event and counts it. That is an
// under-bill (customer-favourable, self-corrects at next top-up), logged, never
// a stalled response and never a silent loss. Close() drains the buffer and
// waits for the writer, so a clean shutdown loses nothing in flight.
//
// Crash-loss window: async widens it from one event (sync Append) to up to
// bufSize events buffered-but-not-yet-fsync'd, lost on a HARD crash. Still
// watermark-safe — those events were never Appended so never got a Seq, so there
// is no gap, only a bounded under-bill. Size bufSize deliberately against that.

// debitAppender is the durable sink the emitter writes to (debitLog satisfies
// it). An interface so the writer's concurrency can be tested against a
// controllable/blocking sink.
type debitAppender interface {
	Append(DebitEvent) (DebitEvent, error)
}

type debitEmitter struct {
	sink     debitAppender
	ch       chan DebitEvent
	logger   *LogMonitor
	wg       sync.WaitGroup
	dropped  int64        // events dropped on overflow (atomic)
	appended int64        // events durably Appended by the writer (atomic)
	failed   int64        // events the writer could not Append, e.g. disk error (atomic)
	closeMu  sync.RWMutex // RLock: Emit; Lock: Close (drains all Emits before closing ch)
	closed   bool
}

// newDebitEmitter starts the background writer draining into sink. bufSize is the
// channel depth (events buffered before Emit starts dropping); <=0 picks a
// sensible default.
func newDebitEmitter(sink debitAppender, bufSize int, logger *LogMonitor) *debitEmitter {
	if bufSize <= 0 {
		bufSize = 1024
	}
	e := &debitEmitter{sink: sink, ch: make(chan DebitEvent, bufSize), logger: logger}
	e.wg.Add(1)
	go e.run()
	return e
}

func (e *debitEmitter) run() {
	defer e.wg.Done()
	for ev := range e.ch {
		if _, err := e.sink.Append(ev); err != nil {
			n := atomic.AddInt64(&e.failed, 1)
			if e.logger != nil {
				e.logger.Warnf("debit emitter: append failed (event %s lost, total failed %d): %v", ev.RequestID, n, err)
			}
			continue
		}
		atomic.AddInt64(&e.appended, 1)
	}
}

// Emit queues an event for durable write WITHOUT blocking the caller. On a full
// buffer it drops + counts (bounded under-bill, logged) rather than stall the
// request path. Safe (a no-op) after Close.
func (e *debitEmitter) Emit(ev DebitEvent) {
	e.closeMu.RLock()
	defer e.closeMu.RUnlock()
	if e.closed {
		return
	}
	select {
	case e.ch <- ev:
	default:
		n := atomic.AddInt64(&e.dropped, 1)
		if e.logger != nil {
			e.logger.Warnf("debit emitter: buffer full, dropped event %s (total dropped %d)", ev.RequestID, n)
		}
	}
}

// Close stops accepting events, drains everything already buffered, and waits
// for the writer to finish — so a clean shutdown persists all in-flight events.
// Idempotent.
func (e *debitEmitter) Close() {
	e.closeMu.Lock()
	if e.closed {
		e.closeMu.Unlock()
		return
	}
	e.closed = true
	close(e.ch)
	e.closeMu.Unlock()
	e.wg.Wait()
}

// droppedCount is the number of events shed on overflow (observability).
func (e *debitEmitter) droppedCount() int64 { return atomic.LoadInt64(&e.dropped) }

// appendedCount is the number of events durably written by the writer.
func (e *debitEmitter) appendedCount() int64 { return atomic.LoadInt64(&e.appended) }

// failedCount is the number of events the writer could not persist (e.g. a disk
// error) — distinct from overflow drops; both are under-bills, surfaced for 2b.
func (e *debitEmitter) failedCount() int64 { return atomic.LoadInt64(&e.failed) }

// mintRequestID returns a random RFC-4122 v4 UUID string — the per-event
// idempotency key finance dedupes on (credit_debits.request_id UNIQUE).
// crypto/rand so it is collision-safe without pulling in a uuid dependency.
func mintRequestID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", err
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}
