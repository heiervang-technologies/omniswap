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
// writer keeps appends serialized (Seq stays monotonic) and lets fsyncs batch
// naturally under load.
//
// Overflow is bounded + LOUD, never blocking: if the buffer is full (writer
// can't keep up — a slow disk), Emit drops the event and counts it. That is an
// under-bill (customer-favourable, self-corrects at next top-up), logged, never
// a stalled response and never a silent loss. Close() drains the buffer and
// waits for the writer, so a clean shutdown loses nothing in flight.

// debitAppender is the durable sink the emitter writes to (debitLog satisfies
// it). An interface so the writer's concurrency can be tested against a
// controllable/blocking sink.
type debitAppender interface {
	Append(DebitEvent) (DebitEvent, error)
}

type debitEmitter struct {
	sink    debitAppender
	ch      chan DebitEvent
	logger  *LogMonitor
	wg      sync.WaitGroup
	dropped int64        // events dropped on overflow (atomic)
	closeMu sync.RWMutex // RLock: Emit; Lock: Close (drains all Emits before closing ch)
	closed  bool
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
		if _, err := e.sink.Append(ev); err != nil && e.logger != nil {
			e.logger.Warnf("debit emitter: append failed (event %s lost): %v", ev.RequestID, err)
		}
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
