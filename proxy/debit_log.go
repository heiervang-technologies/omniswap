package proxy

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// debit_log.go — the durable, append-only debit-event log for usage billing
// (cloud project_gems_monetization, stage 2a). Each COMPLETED inference emits
// one DebitEvent; finance's `credit_debits` ingests them via a HOME-initiated
// pull (the same rail the daily usage backup already uses; a gem never writes
// home). This file is JUST the log core — the emit wiring (mint RequestID at the
// metrics onRecord hook, thread key_fingerprint + serving node) and the pull
// endpoint land in the follow-up; the credit-balance GATE that consumes the
// watermark is stage 2b. All land-dark: the log is a no-op until a path is set.
//
// Field contract (locked with penny — the 9 map 1:1 into finance.credit_debits
// RAW; Seq is the pool-internal watermark ordinal, not a finance column):
//
//	RequestID  pool-minted uuid, one per completed request. The UNIQUE dedupe
//	           key at the finance sink (at-least-once pull + ON CONFLICT DO
//	           NOTHING), so a re-pull / restart never double-counts. Caller-set.
//	Seq        per-INSTANCE monotonic ordinal assigned BY the log on Append. The
//	           gap-free ACK WATERMARK the stage-2b gate ages local debits by:
//	           home acks the highest CONTIGUOUS ingested Seq. Persists across
//	           restart (resumes from the max on load) and NEVER resets — a reset
//	           would reuse ordinals and corrupt the watermark.
//	Ts, KeyFingerprint (per-key join to credit_grants), ClientLabel, Model,
//	InputTokens, OutputTokens, Node (serving gem -> COGS), Country.
//
// Durability: each event is appended to a JSONL file and fsync'd before Append
// returns, so a debit survives a pool restart. The only loss window is a crash
// AFTER the inference completes but BEFORE fsync — that UNDER-bills by at most
// one event and never double-bills (customer-favourable; self-corrects at the
// customer's next top-up on a prepaid model).

// DebitEvent is one billable inference. JSON tags match the finance RAW columns.
type DebitEvent struct {
	RequestID      string    `json:"request_id"`
	Seq            uint64    `json:"seq"`
	Ts             time.Time `json:"ts"`
	KeyFingerprint string    `json:"key_fingerprint"`
	ClientLabel    string    `json:"client_label"`
	Model          string    `json:"model"`
	InputTokens    int64     `json:"input_tokens"`
	OutputTokens   int64     `json:"output_tokens"`
	Node           string    `json:"node"`
	Country        string    `json:"country"`
}

// debitLog is the append-only, fsync'd, seq-ordered event log. Thread-safe.
// A zero path yields a DISABLED log (Append is a no-op) — the land-dark default.
type debitLog struct {
	mu     sync.Mutex
	path   string
	seq    uint64       // last-assigned Seq (monotonic; persisted via the file)
	events []DebitEvent // in-memory ordered log, Seq-ascending
	file   *os.File     // append handle; nil when disabled
	logger *LogMonitor
}

// newDebitLog opens (or creates) the log at path. Empty path => a disabled log
// (enabled() == false; Append no-ops). On an existing file it restores the
// events into memory and resumes Seq from the max, so ordinals never repeat
// across a restart. A load/open error leaves the log disabled and is logged, so
// a bad path degrades to "no billing telemetry" rather than crashing the pool.
func newDebitLog(path string, logger *LogMonitor) *debitLog {
	dl := &debitLog{path: path, logger: logger}
	if path == "" {
		return dl // disabled (land-dark default)
	}
	if err := dl.load(); err != nil {
		if logger != nil {
			logger.Warnf("debit log: load %s failed, disabling: %v", path, err)
		}
		dl.path = ""
		return dl
	}
	f, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o600)
	if err != nil {
		if logger != nil {
			logger.Warnf("debit log: open %s failed, disabling: %v", path, err)
		}
		dl.path = ""
		return dl
	}
	dl.file = f
	// Make the file's directory entry durable so a crash right after a
	// first-create can't lose the freshly-created log file itself. One-time at
	// startup, not per-append. Best-effort: a dir that can't be fsync'd (rare)
	// doesn't disable the log.
	if d, derr := os.Open(filepath.Dir(path)); derr == nil {
		_ = d.Sync()
		_ = d.Close()
	}
	if logger != nil {
		logger.Infof("debit log: %s active (%d events restored, next seq %d)", path, len(dl.events), dl.seq+1)
	}
	return dl
}

// enabled reports whether the log is recording. Gated on the append handle (not
// just the path) so it agrees with Append: after close() — or a bad path —
// enabled() is false and Append no-ops, never a "says-enabled-but-drops" split.
func (dl *debitLog) enabled() bool {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	return dl.file != nil
}

// Append stamps ev with the next monotonic Seq, appends it in memory, and
// writes+fsyncs the JSONL line before returning the stamped event. On a disabled
// log it is a no-op returning ev unchanged (Seq 0). The caller owns RequestID +
// Ts; the log owns Seq. A write error is returned (and not committed to memory)
// so the caller can decide — but the log stays usable for the next Append.
func (dl *debitLog) Append(ev DebitEvent) (DebitEvent, error) {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	if dl.file == nil {
		return ev, nil // disabled / closed: no-op
	}
	ev.Seq = dl.seq + 1
	line, err := json.Marshal(ev)
	if err != nil {
		return ev, fmt.Errorf("debit log: marshal: %w", err)
	}
	if _, err := dl.file.Write(append(line, '\n')); err != nil {
		return ev, fmt.Errorf("debit log: write: %w", err)
	}
	if err := dl.file.Sync(); err != nil {
		return ev, fmt.Errorf("debit log: fsync: %w", err)
	}
	// Commit to memory + advance seq only after the durable write succeeds.
	dl.seq = ev.Seq
	dl.events = append(dl.events, ev)
	return ev, nil
}

// Since returns a COPY of every event with Seq > cursor, in Seq order — the
// home-pull payload. cursor 0 returns the whole log. The copy is independent of
// the live slice so the caller can marshal it without holding the lock.
func (dl *debitLog) Since(cursor uint64) []DebitEvent {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	var out []DebitEvent
	for _, ev := range dl.events {
		if ev.Seq > cursor {
			out = append(out, ev)
		}
	}
	return out
}

// highWater returns the highest Seq assigned so far (0 if empty) — the head of
// the log, for observability and the pull's "you are caught up" signal.
func (dl *debitLog) highWater() uint64 {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	return dl.seq
}

// load reads an existing JSONL file into memory, sets seq to the max Seq seen,
// and TRUNCATES any torn/trailing bytes past the last good, newline-terminated
// record so the O_APPEND handle writes cleanly.
//
// The truncate is load-bearing, not cosmetic: without it a crash-torn final
// fragment (bytes with no newline) survives, the next Append concatenates its
// event onto the fragment into one corrupt merged line, and a LATER restart
// then (a) loses every event from the corruption on and (b) resets maxSeq below
// already-issued ordinals — so the next Append REUSES a Seq, corrupting the 2b
// gap-free ack watermark. Stopping at the last good record and truncating the
// tail keeps Seq strictly non-decreasing across any crash/restart sequence.
//
// A missing file is fine (fresh start). Called under construction, before the
// append handle opens, so it takes no lock.
func (dl *debitLog) load() error {
	f, err := os.Open(dl.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // first run
		}
		return err
	}
	fi, err := f.Stat()
	if err != nil {
		f.Close()
		return err
	}
	fileSize := fi.Size()

	r := bufio.NewReader(f)
	var events []DebitEvent
	var maxSeq uint64
	var goodOffset int64 // bytes through the last good, newline-terminated record
	for {
		line, rerr := r.ReadBytes('\n')
		if rerr == nil {
			var ev DebitEvent
			if json.Unmarshal(line, &ev) != nil {
				break // corrupt complete line: stop; it + any tail get truncated
			}
			events = append(events, ev)
			if ev.Seq > maxSeq {
				maxSeq = ev.Seq
			}
			goodOffset += int64(len(line))
			continue
		}
		if rerr == io.EOF {
			break // trailing bytes w/o newline (torn fragment) are past goodOffset
		}
		f.Close()
		return rerr // real read error
	}
	f.Close()

	dl.events = events
	dl.seq = maxSeq

	// Truncate anything past the last good newline (torn fragment or a corrupt
	// line and its tail) so the append handle can't merge onto it.
	if goodOffset < fileSize {
		if dl.logger != nil {
			dl.logger.Warnf("debit log: truncating %d torn/trailing byte(s) past offset %d during restore", fileSize-goodOffset, goodOffset)
		}
		if err := os.Truncate(dl.path, goodOffset); err != nil {
			return fmt.Errorf("debit log: truncate torn tail: %w", err)
		}
	}
	return nil
}

// close releases the append handle (test teardown / shutdown).
func (dl *debitLog) close() error {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	if dl.file != nil {
		err := dl.file.Close()
		dl.file = nil
		return err
	}
	return nil
}
