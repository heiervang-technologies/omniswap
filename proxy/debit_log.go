package proxy

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
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
	if logger != nil {
		logger.Infof("debit log: %s active (%d events restored, next seq %d)", path, len(dl.events), dl.seq+1)
	}
	return dl
}

// enabled reports whether the log is recording (a non-empty, openable path).
func (dl *debitLog) enabled() bool {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	return dl.path != ""
}

// Append stamps ev with the next monotonic Seq, appends it in memory, and
// writes+fsyncs the JSONL line before returning the stamped event. On a disabled
// log it is a no-op returning ev unchanged (Seq 0). The caller owns RequestID +
// Ts; the log owns Seq. A write error is returned (and not committed to memory)
// so the caller can decide — but the log stays usable for the next Append.
func (dl *debitLog) Append(ev DebitEvent) (DebitEvent, error) {
	dl.mu.Lock()
	defer dl.mu.Unlock()
	if dl.path == "" || dl.file == nil {
		return ev, nil // disabled: no-op
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

// load reads an existing JSONL file into memory and sets seq to the max Seq
// seen. A missing file is fine (fresh start). Called under construction, before
// the append handle is opened, so it takes no lock.
func (dl *debitLog) load() error {
	f, err := os.Open(dl.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil // first run
		}
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 4*1024*1024) // tolerate long lines
	var events []DebitEvent
	var maxSeq uint64
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var ev DebitEvent
		if err := json.Unmarshal(line, &ev); err != nil {
			// A torn final line (crash mid-write) is the plausible cause; stop
			// at the last good record rather than fail the whole restore.
			if dl.logger != nil {
				dl.logger.Warnf("debit log: skipping unparseable line during restore: %v", err)
			}
			break
		}
		events = append(events, ev)
		if ev.Seq > maxSeq {
			maxSeq = ev.Seq
		}
	}
	if err := sc.Err(); err != nil {
		return err
	}
	dl.events = events
	dl.seq = maxSeq
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
