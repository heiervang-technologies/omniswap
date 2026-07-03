package proxy

import (
	"bufio"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func sampleEvent(id string) DebitEvent {
	return DebitEvent{
		RequestID:      id,
		Ts:             time.Unix(1_700_000_000, 0).UTC(),
		KeyFingerprint: "ab12cd34ef56",
		ClientLabel:    "shay",
		Model:          "gemma-4-12b-256k",
		InputTokens:    1234,
		OutputTokens:   567,
		Node:           "amber",
		Country:        "NO",
	}
}

// readRaw parses the JSONL file straight off disk (independent of the in-memory
// state) — proves an event was durably written, not just cached.
func readRaw(t *testing.T, path string) []DebitEvent {
	t.Helper()
	f, err := os.Open(path)
	require.NoError(t, err)
	defer f.Close()
	var out []DebitEvent
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var ev DebitEvent
		require.NoError(t, json.Unmarshal(sc.Bytes(), &ev))
		out = append(out, ev)
	}
	return out
}

// TestDebitLog_Disabled: an empty path is a no-op log — the land-dark default.
func TestDebitLog_Disabled(t *testing.T) {
	dl := newDebitLog("", testLogger)
	assert.False(t, dl.enabled())

	got, err := dl.Append(sampleEvent("r1"))
	assert.NoError(t, err)
	assert.Zero(t, got.Seq, "disabled log must not assign a seq")
	assert.Empty(t, dl.Since(0))
	assert.Zero(t, dl.highWater())
}

// TestDebitLog_BadPathDisables: an unopenable path degrades to disabled, never
// crashes the pool.
func TestDebitLog_BadPathDisables(t *testing.T) {
	dl := newDebitLog(filepath.Join(t.TempDir(), "no-such-dir", "debits.jsonl"), testLogger)
	assert.False(t, dl.enabled())
}

// TestDebitLog_AppendAssignsMonotonicSeq: the log owns Seq; caller owns
// RequestID/Ts; Since filters by cursor.
func TestDebitLog_AppendAssignsMonotonicSeq(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	dl := newDebitLog(path, testLogger)
	require.True(t, dl.enabled())
	defer dl.close()

	for i, id := range []string{"r1", "r2", "r3"} {
		ev := sampleEvent(id)
		ev.Seq = 999 // caller-set seq must be OVERWRITTEN by the log
		got, err := dl.Append(ev)
		require.NoError(t, err)
		assert.Equal(t, uint64(i+1), got.Seq, "seq must be monotonic 1,2,3")
		assert.Equal(t, id, got.RequestID, "caller RequestID preserved")
	}
	assert.Equal(t, uint64(3), dl.highWater())

	all := dl.Since(0)
	require.Len(t, all, 3)
	assert.Equal(t, []uint64{1, 2, 3}, []uint64{all[0].Seq, all[1].Seq, all[2].Seq})

	tail := dl.Since(2)
	require.Len(t, tail, 1)
	assert.Equal(t, uint64(3), tail[0].Seq)
	assert.Equal(t, "r3", tail[0].RequestID)

	assert.Empty(t, dl.Since(3), "cursor at head => caught up")
}

// TestDebitLog_DurableBeforeAppendReturns: the event is on disk (fsync'd) by the
// time Append returns — read the raw file to prove it.
func TestDebitLog_DurableBeforeAppendReturns(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	dl := newDebitLog(path, testLogger)
	defer dl.close()

	_, err := dl.Append(sampleEvent("r1"))
	require.NoError(t, err)

	raw := readRaw(t, path)
	require.Len(t, raw, 1)
	assert.Equal(t, "r1", raw[0].RequestID)
	assert.Equal(t, uint64(1), raw[0].Seq)
	assert.Equal(t, int64(1234), raw[0].InputTokens)
	assert.Equal(t, "amber", raw[0].Node) // full field round-trip
}

// TestDebitLog_ResumesSeqAcrossRestart: a reopened log restores its events and
// continues seq from the max — never reuses an ordinal (the watermark would
// corrupt otherwise).
func TestDebitLog_ResumesSeqAcrossRestart(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")

	dl1 := newDebitLog(path, testLogger)
	for _, id := range []string{"r1", "r2", "r3"} {
		_, err := dl1.Append(sampleEvent(id))
		require.NoError(t, err)
	}
	require.NoError(t, dl1.close())

	dl2 := newDebitLog(path, testLogger)
	defer dl2.close()
	assert.Equal(t, uint64(3), dl2.highWater(), "seq resumes from the persisted max")
	assert.Len(t, dl2.Since(0), 3, "prior events restored")

	got, err := dl2.Append(sampleEvent("r4"))
	require.NoError(t, err)
	assert.Equal(t, uint64(4), got.Seq, "next seq continues, never resets")

	// The file now holds all four in order.
	raw := readRaw(t, path)
	assert.Equal(t, []uint64{1, 2, 3, 4}, []uint64{raw[0].Seq, raw[1].Seq, raw[2].Seq, raw[3].Seq})
}

// TestDebitLog_TornFinalLineTolerated: a crash mid-write leaves a partial last
// line; restore stops at the last good record and resumes cleanly.
func TestDebitLog_TornFinalLineTolerated(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	good, err := json.Marshal(func() DebitEvent { e := sampleEvent("r1"); e.Seq = 1; return e }())
	require.NoError(t, err)
	// one good line + a truncated JSON fragment (no newline) = torn write
	require.NoError(t, os.WriteFile(path, append(append(good, '\n'), []byte(`{"request_id":"r2","seq":2,"inp`)...), 0o600))

	dl := newDebitLog(path, testLogger)
	defer dl.close()
	assert.Equal(t, uint64(1), dl.highWater(), "resume from the last good record")
	require.Len(t, dl.Since(0), 1)

	got, err := dl.Append(sampleEvent("r2"))
	require.NoError(t, err)
	assert.Equal(t, uint64(2), got.Seq)
}

// TestDebitLog_TornThenRestartAppendRestart_NoSeqReuse is the regression for the
// must-fix: a torn fragment must be TRUNCATED on load, not merely skipped, or the
// next Append merges onto it and a SECOND restart loses events + reuses a Seq.
// Exercises torn -> restart -> append -> restart and asserts Seq never regresses.
func TestDebitLog_TornThenRestartAppendRestart_NoSeqReuse(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")

	// crash state: two good records + a torn (no-newline) fragment
	good1, err := json.Marshal(func() DebitEvent { e := sampleEvent("r1"); e.Seq = 1; return e }())
	require.NoError(t, err)
	good2, err := json.Marshal(func() DebitEvent { e := sampleEvent("r2"); e.Seq = 2; return e }())
	require.NoError(t, err)
	content := append(append(good1, '\n'), append(append(good2, '\n'), []byte(`{"request_id":"r3","seq":3,"inp`)...)...)
	require.NoError(t, os.WriteFile(path, content, 0o600))

	// restart #1: load truncates the torn frag; append r3 cleanly at seq 3
	dl1 := newDebitLog(path, testLogger)
	require.Equal(t, uint64(2), dl1.highWater())
	got, err := dl1.Append(sampleEvent("r3"))
	require.NoError(t, err)
	require.Equal(t, uint64(3), got.Seq)
	require.NoError(t, dl1.close())

	// restart #2: r3 must survive as its OWN line (not merged onto the frag),
	// highWater stays 3, and the next Seq is 4 — never a reuse of 3.
	dl2 := newDebitLog(path, testLogger)
	defer dl2.close()
	assert.Equal(t, uint64(3), dl2.highWater(), "seq must not regress after torn-then-append")
	all := dl2.Since(0)
	require.Len(t, all, 3)
	assert.Equal(t, []uint64{1, 2, 3}, []uint64{all[0].Seq, all[1].Seq, all[2].Seq})
	assert.Equal(t, "r3", all[2].RequestID)

	got2, err := dl2.Append(sampleEvent("r4"))
	require.NoError(t, err)
	assert.Equal(t, uint64(4), got2.Seq, "next seq continues at 4 — no reuse")
}

// TestDebitLog_EnabledFalseAfterClose: enabled() agrees with Append after close.
func TestDebitLog_EnabledFalseAfterClose(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	dl := newDebitLog(path, testLogger)
	require.True(t, dl.enabled())
	require.NoError(t, dl.close())
	assert.False(t, dl.enabled(), "closed log reports disabled")
	got, err := dl.Append(sampleEvent("rx"))
	assert.NoError(t, err)
	assert.Zero(t, got.Seq, "closed log no-ops Append (consistent with enabled()=false)")
}

// TestDebitLog_SinceReturnsIndependentCopy: mutating the returned slice must not
// corrupt the live log.
func TestDebitLog_SinceReturnsIndependentCopy(t *testing.T) {
	path := filepath.Join(t.TempDir(), "debits.jsonl")
	dl := newDebitLog(path, testLogger)
	defer dl.close()
	_, err := dl.Append(sampleEvent("r1"))
	require.NoError(t, err)

	snap := dl.Since(0)
	require.Len(t, snap, 1)
	snap[0].RequestID = "TAMPERED"

	fresh := dl.Since(0)
	assert.Equal(t, "r1", fresh[0].RequestID, "live log unaffected by caller mutation")
}
