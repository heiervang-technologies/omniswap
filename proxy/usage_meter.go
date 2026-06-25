package proxy

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"net/http"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
)

// usageHandler serves the per-client token-usage rollup as JSON (auth-gated).
func (pm *ProxyManager) usageHandler(c *gin.Context) {
	if pm.usageMeter == nil {
		c.JSON(http.StatusOK, gin.H{"clients": gin.H{}, "order": []string{}, "totals": gin.H{}})
		return
	}
	c.JSON(http.StatusOK, pm.usageMeter.Snapshot())
}

// clientFromContext reads the usage-attribution label that apiKeyAuth +
// proxyInferenceHandler stash in the request context. Empty when unauth'd or
// not an inference path.
func clientFromContext(r *http.Request) string {
	if r == nil {
		return ""
	}
	if v, ok := r.Context().Value(proxyCtxKey("client")).(string); ok {
		return v
	}
	return ""
}

// usage_meter.go — per-client token-usage analytics for the auth'd pool.
//
// The pool is the auth choke point: it validates the API key AND already parses
// each response's `usage` (via metrics_monitor). This layer attributes that
// usage to a CLIENT LABEL (e.g. "shay", "markus") — never the raw key — and
// keeps a running per-client (+ per-model) rollup, persisted to disk so it
// survives pool restarts. Surfaced at GET /usage (auth-gated) for the dashboard.
//
// Attribution: apiKeyAuth maps the validated key to a label via a
// fingerprint->label table (sha256 prefix; the raw key never appears in the
// labels config). An unlabelled key shows up as its fingerprint, so usage is
// still distinguished without exposing the secret.

// keyFingerprint returns a short, non-reversible id for an API key, so usage can
// be attributed/labelled without ever storing or exposing the raw key value.
func keyFingerprint(key string) string {
	sum := sha256.Sum256([]byte(key))
	return hex.EncodeToString(sum[:])[:12]
}

type modelUsage struct {
	Requests     int64 `json:"requests"`
	InputTokens  int64 `json:"input_tokens"`
	OutputTokens int64 `json:"output_tokens"`
}

type clientUsage struct {
	Requests     int64                  `json:"requests"`
	InputTokens  int64                  `json:"input_tokens"`
	OutputTokens int64                  `json:"output_tokens"`
	ByModel      map[string]*modelUsage `json:"by_model"`
	FirstSeen    time.Time              `json:"first_seen"`
	LastSeen     time.Time              `json:"last_seen"`
}

// UsageMeter aggregates per-client token usage. Thread-safe; persisted to disk.
type UsageMeter struct {
	mu      sync.Mutex
	path    string
	clients map[string]*clientUsage
	dirty   bool
	logger  *LogMonitor
}

func NewUsageMeter(path string, logger *LogMonitor) *UsageMeter {
	m := &UsageMeter{path: path, clients: map[string]*clientUsage{}, logger: logger}
	m.load()
	if m.path != "" {
		go m.persistLoop(15 * time.Second)
	}
	return m
}

func (m *UsageMeter) load() {
	if m.path == "" {
		return
	}
	b, err := os.ReadFile(m.path)
	if err != nil {
		return // first run / no file yet
	}
	var snap map[string]*clientUsage
	if err := json.Unmarshal(b, &snap); err == nil && snap != nil {
		m.clients = snap
		if m.logger != nil {
			m.logger.Infof("usage meter: loaded %d clients from %s", len(snap), m.path)
		}
	}
}

// Record attributes one completed request's token usage to a client. A zero/empty
// client is bucketed as "unknown". Negative tokens (missing usage) clamp to 0.
func (m *UsageMeter) Record(client, model string, inTok, outTok int) {
	if client == "" {
		client = "unknown"
	}
	if model == "" {
		model = "unknown"
	}
	if inTok < 0 {
		inTok = 0
	}
	if outTok < 0 {
		outTok = 0
	}
	now := time.Now()

	m.mu.Lock()
	defer m.mu.Unlock()
	cu := m.clients[client]
	if cu == nil {
		cu = &clientUsage{ByModel: map[string]*modelUsage{}, FirstSeen: now}
		m.clients[client] = cu
	}
	cu.Requests++
	cu.InputTokens += int64(inTok)
	cu.OutputTokens += int64(outTok)
	cu.LastSeen = now

	mu := cu.ByModel[model]
	if mu == nil {
		mu = &modelUsage{}
		cu.ByModel[model] = mu
	}
	mu.Requests++
	mu.InputTokens += int64(inTok)
	mu.OutputTokens += int64(outTok)
	m.dirty = true
}

// Snapshot returns a JSON-serialisable view for the /usage endpoint: the
// per-client rollup plus a deterministically-ordered client list and totals.
func (m *UsageMeter) Snapshot() map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()

	order := make([]string, 0, len(m.clients))
	var totReq, totIn, totOut int64
	for c, cu := range m.clients {
		order = append(order, c)
		totReq += cu.Requests
		totIn += cu.InputTokens
		totOut += cu.OutputTokens
	}
	sort.Slice(order, func(i, j int) bool {
		// busiest first (by total tokens), then name
		a, b := m.clients[order[i]], m.clients[order[j]]
		ai, bj := a.InputTokens+a.OutputTokens, b.InputTokens+b.OutputTokens
		if ai != bj {
			return ai > bj
		}
		return order[i] < order[j]
	})
	return map[string]any{
		"as_of":   time.Now().UTC().Format(time.RFC3339),
		"order":   order,
		"clients": m.clients,
		"totals": map[string]int64{
			"requests":      totReq,
			"input_tokens":  totIn,
			"output_tokens": totOut,
			"total_tokens":  totIn + totOut,
		},
	}
}

func (m *UsageMeter) persistLoop(every time.Duration) {
	for {
		time.Sleep(every)
		m.flush()
	}
}

// flush atomically writes the current rollup to disk if it changed.
func (m *UsageMeter) flush() {
	m.mu.Lock()
	if !m.dirty || m.path == "" {
		m.mu.Unlock()
		return
	}
	b, err := json.MarshalIndent(m.clients, "", "  ")
	if err == nil {
		m.dirty = false
	}
	m.mu.Unlock()
	if err != nil {
		return
	}
	tmp := m.path + ".tmp"
	if err := os.WriteFile(tmp, b, 0o644); err != nil {
		if m.logger != nil {
			m.logger.Warnf("usage meter: write failed: %v", err)
		}
		return
	}
	_ = os.Rename(tmp, m.path)
}
