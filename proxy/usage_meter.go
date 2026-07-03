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

// usageReaderAllowed reports whether the caller's label may access the usage
// analytics (admin-only). config.UsageReaders, default just "admin". apiKeyAuth
// has already set the caller's label in the gin context.
func (pm *ProxyManager) usageReaderAllowed(c *gin.Context) bool {
	readers := pm.config.UsageReaders
	if len(readers) == 0 {
		readers = []string{"admin"}
	}
	caller := c.GetString("client")
	for _, r := range readers {
		if caller == r {
			return true
		}
	}
	return false
}

// usageHandler serves the per-client/country/IP usage rollup as JSON (admin-only).
func (pm *ProxyManager) usageHandler(c *gin.Context) {
	if !pm.usageReaderAllowed(c) {
		pm.sendErrorResponse(c, http.StatusForbidden, "forbidden: /usage is admin-only")
		return
	}
	if pm.usageMeter == nil {
		c.JSON(http.StatusOK, gin.H{"clients": gin.H{}, "order": []string{}, "totals": gin.H{}})
		return
	}
	c.JSON(http.StatusOK, pm.usageMeter.Snapshot())
}

// usageResetHandler clears the usage rollup (admin-only). The clean replacement
// for the alpine-pod hostPath wipe.
func (pm *ProxyManager) usageResetHandler(c *gin.Context) {
	if !pm.usageReaderAllowed(c) {
		pm.sendErrorResponse(c, http.StatusForbidden, "forbidden: /usage is admin-only")
		return
	}
	if pm.usageMeter != nil {
		pm.usageMeter.Reset()
	}
	c.JSON(http.StatusOK, gin.H{"reset": true})
}

// clientFromContext reads the usage-attribution label that apiKeyAuth +
// proxyInferenceHandler stash in the request context. Empty when unauth'd or
// not an inference path.
func clientFromContext(r *http.Request) string {
	return ctxStr(r, "client")
}

// countryFromContext reads the request's Cf-IPCountry (set by apiKeyAuth +
// proxyInferenceHandler). Empty for LAN / non-Cloudflare requests.
func countryFromContext(r *http.Request) string {
	return ctxStr(r, "country")
}

// ipFromContext reads the request's source IP (Cf-Connecting-Ip / ClientIP).
func ipFromContext(r *http.Request) string {
	return ctxStr(r, "ip")
}

func ctxStr(r *http.Request, key string) string {
	if r == nil {
		return ""
	}
	if v, ok := r.Context().Value(proxyCtxKey(key)).(string); ok {
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

// ipUsage is the per-source-IP rollup. Admin-only (IPs are PII); the real client
// IP comes from Cloudflare's Cf-Connecting-Ip when via the tunnel.
type ipUsage struct {
	Requests     int64     `json:"requests"`
	InputTokens  int64     `json:"input_tokens"`
	OutputTokens int64     `json:"output_tokens"`
	Client       string    `json:"client"`  // last key label seen from this IP
	Country      string    `json:"country"` // last country seen
	FirstSeen    time.Time `json:"first_seen"`
	LastSeen     time.Time `json:"last_seen"`
}

// usagePersist is the on-disk shape (clients + country + ip rollups).
type usagePersist struct {
	Clients   map[string]*clientUsage `json:"clients"`
	Countries map[string]*modelUsage  `json:"countries"`
	IPs       map[string]*ipUsage     `json:"ips"`
}

// UsageMeter aggregates per-client / per-country / per-IP token usage.
// Thread-safe; persisted to disk.
type UsageMeter struct {
	mu        sync.Mutex
	path      string
	clients   map[string]*clientUsage
	countries map[string]*modelUsage // ISO country (from Cf-IPCountry) -> totals
	ips       map[string]*ipUsage    // source IP (Cf-Connecting-Ip / ClientIP) -> totals
	dirty     bool
	logger    *LogMonitor
}

func NewUsageMeter(path string, logger *LogMonitor) *UsageMeter {
	m := &UsageMeter{
		path:      path,
		clients:   map[string]*clientUsage{},
		countries: map[string]*modelUsage{},
		ips:       map[string]*ipUsage{},
		logger:    logger,
	}
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
	var snap usagePersist
	if err := json.Unmarshal(b, &snap); err == nil {
		if snap.Clients != nil {
			m.clients = snap.Clients
		}
		if snap.Countries != nil {
			m.countries = snap.Countries
		}
		if snap.IPs != nil {
			m.ips = snap.IPs
		}
		if m.logger != nil {
			m.logger.Infof("usage meter: loaded %d clients / %d countries / %d ips from %s", len(m.clients), len(m.countries), len(m.ips), m.path)
		}
	}
}

// Record attributes one completed request's token usage to a client, a country
// (Cf-IPCountry; "local" for LAN), and a source IP (Cf-Connecting-Ip/ClientIP).
// Empty client -> "unknown". Negative tokens clamp to 0.
func (m *UsageMeter) Record(client, model, country, ip string, inTok, outTok int) {
	if client == "" {
		client = "unknown"
	}
	if model == "" {
		model = "unknown"
	}
	if country == "" {
		country = "local"
	}
	if ip == "" {
		ip = "unknown"
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

	co := m.countries[country]
	if co == nil {
		co = &modelUsage{}
		m.countries[country] = co
	}
	co.Requests++
	co.InputTokens += int64(inTok)
	co.OutputTokens += int64(outTok)

	ipu := m.ips[ip]
	if ipu == nil {
		ipu = &ipUsage{FirstSeen: now}
		m.ips[ip] = ipu
	}
	ipu.Requests++
	ipu.InputTokens += int64(inTok)
	ipu.OutputTokens += int64(outTok)
	ipu.Client = client
	ipu.Country = country
	ipu.LastSeen = now

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
//
// It DEEP-COPIES the rollup maps under the lock so the returned value is fully
// independent of the live maps. The caller (usageHandler) marshals the result
// to JSON without the lock, while Record() keeps mutating the live maps — if we
// returned the live maps here, that concurrent map read (marshal) + write
// (Record) would be a fatal "concurrent map read and map write" runtime panic
// that crashes the pool (regression-guarded by TestUsageMeter_ConcurrentSnapshotMarshal).
func (m *UsageMeter) Snapshot() map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()

	clients := make(map[string]*clientUsage, len(m.clients))
	order := make([]string, 0, len(m.clients))
	var totReq, totIn, totOut int64
	for c, cu := range m.clients {
		byModel := make(map[string]*modelUsage, len(cu.ByModel))
		for mdl, mdu := range cu.ByModel {
			cp := *mdu
			byModel[mdl] = &cp
		}
		clients[c] = &clientUsage{
			Requests:     cu.Requests,
			InputTokens:  cu.InputTokens,
			OutputTokens: cu.OutputTokens,
			ByModel:      byModel,
			FirstSeen:    cu.FirstSeen,
			LastSeen:     cu.LastSeen,
		}
		order = append(order, c)
		totReq += cu.Requests
		totIn += cu.InputTokens
		totOut += cu.OutputTokens
	}
	countries := make(map[string]*modelUsage, len(m.countries))
	for k, v := range m.countries {
		cp := *v
		countries[k] = &cp
	}
	ips := make(map[string]*ipUsage, len(m.ips))
	for k, v := range m.ips {
		cp := *v
		ips[k] = &cp
	}

	sort.Slice(order, func(i, j int) bool {
		// busiest first (by total tokens), then name
		a, b := clients[order[i]], clients[order[j]]
		ai, bj := a.InputTokens+a.OutputTokens, b.InputTokens+b.OutputTokens
		if ai != bj {
			return ai > bj
		}
		return order[i] < order[j]
	})
	return map[string]any{
		"as_of":      time.Now().UTC().Format(time.RFC3339),
		"order":      order,
		"clients":    clients,
		"by_country": countries,
		"by_ip":      ips,
		"totals": map[string]int64{
			"requests":      totReq,
			"input_tokens":  totIn,
			"output_tokens": totOut,
			"total_tokens":  totIn + totOut,
		},
	}
}

// Reset clears all usage rollups (admin op) and flushes the now-empty state to
// disk so the reset survives a restart.
func (m *UsageMeter) Reset() {
	m.mu.Lock()
	m.clients = map[string]*clientUsage{}
	m.countries = map[string]*modelUsage{}
	m.ips = map[string]*ipUsage{}
	m.dirty = true
	m.mu.Unlock()
	m.flush()
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
	b, err := json.MarshalIndent(usagePersist{Clients: m.clients, Countries: m.countries, IPs: m.ips}, "", "  ")
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
