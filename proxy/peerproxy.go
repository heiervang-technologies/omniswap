package proxy

import (
	"fmt"
	"net"
	"net/http"
	"net/http/httputil"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
)

type peerProxyMember struct {
	peerID       string
	reverseProxy *httputil.ReverseProxy
	apiKey       string
	// inFlight is the number of requests currently being proxied to this peer.
	// Used by pickPeerForModel to spread concurrent load across peers that serve
	// the same model. Touched only via sync/atomic.
	inFlight int64
}

type PeerProxy struct {
	peers config.PeerDictionaryConfig
	// modelPeers maps a model id to EVERY peer that serves it (a slice, not a
	// single winner). When more than one peer serves a model, ProxyRequest ranks
	// them warm > vacant > busy so traffic prefers an already-loaded copy and
	// spills to idle GPUs under load instead of serializing on one node.
	modelPeers map[string][]*peerProxyMember

	// memberByPeer indexes proxy members by peer id for node-pinned (`@node`)
	// dispatch, which bypasses the model->peer map.
	memberByPeer map[string]*peerProxyMember
	// peerOrder is the sorted list of peer ids, for deterministic wildcard
	// resolution and error messages.
	peerOrder []string

	logger *LogMonitor

	// loaded-state cache for resolving the `any` wildcard against what each
	// peer currently has up. See peeraddress.go.
	httpClient  *http.Client
	loadedMu    sync.RWMutex
	loadedCache map[string]peerLoadedSet
	loadedTTL   time.Duration
}

func NewPeerProxy(peers config.PeerDictionaryConfig, proxyLogger *LogMonitor) (*PeerProxy, error) {
	modelPeers := make(map[string][]*peerProxyMember)
	memberByPeer := make(map[string]*peerProxyMember)

	// Sort peer IDs for consistent iteration order
	peerIDs := make([]string, 0, len(peers))
	for peerID := range peers {
		peerIDs = append(peerIDs, peerID)
	}
	sort.Strings(peerIDs)

	// Create a shared transport with reasonable timeouts for peer connections
	// these can be tuned with feedback later
	peerTransport := &http.Transport{
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second, // Connection timeout
			KeepAlive: 30 * time.Second,
		}).DialContext,
		TLSHandshakeTimeout: 10 * time.Second,
		// Time to wait for the peer's response headers (i.e. first byte). With
		// load-aware routing a by-name request can be dispatched to a vacant peer
		// that must COLD-LOAD the model first; on a Pascal gem that takes ~60-90s,
		// so a 60s ceiling 502'd every spilled request mid-load. 180s tolerates a
		// cold start while still failing a genuinely hung peer.
		ResponseHeaderTimeout: 180 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   10,
		IdleConnTimeout:       90 * time.Second,
	}

	for _, peerID := range peerIDs {
		peer := peers[peerID]
		// Create reverse proxy for this peer
		reverseProxy := httputil.NewSingleHostReverseProxy(peer.ProxyURL)
		reverseProxy.Transport = peerTransport

		// Wrap Director to set Host header for remote hosts (not localhost)
		originalDirector := reverseProxy.Director
		reverseProxy.Director = func(req *http.Request) {
			originalDirector(req)
			// Ensure Host header matches target URL for remote proxying
			req.Host = req.URL.Host
		}

		reverseProxy.ModifyResponse = func(resp *http.Response) error {
			if strings.Contains(strings.ToLower(resp.Header.Get("Content-Type")), "text/event-stream") {
				resp.Header.Set("X-Accel-Buffering", "no")
			}
			return nil
		}

		reverseProxy.ErrorHandler = func(w http.ResponseWriter, r *http.Request, err error) {
			proxyLogger.Warnf("peer %s: proxy error: %v", peerID, err)
			errMsg := fmt.Sprintf("peer proxy error: %v", err)
			if runtime.GOOS == "darwin" && strings.Contains(err.Error(), "connect: no route to host") {
				errMsg += " (hint: on macOS, check System Settings > Privacy & Security > Local Network permissions)"
			}
			http.Error(w, errMsg, http.StatusBadGateway)
		}

		pp := &peerProxyMember{
			peerID:       peerID,
			reverseProxy: reverseProxy,
			apiKey:       peer.ApiKey,
		}
		memberByPeer[peerID] = pp

		// Register this peer for every model it serves. Unlike the old
		// first-peer-wins map, ALL peers that list a model are kept so
		// ProxyRequest can pick the best one (warm/idle/least-loaded) at
		// dispatch time.
		for _, modelID := range peer.Models {
			modelPeers[modelID] = append(modelPeers[modelID], pp)
		}
	}

	// Surface which models are multi-peer (load-aware routing applies to these).
	for modelID, members := range modelPeers {
		if len(members) > 1 {
			ids := make([]string, 0, len(members))
			for _, m := range members {
				ids = append(ids, m.peerID)
			}
			sort.Strings(ids)
			proxyLogger.Infof("peer routing: model %q served by %d peers %v (load-aware: warm>vacant>busy)", modelID, len(members), ids)
		}
	}

	return &PeerProxy{
		peers:        peers,
		modelPeers:   modelPeers,
		memberByPeer: memberByPeer,
		peerOrder:    peerIDs,
		logger:       proxyLogger,
		// Short-timeout client used only for polling peers' /v1/models to
		// resolve the `any` wildcard; never used for streaming inference.
		httpClient:  &http.Client{Timeout: 5 * time.Second},
		loadedCache: make(map[string]peerLoadedSet),
		loadedTTL:   5 * time.Second,
	}, nil
}

func (p *PeerProxy) HasPeerModel(modelID string) bool {
	return len(p.modelPeers[modelID]) > 0
}

// GetPeerFilters returns the filters for a peer model, or empty filters if not found
func (p *PeerProxy) GetPeerFilters(modelID string) config.Filters {
	members := p.modelPeers[modelID]
	if len(members) == 0 {
		return config.Filters{}
	}
	// Filters are configured per-peer; for a model served by several peers we
	// apply the first peer's (peers serving the same model are expected to share
	// filter config). The pinned `@node` path uses that node's own PeerFilters().
	peer, found := p.peers[members[0].peerID]
	if !found {
		return config.Filters{}
	}
	return peer.Filters
}

func (p *PeerProxy) ListPeers() config.PeerDictionaryConfig {
	return p.peers
}

// ProxyRequest dispatches a by-name (non-pinned) request to the best peer that
// serves the model. When several peers serve it, pickPeerForModel ranks them
// warm > vacant > busy, so we prefer an already-loaded copy and spill to idle
// GPUs under load. The peer's in-flight count is tracked around the dispatch so
// concurrent requests fan out instead of serializing on one node.
func (p *PeerProxy) ProxyRequest(model_id string, writer http.ResponseWriter, request *http.Request) error {
	pp := p.pickPeerForModel(model_id)
	if pp == nil {
		return fmt.Errorf("no peer proxy found for model %s", model_id)
	}

	// Inject API key if configured for this peer
	if pp.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+pp.apiKey)
		request.Header.Set("x-api-key", pp.apiKey)
	}

	atomic.AddInt64(&pp.inFlight, 1)
	defer atomic.AddInt64(&pp.inFlight, -1)
	pp.reverseProxy.ServeHTTP(writer, request)
	return nil
}

// pickPeerForModel selects which peer serves modelID among all peers that list
// it. One candidate -> return it. Several -> rank by
//
//	rank = inFlight*2 + bias,  bias{warm:0, vacant:1, occupied-by-other:3}
//
// (lower wins; ties broken by the deterministic peerOrder, since candidates are
// appended in sorted-peerID order and we keep the first on a tie). So a warm
// idle peer always wins; a warm-but-busy peer yields to a vacant GPU (a one-time
// cold-load, after which it too is warm); a peer running a different model is
// evicted only when nothing better exists. Warm/loaded state is read from the
// same short-TTL loadedCache the `@node`/`any` resolver uses (no extra polling).
func (p *PeerProxy) pickPeerForModel(modelID string) *peerProxyMember {
	candidates := p.modelPeers[modelID]
	switch len(candidates) {
	case 0:
		return nil
	case 1:
		return candidates[0]
	}

	var best *peerProxyMember
	var bestKey int64
	for _, pp := range candidates {
		peer := p.peers[pp.peerID]
		loaded := p.peerLoaded(pp.peerID, peer)
		var bias int64
		switch {
		case loaded.all[modelID]: // model resident here -> warm, no cold load
			bias = 0
		case len(loaded.order) == 0: // nothing loaded -> cold-load onto a free GPU
			bias = 1
		default: // a different model is resident -> cold-load needs an evict
			bias = 3
		}
		key := atomic.LoadInt64(&pp.inFlight)*2 + bias
		if best == nil || key < bestKey {
			best, bestKey = pp, key
		}
	}
	return best
}

// ProxyRequestToPeer dispatches directly to a named peer, bypassing the
// model->peer map. Used for node-pinned (`<model>@<node>`) addressing where the
// caller has already chosen the node.
func (p *PeerProxy) ProxyRequestToPeer(peerID, modelID string, writer http.ResponseWriter, request *http.Request) error {
	pp, found := p.memberByPeer[peerID]
	if !found {
		return fmt.Errorf("no peer proxy found for node %s", peerID)
	}

	if pp.apiKey != "" {
		request.Header.Set("Authorization", "Bearer "+pp.apiKey)
		request.Header.Set("x-api-key", pp.apiKey)
	}

	// Count pinned (`@node`) traffic too, so the load-aware picker sees a node's
	// real occupancy when ranking by-name requests for the same model.
	atomic.AddInt64(&pp.inFlight, 1)
	defer atomic.AddInt64(&pp.inFlight, -1)
	pp.reverseProxy.ServeHTTP(writer, request)
	return nil
}

// PeerFilters returns the filters configured for a peer by id.
func (p *PeerProxy) PeerFilters(peerID string) config.Filters {
	peer, found := p.peers[peerID]
	if !found {
		return config.Filters{}
	}
	return peer.Filters
}
