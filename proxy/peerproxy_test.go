package proxy

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewPeerProxy_EmptyPeers(t *testing.T) {
	peers := config.PeerDictionaryConfig{}
	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	assert.NotNil(t, pm)
	assert.Empty(t, pm.modelPeers)
}

func TestNewPeerProxy_SinglePeer(t *testing.T) {
	proxyURL, _ := url.Parse("http://peer1.example.com:8080")
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    "http://peer1.example.com:8080",
			ProxyURL: proxyURL,
			ApiKey:   "test-key",
			Models:   []string{"model-a", "model-b"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	assert.Len(t, pm.modelPeers, 2)
	assert.True(t, pm.HasPeerModel("model-a"))
	assert.True(t, pm.HasPeerModel("model-b"))
	assert.False(t, pm.HasPeerModel("model-c"))
}

func TestNewPeerProxy_MultiplePeers(t *testing.T) {
	proxyURL1, _ := url.Parse("http://peer1.example.com:8080")
	proxyURL2, _ := url.Parse("http://peer2.example.com:8080")
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    "http://peer1.example.com:8080",
			ProxyURL: proxyURL1,
			Models:   []string{"model-a", "model-b"},
		},
		"peer2": config.PeerConfig{
			Proxy:    "http://peer2.example.com:8080",
			ProxyURL: proxyURL2,
			Models:   []string{"model-c", "model-d"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	assert.Len(t, pm.modelPeers, 4)
	assert.True(t, pm.HasPeerModel("model-a"))
	assert.True(t, pm.HasPeerModel("model-b"))
	assert.True(t, pm.HasPeerModel("model-c"))
	assert.True(t, pm.HasPeerModel("model-d"))
}

func TestNewPeerProxy_SharedModelMultiPeer(t *testing.T) {
	// When the same model is served by multiple peers, ALL of them are kept
	// (in sorted-peerID order) so ProxyRequest can pick the best at dispatch.
	proxyURL1, _ := url.Parse("http://peer1.example.com:8080")
	proxyURL2, _ := url.Parse("http://peer2.example.com:8080")
	peers := config.PeerDictionaryConfig{
		"beta-peer": config.PeerConfig{
			Proxy:    "http://peer2.example.com:8080",
			ProxyURL: proxyURL2,
			Models:   []string{"shared-model"},
		},
		"alpha-peer": config.PeerConfig{
			Proxy:    "http://peer1.example.com:8080",
			ProxyURL: proxyURL1,
			Models:   []string{"shared-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)
	// One model key, mapping to BOTH peers, in sorted-peerID order.
	assert.Len(t, pm.modelPeers, 1)
	assert.True(t, pm.HasPeerModel("shared-model"))
	members := pm.modelPeers["shared-model"]
	require.Len(t, members, 2)
	assert.Equal(t, "alpha-peer", members[0].peerID)
	assert.Equal(t, "beta-peer", members[1].peerID)
}

func TestPickPeerForModel(t *testing.T) {
	now := time.Now()
	// served is populated for any reachable peer (it advertises the model in its
	// /v1/models, loaded or not); an unreachable peer's poll fails -> empty served.
	warm := peerLoadedSet{all: map[string]bool{"m": true}, order: []string{"m"}, served: map[string]bool{"m": true}, fetchedAt: now}
	vacant := peerLoadedSet{all: map[string]bool{}, order: nil, served: map[string]bool{"m": true}, fetchedAt: now}
	other := peerLoadedSet{all: map[string]bool{"x": true}, order: []string{"x"}, served: map[string]bool{"m": true, "x": true}, fetchedAt: now}
	unreachable := peerLoadedSet{served: map[string]bool{}, fetchedAt: now} // poll failed -> nothing served

	// candidates a,b,c (sorted-peerID order, as NewPeerProxy builds them)
	newP := func() (*PeerProxy, map[string]*peerProxyMember) {
		a := &peerProxyMember{peerID: "a"}
		b := &peerProxyMember{peerID: "b"}
		c := &peerProxyMember{peerID: "c"}
		p := &PeerProxy{
			peers:       config.PeerDictionaryConfig{"a": {}, "b": {}, "c": {}},
			modelPeers:  map[string][]*peerProxyMember{"m": {a, b, c}},
			loadedCache: map[string]peerLoadedSet{},
			loadedTTL:   time.Hour, // pre-seeded cache never expires during the test
		}
		return p, map[string]*peerProxyMember{"a": a, "b": b, "c": c}
	}

	t.Run("warm beats vacant and other regardless of position", func(t *testing.T) {
		p, m := newP()
		p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = vacant, warm, other
		assert.Equal(t, m["b"], p.pickPeerForModel("m"))
	})

	t.Run("no warm: vacant beats occupied-by-other", func(t *testing.T) {
		p, m := newP()
		p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = other, vacant, other
		assert.Equal(t, m["b"], p.pickPeerForModel("m"))
	})

	t.Run("warm but busy spills to a vacant GPU", func(t *testing.T) {
		p, m := newP()
		p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = warm, vacant, vacant
		atomic.StoreInt64(&m["a"].inFlight, 1) // warm-busy key=2 > vacant key=1
		assert.Equal(t, m["b"], p.pickPeerForModel("m"))
	})

	t.Run("warm idle beats warm busy", func(t *testing.T) {
		p, m := newP()
		p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = warm, warm, vacant
		atomic.StoreInt64(&m["a"].inFlight, 2)
		assert.Equal(t, m["b"], p.pickPeerForModel("m"))
	})

	t.Run("equal rank breaks to the first (deterministic) peer", func(t *testing.T) {
		p, m := newP()
		p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = warm, warm, vacant
		assert.Equal(t, m["a"], p.pickPeerForModel("m"))
	})

	t.Run("single candidate returns without consulting the cache", func(t *testing.T) {
		solo := &peerProxyMember{peerID: "z"}
		p := &PeerProxy{modelPeers: map[string][]*peerProxyMember{"solo": {solo}}}
		assert.Equal(t, solo, p.pickPeerForModel("solo"))
		assert.Nil(t, p.pickPeerForModel("absent"))
	})

	t.Run("reservation fans out repeated picks across equal peers", func(t *testing.T) {
		// pickPeerForModel reserves (increments inFlight) the winner. Without a
		// matching decrement between picks (i.e. requests still in flight), three
		// equal warm peers should round-robin rather than all stampede the first.
		p, _ := newP()
		p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = warm, warm, warm
		var got []string
		for i := 0; i < 6; i++ {
			got = append(got, p.pickPeerForModel("m").peerID)
		}
		assert.Equal(t, []string{"a", "b", "c", "a", "b", "c"}, got)
	})

	t.Run("unreachable peer is deprioritized below a reachable one", func(t *testing.T) {
		// a dead peer (failed poll -> empty served) must not be picked over a
		// reachable vacant peer, even though both look "not loaded".
		p, m := newP()
		p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = unreachable, vacant, unreachable
		assert.Equal(t, m["b"], p.pickPeerForModel("m"))
	})

	t.Run("unreachable still beaten by a warm peer", func(t *testing.T) {
		p, m := newP()
		p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = unreachable, unreachable, warm
		assert.Equal(t, m["c"], p.pickPeerForModel("m"))
	})

	t.Run("all unreachable still returns one (graceful upstream 502)", func(t *testing.T) {
		p, _ := newP()
		p.loadedCache["a"], p.loadedCache["b"], p.loadedCache["c"] = unreachable, unreachable, unreachable
		assert.NotNil(t, p.pickPeerForModel("m"))
	})
}

func TestHasPeerModel(t *testing.T) {
	proxyURL, _ := url.Parse("http://peer1.example.com:8080")
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    "http://peer1.example.com:8080",
			ProxyURL: proxyURL,
			Models:   []string{"existing-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	assert.True(t, pm.HasPeerModel("existing-model"))
	assert.False(t, pm.HasPeerModel("non-existing-model"))
}

func TestProxyRequest_ModelNotFound(t *testing.T) {
	peers := config.PeerDictionaryConfig{}
	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("non-existing-model", w, req)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no peer proxy found for model non-existing-model")
}

func TestProxyRequest_Success(t *testing.T) {
	// Create a test server to act as the peer
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("response from peer"))
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	assert.Equal(t, http.StatusOK, w.Code)
	assert.Equal(t, "response from peer", w.Body.String())
}

func TestProxyRequest_ApiKeyInjection(t *testing.T) {
	// Create a test server that checks for the Authorization header
	var receivedAuthHeader string
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuthHeader = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			ApiKey:   "secret-api-key",
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	assert.Equal(t, "Bearer secret-api-key", receivedAuthHeader)
}

func TestProxyRequest_NoApiKey(t *testing.T) {
	// Create a test server that checks for the Authorization header
	var receivedAuthHeader string
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuthHeader = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			ApiKey:   "", // No API key
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	assert.Empty(t, receivedAuthHeader)
}

func TestProxyRequest_HostHeaderSet(t *testing.T) {
	// Create a test server that checks the Host header
	var receivedHost string
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedHost = r.Host
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	// The Host header should be set to the target URL's host
	assert.True(t, strings.HasPrefix(receivedHost, "127.0.0.1:"))
}

func TestProxyRequest_SSEHeaderModification(t *testing.T) {
	// Create a test server that returns SSE content type
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
	}))
	defer testServer.Close()

	proxyURL, _ := url.Parse(testServer.URL)
	peers := config.PeerDictionaryConfig{
		"peer1": config.PeerConfig{
			Proxy:    testServer.URL,
			ProxyURL: proxyURL,
			Models:   []string{"test-model"},
		},
	}

	pm, err := NewPeerProxy(peers, testLogger)
	require.NoError(t, err)

	req := httptest.NewRequest("POST", "/v1/chat/completions", nil)
	w := httptest.NewRecorder()

	err = pm.ProxyRequest("test-model", w, req)
	assert.NoError(t, err)
	// The X-Accel-Buffering header should be set to "no" for SSE
	assert.Equal(t, "no", w.Header().Get("X-Accel-Buffering"))
}

func TestWarmLoadedDeadPeerDegradesFast(t *testing.T) {
	// A peer whose host refuses/blackholes the poll must not stall WarmLoaded;
	// it should negative-cache (empty) and return well under the old 5s serial hang.
	p := &PeerProxy{
		peers: config.PeerDictionaryConfig{
			"dead": {Proxy: "http://127.0.0.1:1", ProxyURL: mustURL("http://127.0.0.1:1")},
		},
		peerOrder:   []string{"dead"},
		loadedCache: map[string]peerLoadedSet{},
		loadedTTL:   5 * time.Second,
		httpClient:  &http.Client{Timeout: 2 * time.Second},
	}
	start := time.Now()
	p.WarmLoaded()
	if d := time.Since(start); d > 3*time.Second {
		t.Fatalf("WarmLoaded with a dead peer took %v, expected fast degrade", d)
	}
	ls := p.peerLoaded("dead", p.peers["dead"])
	assert.Empty(t, ls.order, "dead peer should negative-cache to empty loaded set")
}
