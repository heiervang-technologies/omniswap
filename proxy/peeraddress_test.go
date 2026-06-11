package proxy

import (
	"net/url"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
)

func mustURL(s string) *url.URL {
	u, err := url.Parse(s)
	if err != nil {
		panic(err)
	}
	return u
}

// newTestPeerProxy builds a PeerProxy with a pre-seeded loaded-state cache so
// ResolveAddress never touches the network. crystal mirrors the real cluster:
// it declares the alias "crystal" but actually has "gemma-4-12b" loaded.
func newTestPeerProxy() *PeerProxy {
	peers := config.PeerDictionaryConfig{
		"titan":   {Proxy: "http://titan:8080", ProxyURL: mustURL("http://titan:8080"), Models: []string{"gemma-4-31b", "qwen3.6-27b"}},
		"crystal": {Proxy: "http://crystal:8080", ProxyURL: mustURL("http://crystal:8080"), Models: []string{"crystal"}},
		"lithium": {Proxy: "http://lithium:8080", ProxyURL: mustURL("http://lithium:8080"), Models: []string{"nanbeige-3b"}},
	}
	p := &PeerProxy{
		peers:        peers,
		proxyMap:     map[string]*peerProxyMember{},
		memberByPeer: map[string]*peerProxyMember{},
		peerOrder:    []string{"crystal", "lithium", "titan"},
		loadedCache:  map[string]peerLoadedSet{},
		loadedTTL:    time.Hour, // never expire during the test
	}
	now := time.Now()
	p.loadedCache["crystal"] = peerLoadedSet{order: []string{"gemma-4-12b"}, all: map[string]bool{"gemma-4-12b": true, "crystal": true}, fetchedAt: now}
	p.loadedCache["titan"] = peerLoadedSet{order: []string{"gemma-4-31b"}, all: map[string]bool{"gemma-4-31b": true}, fetchedAt: now}
	p.loadedCache["lithium"] = peerLoadedSet{order: nil, all: map[string]bool{}, fetchedAt: now}
	return p
}

func TestSplitAddress(t *testing.T) {
	cases := []struct {
		in        string
		model     string
		node      string
		hasAt     bool
		expectErr bool
	}{
		{"gemma-4-31b", "gemma-4-31b", "", false, false},
		{"gemma-4-12b@crystal", "gemma-4-12b", "crystal", true, false},
		{"@crystal", "", "crystal", true, false},
		{"any@", "any", "", true, false},
		{"  spaced @ node ", "spaced", "node", true, false},
		{"a@b@c", "", "", false, true},
	}
	for _, c := range cases {
		model, node, hasAt, err := splitAddress(c.in)
		if c.expectErr {
			if err == nil {
				t.Errorf("splitAddress(%q): expected error, got none", c.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("splitAddress(%q): unexpected error: %v", c.in, err)
			continue
		}
		if model != c.model || node != c.node || hasAt != c.hasAt {
			t.Errorf("splitAddress(%q) = (%q,%q,%v); want (%q,%q,%v)", c.in, model, node, hasAt, c.model, c.node, c.hasAt)
		}
	}
}

func TestResolveAddress(t *testing.T) {
	p := newTestPeerProxy()
	cases := []struct {
		in     string
		peerID string
		model  string
		rewrote bool
	}{
		// concrete model, any node -> existing behaviour
		{"gemma-4-31b", "", "gemma-4-31b", false},
		{"gemma-4-31b@any", "", "gemma-4-31b", true},
		{"gemma-4-31b@", "", "gemma-4-31b", true},
		// any model, pinned node -> pick what's loaded there
		{"any@crystal", "crystal", "gemma-4-12b", true},
		{"@crystal", "crystal", "gemma-4-12b", true},
		// concrete model, pinned node (loaded but not declared)
		{"gemma-4-12b@crystal", "crystal", "gemma-4-12b", true},
		// concrete model, pinned node (declared)
		{"crystal@crystal", "crystal", "crystal", true},
		{"gemma-4-31b@titan", "titan", "gemma-4-31b", true},
		// global any -> first peer in order with something loaded (crystal)
		{"any", "crystal", "gemma-4-12b", true},
		{"any@any", "crystal", "gemma-4-12b", true},
		{"@", "crystal", "gemma-4-12b", true},
		// pinned node with nothing loaded -> falls back to first declared
		{"any@lithium", "lithium", "nanbeige-3b", true},
	}
	for _, c := range cases {
		res, err := p.ResolveAddress(c.in)
		if err != nil {
			t.Errorf("ResolveAddress(%q): unexpected error: %v", c.in, err)
			continue
		}
		if res.PeerID != c.peerID || res.Model != c.model || res.Rewrote != c.rewrote {
			t.Errorf("ResolveAddress(%q) = {peer:%q model:%q rewrote:%v}; want {peer:%q model:%q rewrote:%v}",
				c.in, res.PeerID, res.Model, res.Rewrote, c.peerID, c.model, c.rewrote)
		}
	}
}

func TestResolveAddressErrors(t *testing.T) {
	p := newTestPeerProxy()
	cases := []struct {
		in   string
		code int
	}{
		{"foo@nosuchnode", 400},
		{"gemma-4-31b@crystal", 400}, // crystal serves neither declared nor loaded gemma-4-31b
		{"a@b@c", 400},
	}
	for _, c := range cases {
		_, err := p.ResolveAddress(c.in)
		if err == nil {
			t.Errorf("ResolveAddress(%q): expected error, got none", c.in)
			continue
		}
		if got := addrErrorCode(err); got != c.code {
			t.Errorf("ResolveAddress(%q): code = %d; want %d (%v)", c.in, got, c.code, err)
		}
	}
}

func TestResolveAddressGlobalAnyNothingLoaded(t *testing.T) {
	p := newTestPeerProxy()
	now := time.Now()
	for _, id := range p.peerOrder {
		p.loadedCache[id] = peerLoadedSet{all: map[string]bool{}, fetchedAt: now}
	}
	_, err := p.ResolveAddress("any")
	if err == nil {
		t.Fatal("expected 503 error when nothing is loaded, got none")
	}
	if got := addrErrorCode(err); got != 503 {
		t.Errorf("global any with nothing loaded: code = %d; want 503", got)
	}
}

func TestParseLoadedModels(t *testing.T) {
	body := []byte(`{"data":[
		{"id":"gemma-4-12b","aliases":["crystal"],"status":{"value":"loaded"}},
		{"id":"qwen3.6-27b","status":{"value":"stopped"}},
		{"id":"plain-no-status"}
	]}`)
	set := parseLoadedModels(body, time.Now())
	if len(set.order) != 2 {
		t.Fatalf("expected 2 loaded ids, got %d: %v", len(set.order), set.order)
	}
	if set.order[0] != "gemma-4-12b" || set.order[1] != "plain-no-status" {
		t.Errorf("unexpected loaded order: %v", set.order)
	}
	if !set.all["crystal"] {
		t.Error("expected alias 'crystal' to be in the membership set")
	}
	if set.all["qwen3.6-27b"] {
		t.Error("stopped model should not be marked loaded")
	}
}
