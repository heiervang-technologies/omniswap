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
		modelPeers:   map[string][]*peerProxyMember{},
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
		in      string
		peerID  string
		model   string
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

func TestParseLoadedModelsModality(t *testing.T) {
	body := []byte(`{"data":[
		{"id":"gemma-4-12b","aliases":["omni"],"status":{"value":"loaded"},
		 "architecture":{"input_modalities":["text","image","audio"],"output_modalities":["text"]}},
		{"id":"sdxl","status":{"value":"stopped"},
		 "architecture":{"input_modalities":["text"],"output_modalities":["image"]}},
		{"id":"plain-text-no-arch","status":{"value":"loaded"}}
	]}`)
	set := parseLoadedModels(body, time.Now())

	// Modality is captured for the loaded multimodal model, keyed by id AND alias.
	mod, ok := set.modalities["gemma-4-12b"]
	if !ok || len(mod.Input) != 3 || mod.Input[1] != "image" || mod.Output[0] != "text" {
		t.Fatalf("gemma modality not captured correctly: %+v ok=%v", mod, ok)
	}
	if a, ok := set.modalities["omni"]; !ok || len(a.Input) != 3 {
		t.Error("expected alias 'omni' to carry the same modality")
	}
	// Modality is independent of loaded status — a cold model still has caps.
	if sd, ok := set.modalities["sdxl"]; !ok || sd.Output[0] != "image" {
		t.Errorf("cold image-out model should still carry modality: %+v ok=%v", sd, ok)
	}
	// A model with no architecture block contributes no modality entry.
	if _, ok := set.modalities["plain-text-no-arch"]; ok {
		t.Error("model without architecture should have no modality entry")
	}
}

func eqStrSlice(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestParseLoadedModelsDims(t *testing.T) {
	body := []byte(`{"data":[
		{"id":"gemma-4-12b","aliases":["omni"],"status":{"value":"loaded"},
		 "meta":{"n_ctx":32768,"n_ctx_train":131072,"size":7441934504,"n_params":7518069290,"n_embd":2560,
		         "n_layer":48,"n_head_kv":8,"n_embd_k_gqa":640,"n_embd_v_gqa":640,"cache_type_k":"q8_0","cache_type_v":"q8_0"}},
		{"id":"cold-no-meta","status":{"value":"unloaded"}},
		{"id":"cold-empty-meta","status":{"value":"unloaded"},"meta":{"n_ctx":0,"n_ctx_train":0}}
	]}`)
	set := parseLoadedModels(body, time.Now())

	d, ok := set.dims["gemma-4-12b"]
	if !ok || d.NCtx != 32768 || d.NCtxTrain != 131072 || d.WeightBytes != 7441934504 || d.NEmbd != 2560 {
		t.Fatalf("dims not captured correctly: %+v ok=%v", d, ok)
	}
	// KV geometry + cache quant for exact KV bytes/token.
	if d.NLayer != 48 || d.NHeadKV != 8 || d.NEmbdKGqa != 640 || d.NEmbdVGqa != 640 || d.CacheTypeK != "q8_0" || d.CacheTypeV != "q8_0" {
		t.Errorf("KV dims not captured correctly: %+v", d)
	}
	if a, ok := set.dims["omni"]; !ok || a.NCtxTrain != 131072 {
		t.Error("expected alias 'omni' to carry the same dims")
	}
	// Cold models with no meta (the common case — router fills meta on load).
	if _, ok := set.dims["cold-no-meta"]; ok {
		t.Error("model with no meta should have no dims entry")
	}
	// An all-zero meta block contributes nothing (treated as not-reported).
	if _, ok := set.dims["cold-empty-meta"]; ok {
		t.Error("model with empty (all-zero) meta should have no dims entry")
	}
}

func TestParseCapability(t *testing.T) {
	cases := []struct {
		in        string
		isCap     bool
		reqIn     []string
		reqOut    []string
		expectErr bool
	}{
		{"any[text,image]to[text]", true, []string{"text", "image"}, []string{"text"}, false},
		{"[text]to[image]", true, []string{"text"}, []string{"image"}, false},
		{"ANY[Text]TO[Audio]", true, []string{"text"}, []string{"audio"}, false},
		{"any[]to[text]", true, []string{}, []string{"text"}, false}, // unconstrained input axis
		{"gemma-4-31b", false, nil, nil, false},                      // plain id
		{"any", false, nil, nil, false},                              // bare wildcard
		{"anything", false, nil, nil, false},                         // not a capability
		{"any[text]image]", false, nil, nil, true},                   // malformed: no 'to['
		{"[text]to", false, nil, nil, true},                          // missing output bracket
	}
	for _, c := range cases {
		in, out, isCap, err := parseCapability(c.in)
		if c.expectErr {
			if err == nil {
				t.Errorf("parseCapability(%q): expected error, got none", c.in)
			}
			continue
		}
		if err != nil {
			t.Errorf("parseCapability(%q): unexpected error: %v", c.in, err)
			continue
		}
		if isCap != c.isCap {
			t.Errorf("parseCapability(%q): isCap=%v want %v", c.in, isCap, c.isCap)
		}
		if isCap && (!eqStrSlice(in, c.reqIn) || !eqStrSlice(out, c.reqOut)) {
			t.Errorf("parseCapability(%q) = in=%v out=%v; want in=%v out=%v", c.in, in, out, c.reqIn, c.reqOut)
		}
	}
}

func TestResolveCapabilityAddress(t *testing.T) {
	p := newTestPeerProxy()
	now := time.Now()
	// titan: vision LLM; crystal: omni-input LLM; lithium: text-only.
	p.loadedCache["titan"] = peerLoadedSet{
		order: []string{"gemma-4-31b"}, all: map[string]bool{"gemma-4-31b": true},
		modalities: map[string]modelModality{"gemma-4-31b": {Input: []string{"text", "image"}, Output: []string{"text"}}},
		fetchedAt:  now,
	}
	p.loadedCache["crystal"] = peerLoadedSet{
		order: []string{"gemma-4-12b"}, all: map[string]bool{"gemma-4-12b": true, "crystal": true},
		modalities: map[string]modelModality{"gemma-4-12b": {Input: []string{"text", "image", "audio"}, Output: []string{"text"}}},
		fetchedAt:  now,
	}
	p.loadedCache["lithium"] = peerLoadedSet{
		order: []string{"nanbeige-3b"}, all: map[string]bool{"nanbeige-3b": true},
		modalities: map[string]modelModality{"nanbeige-3b": {Input: []string{"text"}, Output: []string{"text"}}},
		fetchedAt:  now,
	}

	// any node, vision: first peer in order (crystal) that qualifies.
	if res, err := p.ResolveAddress("any[text,image]to[text]"); err != nil || res.PeerID != "crystal" || res.Model != "gemma-4-12b" {
		t.Errorf("any[text,image]to[text] -> %+v err=%v; want crystal/gemma-4-12b", res, err)
	}
	// pinned to a node.
	if res, err := p.ResolveAddress("any[text,image]to[text]@titan"); err != nil || res.PeerID != "titan" || res.Model != "gemma-4-31b" {
		t.Errorf("@titan -> %+v err=%v; want titan/gemma-4-31b", res, err)
	}
	// audio input: only crystal (omni) qualifies.
	if res, err := p.ResolveAddress("[text,audio]to[text]"); err != nil || res.PeerID != "crystal" {
		t.Errorf("[text,audio]to[text] -> %+v err=%v; want crystal", res, err)
	}
	// image OUTPUT: no LLM peer produces image -> error (lives on comfy-openai).
	if _, err := p.ResolveAddress("any[text]to[image]"); err == nil {
		t.Error("any[text]to[image]: expected error (no LLM peer produces image)")
	}
	// pinned node that can't satisfy -> error.
	if _, err := p.ResolveAddress("[text,audio]to[text]@titan"); err == nil {
		t.Error("[text,audio]to[text]@titan: expected error (titan has no audio-input model)")
	}
}
