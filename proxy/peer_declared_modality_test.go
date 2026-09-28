package proxy

import (
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"gopkg.in/yaml.v3"
)

// A peer served by plain vLLM (the gems' speech servers) lists its model with
// no `architecture` block, so without a declaration it has no modality at all:
// /v1/models shows none and capability addresses cannot reach it.
func declaredModalityProxy(t *testing.T, vllmBody, llamaBody string) *PeerProxy {
	t.Helper()
	serve := func(body string) *httptest.Server {
		s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(body))
		}))
		t.Cleanup(s.Close)
		return s
	}
	asr, llm := serve(vllmBody), serve(llamaBody)
	peers := config.PeerDictionaryConfig{
		"opal-asr": {Proxy: asr.URL, ProxyURL: mustURL(asr.URL), Models: []string{"qwen3-asr"},
			Modalities: map[string]config.PeerModality{"qwen3-asr": {Input: []string{"audio"}, Output: []string{"text"}}}},
		// declares text-only, but advertises omni: the advertisement must win
		"ruby": {Proxy: llm.URL, ProxyURL: mustURL(llm.URL), Models: []string{"gemma-4-12b"},
			Modalities: map[string]config.PeerModality{"gemma-4-12b": {Input: []string{"text"}, Output: []string{"text"}}}},
		// unreachable, never fetched successfully: its declaration still holds
		"opal-tts": {Proxy: "http://127.0.0.1:1", ProxyURL: mustURL("http://127.0.0.1:1"), Models: []string{"qwen3-tts"},
			Modalities: map[string]config.PeerModality{"qwen3-tts": {Input: []string{"text"}, Output: []string{"audio"}}}},
	}
	return &PeerProxy{
		peers:        peers,
		modelPeers:   map[string][]*peerProxyMember{},
		memberByPeer: map[string]*peerProxyMember{},
		peerOrder:    []string{"opal-asr", "opal-tts", "ruby"},
		loadedCache:  map[string]peerLoadedSet{},
		loadedTTL:    time.Hour,
		httpClient:   &http.Client{Timeout: 2 * time.Second},
	}
}

const vllmModels = `{"object":"list","data":[{"id":"qwen3-asr","object":"model","owned_by":"vllm"}]}`
const llamaModels = `{"data":[{"id":"gemma-4-12b","status":{"value":"loaded"},
  "architecture":{"input_modalities":["text","image","audio"],"output_modalities":["text"]}}]}`

func TestDeclaredModalityFillsVLLMGap(t *testing.T) {
	p := declaredModalityProxy(t, vllmModels, llamaModels)
	in, out, ok := p.PeerModelModality("opal-asr", "qwen3-asr")
	if !ok || len(in) != 1 || in[0] != "audio" || len(out) != 1 || out[0] != "text" {
		t.Fatalf("opal-asr/qwen3-asr modality = %v %v %v; want [audio] [text] true", in, out, ok)
	}
}

func TestAdvertisedModalityBeatsDeclaration(t *testing.T) {
	p := declaredModalityProxy(t, vllmModels, llamaModels)
	in, _, ok := p.PeerModelModality("ruby", "gemma-4-12b")
	if !ok || len(in) != 3 {
		t.Fatalf("ruby/gemma-4-12b input = %v ok=%v; want the advertised [text image audio]", in, ok)
	}
}

func TestDeclaredModalitySurvivesAnUnreachablePeer(t *testing.T) {
	p := declaredModalityProxy(t, vllmModels, llamaModels)
	_, out, ok := p.PeerModelModality("opal-tts", "qwen3-tts")
	if !ok || len(out) != 1 || out[0] != "audio" {
		t.Fatalf("opal-tts/qwen3-tts output = %v ok=%v; want [audio] true", out, ok)
	}
}

func TestCapabilityAddressReachesDeclaredPeer(t *testing.T) {
	p := declaredModalityProxy(t, vllmModels, llamaModels)
	// audio in, text out: opal-asr comes first in peer order and now qualifies
	res, err := p.ResolveAddress("any[audio]to[text]")
	if err != nil || res.PeerID != "opal-asr" || res.Model != "qwen3-asr" {
		t.Fatalf("any[audio]to[text] -> %+v err=%v; want opal-asr/qwen3-asr", res, err)
	}
}

// Control arm: the same vLLM peer WITHOUT a declaration has no modality, which
// is the state the gems' speech models were in.
func TestUndeclaredVLLMPeerHasNoModality(t *testing.T) {
	p := declaredModalityProxy(t, vllmModels, llamaModels)
	peer := p.peers["opal-asr"]
	peer.Modalities = nil
	p.peers["opal-asr"] = peer
	if in, out, ok := p.PeerModelModality("opal-asr", "qwen3-asr"); ok {
		t.Fatalf("undeclared vLLM peer reported modality %v %v", in, out)
	}
}

func TestPeerModalitiesConfigValidation(t *testing.T) {
	good := "proxy: http://opal:18081\nmodels: [qwen3-asr]\nmodalities:\n  qwen3-asr: {input: [audio], output: [text]}\n"
	var c config.PeerConfig
	if err := yaml.Unmarshal([]byte(good), &c); err != nil {
		t.Fatalf("valid config rejected: %v", err)
	}
	if got := c.Modalities["qwen3-asr"]; len(got.Input) != 1 || got.Input[0] != "audio" {
		t.Fatalf("parsed modality = %+v", got)
	}
	for name, bad := range map[string]string{
		"unknown model": "proxy: http://opal:18081\nmodels: [qwen3-asr]\nmodalities:\n  qwen3-tts: {input: [text], output: [audio]}\n",
		"empty":         "proxy: http://opal:18081\nmodels: [qwen3-asr]\nmodalities:\n  qwen3-asr: {}\n",
	} {
		var c config.PeerConfig
		if err := yaml.Unmarshal([]byte(bad), &c); err == nil {
			t.Errorf("%s: expected a validation error", name)
		}
	}
}
