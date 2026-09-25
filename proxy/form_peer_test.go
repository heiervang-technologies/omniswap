package proxy

import (
	"bytes"
	"crypto/rand"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"

	"github.com/mostlygeek/llama-swap/proxy/config"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// formPeerCapture is a fake peer that records every multipart request it gets.
type formPeerCapture struct {
	mu       sync.Mutex
	hits     int
	auth     string
	fields   map[string][]string
	fileName string
	file     []byte
}

func newFormPeer(t *testing.T) (*formPeerCapture, *httptest.Server) {
	t.Helper()
	cap := &formPeerCapture{}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		cap.mu.Lock()
		defer cap.mu.Unlock()
		cap.hits++
		cap.auth = r.Header.Get("Authorization")
		if err := r.ParseMultipartForm(1 << 20); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		cap.fields = r.MultipartForm.Value
		if fhs := r.MultipartForm.File["file"]; len(fhs) == 1 {
			cap.fileName = fhs[0].Filename
			f, err := fhs[0].Open()
			if err == nil {
				cap.file, _ = io.ReadAll(f)
				f.Close()
			}
		}
		w.Header().Set("Content-Type", "application/json")
		w.Write([]byte(`{"text":"from-peer"}`))
	}))
	t.Cleanup(srv.Close)
	return cap, srv
}

func transcriptionRequest(t *testing.T, model string, file []byte) *http.Request {
	t.Helper()
	var b bytes.Buffer
	w := multipart.NewWriter(&b)
	require.NoError(t, w.WriteField("model", model))
	require.NoError(t, w.WriteField("language", "nb"))
	require.NoError(t, w.WriteField("response_format", "verbose_json"))
	fw, err := w.CreateFormFile("file", "clip.wav")
	require.NoError(t, err)
	_, err = fw.Write(file)
	require.NoError(t, err)
	require.NoError(t, w.Close())
	req := httptest.NewRequest("POST", "/v1/audio/transcriptions", &b)
	req.Header.Set("Content-Type", w.FormDataContentType())
	req.Header.Set("Authorization", "Bearer client-key-must-not-reach-peer")
	return req
}

func randomBytes(t *testing.T, n int) []byte {
	t.Helper()
	b := make([]byte, n)
	_, err := rand.Read(b)
	require.NoError(t, err)
	return b
}

// A model no local process serves but a peer lists must be forwarded to that
// peer by name — before this, only `<model>@<node>` reached a peer and a bare
// model name got a 400 on every node that didn't run it locally.
func TestProxyManager_FormHandler_ForwardsByNameToPeer(t *testing.T) {
	// 1 MiB stays in memory in ParseMultipartForm; 40 MiB spills past the 32 MiB
	// limit to a temp file. Both must arrive byte-identical.
	for _, size := range []int{1 << 20, 40 << 20} {
		cap, srv := newFormPeer(t)
		proxy := New(config.AddDefaultGroupToConfig(config.Config{
			HealthCheckTimeout: 15,
			Models:             map[string]config.ModelConfig{"local-llm": getTestSimpleResponderConfig("local-llm")},
			Peers: map[string]config.PeerConfig{
				"speech": {Proxy: srv.URL, ProxyURL: mustURL(srv.URL), ApiKey: "peer-key", Models: []string{"qwen3-asr"}},
			},
			LogLevel: "error",
		}))
		defer proxy.StopProcesses(StopWaitForInflightRequest)

		file := randomBytes(t, size)
		rec := CreateTestResponseRecorder()
		proxy.ServeHTTP(rec, transcriptionRequest(t, "qwen3-asr", file))

		assert.Equal(t, http.StatusOK, rec.Code, "size=%d body=%s", size, rec.Body.String())
		assert.JSONEq(t, `{"text":"from-peer"}`, rec.Body.String())
		cap.mu.Lock()
		assert.Equal(t, 1, cap.hits)
		assert.Equal(t, "Bearer peer-key", cap.auth, "peer credential, never the client's")
		assert.Equal(t, []string{"qwen3-asr"}, cap.fields["model"])
		assert.Equal(t, []string{"nb"}, cap.fields["language"])
		assert.Equal(t, []string{"verbose_json"}, cap.fields["response_format"])
		assert.Equal(t, "clip.wav", cap.fileName)
		assert.True(t, bytes.Equal(file, cap.file), "file bytes altered in transit (size=%d got=%d)", size, len(cap.file))
		cap.mu.Unlock()
	}
}

// Neither local nor any peer: still the pre-existing 400, and nothing is sent
// anywhere.
func TestProxyManager_FormHandler_UnknownModelRejected(t *testing.T) {
	cap, srv := newFormPeer(t)
	proxy := New(config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models:             map[string]config.ModelConfig{"local-llm": getTestSimpleResponderConfig("local-llm")},
		Peers: map[string]config.PeerConfig{
			"speech": {Proxy: srv.URL, ProxyURL: mustURL(srv.URL), Models: []string{"qwen3-asr"}},
		},
		LogLevel: "error",
	}))
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	rec := CreateTestResponseRecorder()
	proxy.ServeHTTP(rec, transcriptionRequest(t, "no-such-model", []byte("x")))

	assert.Equal(t, http.StatusBadRequest, rec.Code)
	assert.Contains(t, rec.Body.String(), "could not find real modelID for no-such-model")
	cap.mu.Lock()
	assert.Equal(t, 0, cap.hits)
	cap.mu.Unlock()
}

// Local wins when both a local process and a peer list the model — the peer
// step is a fallback, exactly as in the JSON handler.
func TestProxyManager_FormHandler_LocalPreferredOverPeer(t *testing.T) {
	cap, srv := newFormPeer(t)
	proxy := New(config.AddDefaultGroupToConfig(config.Config{
		HealthCheckTimeout: 15,
		Models:             map[string]config.ModelConfig{"qwen3-asr": getTestSimpleResponderConfig("qwen3-asr")},
		Peers: map[string]config.PeerConfig{
			"speech": {Proxy: srv.URL, ProxyURL: mustURL(srv.URL), Models: []string{"qwen3-asr"}},
		},
		LogLevel: "error",
	}))
	defer proxy.StopProcesses(StopWaitForInflightRequest)

	rec := CreateTestResponseRecorder()
	proxy.ServeHTTP(rec, transcriptionRequest(t, "qwen3-asr", []byte("local audio")))

	assert.Equal(t, http.StatusOK, rec.Code, rec.Body.String())
	assert.Contains(t, rec.Body.String(), "The length of the file is 11 bytes") // simple-responder
	cap.mu.Lock()
	assert.Equal(t, 0, cap.hits)
	cap.mu.Unlock()
}
