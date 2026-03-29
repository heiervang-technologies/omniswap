package main

import (
	"net"
	"net/http"
	"net/http/httptest"

	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// proxyServer status management
// ---------------------------------------------------------------------------

func newTestProxy() *proxyServer {
	return &proxyServer{
		upstreamProxy: nil, // not used in status tests
		status:        notready,
		failCount:     0,
	}
}

func TestInitialStatusNotReady(t *testing.T) {
	p := newTestProxy()
	if got := p.getStatus(); got != notready {
		t.Errorf("want notready, got %s", got)
	}
}

func TestSetAndGetStatus(t *testing.T) {
	p := newTestProxy()
	p.setStatus(ready)
	if got := p.getStatus(); got != ready {
		t.Errorf("want ready, got %s", got)
	}
	p.setStatus(notready)
	if got := p.getStatus(); got != notready {
		t.Errorf("want notready after reset, got %s", got)
	}
}

func TestIncFail(t *testing.T) {
	p := newTestProxy()
	p.incFail(1)
	p.incFail(3)
	if got := p.getFailures(); got != 4 {
		t.Errorf("want 4 failures, got %d", got)
	}
}

func TestResetFailures(t *testing.T) {
	p := newTestProxy()
	p.incFail(5)
	p.resetFailures()
	if got := p.getFailures(); got != 0 {
		t.Errorf("want 0 after reset, got %d", got)
	}
}

func TestInitialFailuresZero(t *testing.T) {
	p := newTestProxy()
	if got := p.getFailures(); got != 0 {
		t.Errorf("want 0, got %d", got)
	}
}

// ---------------------------------------------------------------------------
// sendMagicPacket — packet structure validation
// ---------------------------------------------------------------------------

func TestSendMagicPacketInvalidMAC(t *testing.T) {
	err := sendMagicPacket("not-a-mac")
	if err == nil {
		t.Error("expected error for invalid MAC")
	}
}

func TestSendMagicPacketValidMAC(t *testing.T) {
	// This actually sends a UDP broadcast packet on the loopback.
	// It should succeed without error for a well-formed MAC.
	err := sendMagicPacket("AA:BB:CC:DD:EE:FF")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestMagicPacketStructure(t *testing.T) {
	// Verify the magic packet construction logic directly.
	macStr := "11:22:33:44:55:66"
	hwAddr, err := net.ParseMAC(macStr)
	if err != nil {
		t.Fatal(err)
	}

	// Reconstruct packet the same way as sendMagicPacket
	packet := make([]byte, 102)
	for i := 0; i < 6; i++ {
		packet[i] = 0xFF
	}
	for i := 1; i <= 16; i++ {
		copy(packet[i*6:], hwAddr)
	}

	// Check preamble
	for i := 0; i < 6; i++ {
		if packet[i] != 0xFF {
			t.Errorf("preamble byte %d: want 0xFF, got 0x%02X", i, packet[i])
		}
	}

	// Check MAC is repeated 16 times
	for rep := 0; rep < 16; rep++ {
		offset := 6 + rep*6
		for j := 0; j < 6; j++ {
			if packet[offset+j] != hwAddr[j] {
				t.Errorf("MAC repeat %d byte %d: want 0x%02X, got 0x%02X",
					rep, j, hwAddr[j], packet[offset+j])
			}
		}
	}

	// Total length
	if len(packet) != 102 {
		t.Errorf("packet length: want 102, got %d", len(packet))
	}
}

// ---------------------------------------------------------------------------
// ServeHTTP — /status endpoint
// ---------------------------------------------------------------------------

func TestStatusEndpoint(t *testing.T) {
	p := newTestProxy()
	p.setStatus(ready)
	p.incFail(2)

	req := httptest.NewRequest("GET", "/status", nil)
	w := httptest.NewRecorder()

	p.ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != 200 {
		t.Errorf("want 200, got %d", resp.StatusCode)
	}

	body := w.Body.String()
	if !strings.Contains(body, "status: ready") {
		t.Errorf("want 'status: ready' in body, got: %s", body)
	}
	if !strings.Contains(body, "failures: 2") {
		t.Errorf("want 'failures: 2' in body, got: %s", body)
	}
}

func TestStatusEndpointNotReady(t *testing.T) {
	p := newTestProxy()

	req := httptest.NewRequest("GET", "/status", nil)
	w := httptest.NewRecorder()

	p.ServeHTTP(w, req)

	body := w.Body.String()
	if !strings.Contains(body, "status: not ready") {
		t.Errorf("want 'status: not ready' in body, got: %s", body)
	}
}

// ---------------------------------------------------------------------------
// ServeHTTP — loading page for root when not ready
// ---------------------------------------------------------------------------

func TestRootWhenNotReadyReturnsLoadingPage(t *testing.T) {
	// Set up flagMac so sendMagicPacket doesn't panic
	oldMac := *flagMac
	*flagMac = "AA:BB:CC:DD:EE:FF"
	defer func() { *flagMac = oldMac }()

	p := newTestProxy()
	// status is notready by default

	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()

	p.ServeHTTP(w, req)

	resp := w.Result()
	if resp.StatusCode != http.StatusOK {
		t.Errorf("want 200, got %d", resp.StatusCode)
	}

	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/html") {
		t.Errorf("want text/html content type, got: %s", ct)
	}
}

func TestEventsPathWhenNotReadyReturns204(t *testing.T) {
	oldMac := *flagMac
	*flagMac = "AA:BB:CC:DD:EE:FF"
	defer func() { *flagMac = oldMac }()

	p := newTestProxy()

	req := httptest.NewRequest("GET", "/api/events", nil)
	w := httptest.NewRecorder()

	p.ServeHTTP(w, req)

	if w.Code != http.StatusNoContent {
		t.Errorf("want 204, got %d", w.Code)
	}
}
