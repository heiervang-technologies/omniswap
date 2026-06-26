package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/gin-gonic/gin"
	"github.com/mostlygeek/llama-swap/event"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

type Model struct {
	Id          string `json:"id"`
	Name        string `json:"name"`
	Description string `json:"description"`
	State       string `json:"state"`
	Unlisted    bool   `json:"unlisted"`
	PeerID      string `json:"peerID"`
}

func addApiHandlers(pm *ProxyManager) {
	// Add API endpoints for React to consume
	// Protected with API key authentication
	apiGroup := pm.ginEngine.Group("/api", pm.apiKeyAuth())
	{
		apiGroup.POST("/models/unload", pm.apiUnloadAllModels)
		apiGroup.POST("/models/unload/*model", pm.apiUnloadSingleModelHandler)
		apiGroup.GET("/events", pm.apiSendEvents)
		apiGroup.GET("/metrics", pm.apiGetMetrics)
		apiGroup.GET("/version", pm.apiGetVersion)
	}
}

func (pm *ProxyManager) apiUnloadAllModels(c *gin.Context) {
	pm.StopProcesses(StopImmediately)
	c.JSON(http.StatusOK, gin.H{"msg": "ok"})
}

// routerUnloadHandler implements POST /models/unload with the SAME contract as
// the native llama.cpp router ({"model":"<id>"} -> {"success":true}). Unlike the
// /api unload handlers it PROPAGATES the unload to the peer that owns the model,
// so a model loaded on a remote node (e.g. a gem) can be unloaded from the
// cluster pool — which is what heierchat's unload button needs. The forwarded
// request keeps the /models/unload path, which every peer (native router or a
// nested pool) also serves, and the peer's own response (success, or "model is
// not running") is streamed straight back to the caller.
func (pm *ProxyManager) routerUnloadHandler(c *gin.Context) {
	bodyBytes, err := io.ReadAll(c.Request.Body)
	if err != nil {
		pm.sendErrorResponse(c, http.StatusBadRequest, "could not read request body")
		return
	}
	requestedModel := gjson.GetBytes(bodyBytes, "model").String()

	// No specific model => unload all LOCAL processes. Peers have no unload-all
	// route (their models LRU-evict), so we don't fan out here.
	if requestedModel == "" {
		pm.StopProcesses(StopImmediately)
		c.JSON(http.StatusOK, gin.H{"success": true})
		return
	}

	// Resolve <model>@<node> addressing, mirroring proxyInferenceHandler so the
	// unload path routes identically to the inference path.
	if pm.peerProxy != nil {
		res, rerr := pm.peerProxy.ResolveAddress(requestedModel)
		if rerr != nil {
			pm.sendErrorResponse(c, addrErrorCode(rerr), rerr.Error())
			return
		}
		if res.Rewrote {
			if bodyBytes, err = sjson.SetBytes(bodyBytes, "model", res.Model); err != nil {
				pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error rewriting model name: %s", err.Error()))
				return
			}
		}
		requestedModel = res.Model
		// Pinned peer (e.g. gemma-4-12b-256k@gem) -> forward to that node.
		if res.PeerID != "" {
			pm.rebody(c, bodyBytes)
			pm.forwardUnloadNormalized(c, "peer "+res.PeerID, func(w http.ResponseWriter) error {
				return pm.peerProxy.ProxyRequestToPeer(res.PeerID, res.Model, w, c.Request)
			})
			return
		}
	}

	// Local model -> stop it here.
	if localID, found := pm.config.RealModelName(requestedModel); found {
		processGroup := pm.findGroupByModelName(localID)
		if processGroup == nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("process group not found for model %s", requestedModel))
			return
		}
		if serr := processGroup.StopProcess(localID, StopImmediately); serr != nil {
			pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error stopping process: %s", serr.Error()))
			return
		}
		c.JSON(http.StatusOK, gin.H{"success": true})
		return
	}

	// Model lives on a peer (named, not node-pinned) -> forward to that peer.
	if pm.peerProxy != nil && pm.peerProxy.HasPeerModel(requestedModel) {
		pm.rebody(c, bodyBytes)
		pm.forwardUnloadNormalized(c, "model "+requestedModel, func(w http.ResponseWriter) error {
			return pm.peerProxy.ProxyRequest(requestedModel, w, c.Request)
		})
		return
	}

	pm.sendErrorResponse(c, http.StatusNotFound, "model not found")
}

// unloadNotRunningRe matches a peer router's "already cold" unload response. The
// native llama.cpp router returns 4xx "model is not running" for a CONFIGURED
// model that isn't loaded. We deliberately do NOT match "not found": that means
// an unknown model id — a real error we must surface, not mask as success.
var unloadNotRunningRe = regexp.MustCompile(`(?i)not running`)

// forwardUnloadNormalized dispatches an unload to a peer but BUFFERS the response
// so an already-cold model on a peer router that lacks idempotent-unload (4xx
// "model is not running") is normalized to 200 {success,already_unloaded}. This
// makes pool unload idempotent regardless of the peer router version (mirrors the
// node-router idempotent-unload contract at the pool layer). Unload responses are
// tiny non-streaming JSON, so buffering has no perf cost.
func (pm *ProxyManager) forwardUnloadNormalized(c *gin.Context, who string, dispatch func(http.ResponseWriter) error) {
	buf := newBufferedResponse()
	if err := dispatch(buf); err != nil {
		pm.sendErrorResponse(c, http.StatusBadGateway, fmt.Sprintf("error forwarding unload to %s: %s", who, err.Error()))
		return
	}
	body := buf.body.Bytes()
	if buf.code >= 400 && unloadNotRunningRe.Match(body) {
		pm.proxyLogger.Debugf("unload: normalizing %s %d 'not running' -> 200 already_unloaded", who, buf.code)
		c.JSON(http.StatusOK, gin.H{"success": true, "already_unloaded": true})
		return
	}
	// Pass the peer's response through unchanged.
	for k, vals := range buf.header {
		for _, v := range vals {
			c.Writer.Header().Add(k, v)
		}
	}
	c.Writer.WriteHeader(buf.code)
	_, _ = c.Writer.Write(body)
}

// bufferedResponse is a minimal in-memory http.ResponseWriter used to capture a
// peer's (small, non-streaming) unload response before deciding to normalize it.
type bufferedResponse struct {
	header http.Header
	body   bytes.Buffer
	code   int
}

func newBufferedResponse() *bufferedResponse {
	return &bufferedResponse{header: make(http.Header), code: http.StatusOK}
}

func (b *bufferedResponse) Header() http.Header         { return b.header }
func (b *bufferedResponse) WriteHeader(code int)        { b.code = code }
func (b *bufferedResponse) Write(p []byte) (int, error) { return b.body.Write(p) }

// rebody resets the request body (and its length headers) so a reverse-proxy
// dispatch re-sends the possibly-rewritten bytes. Mirrors proxyInferenceHandler.
func (pm *ProxyManager) rebody(c *gin.Context, bodyBytes []byte) {
	c.Request.Body = io.NopCloser(bytes.NewBuffer(bodyBytes))
	c.Request.Header.Del("transfer-encoding")
	c.Request.Header.Set("content-length", strconv.Itoa(len(bodyBytes)))
	c.Request.ContentLength = int64(len(bodyBytes))
}

func (pm *ProxyManager) getModelStatus() []Model {
	// Extract keys and sort them
	models := []Model{}

	modelIDs := make([]string, 0, len(pm.config.Models))
	for modelID := range pm.config.Models {
		modelIDs = append(modelIDs, modelID)
	}
	sort.Strings(modelIDs)

	// Iterate over sorted keys
	for _, modelID := range modelIDs {
		// Get process state
		processGroup := pm.findGroupByModelName(modelID)
		state := "unknown"
		if processGroup != nil {
			process := processGroup.processes[modelID]
			if process != nil {
				var stateStr string
				switch process.CurrentState() {
				case StateReady:
					stateStr = "ready"
				case StateStarting:
					stateStr = "starting"
				case StateStopping:
					stateStr = "stopping"
				case StateShutdown:
					stateStr = "shutdown"
				case StateStopped:
					stateStr = "stopped"
				default:
					stateStr = "unknown"
				}
				state = stateStr
			}
		}
		models = append(models, Model{
			Id:          modelID,
			Name:        pm.config.Models[modelID].Name,
			Description: pm.config.Models[modelID].Description,
			State:       state,
			Unlisted:    pm.config.Models[modelID].Unlisted,
		})
	}

	// Iterate over the peer models
	if pm.peerProxy != nil {
		for peerID, peer := range pm.peerProxy.ListPeers() {
			for _, modelID := range peer.Models {
				models = append(models, Model{
					Id:     modelID,
					PeerID: peerID,
				})
			}
		}
	}

	return models
}

type messageType string

const (
	msgTypeModelStatus messageType = "modelStatus"
	msgTypeLogData     messageType = "logData"
	msgTypeMetrics     messageType = "metrics"
)

type messageEnvelope struct {
	Type messageType `json:"type"`
	Data string      `json:"data"`
}

// sends a stream of different message types that happen on the server
func (pm *ProxyManager) apiSendEvents(c *gin.Context) {
	c.Header("Content-Type", "text/event-stream")
	c.Header("Cache-Control", "no-cache")
	c.Header("Connection", "keep-alive")
	c.Header("X-Content-Type-Options", "nosniff")
	// prevent nginx from buffering SSE
	c.Header("X-Accel-Buffering", "no")

	sendBuffer := make(chan messageEnvelope, 25)
	ctx, cancel := context.WithCancel(c.Request.Context())
	sendModels := func() {
		data, err := json.Marshal(pm.getModelStatus())
		if err == nil {
			msg := messageEnvelope{Type: msgTypeModelStatus, Data: string(data)}
			select {
			case sendBuffer <- msg:
			case <-ctx.Done():
				return
			default:
			}

		}
	}

	sendLogData := func(source string, data []byte) {
		data, err := json.Marshal(gin.H{
			"source": source,
			"data":   string(data),
		})
		if err == nil {
			select {
			case sendBuffer <- messageEnvelope{Type: msgTypeLogData, Data: string(data)}:
			case <-ctx.Done():
				return
			default:
			}
		}
	}

	sendMetrics := func(metrics []TokenMetrics) {
		jsonData, err := json.Marshal(metrics)
		if err == nil {
			select {
			case sendBuffer <- messageEnvelope{Type: msgTypeMetrics, Data: string(jsonData)}:
			case <-ctx.Done():
				return
			default:
			}
		}
	}

	/**
	 * Send updated models list
	 */
	defer event.On(func(e ProcessStateChangeEvent) {
		sendModels()
	})()
	defer event.On(func(e ConfigFileChangedEvent) {
		sendModels()
	})()

	/**
	 * Send Log data
	 */
	defer pm.proxyLogger.OnLogData(func(data []byte) {
		sendLogData("proxy", data)
	})()
	defer pm.upstreamLogger.OnLogData(func(data []byte) {
		sendLogData("upstream", data)
	})()

	/**
	 * Send Metrics data
	 */
	defer event.On(func(e TokenMetricsEvent) {
		sendMetrics([]TokenMetrics{e.Metrics})
	})()

	// send initial batch of data
	sendLogData("proxy", pm.proxyLogger.GetHistory())
	sendLogData("upstream", pm.upstreamLogger.GetHistory())
	sendModels()
	sendMetrics(pm.metricsMonitor.getMetrics())

	for {
		select {
		case <-c.Request.Context().Done():
			cancel()
			return
		case <-pm.shutdownCtx.Done():
			cancel()
			return
		case msg := <-sendBuffer:
			c.SSEvent("message", msg)
			c.Writer.Flush()
		}
	}
}

func (pm *ProxyManager) apiGetMetrics(c *gin.Context) {
	jsonData, err := pm.metricsMonitor.getMetricsJSON()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "failed to get metrics"})
		return
	}
	c.Data(http.StatusOK, "application/json", jsonData)
}

func (pm *ProxyManager) apiUnloadSingleModelHandler(c *gin.Context) {
	requestedModel := strings.TrimPrefix(c.Param("model"), "/")
	realModelName, found := pm.config.RealModelName(requestedModel)
	if !found {
		pm.sendErrorResponse(c, http.StatusNotFound, "Model not found")
		return
	}

	processGroup := pm.findGroupByModelName(realModelName)
	if processGroup == nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("process group not found for model %s", requestedModel))
		return
	}

	if err := processGroup.StopProcess(realModelName, StopImmediately); err != nil {
		pm.sendErrorResponse(c, http.StatusInternalServerError, fmt.Sprintf("error stopping process: %s", err.Error()))
		return
	} else {
		c.String(http.StatusOK, "OK")
	}
}

func (pm *ProxyManager) apiGetVersion(c *gin.Context) {
	c.JSON(http.StatusOK, map[string]string{
		"version":    pm.version,
		"commit":     pm.commit,
		"build_date": pm.buildDate,
	})
}
