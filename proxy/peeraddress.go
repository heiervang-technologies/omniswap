package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/mostlygeek/llama-swap/proxy/config"
)

// peeraddress.go implements the `<model>@<node>` addressing grammar for the
// peer-mode proxy (a.k.a. the cluster "monolith"). It lets any OpenAI client
// pin a request to a specific cluster node, and resolves the `any` wildcard
// against each peer's live loaded-state.
//
// Grammar (`@` separates the model axis from the node axis; either side may be
// empty or the literal "any", meaning "wildcard"):
//
//	gemma-4-31b            -> route by model name (existing behaviour)
//	gemma-4-31b@any        -> same; the @any is stripped
//	gemma-4-12b@crystal    -> pin to node "crystal", use model "gemma-4-12b"
//	any@crystal / @crystal -> pin to node "crystal", use whatever it has loaded
//	any / @ / any@any / "" -> first node (deterministic order) with a loaded model
//
// Hard rule: node names are only ever valid to the right of '@'. A bare token
// (no '@') is always a model and is never interpreted as a node.

const anyAlias = "any"

// addrError carries an HTTP status code alongside a resolution failure so the
// handler can return 400 for a malformed/unknown address and 503 when a
// wildcard cannot be satisfied because nothing is loaded.
type addrError struct {
	code int
	msg  string
}

func (e *addrError) Error() string { return e.msg }

// addrErrorCode extracts the HTTP status to use for a resolution error,
// defaulting to 400 Bad Request.
func addrErrorCode(err error) int {
	if ae, ok := err.(*addrError); ok {
		return ae.code
	}
	return http.StatusBadRequest
}

// AddressResolution is the outcome of parsing/resolving a `<model>@<node>`
// address against the live peer set.
type AddressResolution struct {
	// PeerID, when non-empty, pins dispatch to that peer, bypassing the
	// model->peer map. Empty means "route by model name with the existing
	// local-then-peer logic".
	PeerID string
	// Model is the concrete model id to dispatch with and to write into the
	// request body/form's "model" field.
	Model string
	// Rewrote is true when Model differs from the raw input, i.e. the request's
	// "model" field must be updated before the request is forwarded.
	Rewrote bool
}

// peerLoadedSet is a short-lived snapshot of one peer's loaded models.
type peerLoadedSet struct {
	order     []string        // loaded model ids, in the order the peer listed them
	all       map[string]bool // loaded ids + their aliases, for membership checks
	fetchedAt time.Time
}

// isWildcard reports whether a model/node component means "any" — either empty
// or the literal "any" (case-insensitive).
func isWildcard(s string) bool {
	return s == "" || strings.EqualFold(s, anyAlias)
}

// splitAddress parses `<model>@<node>` into its components. At most one '@' is
// allowed; either side may be empty (treated as the "any" wildcard).
func splitAddress(raw string) (model, node string, hasAt bool, err error) {
	raw = strings.TrimSpace(raw)
	switch strings.Count(raw, "@") {
	case 0:
		return raw, "", false, nil
	case 1:
		i := strings.IndexByte(raw, '@')
		return strings.TrimSpace(raw[:i]), strings.TrimSpace(raw[i+1:]), true, nil
	default:
		return "", "", false, &addrError{
			code: http.StatusBadRequest,
			msg:  fmt.Sprintf("invalid model address %q: at most one '@' is allowed", raw),
		}
	}
}

// ResolveAddress parses the `<model>@<node>` grammar and resolves any wildcard
// against the live loaded-state of the peers.
func (p *PeerProxy) ResolveAddress(raw string) (AddressResolution, error) {
	model, node, hasAt, err := splitAddress(raw)
	if err != nil {
		return AddressResolution{}, err
	}

	modelWild := isWildcard(model)
	nodeWild := isWildcard(node)

	switch {
	case nodeWild && !modelWild:
		// Concrete model, any node: existing behaviour. Strip a trailing @any
		// so the model field forwarded upstream is clean.
		return AddressResolution{PeerID: "", Model: model, Rewrote: hasAt}, nil

	case nodeWild && modelWild:
		// Global any: first peer (deterministic order) with a model loaded.
		// Deliberately does not trigger a cold load — "any" means "whatever is
		// already up".
		peerID, picked, ok := p.firstLoadedAnywhere()
		if !ok {
			return AddressResolution{}, &addrError{
				code: http.StatusServiceUnavailable,
				msg:  "no peer currently has a model loaded to satisfy 'any'",
			}
		}
		return AddressResolution{PeerID: peerID, Model: picked, Rewrote: true}, nil

	default:
		// Node is concrete.
		peer, ok := p.peers[node]
		if !ok {
			return AddressResolution{}, &addrError{
				code: http.StatusBadRequest,
				msg:  fmt.Sprintf("unknown node %q (known nodes: %s)", node, strings.Join(p.peerOrder, ", ")),
			}
		}
		if modelWild {
			// any@node: a loaded model on that node if there is one (so we don't
			// force a cold swap), else the node's first declared model.
			picked := p.pickModelForPeer(node, peer)
			if picked == "" {
				return AddressResolution{}, &addrError{
					code: http.StatusServiceUnavailable,
					msg:  fmt.Sprintf("node %q has no model available", node),
				}
			}
			return AddressResolution{PeerID: node, Model: picked, Rewrote: true}, nil
		}
		// model@node: confirm the node can serve the model (declared in config
		// or currently loaded) so we fail fast with a clear message.
		if !p.peerServesModel(node, peer, model) {
			return AddressResolution{}, &addrError{
				code: http.StatusBadRequest,
				msg:  fmt.Sprintf("node %q does not serve model %q", node, model),
			}
		}
		return AddressResolution{PeerID: node, Model: model, Rewrote: hasAt}, nil
	}
}

// pickModelForPeer returns a concrete model id for an `any@node` request: a
// currently-loaded model if there is one, otherwise the peer's first declared
// model.
func (p *PeerProxy) pickModelForPeer(peerID string, peer config.PeerConfig) string {
	if loaded := p.peerLoaded(peerID, peer); len(loaded.order) > 0 {
		return loaded.order[0]
	}
	if len(peer.Models) > 0 {
		return peer.Models[0]
	}
	return ""
}

// peerServesModel reports whether the named peer can serve modelID — either it
// is declared in the static config, or the peer currently has it loaded (covers
// ids/aliases a peer exposes that aren't enumerated in config).
func (p *PeerProxy) peerServesModel(peerID string, peer config.PeerConfig, modelID string) bool {
	for _, m := range peer.Models {
		if m == modelID {
			return true
		}
	}
	return p.peerLoaded(peerID, peer).all[modelID]
}

// firstLoadedAnywhere walks peers in deterministic order and returns the first
// peer that has any model loaded, with that model id.
func (p *PeerProxy) firstLoadedAnywhere() (peerID, model string, ok bool) {
	for _, id := range p.peerOrder {
		peer := p.peers[id]
		if loaded := p.peerLoaded(id, peer); len(loaded.order) > 0 {
			return id, loaded.order[0], true
		}
	}
	return "", "", false
}

// LoadedModels returns the currently-loaded model ids for a peer, used by the
// /v1/models listing to surface `<model>@<node>` ids for discovery.
func (p *PeerProxy) LoadedModels(peerID string) []string {
	peer, found := p.peers[peerID]
	if !found {
		return nil
	}
	return p.peerLoaded(peerID, peer).order
}

// PeerOrder returns peer ids in deterministic (sorted) order.
func (p *PeerProxy) PeerOrder() []string {
	return p.peerOrder
}

// IsPeerModelLoaded reports whether the named peer currently has modelID loaded
// (matched against the peer's reported ids and aliases). Used by the /v1/models
// listing to stamp loaded-status on peer entries so aggregating clients (e.g.
// heierchat) can group "loaded" models without per-node configuration. Reads
// the same short-TTL loaded-state cache as the `@node` resolver.
func (p *PeerProxy) IsPeerModelLoaded(peerID, modelID string) bool {
	peer, found := p.peers[peerID]
	if !found {
		return false
	}
	return p.peerLoaded(peerID, peer).all[modelID]
}

// peerLoaded returns a short-TTL snapshot of the peer's loaded models, querying
// the peer's /v1/models when the cache is stale. On query failure it keeps the
// previous snapshot for one more cycle, else returns an empty (but timestamped)
// snapshot so a down peer isn't hammered every request.
func (p *PeerProxy) peerLoaded(peerID string, peer config.PeerConfig) peerLoadedSet {
	p.loadedMu.RLock()
	cached, found := p.loadedCache[peerID]
	p.loadedMu.RUnlock()
	if found && time.Since(cached.fetchedAt) < p.loadedTTL {
		return cached
	}

	snap, err := p.fetchPeerLoaded(peer)
	if err != nil {
		if p.logger != nil {
			p.logger.Debugf("peer %s: loaded-state refresh failed: %v", peerID, err)
		}
		if found {
			return cached // keep the stale snapshot for one more cycle
		}
		empty := peerLoadedSet{all: map[string]bool{}, fetchedAt: time.Now()}
		p.loadedMu.Lock()
		p.loadedCache[peerID] = empty
		p.loadedMu.Unlock()
		return empty
	}

	p.loadedMu.Lock()
	p.loadedCache[peerID] = snap
	p.loadedMu.Unlock()
	return snap
}

// fetchPeerLoaded queries one peer's /v1/models and extracts the loaded set.
func (p *PeerProxy) fetchPeerLoaded(peer config.PeerConfig) (peerLoadedSet, error) {
	if peer.ProxyURL == nil {
		return peerLoadedSet{}, fmt.Errorf("peer has no proxy URL")
	}
	url := strings.TrimRight(peer.ProxyURL.String(), "/") + "/v1/models"
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return peerLoadedSet{}, err
	}
	if peer.ApiKey != "" {
		req.Header.Set("Authorization", "Bearer "+peer.ApiKey)
	}
	resp, err := p.httpClient.Do(req)
	if err != nil {
		return peerLoadedSet{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return peerLoadedSet{}, fmt.Errorf("peer /v1/models returned %d", resp.StatusCode)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, 4<<20))
	if err != nil {
		return peerLoadedSet{}, err
	}
	return parseLoadedModels(body, time.Now()), nil
}

// modelsListPayload is the minimal shape of an OpenAI /v1/models response, plus
// the llama-swap status + aliases enrichment.
type modelsListPayload struct {
	Data []struct {
		ID      string   `json:"id"`
		Aliases []string `json:"aliases"`
		Status  *struct {
			Value string `json:"value"`
		} `json:"status"`
	} `json:"data"`
}

// parseLoadedModels extracts the loaded model ids (and aliases) from a
// /v1/models response. An entry counts as loaded when it has no status field
// (a plain llama-server) or its status.value == "loaded".
func parseLoadedModels(body []byte, at time.Time) peerLoadedSet {
	set := peerLoadedSet{all: map[string]bool{}, fetchedAt: at}
	var payload modelsListPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		return set
	}
	for _, m := range payload.Data {
		if m.ID == "" {
			continue
		}
		if m.Status != nil && !strings.EqualFold(m.Status.Value, "loaded") {
			continue
		}
		set.order = append(set.order, m.ID)
		set.all[m.ID] = true
		for _, a := range m.Aliases {
			if a != "" {
				set.all[a] = true
			}
		}
	}
	return set
}
