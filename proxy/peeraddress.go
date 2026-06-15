package proxy

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
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
// The model axis also accepts a modality-capability expression instead of a
// concrete id — `any[<in>]to[<out>]` (the `any` prefix is optional) — which
// resolves to a peer model whose advertised input_modalities ⊇ <in> and
// output_modalities ⊇ <out>. Combine with @node to constrain to one node:
//
//	any[text,image]to[text]        -> any node's first vision-capable LLM
//	[text,image,audio]to[text]@titan -> an omni-input model on node "titan"
//	any[text]to[image]             -> (no LLM peer produces image -> 503; that
//	                                   modality lives on comfy-openai's endpoint)
//
// An empty bracket set is an unconstrained axis. Capability resolution prefers
// an already-loaded match (no cold swap), else cold-loads a matching model.
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

// modelModality is a peer model's advertised input/output modality signature,
// lifted from its /v1/models `architecture` block. Propagated through the
// aggregated union so clients can group/route by capability (vision, omni, etc.).
type modelModality struct {
	Input  []string
	Output []string
}

// peerLoadedSet is a short-lived snapshot of one peer's advertised models.
type peerLoadedSet struct {
	order      []string                 // loaded model ids, in the order the peer listed them
	all        map[string]bool          // loaded ids + their aliases, for membership checks
	served     map[string]bool          // ALL advertised ids + aliases (loaded or cold), for routing
	modalities map[string]modelModality // id/alias -> advertised modality (only when the peer reports one)
	fetchedAt  time.Time
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

	// Modality-capability model expression (`any[in]to[out]`): resolve to a
	// concrete peer model whose advertised caps satisfy the request, optionally
	// pinned to @node. Intercepted before wildcard handling because the bracket
	// form is neither a plain id nor the bare `any` wildcard.
	if reqIn, reqOut, isCap, capErr := parseCapability(model); capErr != nil {
		return AddressResolution{}, capErr
	} else if isCap {
		return p.resolveCapability(node, isWildcard(node), reqIn, reqOut)
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
		// model@node: confirm the node can serve the model (declared in config,
		// advertised in its /v1/models, or currently loaded) so we fail fast with
		// a clear message; the peer cold-loads it on dispatch if it isn't resident.
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

// peerServesModel reports whether the named peer can serve modelID — i.e. the
// peer can be asked to run it (cold-loading on demand if necessary). True when
// the model is declared in static config, or the peer currently advertises it
// in /v1/models (loaded OR cold-but-swappable), or it is a live id/alias. This
// makes `<model>@<node>` a generic "pin node, pick any model that node serves"
// primitive rather than only resolving the pre-advertised alias entries.
func (p *PeerProxy) peerServesModel(peerID string, peer config.PeerConfig, modelID string) bool {
	for _, m := range peer.Models {
		if m == modelID {
			return true
		}
	}
	loaded := p.peerLoaded(peerID, peer)
	return loaded.served[modelID] || loaded.all[modelID]
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

// parseCapability detects and parses a modality-capability model expression of
// the form `any[in1,in2]to[out1,out2]` (the `any` prefix is optional). Returns
// the requested input/output modality sets (empty = unconstrained on that axis)
// and ok=true when raw is such an expression. A plain model id or the bare
// `any` wildcard returns ok=false so the caller falls back to id/wildcard
// resolution. Modality tokens are lowercased (text/image/audio/video).
func parseCapability(raw string) (reqIn, reqOut []string, ok bool, err error) {
	s := strings.ToLower(strings.TrimSpace(raw))
	s = strings.TrimSpace(strings.TrimPrefix(s, anyAlias))
	if !strings.HasPrefix(s, "[") {
		return nil, nil, false, nil // not a capability expression
	}
	bad := func() (s []string, r []string, ok bool, e error) {
		return nil, nil, false, &addrError{
			code: http.StatusBadRequest,
			msg:  fmt.Sprintf("invalid capability address %q: expected any[<in>]to[<out>]", raw),
		}
	}
	c1 := strings.IndexByte(s, ']')
	if c1 < 0 {
		return bad()
	}
	rest := strings.TrimSpace(s[c1+1:])
	if !strings.HasPrefix(rest, "to") {
		return bad()
	}
	rest = strings.TrimSpace(strings.TrimPrefix(rest, "to"))
	if !strings.HasPrefix(rest, "[") || !strings.HasSuffix(rest, "]") {
		return bad()
	}
	return splitModalityCSV(s[1:c1]), splitModalityCSV(rest[1 : len(rest)-1]), true, nil
}

func splitModalityCSV(s string) []string {
	out := []string{}
	for _, t := range strings.Split(s, ",") {
		if t = strings.TrimSpace(t); t != "" {
			out = append(out, t)
		}
	}
	return out
}

// modalitySatisfies reports whether a model's advertised modality covers the
// request: every requested input modality is in mod.Input and every requested
// output modality is in mod.Output. An empty request set is unconstrained.
func modalitySatisfies(mod modelModality, reqIn, reqOut []string) bool {
	return containsAllFold(mod.Input, reqIn) && containsAllFold(mod.Output, reqOut)
}

func containsAllFold(have, want []string) bool {
	for _, w := range want {
		found := false
		for _, h := range have {
			if strings.EqualFold(h, w) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}
	return true
}

// firstModelMatchingCaps returns a model on the peer whose advertised modality
// satisfies the request, preferring an already-loaded model (no cold swap),
// else a cold-but-advertised one (resolved deterministically). Only models the
// peer advertises a modality signature for are eligible.
func (p *PeerProxy) firstModelMatchingCaps(peerID string, peer config.PeerConfig, reqIn, reqOut []string) (string, bool) {
	loaded := p.peerLoaded(peerID, peer)
	for _, id := range loaded.order { // prefer loaded, in advertised order
		if mod, ok := loaded.modalities[id]; ok && modalitySatisfies(mod, reqIn, reqOut) {
			return id, true
		}
	}
	ids := make([]string, 0, len(loaded.modalities))
	for id := range loaded.modalities {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		if mod := loaded.modalities[id]; modalitySatisfies(mod, reqIn, reqOut) {
			return id, true
		}
	}
	return "", false
}

// resolveCapability resolves a modality-capability address to a concrete
// PeerID+Model. With a wildcard node it scans peers in deterministic order;
// with a concrete node it constrains to that node.
func (p *PeerProxy) resolveCapability(node string, nodeWild bool, reqIn, reqOut []string) (AddressResolution, error) {
	desc := fmt.Sprintf("[%s]to[%s]", strings.Join(reqIn, ","), strings.Join(reqOut, ","))
	if nodeWild {
		for _, peerID := range p.peerOrder {
			peer := p.peers[peerID]
			if picked, ok := p.firstModelMatchingCaps(peerID, peer, reqIn, reqOut); ok {
				return AddressResolution{PeerID: peerID, Model: picked, Rewrote: true}, nil
			}
		}
		return AddressResolution{}, &addrError{
			code: http.StatusServiceUnavailable,
			msg:  fmt.Sprintf("no peer has a model satisfying capability %s", desc),
		}
	}
	peer, ok := p.peers[node]
	if !ok {
		return AddressResolution{}, &addrError{
			code: http.StatusBadRequest,
			msg:  fmt.Sprintf("unknown node %q (known nodes: %s)", node, strings.Join(p.peerOrder, ", ")),
		}
	}
	if picked, ok := p.firstModelMatchingCaps(node, peer, reqIn, reqOut); ok {
		return AddressResolution{PeerID: node, Model: picked, Rewrote: true}, nil
	}
	return AddressResolution{}, &addrError{
		code: http.StatusServiceUnavailable,
		msg:  fmt.Sprintf("node %q has no model satisfying capability %s", node, desc),
	}
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

// PeerModelModality returns the input/output modality signature a peer advertises
// for modelID (from its /v1/models `architecture` block), if any. Reads the same
// short-TTL cache as the loaded-status stamping, so it costs no extra peer poll.
// Used by the /v1/models union to propagate capability metadata that the bare
// peer model-id list would otherwise drop.
func (p *PeerProxy) PeerModelModality(peerID, modelID string) (input, output []string, ok bool) {
	peer, found := p.peers[peerID]
	if !found {
		return nil, nil, false
	}
	mod, ok := p.peerLoaded(peerID, peer).modalities[modelID]
	if !ok {
		return nil, nil, false
	}
	return mod.Input, mod.Output, true
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
		empty := peerLoadedSet{all: map[string]bool{}, served: map[string]bool{}, fetchedAt: time.Now()}
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
		Architecture *struct {
			InputModalities  []string `json:"input_modalities"`
			OutputModalities []string `json:"output_modalities"`
		} `json:"architecture"`
	} `json:"data"`
}

// parseLoadedModels extracts a peer's advertised models from a /v1/models
// response. Every entry is recorded in `served` (routable, possibly via a cold
// swap). The subset that is actually resident — an entry with no status field
// (a plain llama-server) or status.value == "loaded" — additionally populates
// `order`/`all` (the loaded set surfaced as hot ids and `<id>@<node>` aliases).
func parseLoadedModels(body []byte, at time.Time) peerLoadedSet {
	set := peerLoadedSet{all: map[string]bool{}, served: map[string]bool{}, modalities: map[string]modelModality{}, fetchedAt: at}
	var payload modelsListPayload
	if err := json.Unmarshal(body, &payload); err != nil {
		return set
	}
	for _, m := range payload.Data {
		if m.ID == "" {
			continue
		}
		// Everything the peer advertises is routable (may cold-load).
		set.served[m.ID] = true
		for _, a := range m.Aliases {
			if a != "" {
				set.served[a] = true
			}
		}
		// Capture the advertised modality signature (independent of loaded
		// status — a cold model still has a fixed capability). Keyed by id and
		// every alias so a union lookup hits regardless of which name is used.
		if m.Architecture != nil && (len(m.Architecture.InputModalities) > 0 || len(m.Architecture.OutputModalities) > 0) {
			mod := modelModality{Input: m.Architecture.InputModalities, Output: m.Architecture.OutputModalities}
			set.modalities[m.ID] = mod
			for _, a := range m.Aliases {
				if a != "" {
					set.modalities[a] = mod
				}
			}
		}
		// Only resident models count as "loaded".
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
