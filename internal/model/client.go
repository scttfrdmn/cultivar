package model

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultEndpoint is the Hugging Face Hub API root.
const DefaultEndpoint = "https://huggingface.co"

// maxBodyBytes caps a metadata response. The model API returns a few KB; a
// megabyte is generous and bounds a hostile or broken response.
const maxBodyBytes = 1 << 20

// ErrNotFound reports that the repo does not exist, or exists but is invisible to
// the supplied token. The Hub returns 401 for a gated repo read without an
// authorized token, which is indistinguishable from "no such repo" — so callers
// must not report "typo?" without noting the gating possibility.
var ErrNotFound = errors.New("model repo not found or not visible with this token")

// Client reads model metadata from the Hugging Face Hub.
type Client struct {
	endpoint string
	token    string
	http     *http.Client
	now      func() time.Time
}

// Option configures a [Client].
type Option func(*Client)

// WithToken supplies an HF token, needed to read gated repos.
func WithToken(token string) Option {
	return func(c *Client) { c.token = strings.TrimSpace(token) }
}

// WithEndpoint overrides the Hub root, for fixtures and tests.
func WithEndpoint(endpoint string) Option {
	return func(c *Client) { c.endpoint = strings.TrimRight(endpoint, "/") }
}

// WithHTTPClient overrides the HTTP client.
func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) { c.http = h }
}

// WithClock overrides the clock, so fixtures produce stable observation times.
func WithClock(now func() time.Time) Option {
	return func(c *Client) { c.now = now }
}

// NewClient returns a Hub client.
func NewClient(opts ...Option) *Client {
	c := &Client{
		endpoint: DefaultEndpoint,
		http:     &http.Client{Timeout: 30 * time.Second},
		now:      time.Now,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// modelInfo mirrors the fields of /api/models/{id} that this package needs.
type modelInfo struct {
	ID string `json:"id"`
	// Gated is polymorphic: false for public repos, "auto" or "manual" for gated
	// ones. json.RawMessage defers the decision to parseGate.
	Gated       json.RawMessage `json:"gated"`
	Safetensors *struct {
		Parameters map[string]int64 `json:"parameters"`
		Total      int64            `json:"total"`
	} `json:"safetensors"`
	Config *struct {
		Architectures      []string `json:"architectures"`
		ModelType          string   `json:"model_type"`
		QuantizationConfig *struct {
			QuantMethod string `json:"quant_method"`
		} `json:"quantization_config"`
	} `json:"config"`
	Siblings []struct {
		Filename string `json:"rfilename"`
	} `json:"siblings"`
	Tags []string `json:"tags"`
}

// configJSON mirrors the sizing fields of a repo's config.json.
//
// These come from the raw file rather than the API's `config` object, which omits
// them: the API returns only architectures, model_type, and tokenizer_config, so
// the KV cache cannot be sized from it at all.
type configJSON struct {
	MaxPositionEmbeddings int    `json:"max_position_embeddings"`
	HiddenSize            int    `json:"hidden_size"`
	NumHiddenLayers       int    `json:"num_hidden_layers"`
	NumAttentionHeads     int    `json:"num_attention_heads"`
	NumKeyValueHeads      int    `json:"num_key_value_heads"`
	HeadDim               int    `json:"head_dim"`
	TorchDType            string `json:"torch_dtype"`
	TextConfig            *struct {
		MaxPositionEmbeddings int `json:"max_position_embeddings"`
		HiddenSize            int `json:"hidden_size"`
		NumHiddenLayers       int `json:"num_hidden_layers"`
		NumAttentionHeads     int `json:"num_attention_heads"`
		NumKeyValueHeads      int `json:"num_key_value_heads"`
		HeadDim               int `json:"head_dim"`
	} `json:"text_config"`
}

// Get resolves a repo id to a [Model].
//
// It makes two calls: /api/models/{id} for gating and parameter counts, and
// raw/main/config.json for the sizing fields the API omits. A missing or
// unreadable config.json is not fatal — the weights can still be sized, and the KV
// cache is then reported as unsizable, which is a more useful answer than failing
// the whole lookup.
func (c *Client) Get(ctx context.Context, id string) (*Model, error) {
	id = strings.Trim(strings.TrimSpace(id), "/")
	if id == "" {
		return nil, fmt.Errorf("empty model id")
	}

	var info modelInfo
	if err := c.getJSON(ctx, "/api/models/"+id, &info); err != nil {
		return nil, err
	}

	gate, err := parseGate(info.Gated)
	if err != nil {
		return nil, fmt.Errorf("model %s: %w", id, err)
	}

	m := &Model{
		ID:         id,
		Gate:       gate,
		ObservedAt: c.now().UTC(),
	}
	if info.Safetensors != nil {
		m.Parameters = info.Safetensors.Parameters
	}
	if info.Config != nil {
		m.Architectures = info.Config.Architectures
		if info.Config.QuantizationConfig != nil {
			m.QuantMethod = info.Config.QuantizationConfig.QuantMethod
		}
	}
	// The API's safetensors block is absent for GGUF-only repos, but so is it for
	// repos whose weights the Hub has not indexed, so check the file list too.
	m.HasSafetensors = info.Safetensors != nil
	for _, s := range info.Siblings {
		if strings.HasSuffix(s.Filename, ".safetensors") {
			m.HasSafetensors = true
			break
		}
	}

	var cfg configJSON
	if err := c.getJSON(ctx, "/"+id+"/raw/main/config.json", &cfg); err == nil {
		m.Config = cfg.sizing()
	} else if !errors.Is(err, ErrNotFound) {
		// A 404 on config.json is ordinary (GGUF-only repos, datasets). Anything
		// else — a throttle, a network failure — would silently produce a model with
		// an unsizable KV cache, so it is worth failing on.
		return nil, fmt.Errorf("model %s: read config.json: %w", id, err)
	}

	return m, nil
}

// sizing flattens config.json into a [Config], preferring the top-level fields and
// falling back to text_config. Multimodal repos (Llama 4, Gemma 3) nest the
// language-model fields under text_config, and reading only the top level leaves
// the KV cache unsizable for them.
func (c configJSON) sizing() Config {
	out := Config{
		MaxPositionEmbeddings: c.MaxPositionEmbeddings,
		HiddenSize:            c.HiddenSize,
		NumHiddenLayers:       c.NumHiddenLayers,
		NumAttentionHeads:     c.NumAttentionHeads,
		NumKeyValueHeads:      c.NumKeyValueHeads,
		HeadDim:               c.HeadDim,
		TorchDType:            c.TorchDType,
	}
	if t := c.TextConfig; t != nil {
		if out.MaxPositionEmbeddings == 0 {
			out.MaxPositionEmbeddings = t.MaxPositionEmbeddings
		}
		if out.HiddenSize == 0 {
			out.HiddenSize = t.HiddenSize
		}
		if out.NumHiddenLayers == 0 {
			out.NumHiddenLayers = t.NumHiddenLayers
		}
		if out.NumAttentionHeads == 0 {
			out.NumAttentionHeads = t.NumAttentionHeads
		}
		if out.NumKeyValueHeads == 0 {
			out.NumKeyValueHeads = t.NumKeyValueHeads
		}
		if out.HeadDim == 0 {
			out.HeadDim = t.HeadDim
		}
	}
	// Multi-query attention repos omit num_key_value_heads, meaning it equals
	// num_attention_heads. Defaulting it to zero would make the cache unsizable for
	// every such model.
	if out.NumKeyValueHeads == 0 {
		out.NumKeyValueHeads = out.NumAttentionHeads
	}
	return out
}

// parseGate decodes the polymorphic `gated` field: false, "auto", or "manual".
//
// An unrecognized value is an error rather than a default. Defaulting to
// GateNone would mean launching a GPU for a repo that then 401s on download —
// paying for an instance that can never serve — and defaulting to GateManual
// would block public models.
func parseGate(raw json.RawMessage) (Gate, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return GateNone, nil
	}
	var b bool
	if err := json.Unmarshal(raw, &b); err == nil {
		if b {
			// `true` without a mode: gated, mode unknown. Treat as manual, the
			// stricter reading, so a token is demanded rather than assumed absent.
			return GateManual, nil
		}
		return GateNone, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", fmt.Errorf("unexpected gated value %s", raw)
	}
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "", "false":
		return GateNone, nil
	case "auto":
		return GateAuto, nil
	case "manual":
		return GateManual, nil
	default:
		return "", fmt.Errorf("unknown gated mode %q", s)
	}
}

func (c *Client) getJSON(ctx context.Context, path string, out any) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.endpoint+path, nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "application/json")
	if c.token != "" {
		req.Header.Set("Authorization", "Bearer "+c.token)
	}

	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("GET %s: %w", path, err)
	}
	defer resp.Body.Close()

	switch {
	case resp.StatusCode == http.StatusNotFound,
		resp.StatusCode == http.StatusUnauthorized,
		resp.StatusCode == http.StatusForbidden:
		// 401/403 on a gated repo and 404 on a nonexistent one are the same signal
		// from the caller's perspective: not readable with these credentials.
		return fmt.Errorf("GET %s: %d: %w", path, resp.StatusCode, ErrNotFound)
	case resp.StatusCode != http.StatusOK:
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return fmt.Errorf("GET %s: %d: %s", path, resp.StatusCode, strings.TrimSpace(string(body)))
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxBodyBytes))
	if err != nil {
		return fmt.Errorf("GET %s: read body: %w", path, err)
	}
	if err := json.Unmarshal(data, out); err != nil {
		return fmt.Errorf("GET %s: decode: %w", path, err)
	}
	return nil
}

// CanRead verifies that the configured token can actually fetch this repo's
// weights, which is the precondition `deploy` must check before spending money.
//
// It probes the resolve endpoint for a weight file rather than the metadata API,
// because the two are separately authorized: metadata for a gated repo is
// readable while the download 401s. Checking only that a token is *set* is what
// lets a $55/hr p5.48xlarge come up and then fail to download.
func (c *Client) CanRead(ctx context.Context, m *Model) error {
	if !m.Gate.RequiresToken() {
		return nil
	}
	if c.token == "" {
		return fmt.Errorf("model %s is gated (%s) and no HF token was supplied", m.ID, m.Gate)
	}
	// HEAD the config.json resolve path: cheap, and gated behind the same grant as
	// the weights.
	req, err := http.NewRequestWithContext(ctx, http.MethodHead,
		c.endpoint+"/"+m.ID+"/resolve/main/config.json", nil)
	if err != nil {
		return err
	}
	req.Header.Set("Authorization", "Bearer "+c.token)
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("verify read access to %s: %w", m.ID, err)
	}
	defer resp.Body.Close()
	switch resp.StatusCode {
	case http.StatusOK, http.StatusFound, http.StatusTemporaryRedirect:
		return nil
	case http.StatusUnauthorized, http.StatusForbidden:
		return fmt.Errorf("model %s is gated (%s) and this token cannot read it; "+
			"accept the license at https://huggingface.co/%s", m.ID, m.Gate, m.ID)
	default:
		return fmt.Errorf("verify read access to %s: unexpected status %d", m.ID, resp.StatusCode)
	}
}
