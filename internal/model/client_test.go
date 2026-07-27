package model

import (
	"context"
	"errors"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// hub is a fake Hugging Face Hub. Handlers are keyed by request path, so a test
// can omit config.json to exercise the GGUF path, or return a 401 for it to
// exercise the gated path.
type hub struct {
	t       *testing.T
	routes  map[string]func(w http.ResponseWriter, r *http.Request)
	authSaw []string
}

func newHub(t *testing.T) *hub {
	return &hub{t: t, routes: map[string]func(http.ResponseWriter, *http.Request){}}
}

func (h *hub) on(path string, fn func(w http.ResponseWriter, r *http.Request)) *hub {
	h.routes[path] = fn
	return h
}

func (h *hub) json(path, body string) *hub {
	return h.on(path, func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	})
}

func (h *hub) status(path string, code int) *hub {
	return h.on(path, func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(code) })
}

func (h *hub) client(opts ...Option) *Client {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		h.authSaw = append(h.authSaw, r.Header.Get("Authorization"))
		if fn, ok := h.routes[r.URL.Path]; ok {
			fn(w, r)
			return
		}
		http.NotFound(w, r)
	}))
	h.t.Cleanup(srv.Close)
	base := []Option{
		WithEndpoint(srv.URL),
		WithClock(func() time.Time { return observed }),
	}
	return NewClient(append(base, opts...)...)
}

// Trimmed to the fields the client reads, in the shape the live API returned on
// 2026-07-27.
const qwenInfoJSON = `{
  "id": "Qwen/Qwen3-32B",
  "gated": false,
  "safetensors": {"parameters": {"BF16": 32762123264}, "total": 32762123264},
  "config": {"architectures": ["Qwen3ForCausalLM"], "model_type": "qwen3"},
  "siblings": [{"rfilename": "config.json"}, {"rfilename": "model-00001-of-00017.safetensors"}]
}`

const qwenConfigJSON = `{
  "architectures": ["Qwen3ForCausalLM"],
  "hidden_size": 5120,
  "head_dim": 128,
  "max_position_embeddings": 40960,
  "num_attention_heads": 64,
  "num_hidden_layers": 64,
  "num_key_value_heads": 8,
  "torch_dtype": "bfloat16"
}`

func TestGetParsesLiveShape(t *testing.T) {
	c := newHub(t).
		json("/api/models/Qwen/Qwen3-32B", qwenInfoJSON).
		json("/Qwen/Qwen3-32B/raw/main/config.json", qwenConfigJSON).
		client()

	m, err := c.Get(context.Background(), "Qwen/Qwen3-32B")
	if err != nil {
		t.Fatal(err)
	}
	if m.Gate != GateNone {
		t.Errorf("gate = %q, want none", m.Gate)
	}
	if got := m.Parameters["BF16"]; got != 32762123264 {
		t.Errorf("BF16 params = %d, want 32762123264", got)
	}
	if !m.HasSafetensors {
		t.Error("HasSafetensors = false for a repo publishing safetensors")
	}
	// The sizing fields must come from the raw file: the API's config object carries
	// only architectures and model_type, so a client reading just that call would
	// report every KV cache as unsizable.
	want := Config{
		MaxPositionEmbeddings: 40960, HiddenSize: 5120, NumHiddenLayers: 64,
		NumAttentionHeads: 64, NumKeyValueHeads: 8, HeadDim: 128, TorchDType: "bfloat16",
	}
	if m.Config != want {
		t.Errorf("config = %+v, want %+v", m.Config, want)
	}
	if m.ObservedAt != observed {
		t.Errorf("ObservedAt = %v, want the injected clock", m.ObservedAt)
	}
	// End to end: the parsed model must produce the same total as the fixture.
	if v := m.Size(SizingRequest{}).Total.MustValue(); math.Abs(v-81.68) > 0.02 {
		t.Errorf("total = %.2f GiB, want 81.68", v)
	}
}

func TestGetTrimsAndValidatesID(t *testing.T) {
	c := newHub(t).
		json("/api/models/Qwen/Qwen3-32B", qwenInfoJSON).
		json("/Qwen/Qwen3-32B/raw/main/config.json", qwenConfigJSON).
		client()
	for _, id := range []string{"  Qwen/Qwen3-32B  ", "/Qwen/Qwen3-32B", "Qwen/Qwen3-32B/"} {
		m, err := c.Get(context.Background(), id)
		if err != nil {
			t.Errorf("Get(%q): %v", id, err)
			continue
		}
		if m.ID != "Qwen/Qwen3-32B" {
			t.Errorf("Get(%q) → ID %q", id, m.ID)
		}
	}
	if _, err := c.Get(context.Background(), "   "); err == nil {
		t.Error("an empty id was accepted")
	}
}

func TestGetMissingRepoIsErrNotFound(t *testing.T) {
	c := newHub(t).client()
	_, err := c.Get(context.Background(), "nobody/nothing")
	if !errors.Is(err, ErrNotFound) {
		t.Fatalf("err = %v, want ErrNotFound", err)
	}
	// 401 must map to the same sentinel: a gated repo read without a token is
	// indistinguishable from a nonexistent one, and telling the user "typo?" when the
	// real answer is "you need to accept a license" sends them the wrong way.
	c2 := newHub(t).status("/api/models/meta-llama/Llama-3.3-70B-Instruct", http.StatusUnauthorized).client()
	if _, err := c2.Get(context.Background(), "meta-llama/Llama-3.3-70B-Instruct"); !errors.Is(err, ErrNotFound) {
		t.Errorf("401 → %v, want ErrNotFound", err)
	}
}

func TestGetToleratesMissingConfigButNotOtherFailures(t *testing.T) {
	// A GGUF-only repo has no config.json. Weights are unsizable from safetensors
	// either way, so failing the lookup would deny the caller the gating and file
	// list it could still act on.
	gguf := newHub(t).
		json("/api/models/someone/Qwen3-32B-GGUF", `{"id":"someone/Qwen3-32B-GGUF","gated":false,
		  "siblings":[{"rfilename":"Qwen3-32B-Q4_K_M.gguf"}]}`).
		client()
	m, err := gguf.Get(context.Background(), "someone/Qwen3-32B-GGUF")
	if err != nil {
		t.Fatalf("a missing config.json failed the lookup: %v", err)
	}
	if m.HasSafetensors {
		t.Error("HasSafetensors = true for a GGUF-only repo")
	}
	if _, ok := m.WeightBytes().Value(); ok {
		t.Error("a GGUF-only repo was sized from safetensors")
	}

	// A throttle is different: it would silently produce a model with no sizing
	// fields, which reads as "this model has no KV cache" instead of "ask again".
	throttled := newHub(t).
		json("/api/models/Qwen/Qwen3-32B", qwenInfoJSON).
		status("/Qwen/Qwen3-32B/raw/main/config.json", http.StatusTooManyRequests).
		client()
	if _, err := throttled.Get(context.Background(), "Qwen/Qwen3-32B"); err == nil {
		t.Error("a 429 on config.json was swallowed")
	}
}

func TestGetSendsTokenOnlyWhenSet(t *testing.T) {
	h := newHub(t).
		json("/api/models/Qwen/Qwen3-32B", qwenInfoJSON).
		json("/Qwen/Qwen3-32B/raw/main/config.json", qwenConfigJSON)
	c := h.client(WithToken("  hf_secret  "))
	if _, err := c.Get(context.Background(), "Qwen/Qwen3-32B"); err != nil {
		t.Fatal(err)
	}
	for _, saw := range h.authSaw {
		if saw != "Bearer hf_secret" {
			t.Errorf("Authorization = %q, want the trimmed bearer token", saw)
		}
	}

	h2 := newHub(t).
		json("/api/models/Qwen/Qwen3-32B", qwenInfoJSON).
		json("/Qwen/Qwen3-32B/raw/main/config.json", qwenConfigJSON)
	if _, err := h2.client().Get(context.Background(), "Qwen/Qwen3-32B"); err != nil {
		t.Fatal(err)
	}
	for _, saw := range h2.authSaw {
		if saw != "" {
			t.Errorf("Authorization = %q with no token configured", saw)
		}
	}
}

func TestGetReadsQuantMethod(t *testing.T) {
	c := newHub(t).
		json("/api/models/openai/gpt-oss-120b", `{"id":"openai/gpt-oss-120b","gated":false,
		  "safetensors":{"parameters":{"BF16":2167371072,"U8":118244966400}},
		  "config":{"architectures":["GptOssForCausalLM"],"model_type":"gpt_oss",
		    "quantization_config":{"quant_method":"mxfp4"}}}`).
		json("/openai/gpt-oss-120b/raw/main/config.json", `{"hidden_size":2880,"head_dim":64,
		  "max_position_embeddings":131072,"num_attention_heads":64,"num_hidden_layers":36,
		  "num_key_value_heads":8}`).
		client()
	m, err := c.Get(context.Background(), "openai/gpt-oss-120b")
	if err != nil {
		t.Fatal(err)
	}
	if m.QuantMethod != "mxfp4" {
		t.Errorf("quant method = %q, want mxfp4", m.QuantMethod)
	}
	if v := m.WeightBytes().MustValue(); math.Abs(v-114.16) > 0.01 {
		t.Errorf("weights = %.2f GiB, want 114.16 (per-dtype, not summed)", v)
	}
}

func TestParseGate(t *testing.T) {
	for _, tc := range []struct {
		raw  string
		want Gate
	}{
		{`false`, GateNone},
		{`"auto"`, GateAuto},
		{`"manual"`, GateManual},
		{`"MANUAL"`, GateManual},
		{`null`, GateNone},
		{``, GateNone},
		// `true` without a mode: gated, mode unspecified. Reading it as manual demands
		// a token, which is recoverable; reading it as public launches a GPU that
		// cannot download the weights, which costs money.
		{`true`, GateManual},
	} {
		got, err := parseGate([]byte(tc.raw))
		if err != nil {
			t.Errorf("parseGate(%s): %v", tc.raw, err)
			continue
		}
		if got != tc.want {
			t.Errorf("parseGate(%s) = %q, want %q", tc.raw, got, tc.want)
		}
	}
	// A mode the Hub adds later must fail loudly. Defaulting either way is a guess
	// about whether money can be spent safely.
	for _, raw := range []string{`"enterprise-only"`, `42`, `{"mode":"manual"}`} {
		if got, err := parseGate([]byte(raw)); err == nil {
			t.Errorf("parseGate(%s) = %q, want an error", raw, got)
		}
	}
}

func TestSizingFallsBackToTextConfig(t *testing.T) {
	// Multimodal repos (Llama 4, Gemma 3) nest the language-model fields under
	// text_config. Reading only the top level leaves the cache unsizable for a whole
	// class of models, which reads as "we can't help you" for models we can size.
	cfg := configJSON{TorchDType: "bfloat16"}
	cfg.TextConfig = &struct {
		MaxPositionEmbeddings int `json:"max_position_embeddings"`
		HiddenSize            int `json:"hidden_size"`
		NumHiddenLayers       int `json:"num_hidden_layers"`
		NumAttentionHeads     int `json:"num_attention_heads"`
		NumKeyValueHeads      int `json:"num_key_value_heads"`
		HeadDim               int `json:"head_dim"`
	}{
		MaxPositionEmbeddings: 8192, HiddenSize: 5120, NumHiddenLayers: 48,
		NumAttentionHeads: 40, NumKeyValueHeads: 8, HeadDim: 128,
	}
	got := cfg.sizing()
	want := Config{
		MaxPositionEmbeddings: 8192, HiddenSize: 5120, NumHiddenLayers: 48,
		NumAttentionHeads: 40, NumKeyValueHeads: 8, HeadDim: 128, TorchDType: "bfloat16",
	}
	if got != want {
		t.Errorf("sizing() = %+v, want %+v", got, want)
	}
}

func TestSizingPrefersTopLevelOverTextConfig(t *testing.T) {
	cfg := configJSON{NumHiddenLayers: 64, NumKeyValueHeads: 8, HeadDim: 128}
	cfg.TextConfig = &struct {
		MaxPositionEmbeddings int `json:"max_position_embeddings"`
		HiddenSize            int `json:"hidden_size"`
		NumHiddenLayers       int `json:"num_hidden_layers"`
		NumAttentionHeads     int `json:"num_attention_heads"`
		NumKeyValueHeads      int `json:"num_key_value_heads"`
		HeadDim               int `json:"head_dim"`
	}{NumHiddenLayers: 4, NumKeyValueHeads: 1, HeadDim: 16, MaxPositionEmbeddings: 2048}
	got := cfg.sizing()
	if got.NumHiddenLayers != 64 || got.NumKeyValueHeads != 8 || got.HeadDim != 128 {
		t.Errorf("text_config overrode top-level fields: %+v", got)
	}
	// Absent top-level fields still fill from text_config.
	if got.MaxPositionEmbeddings != 2048 {
		t.Errorf("max_position_embeddings = %d, want 2048 from text_config", got.MaxPositionEmbeddings)
	}
}

func TestSizingDefaultsKVHeadsToAttentionHeadsForMQA(t *testing.T) {
	// MQA/MHA repos omit num_key_value_heads, meaning it equals num_attention_heads.
	// Leaving it zero makes the cache unsizable for every such model.
	got := configJSON{HiddenSize: 4096, NumHiddenLayers: 32, NumAttentionHeads: 32}.sizing()
	if got.NumKeyValueHeads != 32 {
		t.Errorf("kv heads = %d, want 32 (defaulted from attention heads)", got.NumKeyValueHeads)
	}
}

func TestCanReadUngatedNeedsNoToken(t *testing.T) {
	c := newHub(t).client()
	if err := c.CanRead(context.Background(), &Model{ID: "Qwen/Qwen3-32B", Gate: GateNone}); err != nil {
		t.Errorf("a public repo required a token: %v", err)
	}
}

func TestCanReadGatedWithoutTokenFailsBeforeSpending(t *testing.T) {
	c := newHub(t).client()
	err := c.CanRead(context.Background(), &Model{ID: "meta-llama/Llama-3.3-70B-Instruct", Gate: GateManual})
	if err == nil {
		t.Fatal("a gated repo with no token passed the precondition")
	}
	if !strings.Contains(err.Error(), "gated") {
		t.Errorf("error %q does not say the repo is gated", err)
	}
}

func TestCanReadProbesTheDownloadPathNotMetadata(t *testing.T) {
	// The failure being prevented: metadata for a gated repo is readable with a
	// token that cannot download the weights. Checking only that a token is set is
	// what lets a $55/hr p5.48xlarge come up and then 401 on the download.
	h := newHub(t).
		json("/api/models/meta-llama/Llama-3.3-70B-Instruct", `{"id":"x","gated":"manual"}`).
		status("/meta-llama/Llama-3.3-70B-Instruct/resolve/main/config.json", http.StatusForbidden)
	c := h.client(WithToken("hf_unapproved"))
	m := &Model{ID: "meta-llama/Llama-3.3-70B-Instruct", Gate: GateManual}
	err := c.CanRead(context.Background(), m)
	if err == nil {
		t.Fatal("a token that cannot download the weights passed the precondition")
	}
	if !strings.Contains(err.Error(), "huggingface.co/meta-llama/Llama-3.3-70B-Instruct") {
		t.Errorf("error %q does not point at the license page", err)
	}
}

func TestCanReadSucceedsWithAnApprovedToken(t *testing.T) {
	for _, code := range []int{http.StatusOK, http.StatusFound, http.StatusTemporaryRedirect} {
		// The Hub 302s weight downloads to CDN storage, so a redirect is success.
		h := newHub(t).status("/meta-llama/Llama-3.3-70B-Instruct/resolve/main/config.json", code)
		c := h.client(WithToken("hf_approved"))
		m := &Model{ID: "meta-llama/Llama-3.3-70B-Instruct", Gate: GateManual}
		if err := c.CanRead(context.Background(), m); err != nil {
			t.Errorf("status %d → %v, want success", code, err)
		}
	}
}

func TestContextCancellationPropagates(t *testing.T) {
	c := newHub(t).json("/api/models/a/b", `{"id":"a/b"}`).client()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Get(ctx, "a/b"); !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled", err)
	}
}
