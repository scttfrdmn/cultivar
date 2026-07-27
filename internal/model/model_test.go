package model

import (
	"math"
	"strings"
	"testing"
	"time"

	"github.com/scttfrdmn/cultivar/internal/report"
)

var observed = time.Date(2026, 7, 27, 12, 0, 0, 0, time.UTC)

// Metadata read live from the Hub on 2026-07-27. These are fixtures for the
// arithmetic, and the numbers asserted against them are the ones the tool must
// produce.
var (
	// Single-dtype, GQA (64 query heads, 8 KV heads), 40,960-token ceiling.
	qwen3_32b = Model{
		ID:             "Qwen/Qwen3-32B",
		Gate:           GateNone,
		Parameters:     map[string]int64{"BF16": 32762123264},
		Architectures:  []string{"Qwen3ForCausalLM"},
		HasSafetensors: true,
		Config: Config{
			MaxPositionEmbeddings: 40960,
			HiddenSize:            5120,
			NumHiddenLayers:       64,
			NumAttentionHeads:     64,
			NumKeyValueHeads:      8,
			HeadDim:               128,
			TorchDType:            "bfloat16",
		},
		ObservedAt: observed,
	}

	// Mixed dtype: MXFP4 experts reported as U8, attention kept in BF16. This is
	// the model that makes summing parameters a 2x error.
	gptOSS120b = Model{
		ID:             "openai/gpt-oss-120b",
		Gate:           GateNone,
		Parameters:     map[string]int64{"BF16": 2167371072, "U8": 118244966400},
		Architectures:  []string{"GptOssForCausalLM"},
		HasSafetensors: true,
		QuantMethod:    "mxfp4",
		Config: Config{
			MaxPositionEmbeddings: 131072,
			HiddenSize:            2880,
			NumHiddenLayers:       36,
			NumAttentionHeads:     64,
			NumKeyValueHeads:      8,
			HeadDim:               64,
		},
		ObservedAt: observed,
	}

	// Gated "manual": deploy fails without an approved token.
	//
	// Config is deliberately empty. This repo's config.json is behind the gate, so
	// an unauthenticated read returns an HTML error page rather than JSON — which is
	// exactly the state the tool sees for a gated model without a token, and the
	// reason [Client.CanRead] exists. Writing layer and head counts here from
	// recollection would put unverified numbers in a fixture that pins the
	// arithmetic, so the sizing assertions use the two repos whose configs were
	// actually read.
	llama33_70b = Model{
		ID:             "meta-llama/Llama-3.3-70B-Instruct",
		Gate:           GateManual,
		Parameters:     map[string]int64{"BF16": 70553706496},
		Architectures:  []string{"LlamaForCausalLM"},
		HasSafetensors: true,
		ObservedAt:     observed,
	}
)

func TestWeightBytesUsesEachDTypesOwnWidth(t *testing.T) {
	// The trap, stated as a test. gpt-oss-120b is {BF16: 2.17e9, U8: 118.2e9}:
	// correct sizing is 114.16 GiB, and summing the counts then multiplying by 2
	// bytes gives 224.29 GiB — which would rule out every single GPU instance AWS
	// sells and send the user to Bedrock or to a two-node deployment for no reason.
	got := gptOSS120b.WeightBytes()
	want := 114.16
	if v := got.MustValue(); math.Abs(v-want) > 0.01 {
		t.Errorf("weights = %.2f GiB, want %.2f", v, want)
	}
	naive := float64(gptOSS120b.TotalParameters()) * 2 / GiB
	if math.Abs(naive-224.29) > 0.01 {
		t.Fatalf("sanity check: naive sum = %.2f GiB, expected 224.29", naive)
	}
	if naive/want < 1.9 {
		t.Errorf("expected the naive sum to be ~2x the correct figure; got %.2fx", naive/want)
	}
	// The source must show the per-dtype widths, so a reader can check the math.
	for _, want := range []string{"BF16", "U8", "safetensors.parameters"} {
		if !strings.Contains(got.Source(), want) {
			t.Errorf("source %q is missing %q", got.Source(), want)
		}
	}
}

func TestWeightBytesSingleDType(t *testing.T) {
	for _, tc := range []struct {
		m    Model
		want float64
	}{
		{qwen3_32b, 61.02},
	} {
		got := tc.m.WeightBytes()
		if v := got.MustValue(); math.Abs(v-tc.want) > 0.01 {
			t.Errorf("%s weights = %.2f GiB, want %.2f", tc.m.ID, v, tc.want)
		}
		if got.Provenance() != report.ProvenanceLive {
			t.Errorf("%s provenance = %s, want live", tc.m.ID, got.Provenance())
		}
	}
}

func TestUnknownDTypeIsUnavailableNotGuessed(t *testing.T) {
	// A dtype whose width we don't know makes the total wrong by an integer factor.
	// A partial sum would be worse than no answer because it would look plausible.
	m := qwen3_32b
	m.Parameters = map[string]int64{"BF16": 1e9, "SOMETHING_NEW": 5e9}
	got := m.WeightBytes()
	if v, ok := got.Value(); ok {
		t.Errorf("unknown dtype produced %.2f GiB", v)
	}
	if !strings.Contains(got.Source(), "SOMETHING_NEW") {
		t.Errorf("source %q does not name the unrecognized dtype", got.Source())
	}
}

func TestGGUFOnlyRepoIsNotSizableFromSafetensors(t *testing.T) {
	// A GGUF-only repo has no safetensors.parameters. It is servable via llama.cpp
	// but must be sized from the GGUF file, not guessed at from nothing.
	m := Model{ID: "someone/Qwen3-32B-GGUF", HasSafetensors: false, ObservedAt: observed}
	got := m.WeightBytes()
	if _, ok := got.Value(); ok {
		t.Error("a GGUF-only repo produced a safetensors-derived size")
	}
	if !strings.Contains(got.Source(), "GGUF") {
		t.Errorf("source %q does not point at the GGUF path", got.Source())
	}
}

func TestKVCacheUsesKVHeadsNotAttentionHeads(t *testing.T) {
	// Qwen3-32B is GQA: 64 query heads, 8 KV heads. Using attention heads would
	// overstate the cache 8x — 80 GiB instead of 10 GiB at full context, which alone
	// would disqualify every single-GPU instance.
	got := qwen3_32b.KVCacheBytesPerToken(2)
	want := float64(2 * 64 * 8 * 128 * 2) // 262,144 B = 256 KiB per token
	if v := got.MustValue(); v != want {
		t.Errorf("per-token cache = %.0f B, want %.0f", v, want)
	}
	wrong := float64(2 * 64 * 64 * 128 * 2)
	if wrong/want != 8 {
		t.Fatalf("sanity check: attention-head figure is %.1fx, expected 8x", wrong/want)
	}
	if !strings.Contains(got.Source(), "kv_heads") {
		t.Errorf("source %q does not name the head count used", got.Source())
	}
}

func TestKVCacheUnsizableWithoutConfig(t *testing.T) {
	m := Model{ID: "x/y", Parameters: map[string]int64{"BF16": 1e9}, HasSafetensors: true, ObservedAt: observed}
	got := m.KVCacheBytesPerToken(2)
	if _, ok := got.Value(); ok {
		t.Error("sized a KV cache with no layer/head fields")
	}
}

func TestGatedModelWithUnreadableConfigSizesWeightsOnly(t *testing.T) {
	// The real shape of a gated repo read without a token: the metadata API answers
	// (so the parameter count is known) while config.json 401s (so the cache is not).
	// Weights alone are a floor, not a requirement, so the total must stay unknown
	// rather than quietly become weights-plus-overhead.
	if v := llama33_70b.WeightBytes().MustValue(); math.Abs(v-131.42) > 0.01 {
		t.Errorf("weights = %.2f GiB, want 131.42", v)
	}
	s := llama33_70b.Size(SizingRequest{ContextTokens: 8192})
	if _, ok := s.KVCache.Value(); ok {
		t.Error("sized the KV cache for a repo whose config.json was never read")
	}
	if _, ok := s.Total.Value(); ok {
		t.Error("produced a total from weights alone; that understates the requirement")
	}
	// And an 80 GiB GPU must not look like a candidate just because the total is
	// unknown, even though the known part alone already exceeds it.
	if fits, _ := s.FitsIn(report.Live(80, report.UnitGiB, "1xH100", observed)); fits {
		t.Error("an unsizable gated model was reported as fitting")
	}
}

func TestSizeQwen3_32B(t *testing.T) {
	// The headline example. At the model's full 40,960-token ceiling the cache adds
	// 10 GiB on top of 61 GiB of weights, so the total is 81.68 GiB — which does NOT
	// fit a single 80 GiB H100 but does fit a 96 GiB g7e.
	s := qwen3_32b.Size(SizingRequest{})
	if s.ContextTokens != 40960 {
		t.Errorf("context = %d, want the model's max of 40960", s.ContextTokens)
	}
	for _, tc := range []struct {
		name string
		got  report.Amount
		want float64
	}{
		{"weights", s.Weights, 61.02},
		{"kv cache", s.KVCache, 10.00},
		{"overhead", s.Overhead, 10.65},
		{"total", s.Total, 81.68},
	} {
		if v := tc.got.MustValue(); math.Abs(v-tc.want) > 0.02 {
			t.Errorf("%s = %.2f GiB, want %.2f", tc.name, v, tc.want)
		}
	}
	if s.Total.Provenance() != report.ProvenanceDerived {
		t.Errorf("total provenance = %s, want derived", s.Total.Provenance())
	}
}

func TestContextLengthIsARealLever(t *testing.T) {
	// Capping context is how a model that doesn't fit starts fitting. Qwen3-32B at
	// its 40,960 ceiling needs 81.68 GiB; at 8,192 it needs 72.48 GiB, which is the
	// difference between "no single 80 GiB GPU works" and "an H100 works".
	full := qwen3_32b.Size(SizingRequest{})
	capped := qwen3_32b.Size(SizingRequest{ContextTokens: 8192})
	if v := capped.Total.MustValue(); math.Abs(v-72.48) > 0.02 {
		t.Errorf("capped total = %.2f GiB, want 72.48", v)
	}
	h100 := report.Live(80, report.UnitGiB, "p5.4xlarge 1xH100", observed)
	if fits, _ := full.FitsIn(h100); fits {
		t.Error("full-context Qwen3-32B reported as fitting an 80 GiB H100")
	}
	if fits, why := capped.FitsIn(h100); !fits {
		t.Errorf("8k-context Qwen3-32B does not fit an 80 GiB H100: %s", why)
	}
}

func TestConcurrencyMakesCacheDominate(t *testing.T) {
	// At concurrency 8 with full context, Qwen3-32B's cache is 80 GiB against 61 GiB
	// of weights. A sizing that assumes a single sequence understates a served
	// endpoint badly, which is why concurrency is an explicit input.
	s := qwen3_32b.Size(SizingRequest{Concurrency: 8})
	cache := s.KVCache.MustValue()
	weights := s.Weights.MustValue()
	if cache <= weights {
		t.Errorf("cache %.1f GiB did not exceed weights %.1f GiB at concurrency 8", cache, weights)
	}
	if math.Abs(cache-80.0) > 0.05 {
		t.Errorf("cache at concurrency 8 = %.2f GiB, want 80.00", cache)
	}
}

func TestSizeRejectsTheG5PairingThatMotivatedThisPackage(t *testing.T) {
	// An early draft of this project recommended g5.2xlarge (22.9 GiB of VRAM) for
	// Qwen3-32B. The model cannot load. Emitting that must be impossible.
	s := qwen3_32b.Size(SizingRequest{})
	g5 := report.Live(22.9, report.UnitGiB, "g5.2xlarge 1xA10G", observed)
	fits, why := s.FitsIn(g5)
	if fits {
		t.Fatal("Qwen3-32B reported as fitting a g5.2xlarge")
	}
	if !strings.Contains(why, "22.9") || !strings.Contains(why, "needs") {
		t.Errorf("reason %q does not state the requirement against the capacity", why)
	}
	// Headroom should say how badly, not just "no".
	head := s.Headroom(g5).MustValue()
	if head > -2.0 {
		t.Errorf("headroom = %.2f, expected deeply negative (needs ~3.5x the VRAM)", head)
	}
}

func TestUnsizableModelDoesNotFitAnything(t *testing.T) {
	// Refusing to recommend is the correct answer when the requirement is unknown.
	m := Model{ID: "x/y", HasSafetensors: false, ObservedAt: observed}
	s := m.Size(SizingRequest{ContextTokens: 4096})
	huge := report.Live(1000, report.UnitGiB, "hypothetical", observed)
	fits, why := s.FitsIn(huge)
	if fits {
		t.Error("an unsizable model was reported as fitting")
	}
	if !strings.Contains(why, "cannot size") {
		t.Errorf("reason %q does not say the model could not be sized", why)
	}
	if _, ok := s.Total.Value(); ok {
		t.Error("an unsizable model produced a total")
	}
}

func TestUnknownGPUMemoryDoesNotFit(t *testing.T) {
	s := qwen3_32b.Size(SizingRequest{})
	unknown := report.Unavailable(report.UnitGiB, "DescribeInstanceTypes returned no GpuInfo")
	if fits, _ := s.FitsIn(unknown); fits {
		t.Error("fit reported against unknown GPU memory")
	}
	if _, ok := s.Headroom(unknown).Value(); ok {
		t.Error("headroom computed against unknown GPU memory")
	}
}

func TestFitsInMultiGPUPool(t *testing.T) {
	// Sharding is what makes big models servable: gpt-oss-120b needs 141.64 GiB, so
	// it fits 4xL40S (192 GiB) but not 1xH200 (141 GiB) — narrowly, which is exactly
	// the case where a sloppy overhead estimate would give the wrong answer.
	s := gptOSS120b.Size(SizingRequest{})
	if v := s.Total.MustValue(); math.Abs(v-141.64) > 0.05 {
		t.Fatalf("total = %.2f GiB, want 141.64", v)
	}
	l40s4 := report.Live(4*48, report.UnitGiB, "g6e.12xlarge 4xL40S", observed)
	if fits, why := s.FitsIn(l40s4); !fits {
		t.Errorf("gpt-oss-120b does not fit 4xL40S: %s", why)
	}
	h200 := report.Live(141, report.UnitGiB, "1xH200", observed)
	if fits, _ := s.FitsIn(h200); fits {
		t.Error("gpt-oss-120b reported as fitting a single 141 GiB H200 with cache and overhead")
	}
}

func TestOverheadFractionIsExplicit(t *testing.T) {
	// The 15% default is an unmeasured allowance, so it must be overridable and
	// visible in the output rather than baked into the arithmetic.
	base := qwen3_32b.Size(SizingRequest{ContextTokens: 8192})
	tighter := qwen3_32b.Size(SizingRequest{ContextTokens: 8192, OverheadFraction: 0.05})
	if tighter.Total.MustValue() >= base.Total.MustValue() {
		t.Error("a smaller overhead fraction did not reduce the total")
	}
	if !strings.Contains(base.Overhead.Source(), "15%") {
		t.Errorf("overhead source %q does not state the fraction used", base.Overhead.Source())
	}
	if !strings.Contains(tighter.Overhead.Source(), "5%") {
		t.Errorf("overhead source %q does not state the overridden fraction", tighter.Overhead.Source())
	}
}

func TestKVCacheDTypeWidth(t *testing.T) {
	// fp8 cache halves the cache cost, which can bring a model under a GPU's
	// ceiling. Reporting it as if it were fp16 would understate what fits.
	fp16 := qwen3_32b.Size(SizingRequest{})
	fp8 := qwen3_32b.Size(SizingRequest{KVCacheDTypeBytes: 1})
	if ratio := fp16.KVCache.MustValue() / fp8.KVCache.MustValue(); math.Abs(ratio-2) > 0.001 {
		t.Errorf("fp16/fp8 cache ratio = %.3f, want 2", ratio)
	}
}

func TestEffectiveHeadDimPrefersTheExplicitField(t *testing.T) {
	// Qwen3-32B publishes head_dim 128 while hidden_size/num_attention_heads is 80.
	// Deriving would understate the cache 1.6x.
	got, ok := qwen3_32b.Config.EffectiveHeadDim()
	if !ok || got != 128 {
		t.Errorf("head dim = %d (%v), want 128 from the explicit field", got, ok)
	}
	derived := Config{HiddenSize: 5120, NumAttentionHeads: 64}
	if got, ok := derived.EffectiveHeadDim(); !ok || got != 80 {
		t.Errorf("derived head dim = %d (%v), want 80", got, ok)
	}
	if _, ok := (Config{}).EffectiveHeadDim(); ok {
		t.Error("an empty config produced a head dim")
	}
}

func TestGateRequiresToken(t *testing.T) {
	if !GateManual.RequiresToken() || !GateAuto.RequiresToken() {
		t.Error("a gated repo was reported as needing no token")
	}
	if GateNone.RequiresToken() {
		t.Error("a public repo was reported as needing a token")
	}
	if !llama33_70b.Gate.RequiresToken() {
		t.Error("Llama-3.3-70B-Instruct is gated manual and must require a token")
	}
}

func TestDTypeBytes(t *testing.T) {
	cases := map[string]float64{
		"BF16": 2, "F16": 2, "F32": 4, "F64": 8,
		"U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
		"MXFP4": 0.5, "U4": 0.5,
		"bf16": 2, " BF16 ": 2, // case and whitespace tolerant
	}
	for dtype, want := range cases {
		got, ok := DTypeBytes(dtype)
		if !ok {
			t.Errorf("%q not recognized", dtype)
			continue
		}
		if got != want {
			t.Errorf("%q = %v bytes, want %v", dtype, got, want)
		}
	}
	for _, dtype := range []string{"", "FLOAT13", "who knows"} {
		if _, ok := DTypeBytes(dtype); ok {
			t.Errorf("%q was recognized; unknown dtypes must not be guessed", dtype)
		}
	}
}
