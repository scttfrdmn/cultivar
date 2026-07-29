package model

import (
	"testing"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// The reason [Sizing] echoes its resolved parameters back: a report has to state the
// values the arithmetic used, not the ones the caller asked for. A zero request for
// context becomes Qwen3-32B's 40,960-token ceiling — ~10 GiB of cache, enough to
// change which instances qualify — so recording the request would describe a fit
// decision that was never made.
func TestSizingRecordsTheValuesItActuallyUsed(t *testing.T) {
	defaulted := qwen3_32b.Size(SizingRequest{})
	got := defaulted.Record(report.Assumptions{})

	if got.ContextTokens != 40960 {
		t.Errorf("recorded context = %d, want the model's 40960 ceiling that Size resolved to, "+
			"not the 0 that was requested", got.ContextTokens)
	}
	if got.Concurrency != 1 {
		t.Errorf("recorded concurrency = %d, want 1", got.Concurrency)
	}
	if got.KVCacheDTypeBytes != 2 {
		t.Errorf("recorded cache width = %g, want the fp16 default of 2", got.KVCacheDTypeBytes)
	}
	if got.OverheadFraction != DefaultOverheadFraction {
		t.Errorf("recorded overhead = %g, want %g", got.OverheadFraction, DefaultOverheadFraction)
	}

	// Explicit values pass through unchanged, or the echo is just a constant.
	explicit := qwen3_32b.Size(SizingRequest{
		ContextTokens: 4096, Concurrency: 8, KVCacheDTypeBytes: 1, OverheadFraction: 0.25,
	}).Record(report.Assumptions{})
	if explicit.ContextTokens != 4096 || explicit.Concurrency != 8 ||
		explicit.KVCacheDTypeBytes != 1 || explicit.OverheadFraction != 0.25 {
		t.Errorf("explicit request not recorded verbatim: %+v", explicit)
	}

	// The two must differ, or the defaulting path was never exercised.
	if explicit.ContextTokens == got.ContextTokens {
		t.Error("the defaulted and explicit cases record the same context; the test proves nothing")
	}
}

// Record must not disturb the fields other owners fill in. The envelope's assumption
// block is assembled from three sources, and one that clobbers another produces a
// report stating a blend ratio the break-even figure was not computed at.
func TestRecordingSizingLeavesTheBlendAlone(t *testing.T) {
	existing := report.Assumptions{
		InputTokenWeight: 3, OutputTokenWeight: 1, Utilization: 0.4,
		Throughput: report.External(1200, report.UnitTokensPerSecond, "vLLM benchmark", observed),
	}
	got := qwen3_32b.Size(SizingRequest{ContextTokens: 4096}).Record(existing)

	if got.Ratio() != "3:1" {
		t.Errorf("blend ratio = %q after recording sizing, want 3:1", got.Ratio())
	}
	if got.Utilization != 0.4 {
		t.Errorf("utilization = %g after recording sizing, want 0.4", got.Utilization)
	}
	if !got.Throughput.Known() || got.Throughput.MustValue() != 1200 {
		t.Errorf("throughput = %s after recording sizing, want 1200 tok/s", got.Throughput)
	}
	// And the receiver is unchanged: these are value semantics, so a caller holding
	// the original must still see it.
	if existing.ContextTokens != 0 {
		t.Errorf("Record mutated its argument: context = %d", existing.ContextTokens)
	}
}

// Subject carries the facts the model package owns and nothing else. The Bedrock id
// is deliberately absent — that is a different package's finding — and an empty one
// is a real answer, since 94 of 132 mappable repos are marketplace-only.
func TestSubjectCarriesTheRepoFacts(t *testing.T) {
	got := qwen3_32b.Subject()
	if got.ModelID != "Qwen/Qwen3-32B" {
		t.Errorf("model id = %q", got.ModelID)
	}
	if got.Gated {
		t.Error("Qwen3-32B is ungated; a false gate reading blocks a deploy that would work")
	}
	if got.Quantization != "" {
		t.Errorf("quantization = %q, want empty for an unquantized checkpoint", got.Quantization)
	}
	if !got.ObservedAt.Equal(observed) {
		t.Errorf("observedAt = %v, want %v", got.ObservedAt, observed)
	}
	if got.BedrockModelID != "" {
		t.Errorf("bedrock id = %q; the model package does not know it and must not guess",
			got.BedrockModelID)
	}
	if err := got.Validate(); err != nil {
		t.Errorf("a subject built from a resolved model must validate: %v", err)
	}

	// A gated repo says so, because it is a hard deploy blocker: the download 401s
	// after the GPU is already running and billing.
	if !llama33_70b.Subject().Gated {
		t.Error("Llama-3.3-70B is gated:manual and must be recorded as gated")
	}
	// GateAuto counts too — it still needs a token.
	auto := qwen3_32b
	auto.Gate = GateAuto
	if !auto.Subject().Gated {
		t.Error("gate:auto still requires a token and must record as gated")
	}

	// Quantization is reported when present: it is why gpt-oss-120b sizes at 114 GiB
	// rather than 224, and a reader comparing two runs needs to see it changed.
	if got := gptOSS120b.Subject().Quantization; got != "mxfp4" {
		t.Errorf("quantization = %q, want mxfp4", got)
	}
}
