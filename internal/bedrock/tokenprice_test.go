package bedrock

import (
	"strings"
	"testing"
)

// The Qwen3-32B rows for us-east-1, read live 2026-07-27, verbatim. All 16 of them:
// every rate is published twice, once per naming convention, and the two conventions
// carry the tier in different attributes. This is the fixture the tier logic is
// pinned against because it contains every populated/null combination AWS uses.
var qwen3_32bRows = []rowAttrs{
	// Display-name convention: feature is set, service_tier is empty.
	{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-input-tokens", InferenceType: "Input tokens", Feature: "On-demand Inference", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-output-tokens", InferenceType: "Output tokens", Feature: "On-demand Inference", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-input-tokens-flex", InferenceType: "Input tokens flex", Feature: "On-demand Inference", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-output-tokens-flex", InferenceType: "Output tokens flex", Feature: "On-demand Inference", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-input-tokens-priority", InferenceType: "Input tokens priority", Feature: "On-demand Inference", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-output-tokens-priority", InferenceType: "Output tokens priority", Feature: "On-demand Inference", RegionCode: "us-east-1"},
	// The two rows where batch is carried ONLY by feature: inferenceType is
	// byte-identical to standard's.
	{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-input-tokens-batch", InferenceType: "Input tokens", Feature: "Batch Inference", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-output-tokens-batch", InferenceType: "Output tokens", Feature: "Batch Inference", RegionCode: "us-east-1"},

	// Mantle convention: service_tier is set, feature is empty. Note the casing drift
	// on the batch rows ("input tokens batch" lowercase).
	{Model: "Qwen3 32B", UsageType: "USE1-qwen.qwen3-32b-mantle-input-tokens-standard", InferenceType: "Input tokens", ServiceTier: "standard", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-qwen.qwen3-32b-mantle-output-tokens-standard", InferenceType: "Output tokens", ServiceTier: "standard", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-qwen.qwen3-32b-mantle-input-tokens-flex", InferenceType: "Input tokens flex", ServiceTier: "flex", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-qwen.qwen3-32b-mantle-output-tokens-flex", InferenceType: "Output tokens flex", ServiceTier: "flex", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-qwen.qwen3-32b-mantle-input-tokens-priority", InferenceType: "Input tokens priority", ServiceTier: "priority", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-qwen.qwen3-32b-mantle-output-tokens-priority", InferenceType: "Output tokens priority", ServiceTier: "priority", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-qwen.qwen3-32b-mantle-input-tokens-batch", InferenceType: "input tokens batch", ServiceTier: "batch", RegionCode: "us-east-1"},
	{Model: "Qwen3 32B", UsageType: "USE1-qwen.qwen3-32b-mantle-output-tokens-batch", InferenceType: "output tokens batch", ServiceTier: "batch", RegionCode: "us-east-1"},
}

func TestTierFromServiceTierWhenPresent(t *testing.T) {
	for _, tc := range []struct {
		serviceTier string
		want        Tier
	}{
		{"standard", TierStandard},
		{"priority", TierPriority},
		{"flex", TierFlex},
		{"batch", TierBatch},
		{"  BATCH  ", TierBatch},
	} {
		got, ok := tierOf(rowAttrs{
			UsageType:   "USE1-x-mantle-input-tokens-standard",
			ServiceTier: tc.serviceTier,
		})
		if !ok || got != tc.want {
			t.Errorf("service_tier %q → %q (%v), want %q", tc.serviceTier, got, ok, tc.want)
		}
	}
}

func TestBatchIsDetectedFromFeatureAlone(t *testing.T) {
	// The single most expensive misread available. USE1-Qwen3-32B-input-tokens-batch
	// carries inferenceType "Input tokens" — identical to the standard row — and
	// service_tier null. Only `feature: "Batch Inference"` says it is batch. Reading
	// inferenceType alone files a $0.075/1M batch rate as the $0.15/1M standard rate,
	// making Bedrock look half price and pushing the break-even verdict the wrong way.
	got, ok := tierOf(rowAttrs{
		Model:         "Qwen3 32B",
		UsageType:     "USE1-Qwen3-32B-input-tokens-batch",
		InferenceType: "Input tokens",
		Feature:       "Batch Inference",
	})
	if !ok {
		t.Fatal("a batch row was rejected outright")
	}
	if got != TierBatch {
		t.Errorf("tier = %q, want batch; inferenceType is indistinguishable from standard "+
			"on this row, so feature is the only signal", got)
	}
}

func TestTierFromUsageTypeSuffixWhenNothingElseIsSet(t *testing.T) {
	// xai.grok-4.3 publishes every tier with inferenceType and feature both null.
	for _, tc := range []struct {
		usage string
		want  Tier
	}{
		{"USE1-xai.grok-4.3-mantle-input-tokens-standard", TierStandard},
		{"USE1-xai.grok-4.3-mantle-output-tokens-priority", TierPriority},
		{"USE1-xai.grok-4.3-mantle-input-tokens-flex", TierFlex},
		{"USE1-xai.grok-4.3-mantle-output-tokens-batch", TierBatch},
		// The display-name standard rows have no tier suffix at all.
		{"USE1-Qwen3-32B-input-tokens", TierStandard},
	} {
		got, ok := tierOf(rowAttrs{UsageType: tc.usage})
		if !ok || got != tc.want {
			t.Errorf("usagetype %q → %q (%v), want %q", tc.usage, got, ok, tc.want)
		}
	}
}

func TestEveryQwenRowClassifiesAndTheTwoConventionsAgree(t *testing.T) {
	// All 16 live rows must classify, and the display-name row and the mantle row for
	// the same (tier, meter) must land on the same key. If they disagree, the
	// duplicate-detection in Lookup turns a real rate into an error.
	type key struct {
		tier  Tier
		meter Meter
	}
	got := map[key][]string{}
	for _, r := range qwen3_32bRows {
		tier, ok := tierOf(r)
		if !ok {
			t.Errorf("%s: no tier", r.UsageType)
			continue
		}
		meter, ok := meterOf(r)
		if !ok {
			t.Errorf("%s: no meter", r.UsageType)
			continue
		}
		got[key{tier, meter}] = append(got[key{tier, meter}], r.UsageType)
	}
	if len(got) != 8 {
		t.Errorf("16 rows produced %d distinct (tier, meter) keys, want 8 — "+
			"each rate is published once per naming convention", len(got))
	}
	for k, uts := range got {
		if len(uts) != 2 {
			t.Errorf("%v has %d rows (%v), want the display-name and mantle pair", k, len(uts), uts)
		}
	}
	for _, tier := range []Tier{TierStandard, TierPriority, TierFlex, TierBatch} {
		for _, meter := range []Meter{MeterInput, MeterOutput} {
			if _, ok := got[key{tier, meter}]; !ok {
				t.Errorf("no %s %s row classified", tier, meter)
			}
		}
	}
}

func TestNonInferenceFeaturesAreNotTiers(t *testing.T) {
	// A feature that is not inference is a different product. Model Customization
	// publishes input and output token rates for a fine-tuned model, with an
	// inferenceType of plain "Input tokens" — reading it as a tier would give the base
	// model a second, conflicting standard rate.
	for _, r := range []rowAttrs{
		{Model: "Nova Lite", UsageType: "USE1-NovaLite-input-tokens-custom-model", InferenceType: "Input tokens", Feature: "Model Customization"},
		{Model: "Nova Pro", UsageType: "USE1-NovaPro-Customization-Training", Feature: "Model Customization"},
		{UsageType: "USE1-OSS-CustomModelImport-Inference-v1:0", Feature: "Custom Model Import"},
		{UsageType: "USE1-TitanText-Premier-ProvisionedThroughput-Custom-NoCommit-ModelUnits", Feature: "Provisioned Throughput Inference - Custom - no commit"},
	} {
		if tier, ok := tierOf(r); ok {
			t.Errorf("%s (feature %q) classified as tier %q; it is not on-demand inference "+
				"of a published model", r.UsageType, r.Feature, tier)
		}
	}
}

func TestCustomModelRowsAreRejectedByMeterToo(t *testing.T) {
	// Defence in depth: the customization rows are excluded on the usagetype as well
	// as the feature, because a row can carry the marker in only one of them.
	for _, r := range []rowAttrs{
		{Model: "Nova Lite", UsageType: "USE1-NovaLite-input-tokens-custom-model", InferenceType: "Input tokens"},
		{Model: "Nova Pro", UsageType: "USE1-NovaPro-output-tokens-custom-model", InferenceType: "Output tokens"},
		{Model: "Nova 2.0 Lite", UsageType: "USE1-Nova2.0Lite-Customization-Training"},
	} {
		if meter, ok := meterOf(r); ok {
			t.Errorf("%s classified as %s tokens; it is a fine-tuned model's meter, "+
				"not the base model's", r.UsageType, meter)
		}
	}
}

func TestCacheAndModalityMetersAreNotTextTokens(t *testing.T) {
	// Prompt cache reads are billed *in addition* to input, so counting one as input
	// double-counts. Modality meters are priced per 1K tokens like text but measure
	// pixels and audio seconds, so a text comparison using them is meaningless.
	for _, r := range []rowAttrs{
		{Model: "Nova Pro", UsageType: "USE1-NovaPro-cache-read-input-token-count", InferenceType: "Cache read input tokens"},
		{Model: "Nova Pro", UsageType: "USE1-NovaPro-cache-write-input-token-count", InferenceType: "Cache write input tokens"},
		{Model: "xai.grok-4.3", UsageType: "USE1-xai.grok-4.3-mantle-cache-read-tokens-standard", TokenType: "Cache Read Input Tokens", ServiceTier: "standard"},
		{Model: "Nova 2.0 Pro", UsageType: "USE1-Nova2.0Pro-input-image-token-count", InferenceType: "Input Image Token Count"},
		{Model: "Nova 2.0 Omni", UsageType: "USE1-Nova2.0Omni-input-audio-token-count", InferenceType: "Input Audio Token Count"},
		{Model: "Nova 2.0 Omni", UsageType: "USE1-Nova2.0Omni-input-video-token-count", InferenceType: "Input Video Token Count"},
		{Model: "Nova Sonic", UsageType: "USE1-NovaSonic-speech-understanding-input-token", InferenceType: "Speech Understanding input token"},
	} {
		if meter, ok := meterOf(r); ok {
			t.Errorf("%s classified as %s tokens; it is not text throughput", r.UsageType, meter)
		}
	}
}

func TestCrossRegionMetersAreExcluded(t *testing.T) {
	// Cross-region ("global") inference is a separate, cheaper product: Nova 2.0 Pro
	// standard input is $1.375/1M regionally and $1.25/1M cross-region. Including both
	// gives one (model, tier, meter) key two different rates. Excluding them takes the
	// us-east-1 catalogue from 21 conflicting keys to zero, which is the evidence that
	// this is a distinct meter rather than a discount on the same one.
	for _, r := range []rowAttrs{
		{Model: "Nova 2.0 Pro", UsageType: "USE1-Nova2.0Pro-text-input-tokens-cross-region-global", InferenceType: "Input tokens", Feature: "On-demand Inference"},
		{Model: "Nova 2.0 Lite", UsageType: "USE1-Nova2.0Lite-output-tokens-flex-cross-region-global", InferenceType: "Output tokens flex", Feature: "On-demand Inference"},
	} {
		if meter, ok := meterOf(r); ok {
			t.Errorf("%s classified as %s; cross-region routing is priced separately "+
				"and mixing it in makes one meter carry two rates", r.UsageType, meter)
		}
	}
	// The regional equivalent must still classify, or the exclusion has eaten the
	// rate it was meant to protect.
	if meter, ok := meterOf(rowAttrs{
		Model: "Nova 2.0 Pro", UsageType: "USE1-Nova2.0Pro-text-input-tokens",
		InferenceType: "Input tokens", Feature: "On-demand Inference",
	}); !ok || meter != MeterInput {
		t.Errorf("the regional row was excluded along with the cross-region one: %q (%v)", meter, ok)
	}
}

func TestMeterFromTokenTypeWhenInferenceTypeIsNull(t *testing.T) {
	// grok-4.3's only meter signal is tokenType, e.g. "input_tokens_mantle".
	for _, tc := range []struct {
		attrs rowAttrs
		want  Meter
	}{
		{rowAttrs{UsageType: "USE1-xai.grok-4.3-mantle-input-tokens-standard", TokenType: "input_tokens_mantle"}, MeterInput},
		{rowAttrs{UsageType: "USE1-xai.grok-4.3-mantle-output-tokens-flex", TokenType: "output_tokens_mantle"}, MeterOutput},
		// And with tokenType absent too, the usagetype still carries it.
		{rowAttrs{UsageType: "USE1-Qwen3-32B-input-tokens"}, MeterInput},
		{rowAttrs{UsageType: "USE1-Qwen3-32B-output-tokens"}, MeterOutput},
	} {
		got, ok := meterOf(tc.attrs)
		if !ok || got != tc.want {
			t.Errorf("%s → %q (%v), want %q", tc.attrs.UsageType, got, ok, tc.want)
		}
	}
}

func TestPerMillionTokensReadsTheUnit(t *testing.T) {
	// The 1000x trap. 828 of the 840 token-denominated dimensions in us-east-1 are
	// per "1K tokens" and 12 per "1M tokens" — all 12 belonging to xai.grok-4.3, whose
	// standard input is $1.25/1M. Assuming "1K tokens" everywhere reports that as
	// $1,250/1M, in the direction that makes self-hosting look free.
	for _, tc := range []struct {
		rate float64
		unit string
		want float64
	}{
		{0.00015, "1K tokens", 0.15},   // Qwen3 32B standard input
		{0.0006, "1K tokens", 0.60},    // Qwen3 32B standard output
		{1.25, "1M tokens", 1.25},      // grok-4.3 standard input — NOT 1250
		{2.5, "1M tokens", 2.5},        // grok-4.3 standard output
		{0.00015, "1000 tokens", 0.15}, // spelling variants
		{1.25, "1000000 tokens", 1.25},
	} {
		got, err := perMillionTokens(tc.rate, tc.unit)
		if err != nil {
			t.Errorf("%v per %q: %v", tc.rate, tc.unit, err)
			continue
		}
		if !sameRate(got, tc.want) {
			t.Errorf("%v per %q = %v, want %v", tc.rate, tc.unit, got, tc.want)
		}
	}
}

func TestUnknownUnitIsRejectedNotDefaulted(t *testing.T) {
	// A wrong scale factor is worse than a missing price, because only the missing
	// price is visible. If AWS introduces "1B tokens", this must fail loudly.
	for _, unit := range []string{"1B tokens", "", "1K Requests", "characters"} {
		if got, err := perMillionTokens(0.00015, unit); err == nil {
			t.Errorf("unit %q accepted, returning %v; an unrecognized scale must be an error", unit, got)
		}
	}
}

func TestIsTokenUnit(t *testing.T) {
	// The same model publishes non-token dimensions alongside its token rates.
	for _, u := range []string{"1K tokens", "1M tokens", "tokens"} {
		if !isTokenUnit(u) {
			t.Errorf("%q not recognized as a token unit", u)
		}
	}
	for _, u := range []string{"hour", "hours", "Model/month", "Requests", "image", "video",
		"Images processed", "Custom Model Unit per Min", "TextUnit"} {
		if isTokenUnit(u) {
			t.Errorf("%q treated as a token unit", u)
		}
	}
}

func TestJoinKeyCollapsesBothNamingConventions(t *testing.T) {
	// The join exists because hf-bedrock-map yields "qwen.qwen3-32b-v1:0" while the
	// Price List keys on "Qwen3 32B".
	for _, tc := range []struct {
		modelID, displayName string
	}{
		{"qwen.qwen3-32b-v1:0", "Qwen3 32B"},
		{"meta.llama3-3-70b-instruct-v1:0", "Llama 3.3 70B"},
		{"meta.llama4-maverick-17b-instruct-v1:0:128k", "Llama 4 Maverick 17B"},
		{"qwen.qwen3-235b-a22b-2507-v1:0", "Qwen3 235B A22B 2507"},
		{"openai.gpt-oss-120b-1:0", "gpt-oss-120b"}, // the v-less version suffix
		{"mistral.mixtral-8x7b-instruct-v0:1", "Mixtral 8x7B"},
		{"minimax.minimax-m2.5", "MiniMax M2.5"},
		{"zai.glm-4.7-flash", "GLM 4.7 Flash"},
		{"google.gemma-3-12b-it", "Gemma 3 12B"}, // "it" is noise
		// The release-stamp strip: "2407" on the id side against a display name that
		// carries no version at all. Without it this pair misses.
		{"mistral.mistral-large-2407-v1:0", "Mistral Large"},
	} {
		id := joinKey(modelIDBase(tc.modelID), true)
		name := joinKey(tc.displayName, false)
		if id != name {
			t.Errorf("%q → %q but %q → %q; the join would miss", tc.modelID, id, tc.displayName, name)
		}
	}
}

func TestJoinKeySeparatesTheLatencyOptimizedVariant(t *testing.T) {
	// "Llama 3.1 70B Latency Optimized" is a real, pricier product ($0.90/1M both
	// directions against the base model's $0.72). Collapsing it into the base model
	// overstates Bedrock by 25% and would make self-hosting look better than it is.
	base := joinKey("Llama 3.1 70B", false)
	variant := joinKey("Llama 3.1 70B Latency Optimized", false)
	if base == variant {
		t.Fatalf("both names produced %q; the variant must not collapse into the base model", base)
	}
	if got := joinKey(modelIDBase("meta.llama3-1-70b-instruct-v1:0"), true); got != base {
		t.Errorf("the model id produced %q but the base display name %q", got, base)
	}
}

func TestJoinKeyDoesNotCollideAcrossTheLiveDisplayNames(t *testing.T) {
	// Every `model` attribute value in the us-east-1 catalogue, 2026-07-27. The join
	// is only safe if no two of these share a key, under either provider treatment —
	// a collision is an ambiguous match, and the candidates can differ several-fold
	// in price. Concatenating rather than comparing token sets is what buys this: a
	// set comparison has to decide what "3" and "1" mean in "llama3-1-70b" versus
	// "Llama 3.1 70B", and every rule for rejoining them creates a new collision.
	names := []string{
		"Claude 2.0", "Claude 2.1", "Claude 3 Haiku", "Claude 3 Sonnet", "Claude Instant",
		"DeepSeek V3.1", "DeepSeek v3.2", "Devstral", "GLM 4.7", "GLM 4.7 Flash", "GLM 5",
		"GPT OSS Safeguard 120B", "GPT OSS Safeguard 20B", "Gemma 3 12B", "Gemma 3 27B",
		"Gemma 3 4B", "Kimi K2 Thinking", "Kimi K2.5", "Llama 3 70B", "Llama 3 8B",
		"Llama 3.1 405B", "Llama 3.1 70B", "Llama 3.1 70B Latency Optimized", "Llama 3.1 8B",
		"Llama 3.2 11B", "Llama 3.2 1B", "Llama 3.2 3B", "Llama 3.2 90B", "Llama 3.3 70B",
		"Llama 4 Maverick 17B", "Llama 4 Scout 17B", "Magistral Small 1.2", "MiniMax M2.5",
		"Minimax M2", "Minimax M2.1", "Ministral 14B 3.0", "Ministral 3B 3.0",
		"Ministral 8B 3.0", "Mistral 7B", "Mistral Large", "Mistral Large 3", "Mistral Small",
		"Mixtral 8x7B", "NVIDIA Nemotron 3 Super 120B A12B", "NVIDIA Nemotron Nano 2",
		"NVIDIA Nemotron Nano 2 VL", "Nemotron Nano 3 30B", "Nova 2.0 Lite", "Nova 2.0 Omni",
		"Nova 2.0 Pro", "Nova Canvas", "Nova Lite", "Nova Micro", "Nova Premier", "Nova Pro",
		"Nova Pro Latency Optimized", "Nova Reel", "Nova Sonic", "Nova Sonic 2.0",
		"Pixtral Large 25.02", "Qwen3 235B A22B 2507", "Qwen3 32B", "Qwen3 Coder 30B A3B",
		"Qwen3 Coder 480B A35B", "Qwen3 Coder Next", "Qwen3 Next 80B A3B", "Qwen3 VL 235B A22B",
		"R1", "Voxtral Mini 1.0", "Voxtral Small 1.0", "Writer Palmyra Vision 7B",
		"google.gemma-4-26b-a4b", "google.gemma-4-31b", "google.gemma-4-e2b",
		"gpt-oss-120b", "gpt-oss-20b", "xai.grok-4.3",
	}
	if len(names) != 77 {
		t.Fatalf("fixture has %d names, the live catalogue had 77", len(names))
	}
	seen := map[string]string{}
	for _, n := range names {
		// Names are indexed under both provider treatments, as matchByKey does.
		for _, stripped := range []bool{false, true} {
			k := joinKey(n, stripped)
			if k == "" {
				t.Errorf("%q produced an empty join key", n)
				continue
			}
			if prev, dup := seen[k]; dup && prev != n {
				t.Errorf("%q and %q both key to %q; the join would be ambiguous", prev, n, k)
			}
			seen[k] = n
		}
	}
}

func TestResolveModelNameUsesTheMantleAnchor(t *testing.T) {
	got, err := resolveModelName("qwen.qwen3-32b-v1:0", "us-east-1", qwen3_32bRows)
	if err != nil {
		t.Fatal(err)
	}
	if got != "Qwen3 32B" {
		t.Errorf("resolved to %q, want Qwen3 32B", got)
	}
}

func TestMantleAnchorRequiresTheDelimiter(t *testing.T) {
	// Bare substring containment collides on prefixes: "minimax.minimax-m2" is a
	// substring of the usagetype for "MiniMax M2.5", and
	// "mistral.magistral-small-2509" of "Magistral Small 1.2" — both wrong models at
	// different prices. The "-mantle" boundary is what makes stage 1 an identity.
	rows := []rowAttrs{
		{Model: "Minimax M2", UsageType: "USE1-minimax.minimax-m2-mantle-input-tokens-standard", ServiceTier: "standard", InferenceType: "Input tokens"},
		{Model: "MiniMax M2.5", UsageType: "USE1-minimax.minimax-m2.5-mantle-input-tokens-standard", ServiceTier: "standard", InferenceType: "Input tokens"},
		{Model: "Minimax M2.1", UsageType: "USE1-minimax.minimax-m2.1-mantle-input-tokens-standard", ServiceTier: "standard", InferenceType: "Input tokens"},
	}
	for id, want := range map[string]string{
		"minimax.minimax-m2":   "Minimax M2",
		"minimax.minimax-m2.5": "MiniMax M2.5",
		"minimax.minimax-m2.1": "Minimax M2.1",
	} {
		got, err := resolveModelName(id, "us-east-1", rows)
		if err != nil {
			t.Errorf("%s: %v", id, err)
			continue
		}
		if got != want {
			t.Errorf("%s resolved to %q, want %q — a shorter id is a prefix of a longer one",
				id, got, want)
		}
	}
}

func TestNamesTheJoinKeyCannotReachRelyOnTheAnchor(t *testing.T) {
	// The two stages are not redundant. "Magistral Small 1.2" and
	// "mistral.magistral-small-2509" encode the same release incompatibly — a minor
	// version against a YYMM stamp — so no key normalization links them. Stage 1 does,
	// because the usagetype embeds the id verbatim. Equally, "Llama 3.3 70B" has no
	// mantle usagetype at all and only stage 2 reaches it. Dropping either stage loses
	// models silently, as a "no token price" verdict.
	rows := []rowAttrs{
		{Model: "Magistral Small 1.2", UsageType: "USE1-mistral.magistral-small-2509-mantle-input-tokens-standard", ServiceTier: "standard", InferenceType: "Input tokens"},
		{Model: "Voxtral Mini 1.0", UsageType: "USE1-mistral.voxtral-mini-3b-2507-mantle-input-tokens-standard", ServiceTier: "standard", InferenceType: "Input tokens"},
	}
	for id, want := range map[string]string{
		"mistral.magistral-small-2509": "Magistral Small 1.2",
		"mistral.voxtral-mini-3b-2507": "Voxtral Mini 1.0",
	} {
		if joinKey(modelIDBase(id), true) == joinKey(want, false) {
			t.Errorf("%q and %q now share a join key; this test is no longer about the anchor", id, want)
		}
		got, err := resolveModelName(id, "us-east-1", rows)
		if err != nil {
			t.Errorf("%s: %v", id, err)
			continue
		}
		if got != want {
			t.Errorf("%s resolved to %q, want %q", id, got, want)
		}
	}
}

func TestResolveModelNameFallsBackToTheJoinKey(t *testing.T) {
	// 22 of the 46 foundation models have no mantle-convention usagetype, so stage 2
	// carries them.
	rows := []rowAttrs{
		{Model: "Llama 3.3 70B", UsageType: "USE1-Llama3-3-70B-input-tokens", InferenceType: "Input tokens", Feature: "On-demand Inference"},
		{Model: "Llama 3.1 70B", UsageType: "USE1-Llama3-1-70B-input-tokens", InferenceType: "Input tokens", Feature: "On-demand Inference"},
		{Model: "Llama 3.1 70B Latency Optimized", UsageType: "USE1-Llama3-1-70B-LatencyOptimized-input-tokens", InferenceType: "Input tokens", Feature: "On-demand Inference"},
	}
	for id, want := range map[string]string{
		"meta.llama3-3-70b-instruct-v1:0":      "Llama 3.3 70B",
		"meta.llama3-1-70b-instruct-v1:0":      "Llama 3.1 70B",
		"meta.llama3-1-70b-instruct-v1:0:128k": "Llama 3.1 70B",
	} {
		got, err := resolveModelName(id, "us-east-1", rows)
		if err != nil {
			t.Errorf("%s: %v", id, err)
			continue
		}
		if got != want {
			t.Errorf("%s resolved to %q, want %q", id, got, want)
		}
	}
}

func TestResolveModelNameNeverGuessesTheLatencyOptimizedVariant(t *testing.T) {
	// The specific 25% overstatement: $0.90/1M against $0.72/1M. Exact key equality
	// excludes the variant, so no tie-breaking heuristic is needed — but if a future
	// key change made them collide, this must become an error rather than a coin flip.
	rows := []rowAttrs{
		{Model: "Llama 3.1 70B", UsageType: "USE1-Llama3-1-70B-input-tokens", InferenceType: "Input tokens"},
		{Model: "Llama 3.1 70B Latency Optimized", UsageType: "USE1-Llama3-1-70B-LatencyOptimized-input-tokens", InferenceType: "Input tokens"},
	}
	got, err := resolveModelName("meta.llama3-1-70b-instruct-v1:0", "us-east-1", rows)
	if err != nil {
		var ambiguous *ErrAmbiguousModel
		if !asErr(err, &ambiguous) {
			t.Fatalf("unexpected error type: %v", err)
		}
		return // an honest refusal is acceptable; a wrong pick is not
	}
	if got != "Llama 3.1 70B" {
		t.Errorf("resolved to %q, want the base variant Llama 3.1 70B", got)
	}
}

func TestResolveModelNameReportsAmbiguityRatherThanPicking(t *testing.T) {
	rows := []rowAttrs{
		{Model: "Thing One", UsageType: "USE1-vendor.thing-mantle-input-tokens-standard"},
		{Model: "Thing Two", UsageType: "USE1-vendor.thing-mantle-output-tokens-standard"},
	}
	_, err := resolveModelName("vendor.thing-v1:0", "us-east-1", rows)
	var ambiguous *ErrAmbiguousModel
	if !asErr(err, &ambiguous) {
		t.Fatalf("err = %v, want ErrAmbiguousModel", err)
	}
	if len(ambiguous.Candidates) != 2 {
		t.Errorf("candidates = %v, want both names listed so the user can disambiguate",
			ambiguous.Candidates)
	}
	if !strings.Contains(err.Error(), "us-east-1") {
		t.Errorf("error %q should name the region", err)
	}
}

func TestUnmatchableModelIsAnErrorNotANearestMatch(t *testing.T) {
	// deepseek.v3-v1:0 keys to bare "v3" against display names "DeepSeek V3.1" and
	// "DeepSeek v3.2" — different models, not spellings of this one. Returning either
	// would price a model against a rate that is not its own; the correct answer is
	// that us-east-1 publishes no rate for it.
	rows := []rowAttrs{
		{Model: "DeepSeek V3.1", UsageType: "USE1-DeepSeekV3-1-input-tokens", InferenceType: "Input tokens"},
		{Model: "DeepSeek v3.2", UsageType: "USE1-deepseek.v3.2-mantle-input-tokens-standard", ServiceTier: "standard"},
	}
	_, err := resolveModelName("deepseek.v3-v1:0", "us-east-1", rows)
	var missing *ErrNoPriceListModel
	if !asErr(err, &missing) {
		t.Fatalf("err = %v, want ErrNoPriceListModel", err)
	}
	if missing.ModelID != "deepseek.v3-v1:0" || missing.Region != "us-east-1" {
		t.Errorf("error carries %+v, want the id and region for the report", missing)
	}
}

func TestModelIDBaseStripsEveryVersionForm(t *testing.T) {
	for id, want := range map[string]string{
		"qwen.qwen3-32b-v1:0":                     "qwen.qwen3-32b",
		"meta.llama3-1-70b-instruct-v1:0:128k":    "meta.llama3-1-70b-instruct",
		"meta.llama4-scout-17b-instruct-v1:0:10m": "meta.llama4-scout-17b-instruct",
		"mistral.mixtral-8x7b-instruct-v0:1":      "mistral.mixtral-8x7b-instruct",
		"openai.gpt-oss-120b-1:0":                 "openai.gpt-oss-120b",
		// No version suffix at all: the newer ids omit it.
		"zai.glm-4.7":          "zai.glm-4.7",
		"minimax.minimax-m2.5": "minimax.minimax-m2.5",
	} {
		if got := modelIDBase(id); got != want {
			t.Errorf("modelIDBase(%q) = %q, want %q", id, got, want)
		}
	}
}

func TestTierValid(t *testing.T) {
	for _, t2 := range []Tier{TierStandard, TierPriority, TierFlex, TierBatch} {
		if !t2.Valid() {
			t.Errorf("%q reported invalid", t2)
		}
	}
	for _, t2 := range []Tier{"", "Standard", "on-demand", "provisioned"} {
		if t2.Valid() {
			t.Errorf("%q reported valid", t2)
		}
	}
}

// asErr is errors.As with a local name, so the test file does not need to import
// errors for one call site per assertion.
func asErr(err error, target any) bool {
	switch t := target.(type) {
	case **ErrAmbiguousModel:
		if e, ok := err.(*ErrAmbiguousModel); ok {
			*t = e
			return true
		}
	case **ErrNoPriceListModel:
		if e, ok := err.(*ErrNoPriceListModel); ok {
			*t = e
			return true
		}
	}
	return false
}
