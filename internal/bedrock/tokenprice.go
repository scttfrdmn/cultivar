package bedrock

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Tier is a Bedrock inference service tier. The four tiers span 3.5x on the same
// model — Qwen3-32B in us-east-1 is $0.075/1M input on flex and batch, $0.15 on
// standard, $0.2625 on priority — so picking the wrong one is the largest single
// error available in the comparison.
type Tier string

const (
	// TierStandard is ordinary synchronous on-demand inference. It is the default
	// and the only tier a plain "what would Bedrock cost" question should quote.
	TierStandard Tier = "standard"
	// TierPriority is a premium latency tier: 75% above standard for Qwen3-32B.
	// Taking the first row the Price List returns often lands here.
	TierPriority Tier = "priority"
	// TierFlex is a cheaper best-effort tier — half of standard, and often fine for
	// non-interactive work. Worth surfacing because it moves break-even a long way.
	TierFlex Tier = "flex"
	// TierBatch is asynchronous bulk inference, also half of standard. Not
	// comparable to an interactive endpoint, but it is the right tier for a nightly
	// job, and for Llama 3.1 405B it is the *only* tier published.
	TierBatch Tier = "batch"
)

// Valid reports whether t is one of the four known tiers.
func (t Tier) Valid() bool {
	switch t {
	case TierStandard, TierPriority, TierFlex, TierBatch:
		return true
	}
	return false
}

// Meter is which token meter a rate applies to. Input and output are separately
// metered and differ up to 4x, so a single "$/1M tokens" figure requires an assumed
// traffic mix. That assumption belongs in the compare layer with an explicit ratio,
// not here.
type Meter string

const (
	// MeterInput is prompt tokens.
	MeterInput Meter = "input"
	// MeterOutput is generated tokens.
	MeterOutput Meter = "output"
)

// Rate is one published per-token rate, normalized to USD per million tokens.
type Rate struct {
	// Tier is the service tier.
	Tier Tier
	// Meter is input or output.
	Meter Meter
	// USDPerMillionTokens is the rate, normalized. See [perMillionTokens] — the
	// published unit is not constant, and assuming it is a 1000x error.
	USDPerMillionTokens float64
	// UsageType is the Price List usagetype the rate came from, and Unit the unit it
	// was published in. Both are provenance: a reader checking a number against the
	// AWS console needs to know which meter and which scale produced it.
	UsageType string
	Unit      string
}

// tierOf extracts the tier from a Price List row, or reports that the row is not a
// tiered inference meter at all.
//
// Three attributes have to be consulted because AWS populates them inconsistently
// within a single model. Every Qwen3-32B rate in us-east-1 is published twice, once
// per naming convention, and the two carry the tier in different fields:
//
//	usagetype=USE1-Qwen3-32B-input-tokens-flex           inferenceType="Input tokens flex"   feature="On-demand Inference"  service_tier=""
//	usagetype=USE1-qwen.qwen3-32b-mantle-input-tokens-flex  inferenceType="Input tokens flex" feature=""                    service_tier="flex"
//	usagetype=USE1-Qwen3-32B-input-tokens-batch           inferenceType="Input tokens"       feature="Batch Inference"      service_tier=""
//	usagetype=USE1-qwen.qwen3-32b-mantle-input-tokens-batch inferenceType="input tokens batch" feature=""                  service_tier="batch"
//
// The third line is the trap: batch is signalled *only* by feature, with an
// inferenceType byte-identical to standard's. Reading inferenceType alone files a
// batch rate as standard, 50% low. The fourth line shows the casing drift
// ("input tokens batch" against "Input tokens" elsewhere), so all matching is
// case-folded. service_tier is populated on only 332 of 1013 us-east-1 rows, which
// is why it cannot be the sole source either.
func tierOf(attrs rowAttrs) (Tier, bool) {
	// service_tier is explicit when present, so trust it first.
	if t := Tier(strings.ToLower(strings.TrimSpace(attrs.ServiceTier))); t.Valid() {
		return t, true
	}

	feature := strings.ToLower(strings.TrimSpace(attrs.Feature))
	inference := strings.ToLower(strings.TrimSpace(attrs.InferenceType))
	usage := strings.ToLower(attrs.UsageType)

	// A feature that is not an inference feature is a different product entirely —
	// "Custom Model Import", "Model Customization", "Model Evaluation" — and must not
	// be read as a tier of ordinary inference.
	if feature != "" && !strings.Contains(feature, "inference") {
		return "", false
	}
	// Batch before the others: feature carries it on display-name rows where
	// inferenceType is indistinguishable from standard.
	if strings.Contains(feature, "batch") || strings.Contains(inference, "batch") ||
		strings.HasSuffix(usage, "-batch") {
		return TierBatch, true
	}
	switch {
	case strings.Contains(inference, "priority"), strings.HasSuffix(usage, "-priority"):
		return TierPriority, true
	case strings.Contains(inference, "flex"), strings.HasSuffix(usage, "-flex"):
		return TierFlex, true
	}
	// xai.grok-4.3 publishes every tier with inferenceType and feature both null,
	// distinguished only by the usagetype suffix.
	if strings.HasSuffix(usage, "-standard") || strings.HasSuffix(usage, "-tokens") {
		return TierStandard, true
	}
	if inference != "" {
		return TierStandard, true
	}
	return "", false
}

// crossRegionMeters are the tokens marking a cross-region ("global") routing meter.
//
// These are a separate, cheaper product, not a cheaper way to buy the same thing:
// Nova 2.0 Pro standard input is $1.375/1M regionally and $1.25/1M cross-region.
// Including both makes one (model, tier, meter) key carry two different rates, which
// is what a 10%-off row looks like when it is silently mixed in. Excluding them
// removes every rate conflict in the us-east-1 catalogue — 21 conflicting keys go to
// zero — so the exclusion is also the evidence that the classification is right.
// Cross-region inference is worth pricing on its own; it is not this function's job.
var crossRegionMeters = []string{"cross-region", "global"}

// nonTextMeters are meters that are billed per token but are not text throughput.
// Counting any of them as input or output produces a number that looks plausible
// and means nothing.
var nonTextMeters = []string{
	// Prompt caching is billed in addition to input, so folding it in double-counts.
	"cache",
	// Modality meters on multimodal models. "Input Image Token Count" is priced per
	// 1K tokens like text but measures pixels.
	"image", "audio", "video", "speech", "document",
}

// nonInferenceUsage are usagetype markers for products that are not on-demand
// inference of a published model.
//
// usagetype has to be consulted because the other attributes do not always say so.
// "USE1-NovaPro-Customization-Training" carries inferenceType and service_tier both
// null, and "USE1-NovaLite-input-tokens-custom-model" carries a perfectly ordinary
// inferenceType of "Input tokens" — it is the *fine-tuned* model's inference rate,
// which is a different product from the base model's and would otherwise be read as
// a second, conflicting standard rate.
var nonInferenceUsage = []string{
	"customization", "custommodelimport", "custom-model", "training",
	"evaluation", "distillation", "embed", "provisionedthroughput",
}

// meterOf classifies a row as input or output text tokens, or rejects it.
//
// Rejection is the important half. The us-east-1 catalogue has 869 rows carrying an
// inferenceType across 38 distinct values, and a comparison of text throughput wants
// only four of them per model.
func meterOf(attrs rowAttrs) (Meter, bool) {
	inference := strings.ToLower(strings.TrimSpace(attrs.InferenceType))
	usage := strings.ToLower(attrs.UsageType)
	tokenType := strings.ToLower(strings.TrimSpace(attrs.TokenType))
	blob := inference + " " + usage + " " + tokenType

	for _, m := range crossRegionMeters {
		if strings.Contains(usage, m) {
			return "", false
		}
	}
	for _, m := range nonTextMeters {
		if strings.Contains(blob, m) {
			return "", false
		}
	}
	for _, m := range nonInferenceUsage {
		if strings.Contains(usage, m) {
			return "", false
		}
	}

	// inferenceType first, then tokenType (grok's only signal, e.g.
	// "input_tokens_mantle"), then the usagetype.
	for _, field := range []string{inference, tokenType} {
		switch {
		case strings.Contains(field, "input"):
			return MeterInput, true
		case strings.Contains(field, "output"):
			return MeterOutput, true
		}
	}
	switch {
	case strings.Contains(usage, "input-tokens"), strings.Contains(usage, "-input-"):
		return MeterInput, true
	case strings.Contains(usage, "output-tokens"), strings.Contains(usage, "-output-"):
		return MeterOutput, true
	}
	return "", false
}

// perMillionTokens converts a published rate to USD per million tokens.
//
// The unit is not constant across the catalogue. Of the 840 token-denominated price
// dimensions in us-east-1, 828 are per "1K tokens" and 12 per "1M tokens" — every
// one of the latter belonging to xai.grok-4.3, whose standard input is $1.25 per 1M.
// Assuming "1K tokens" everywhere reports that as $1,250 per 1M: a 1000x error, in
// the direction that makes self-hosting look free. So the unit is read, never
// assumed, and an unrecognized one is rejected rather than defaulted — a wrong scale
// factor is worse than a missing price, because only the missing price is visible.
func perMillionTokens(rate float64, unit string) (float64, error) {
	switch strings.ToLower(strings.TrimSpace(unit)) {
	case "1k tokens", "1000 tokens":
		return rate * 1000, nil
	case "1m tokens", "1000000 tokens":
		return rate, nil
	case "tokens", "token":
		return rate * 1e6, nil
	default:
		return 0, fmt.Errorf("unrecognized token unit %q; refusing to assume a scale factor", unit)
	}
}

// isTokenUnit reports whether a price dimension is denominated in tokens at all.
// The same model can publish "hour" (provisioned throughput), "Model/month",
// "Requests", "image", and "Custom Model Unit per Min" dimensions alongside its
// token rates.
func isTokenUnit(unit string) bool {
	return strings.Contains(strings.ToLower(unit), "token")
}

// ErrAmbiguousModel reports that a Bedrock model id matched several Price List
// model names, so the resolver refused to pick one. Reported rather than resolved
// because the candidates can differ several-fold in price.
type ErrAmbiguousModel struct {
	ModelID    string
	Region     string
	Candidates []string
}

func (e *ErrAmbiguousModel) Error() string {
	return fmt.Sprintf("bedrock model %q matches %d Price List models in %s (%s); refusing to guess",
		e.ModelID, len(e.Candidates), e.Region, strings.Join(e.Candidates, ", "))
}

// ErrNoPriceListModel reports that no Price List model name matched. For a
// foundation model this means the region does not publish a rate for it, which is a
// real answer: not every model is offered in every region.
type ErrNoPriceListModel struct {
	ModelID string
	Region  string
}

func (e *ErrNoPriceListModel) Error() string {
	return fmt.Sprintf("no Price List model matches bedrock model %q in %s", e.ModelID, e.Region)
}

// joinNoise are tokens that appear in one naming convention but not the other while
// carrying no identity. "mantle" is an AWS-internal codename that appears only in
// usagetype; "instruct" and "it" appear in model ids but never in display names.
var joinNoise = map[string]bool{
	"instruct": true, "it": true, "mantle": true, "tokens": true, "token": true,
}

// joinProviders are provider prefixes. The Price List `model` attribute usually
// omits them ("Qwen3 32B" for "qwen.qwen3-32b-v1:0") but occasionally keeps them
// ("xai.grok-4.3", which is a model id sitting in the display-name field), so both
// the stripped and unstripped forms are tried.
//
// Note the list is of *id* prefixes, not the `provider` attribute, which has its own
// inconsistency: Devstral is published under provider "Mistral AI" while every other
// Mistral model is under "Mistral", and Kimi appears as both "Moonshot AI" and
// "Kimi AI". That attribute is unusable as a join key.
var joinProviders = map[string]bool{
	"meta": true, "qwen": true, "openai": true, "google": true, "mistral": true,
	"deepseek": true, "minimax": true, "moonshot": true, "moonshotai": true,
	"zai": true, "xai": true, "amazon": true, "anthropic": true, "nvidia": true,
	"writer": true, "ai21": true, "cohere": true, "kimi": true, "stability": true,
	"twelvelabs": true, "luma": true,
}

// versionSuffix matches the version and context-window tail of a Bedrock model id:
// "-v1:0", "-v1:0:128k", "-v0:1", and the v-less "-1:0" that openai.gpt-oss-120b-1:0
// uses. Everything from the hyphen on is dropped.
var versionSuffix = regexp.MustCompile(`-v?\d+(\.\d+)?:.*$`)

// releaseDate matches a trailing YYMM release stamp, as in
// "mistral.magistral-small-2509" whose display name is "Magistral Small 1.2". The
// two encode the same release differently, so neither survives into the join key.
var releaseDate = regexp.MustCompile(`(2[0-5](0[1-9]|1[0-2]))$`)

// nonAlphanumeric splits on everything that is not a letter or digit, which is what
// lets "qwen3-32b" and "Qwen3 32B" reduce to the same string.
var nonAlphanumeric = regexp.MustCompile(`[^a-z0-9]+`)

// joinKey reduces a Bedrock model id or a Price List display name to one comparable
// string: lowercased, punctuation removed, noise words dropped, trailing release
// stamp stripped.
//
// Concatenating rather than comparing token sets is deliberate. A set comparison has
// to decide what "3" and "1" mean in "llama3-1-70b" versus "Llama 3.1 70B", and
// every rule for rejoining them creates a new collision. Concatenation sidesteps the
// question: both become "llama3170b". Measured over the 77 distinct display names in
// us-east-1, no two collide under this key, and it separates "Llama 3.1 70B" from
// "Llama 3.1 70B Latency Optimized" — a $0.72 versus $0.90 distinction — without
// needing a variant-marker list.
func joinKey(s string, stripProvider bool) string {
	s = versionSuffix.ReplaceAllString(strings.ToLower(s), "")
	groups := nonAlphanumeric.Split(s, -1)

	out := make([]string, 0, len(groups))
	for i, g := range groups {
		switch {
		case g == "", joinNoise[g]:
			continue
		case i == 0 && stripProvider && joinProviders[g]:
			continue
		}
		out = append(out, g)
	}
	return releaseDate.ReplaceAllString(strings.Join(out, ""), "")
}

// modelIDBase strips the version and context-window suffix from a Bedrock model id,
// leaving the part that appears verbatim inside a usagetype.
func modelIDBase(modelID string) string {
	return versionSuffix.ReplaceAllString(strings.ToLower(strings.TrimSpace(modelID)), "")
}

// resolveModelName joins a Bedrock model id to the Price List `model` attribute
// value that carries its rates.
//
// The two namespaces are not joinable by key and the mismatch is not cosmetic:
// hf-bedrock-map yields "qwen.qwen3-32b-v1:0" while the Price List keys on
// "Qwen3 32B". Both conventions even appear inside the same attribute — the model
// list contains "Qwen3 Next 80B A3B" and "xai.grok-4.3" side by side.
//
// Two stages, because neither works alone. Measured against all 46 foundation-model
// ids in the live mapping against us-east-1: stage 1 resolves 23, stage 2 another 22,
// 1 is unmatched, and none are ambiguous.
//
//  1. The mantle-anchored usagetype match. Many usagetypes embed the model id
//     followed by "-mantle": "USE1-qwen.qwen3-32b-mantle-input-tokens-standard".
//     That is an identity rather than a heuristic. The "-mantle" boundary is
//     required — plain substring containment matches "minimax.minimax-m2" against
//     "MiniMax M2.5" and "mistral.magistral-small-2509" against "Magistral Small
//     1.2", both wrong, because a shorter id is a prefix of a longer one.
//  2. Exact equality of [joinKey], trying both the provider-stripped and
//     unstripped forms, which links "meta.llama3-3-70b-instruct-v1:0" to
//     "Llama 3.3 70B" and "xai.grok-4.3-v1:0" to "xai.grok-4.3".
//
// The one id that resolves to nothing is "deepseek.v3-v1:0", whose key is bare "v3"
// against display names "DeepSeek V3.1" and "DeepSeek v3.2" — different models, not
// spellings of this one. Reporting no price there is the correct answer, and it is
// why the fallthrough is an error rather than a nearest match.
func resolveModelName(modelID, region string, rows []rowAttrs) (string, error) {
	base := modelIDBase(modelID)
	anchor := base + "-mantle"

	anchored := map[string]bool{}
	names := map[string]bool{}
	for _, r := range rows {
		if r.Model == "" {
			continue
		}
		names[r.Model] = true
		if strings.Contains(strings.ToLower(r.UsageType), anchor) {
			anchored[r.Model] = true
		}
	}

	for _, candidates := range []map[string]bool{anchored, matchByKey(base, names)} {
		switch len(candidates) {
		case 1:
			return soleKey(candidates), nil
		case 0:
			continue
		default:
			return "", &ErrAmbiguousModel{
				ModelID: modelID, Region: region, Candidates: sortedKeys(candidates),
			}
		}
	}
	return "", &ErrNoPriceListModel{ModelID: modelID, Region: region}
}

// matchByKey returns the display names whose join key equals the model id's, under
// either provider treatment.
func matchByKey(base string, names map[string]bool) map[string]bool {
	want := map[string]bool{
		joinKey(base, false): true,
		joinKey(base, true):  true,
	}
	delete(want, "")

	out := map[string]bool{}
	for name := range names {
		if want[joinKey(name, false)] || want[joinKey(name, true)] {
			out[name] = true
		}
	}
	return out
}

func soleKey(m map[string]bool) string {
	for k := range m {
		return k
	}
	return ""
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
