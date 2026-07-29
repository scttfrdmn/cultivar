package report

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"time"
)

// The envelope: what a report is, when it was made, and what it assumed.
//
// A recommendation without its inputs is unfalsifiable. Two runs a day apart can
// disagree because a price moved, because a region was added, or because the
// assumed traffic mix changed — and only the first two are facts about AWS. So the
// gating inputs are part of the document rather than prose around it, which is also
// what makes the history log comparable across runs.
//
// Everything here is plain data with no AWS dependency, deliberately: the schema is
// the contract the page, the history log, and `deploy` all serialize behind, and it
// should be readable and testable without credentials.

// SchemaVersion is the version of the report wire format.
//
// It is independent of the tool version: a bug fix that changes no field shape must
// not invalidate a consumer's parser, and a field addition must not require a new
// binary to read old reports. See [CompatibilityPolicy] for what a consumer may
// rely on.
const SchemaVersion = "report.v1"

// CompatibilityPolicy states what may change within a schema version, so a consumer
// knows what it is allowed to depend on. It is exported because it belongs in the
// generated documentation next to the schema itself, not only in a comment here.
//
// Within report.v1:
//
//   - Fields may be ADDED. A consumer must ignore fields it does not recognize.
//   - Fields are never removed or renamed, and a field's type never changes.
//   - A field's meaning never changes. A different meaning gets a different name.
//   - Enum-valued strings ([Provenance], [Unit], [RegionState], and the outcome
//     values) may gain members. A consumer must treat an unrecognized member as
//     unknown rather than failing — new AWS states are the normal case, not an
//     error.
//   - An absent optional field and its zero value are not interchangeable: money
//     that could not be resolved is null with an "unavailable" provenance, never 0.
//
// Anything else is report.v2.
const CompatibilityPolicy = "additive within a major version; unknown fields and unknown enum members must be ignored, not rejected"

// RegionState is what happened when a region was queried.
//
// The distinction that matters is between "asked, and there is nothing" and "never
// successfully asked". Both produce zero candidates, and conflating them turns an
// AccessDenied into a claim about AWS inventory — which then sends a user to a
// costlier region with no indication anything went wrong.
//
// More states arrive with the obtainability work (#22: ineligible, throttled). New
// members are additive per [CompatibilityPolicy], so consumers must not assume this
// list is closed.
type RegionState string

const (
	// RegionOK means the query succeeded and returned candidates.
	RegionOK RegionState = "ok"

	// RegionEmpty means the query succeeded and the region genuinely offers nothing
	// matching. A real answer: us-west-1 has 9 GPU types against us-east-1's 69.
	RegionEmpty RegionState = "empty"

	// RegionFailed means the query did not complete, so this region contributed no
	// information. Never to be rendered as an absence of capacity.
	RegionFailed RegionState = "failed"
)

// Valid reports whether s is a defined state.
func (s RegionState) Valid() bool {
	switch s {
	case RegionOK, RegionEmpty, RegionFailed:
		return true
	}
	return false
}

// Informative reports whether the query actually ran. False means any conclusion
// drawn about this region is a conclusion about the query, not about AWS.
func (s RegionState) Informative() bool { return s == RegionOK || s == RegionEmpty }

// Region is the per-region outcome of a query.
type Region struct {
	// Name is the region code, e.g. "us-east-2".
	Name string `json:"name"`

	// State is what happened.
	State RegionState `json:"state"`

	// Considered is how many candidates were evaluated, and Usable how many passed.
	// Both are recorded even on failure (as zero) so a reader can see that a region
	// contributing nothing contributed nothing because it was never read.
	Considered int `json:"considered"`
	Usable     int `json:"usable"`

	// Error is the failure text when State is [RegionFailed]. A string rather than a
	// wrapped error because this is a wire format; the live error stays in the Go
	// value that produced it.
	Error string `json:"error,omitempty"`
}

// Validate reports whether r is internally consistent.
func (r Region) Validate() error {
	if strings.TrimSpace(r.Name) == "" {
		return fmt.Errorf("region has no name")
	}
	if !r.State.Valid() {
		return fmt.Errorf("region %s: state %q is not one of ok, empty, failed", r.Name, r.State)
	}
	if r.Considered < 0 || r.Usable < 0 {
		return fmt.Errorf("region %s: negative counts (%d considered, %d usable)", r.Name, r.Considered, r.Usable)
	}
	if r.Usable > r.Considered {
		return fmt.Errorf("region %s: %d usable exceeds %d considered", r.Name, r.Usable, r.Considered)
	}
	// The two directions of the distinction this type exists to preserve.
	if r.State == RegionFailed && r.Error == "" {
		return fmt.Errorf("region %s: failed with no error text, so the failure cannot be told from an empty result", r.Name)
	}
	if r.State != RegionFailed && r.Error != "" {
		return fmt.Errorf("region %s: state %s carries an error (%q); a region that errored is not ok", r.Name, r.State, r.Error)
	}
	if r.State == RegionOK && r.Usable == 0 {
		return fmt.Errorf("region %s: state ok with 0 usable candidates should be %q", r.Name, RegionEmpty)
	}
	if r.State == RegionEmpty && r.Usable > 0 {
		return fmt.Errorf("region %s: state empty with %d usable candidates", r.Name, r.Usable)
	}
	return nil
}

// Assumptions are the inputs that change the answer without changing any fact about
// AWS. They are structured rather than free-text because the history log has to be
// able to tell a price change from an assumption change.
type Assumptions struct {
	// InputTokenWeight and OutputTokenWeight are the traffic mix used to blend
	// Bedrock's two meters. Recorded as the pair rather than the resulting ratio so
	// the arithmetic is reproducible without parsing a string. Output costs 4x input
	// for Qwen3-32B, so this moves the break-even point by ~40%.
	InputTokenWeight  float64 `json:"inputTokenWeight"`
	OutputTokenWeight float64 `json:"outputTokenWeight"`

	// Utilization is the fraction of billed hours actually serving. There is no
	// default anywhere in this tool: a silent 1.0 is the most flattering assumption
	// available to a self-hosting recommendation and is almost never true.
	Utilization float64 `json:"utilization"`

	// ContextTokens and Concurrency size the KV cache, which decides which instances
	// qualify at all — at concurrency 8, Qwen3-32B's cache exceeds its weights.
	ContextTokens int `json:"contextTokens"`
	Concurrency   int `json:"concurrency"`

	// KVCacheDTypeBytes is the cache element width (2 for fp16/bf16).
	KVCacheDTypeBytes float64 `json:"kvCacheDTypeBytes"`

	// OverheadFraction is the allowance for activations, CUDA graphs, and
	// fragmentation. Recorded because it is an estimate, not a measurement.
	OverheadFraction float64 `json:"overheadFraction"`

	// Throughput is the tok/s figure the comparison used, with its own provenance.
	// Usually external (a published benchmark) or unavailable until one has been
	// measured on this hardware — which is exactly why it must be visible.
	Throughput Amount `json:"throughput"`
}

// BlendRatio renders an input:output weight pair the way a user states it: "3:1".
//
// One function rather than a format string at each site that needs it. The ratio is
// printed in the verdict, echoed in a CLI flag, and written into the provenance of
// every derived token price, and a separator or an ordering that disagrees between
// those places is a silently misstated assumption.
func BlendRatio(input, output float64) string {
	return fmt.Sprintf("%g:%g", input, output)
}

// Ratio renders the blend as a user states it: "3:1".
func (a Assumptions) Ratio() string {
	return BlendRatio(a.InputTokenWeight, a.OutputTokenWeight)
}

// WithBlend returns a copy carrying the traffic mix and duty cycle. Callers should
// prefer the Record method on the type that owns those values, so the mapping is
// written once.
func (a Assumptions) WithBlend(input, output, utilization float64) Assumptions {
	a.InputTokenWeight = input
	a.OutputTokenWeight = output
	a.Utilization = utilization
	return a
}

// WithSizing returns a copy carrying the memory-sizing parameters.
func (a Assumptions) WithSizing(contextTokens, concurrency int, dtypeBytes, overhead float64) Assumptions {
	a.ContextTokens = contextTokens
	a.Concurrency = concurrency
	a.KVCacheDTypeBytes = dtypeBytes
	a.OverheadFraction = overhead
	return a
}

// WithThroughput returns a copy carrying the tok/s figure the comparison used.
func (a Assumptions) WithThroughput(t Amount) Assumptions {
	a.Throughput = t
	return a
}

// Validate reports whether these assumptions could have produced a report.
func (a Assumptions) Validate() error {
	if a.InputTokenWeight < 0 || a.OutputTokenWeight < 0 {
		return fmt.Errorf("assumptions: negative token weight (%g:%g)", a.InputTokenWeight, a.OutputTokenWeight)
	}
	if a.InputTokenWeight+a.OutputTokenWeight == 0 {
		return fmt.Errorf("assumptions: token weights sum to zero; the traffic mix is unstated")
	}
	if a.Utilization <= 0 || a.Utilization > 1 {
		return fmt.Errorf("assumptions: utilization %g is not in (0, 1]", a.Utilization)
	}
	if a.ContextTokens <= 0 {
		return fmt.Errorf("assumptions: context length %d is not positive", a.ContextTokens)
	}
	if a.Concurrency <= 0 {
		return fmt.Errorf("assumptions: concurrency %d is not positive", a.Concurrency)
	}
	if a.KVCacheDTypeBytes <= 0 {
		return fmt.Errorf("assumptions: KV cache dtype width %g is not positive", a.KVCacheDTypeBytes)
	}
	if a.OverheadFraction < 0 {
		return fmt.Errorf("assumptions: overhead fraction %g is negative", a.OverheadFraction)
	}
	if err := a.Throughput.Valid(); err != nil {
		return fmt.Errorf("assumptions: throughput: %w", err)
	}
	if a.Throughput.Unit() != UnitTokensPerSecond {
		return fmt.Errorf("assumptions: throughput has unit %s, want %s", a.Throughput.Unit(), UnitTokensPerSecond)
	}
	return nil
}

// Query records the parameters that gate a result, as opposed to those that merely
// select from it.
//
// This distinction is load-bearing. truffle#109 found that asking for 8 instances
// where 1 was requested flipped "no capacity exists" to "offerings available" — so a
// capacity report that does not state its instance count is not interpretable, and
// two runs with different counts look like AWS changed its mind.
type Query struct {
	// Regions is the region set as requested, in order. Kept alongside the per-region
	// results so a region that was asked for and vanished from the results is visible.
	Regions []string `json:"regions"`

	// InstanceCount is how many instances the query asked to be available at once.
	// Zero means the dimension did not apply to this report.
	InstanceCount int `json:"instanceCount,omitempty"`

	// DurationHours is the capacity-block duration requested. The *returned*
	// duration goes with each offering, never here: a 24h request came back with a
	// 19h block, and dividing the fee by the requested duration understates the rate
	// by 21%.
	DurationHours int `json:"durationHours,omitempty"`

	// InstanceTypes is an explicit type restriction, empty when the whole catalogue
	// was enumerated. Recorded because "nothing fits" and "nothing fits among the
	// three types you named" are different findings.
	InstanceTypes []string `json:"instanceTypes,omitempty"`
}

// Validate reports whether q is well-formed.
func (q Query) Validate() error {
	if len(q.Regions) == 0 {
		return fmt.Errorf("query: no regions")
	}
	seen := make(map[string]bool, len(q.Regions))
	for _, r := range q.Regions {
		if strings.TrimSpace(r) == "" {
			return fmt.Errorf("query: empty region name")
		}
		if seen[r] {
			return fmt.Errorf("query: region %s listed twice", r)
		}
		seen[r] = true
	}
	if q.InstanceCount < 0 {
		return fmt.Errorf("query: negative instance count %d", q.InstanceCount)
	}
	if q.DurationHours < 0 {
		return fmt.Errorf("query: negative duration %d", q.DurationHours)
	}
	return nil
}

// Subject is what the report is about.
type Subject struct {
	// ModelID is the Hugging Face repo id, e.g. "Qwen/Qwen3-32B".
	ModelID string `json:"modelId"`

	// BedrockModelID is the Bedrock foundation-model id when one exists, empty
	// otherwise. Emptiness is a finding, not a gap: 94 of 132 mappable models are
	// marketplace-only, with no token meter at all.
	BedrockModelID string `json:"bedrockModelId,omitempty"`

	// Gated reports whether the repo needs an HF token, which blocks deploy rather
	// than comparison.
	Gated bool `json:"gated"`

	// Quantization is the checkpoint's quantization method, empty when unquantized.
	Quantization string `json:"quantization,omitempty"`

	// ObservedAt is when the model metadata was read.
	ObservedAt time.Time `json:"observedAt"`
}

// MarshalJSON implements [json.Marshaler], normalizing the observation time to UTC.
//
// Subject carries its own marshaller for the same reason [Amount] does: the normalization
// has to happen wherever the type is written, not only where it happens to be embedded
// today. Without it a report generated in a non-UTC zone emits generatedAt as Z and
// observedAt with an offset — the two timestamps a reader compares to judge staleness,
// rendered in different frames, which sorts wrongly in the history log and reads as a
// negative age.
func (s Subject) MarshalJSON() ([]byte, error) {
	type alias Subject // avoids recursing into this method
	out := alias(s)
	out.ObservedAt = s.ObservedAt.UTC()
	return json.Marshal(out)
}

// Validate reports whether s identifies a model.
func (s Subject) Validate() error {
	if strings.TrimSpace(s.ModelID) == "" {
		return fmt.Errorf("subject: no model id")
	}
	if s.ObservedAt.IsZero() {
		return fmt.Errorf("subject: %s has no observation time", s.ModelID)
	}
	return nil
}

// Envelope is the header every cultivar report carries.
//
// A report body ([Selection], a verdict, a benchmark) is attached by the command
// that produced it; this part is identical across all of them, so the page and the
// history log can read provenance and staleness from any report without knowing
// which command wrote it.
type Envelope struct {
	// SchemaVersion is the wire format version. Always [SchemaVersion] on write; a
	// reader must check it before trusting field meanings.
	SchemaVersion string `json:"schemaVersion"`

	// Kind is the report type, e.g. "compare" or "price". Lets one consumer handle a
	// mixed directory of history records without inferring shape from content.
	Kind string `json:"kind"`

	// ToolVersion is the cultivar build that produced this. Separate from
	// SchemaVersion: the same schema outlives many builds, and a wrong number is
	// traceable to a build only if the build is recorded.
	ToolVersion string `json:"toolVersion"`

	// GeneratedAt is when the report was produced, in UTC.
	//
	// Not the same as the freshness of its contents: availability data has a shelf
	// life of hours, and each [Amount] carries its own observedAt. This is the outer
	// bound — nothing in the report is newer than this — which is what lets a page
	// degrade its own claims rather than presenting a cached figure as current.
	GeneratedAt time.Time `json:"generatedAt"`

	// Account and Partition identify whose AWS view this is. Both optional: prices
	// are not account-specific, but quotas and offered-type sets are, so a quota
	// finding is only interpretable with the account recorded. Partition matters
	// because rates and available services differ in aws-us-gov and aws-cn.
	Account   string `json:"account,omitempty"`
	Partition string `json:"partition,omitempty"`

	// Subject is the model under consideration.
	Subject Subject `json:"subject"`

	// Query is the parameter set that gated the result.
	Query Query `json:"query"`

	// Regions is the per-region outcome, in the order requested.
	Regions []Region `json:"regions"`

	// Assumptions are the inputs that change the answer without changing any fact.
	Assumptions Assumptions `json:"assumptions"`
}

// NewEnvelope returns an envelope stamped with the current schema version.
//
// generatedAt is passed rather than read from the clock so that a report is
// reproducible under test and so the caller can stamp a whole multi-part run with
// one time. It is normalized to UTC, because a history log spanning timezones sorts
// wrongly otherwise.
func NewEnvelope(kind, toolVersion string, generatedAt time.Time) Envelope {
	return Envelope{
		SchemaVersion: SchemaVersion,
		Kind:          kind,
		ToolVersion:   toolVersion,
		GeneratedAt:   generatedAt.UTC(),
	}
}

// Validate reports whether e is complete enough to publish.
//
// Called before writing a report anywhere durable. The failure mode it exists to
// catch is a report that looks fine and is missing the one field that made it
// interpretable — an assumption, a region status, the tool version — because such a
// report cannot be repaired after the fact: the run that produced it is gone.
func (e Envelope) Validate() error {
	if e.SchemaVersion != SchemaVersion {
		return fmt.Errorf("envelope: schema version %q, want %q", e.SchemaVersion, SchemaVersion)
	}
	if strings.TrimSpace(e.Kind) == "" {
		return fmt.Errorf("envelope: no kind")
	}
	if strings.TrimSpace(e.ToolVersion) == "" {
		return fmt.Errorf("envelope: no tool version, so a wrong number here is untraceable to a build")
	}
	if e.GeneratedAt.IsZero() {
		return fmt.Errorf("envelope: no generatedAt, so staleness is unknowable")
	}
	if err := e.validateIdentity(); err != nil {
		return fmt.Errorf("envelope: %w", err)
	}
	if err := e.Subject.Validate(); err != nil {
		return fmt.Errorf("envelope: %w", err)
	}
	if err := e.Query.Validate(); err != nil {
		return fmt.Errorf("envelope: %w", err)
	}
	if err := e.Assumptions.Validate(); err != nil {
		return fmt.Errorf("envelope: %w", err)
	}
	if len(e.Regions) == 0 {
		return fmt.Errorf("envelope: no region results for %d requested regions", len(e.Query.Regions))
	}
	reported := make(map[string]bool, len(e.Regions))
	for _, r := range e.Regions {
		if err := r.Validate(); err != nil {
			return fmt.Errorf("envelope: %w", err)
		}
		if reported[r.Name] {
			return fmt.Errorf("envelope: region %s reported twice", r.Name)
		}
		reported[r.Name] = true
	}
	// Every requested region must appear. A region that was asked for and is missing
	// from the results is the silent-drop failure this envelope exists to expose.
	for _, want := range e.Query.Regions {
		if !reported[want] {
			return fmt.Errorf("envelope: region %s was queried but has no result; "+
				"a dropped region reads as one with no capacity", want)
		}
	}
	// And nothing may appear that was not asked for, which would mean the region set
	// in the report is not the region set that produced it.
	requested := make(map[string]bool, len(e.Query.Regions))
	for _, r := range e.Query.Regions {
		requested[r] = true
	}
	for _, r := range e.Regions {
		if !requested[r.Name] {
			return fmt.Errorf("envelope: region %s has a result but was not queried", r.Name)
		}
	}
	return nil
}

// validateIdentity checks the optional AWS identity fields, which are optional in the
// sense that absent is allowed — not in the sense that anything goes.
//
// Both are free-text on the wire and both are shaped: an account is twelve digits, a
// partition is one of a known set. The check exists because these two fields have no
// unit and no provenance, so nothing else downstream would notice a profile name or
// a whole principal ARN sitting in Account. And a partition is the field a reader
// would use to decide whether the report's prices apply to them at all, which makes a
// malformed one worse than a missing one.
func (e Envelope) validateIdentity() error {
	if e.Account != "" {
		if len(e.Account) != 12 {
			return fmt.Errorf("account %q is not a 12-digit account id", e.Account)
		}
		for _, c := range e.Account {
			if c < '0' || c > '9' {
				return fmt.Errorf("account %q is not all digits", e.Account)
			}
		}
	}
	// Checked as a prefix rather than an exact set: the aws-iso family has members
	// this tool will never see, and a partition it does not recognize is still a
	// partition. What it must not accept is a region name or a service namespace,
	// which is what a mis-wired field would put here.
	if e.Partition != "" && e.Partition != "aws" && !strings.HasPrefix(e.Partition, "aws-") {
		return fmt.Errorf("partition %q is not an AWS partition; a report whose partition is "+
			"wrong is one whose prices may not apply at all", e.Partition)
	}
	return nil
}

// Age returns how long ago the report was generated. It is the floor on staleness,
// not the whole story: individual amounts may be older, since a price read from a
// cache is older than the report that contains it.
func (e Envelope) Age(now time.Time) time.Duration { return now.Sub(e.GeneratedAt) }

// Informative returns the regions that were actually read. A report where this is
// empty has no findings about AWS at all, whatever else it contains.
func (e Envelope) Informative() []Region {
	var out []Region
	for _, r := range e.Regions {
		if r.State.Informative() {
			out = append(out, r)
		}
	}
	return out
}

// Failures returns the regions that could not be read.
func (e Envelope) Failures() []Region {
	var out []Region
	for _, r := range e.Regions {
		if !r.State.Informative() {
			out = append(out, r)
		}
	}
	return out
}

// Degraded reports whether any region failed, meaning the report is incomplete in a
// way a reader must be told about. A partial answer presented as a full one is the
// same class of error as an estimate presented as a price.
func (e Envelope) Degraded() bool { return len(e.Failures()) > 0 }

// MarshalJSON implements [json.Marshaler], normalizing before writing.
//
// Regions are emitted in the order they were requested rather than the order they
// completed. Discovery fans out across goroutines, so completion order varies run to
// run, and a history log full of reordered-but-identical reports is one where a real
// change cannot be spotted in a diff.
func (e Envelope) MarshalJSON() ([]byte, error) {
	type alias Envelope // avoids recursing into this method
	out := alias(e)
	out.GeneratedAt = e.GeneratedAt.UTC()
	out.Regions = orderRegions(e.Regions, e.Query.Regions)
	return json.Marshal(out)
}

// orderRegions returns rs sorted to match want, with any region not in want appended
// alphabetically. Nothing is dropped: a region present in the results and absent
// from the request is a bug, and [Envelope.Validate] reports it — silently
// discarding it here would hide it.
func orderRegions(rs []Region, want []string) []Region {
	if len(rs) == 0 {
		return rs
	}
	rank := make(map[string]int, len(want))
	for i, r := range want {
		rank[r] = i
	}
	out := make([]Region, len(rs))
	copy(out, rs)
	sort.SliceStable(out, func(i, j int) bool {
		ri, iok := rank[out[i].Name]
		rj, jok := rank[out[j].Name]
		switch {
		case iok && jok:
			return ri < rj
		case iok != jok:
			return iok // requested regions first
		default:
			return out[i].Name < out[j].Name
		}
	})
	return out
}
