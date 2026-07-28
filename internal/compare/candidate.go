// Package compare turns a model and a set of regions into a ranked answer.
//
// This file covers candidate selection: given how much GPU memory a model needs,
// which instance types can actually serve it. Everything is derived from live
// DescribeInstanceTypes data — GPU memory, GPU count, architecture, manufacturer —
// because a hardcoded family list is the same bug as a hardcoded price, and this
// ecosystem has now shipped that bug four times. g7e did not exist when this
// project started; whatever replaces it will not exist when this is read.
//
// Selection rejects rather than filters. Every type considered comes back with a
// [Rejection] saying why it was dropped, because "nothing can serve this model" and
// "everything was excluded by a name pattern" look identical in a verdict, and only
// one of them is a fact about AWS.
package compare

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	truffle "github.com/spore-host/truffle/pkg/aws"

	"github.com/scttfrdmn/cultivar/internal/model"
	"github.com/scttfrdmn/cultivar/internal/report"
)

// mibPerGiB converts the MiB that DescribeInstanceTypes reports into the GiB that
// [model.Sizing] works in.
const mibPerGiB = 1024

// anyInstanceType matches every instance type, on purpose.
//
// The GPU question is answered from each type's own GpuInfo, not from its name. All
// 69 GPU types in us-east-1 do currently start with g or p, so a "^(g|p)" pattern
// would work today — which is exactly what makes it dangerous. It would be a
// hardcoded family list wearing a regexp, betting that AWS never names an accelerator
// outside those two letters, and the whole reason this package reads the live
// catalogue is that this ecosystem has lost that class of bet four times. Reading
// GpuInfo costs nothing extra and cannot go stale.
//
// The literal ".*" matters beyond matching: truffle's extractSpecificTypes inspects
// the pattern and, when it finds no wildcard, pushes it down as a
// DescribeInstanceTypes InstanceTypes filter. ".*" is on its wildcard list, so this
// correctly enumerates the region instead of asking EC2 for a type named ".*".
var anyInstanceType = regexp.MustCompile(`.*`)

// Rejection says why a type cannot serve the model. The empty value means it can.
type Rejection string

const (
	// RejectNone means the type is a usable candidate.
	RejectNone Rejection = ""

	// RejectNoGPU is a type with no GPU at all — the overwhelming majority of the
	// ~900 types in a region.
	RejectNoGPU Rejection = "no-gpu"

	// RejectFractionalGPU is a type that gets a slice of a physical GPU rather than
	// a whole one. See [Candidate.GPUs].
	RejectFractionalGPU Rejection = "fractional-gpu"

	// RejectNotCUDA is a non-NVIDIA accelerator. vLLM, SGLang and llama.cpp's server
	// path all need CUDA, so g4ad's AMD Radeon Pro V520 will not run any of them and
	// recommending one produces a launch that cannot serve.
	RejectNotCUDA Rejection = "not-cuda"

	// RejectUnknownGPUMemory is a type reporting GPUs with no memory figure. It
	// cannot be checked, so it is not offered — but it is reported, not dropped.
	RejectUnknownGPUMemory Rejection = "unknown-gpu-memory"

	// RejectUnknownRequirement means the model could not be sized. It applies to
	// every type equally, and its reason names the model rather than the hardware.
	RejectUnknownRequirement Rejection = "unknown-requirement"

	// RejectShardingUndivisible is a multi-GPU type whose GPUs cannot be combined for
	// this model because the attention heads do not distribute. See
	// [tensorParallelSizes].
	RejectShardingUndivisible Rejection = "sharding-undivisible"

	// RejectInsufficientVRAM is the ordinary case: the GPUs this model can use do not
	// add up to enough memory.
	RejectInsufficientVRAM Rejection = "insufficient-vram"
)

// Usable reports whether this rejection permits serving.
func (r Rejection) Usable() bool { return r == RejectNone }

// Candidate is one instance type evaluated against one model in one region.
type Candidate struct {
	// InstanceType and Region identify the offering.
	InstanceType string
	Region       string

	// GPUModel and GPUManufacturer are as reported, e.g. "L40S" / "NVIDIA". Carried
	// because the model name is what a reader recognizes, and what an engine's
	// compatibility notes are written against.
	GPUModel        string
	GPUManufacturer string

	// Architecture is the host CPU architecture. arm64 GPU types exist — g5g pairs
	// Graviton2 with an NVIDIA T4g — and container images are not interchangeable
	// across it, so a g5g recommendation needs an arm64 image or it fails at pull.
	Architecture string

	// GPUs is the count of whole physical GPUs.
	//
	// Zero with non-zero GPU memory is a real state, not a bug: g6f and gr6f report
	// Count 0 alongside a LogicalGpuCount of 1 and a GpuPartitionSize between 0.125
	// and 0.5. They are time-sliced fractions of an L4. Dividing by that count to get
	// per-GPU memory would divide by zero, which is why the fractional case is
	// answered before any arithmetic.
	GPUs int

	// GPUMemory is the total across all of the instance's GPUs, in GiB.
	GPUMemory report.Amount

	// PerGPUMemory is one GPU's share, in GiB. This is the bound that matters when a
	// model cannot shard: an 8x L40S instance with 358 GiB aggregate still holds only
	// 44.7 GiB of a model pinned to tensor-parallel 1.
	PerGPUMemory report.Amount

	// TensorParallel is how many of the instance's GPUs the weights would split
	// across, after divisibility. See [tensorParallelSizes].
	TensorParallel int

	// UsableMemory is the memory reachable at TensorParallel, in GiB — the figure the
	// fit is decided against. It equals GPUMemory only when every GPU can be used.
	UsableMemory report.Amount

	// VCPUs and HostMemory describe the host. Recorded, not gated on: weights are
	// mmapped from disk, so host RAM below the model size slows loading rather than
	// preventing it, and gating on it would drop configurations that work.
	VCPUs      int
	HostMemory report.Amount

	// OnDemand is the hourly rate, or unpriced. Looked up only for usable candidates.
	// Unpriced does not disqualify — p5e.48xlarge has no on-demand offering at all
	// yet is purchasable through capacity blocks — but see [Selection.Cheapest] for
	// why it cannot rank first either.
	OnDemand report.Amount

	// Rejection is why this type cannot serve the model, empty if it can.
	Rejection Rejection

	// Reason is the human-readable form, always populated when Rejection is not
	// empty. It states figures rather than verdicts — "needs 61.0 GiB, has 44.7 GiB"
	// — because the verdict is the part the reader is checking.
	Reason string

	// Headroom is the unused fraction of UsableMemory after the model loads, for
	// usable candidates. A fraction rather than GiB because 8 GiB spare means
	// something different on a 24 GiB card than on a 180 GiB one.
	Headroom report.Amount
}

// Usable reports whether this candidate can serve the model.
func (c Candidate) Usable() bool { return c.Rejection.Usable() }

// RegionStatus records what happened in one region, so an empty result is
// attributable.
//
// This exists because truffle's SearchInstanceTypes returns an error only when
// *every* requested region fails; a partial failure prints to stderr and returns
// nil, which makes a denied or throttled region indistinguishable from one that
// offers no GPUs. That distinction decides whether cultivar says "us-east-2 has
// nothing that fits" or "us-east-2 could not be checked", and the first, said
// wrongly, sends a user to a pricier region. So regions are queried one at a time,
// where "all failed" means "this one failed" and the error is attributable.
type RegionStatus struct {
	// Region is the region queried.
	Region string

	// Err is the query failure, nil on success. A region with Err set has no
	// candidates *and no information*, which is not the same as no candidates.
	Err error

	// Considered is how many instance types were evaluated.
	Considered int

	// Usable is how many of them can serve the model.
	Usable int
}

// Selection is the outcome of evaluating a model against a set of regions.
type Selection struct {
	// Candidates is every type evaluated: usable ones first, cheapest first within
	// that. See [Selection.Usable] for just the servable set.
	Candidates []Candidate

	// Regions is the per-region query status, in the order requested.
	Regions []RegionStatus

	// Required is the memory requirement the candidates were judged against.
	Required report.Amount

	// ObservedAt is when the instance data was read.
	ObservedAt time.Time
}

// Usable returns the candidates that can serve the model, cheapest first.
func (s *Selection) Usable() []Candidate {
	if s == nil {
		return nil
	}
	var out []Candidate
	for _, c := range s.Candidates {
		if c.Usable() {
			out = append(out, c)
		}
	}
	return out
}

// Cheapest returns the lowest-priced usable candidate.
//
// An unpriced candidate is never returned, even when it is the only usable one,
// because "cheapest: unpriced" reads as free — the inversion [report.Compare] exists
// to prevent. The bool reports whether any priced usable candidate exists, so a
// caller holding one unpriced option can say so explicitly instead of implying it is
// a bargain.
func (s *Selection) Cheapest() (Candidate, bool) {
	for _, c := range s.Usable() {
		if c.OnDemand.Known() {
			return c, true
		}
	}
	return Candidate{}, false
}

// Failures returns the regions that could not be queried.
func (s *Selection) Failures() []RegionStatus {
	if s == nil {
		return nil
	}
	var out []RegionStatus
	for _, r := range s.Regions {
		if r.Err != nil {
			out = append(out, r)
		}
	}
	return out
}

// instanceSearcher is the slice of truffle's client this package uses.
// [truffle.Client] satisfies it.
type instanceSearcher interface {
	SearchInstanceTypes(ctx context.Context, regions []string, matcher *regexp.Regexp, opts truffle.FilterOptions) ([]truffle.InstanceTypeResult, error)
}

// onDemandPricer is the slice of [ec2.Pricer] this package uses.
type onDemandPricer interface {
	OnDemand(ctx context.Context, instanceType, region string) (report.Amount, error)
}

// Selector finds the instance types that can serve a model.
type Selector struct {
	search instanceSearcher
	pricer onDemandPricer
	now    func() time.Time
}

// NewSelector returns a Selector over a truffle client and a pricer. A nil pricer
// leaves every candidate unpriced, which is how the fit logic is tested in isolation.
func NewSelector(search instanceSearcher, pricer onDemandPricer, now func() time.Time) *Selector {
	if now == nil {
		now = time.Now
	}
	return &Selector{search: search, pricer: pricer, now: now}
}

// Select evaluates every instance type in each region against a model's sizing.
//
// The model's [model.Config] is needed as well as its [model.Sizing], because fit is
// not only about memory: attention-head divisibility decides how many of an
// instance's GPUs can be combined, and therefore how much of its VRAM is reachable
// at all.
//
// Regions are queried one at a time so that a failure is attributable to one of them
// — see [RegionStatus]. Select returns an error only when every region failed, since
// a partial result is still a usable answer as long as the gaps are visible in
// [Selection.Failures].
func (s *Selector) Select(ctx context.Context, m model.Model, sizing model.Sizing, regions []string) (*Selection, error) {
	if len(regions) == 0 {
		return nil, fmt.Errorf("candidate selection: at least one region is required")
	}

	out := &Selection{
		Required:   sizing.Total,
		ObservedAt: s.now().UTC(),
		Regions:    make([]RegionStatus, 0, len(regions)),
	}

	for _, region := range regions {
		status := RegionStatus{Region: region}

		found, err := s.search.SearchInstanceTypes(ctx, []string{region}, anyInstanceType, truffle.FilterOptions{})
		if err != nil {
			status.Err = err
			out.Regions = append(out.Regions, status)
			continue
		}

		for _, it := range found {
			c := s.evaluate(m, sizing, it, out.ObservedAt)
			status.Considered++
			if c.Usable() {
				status.Usable++
			}
			out.Candidates = append(out.Candidates, c)
		}
		out.Regions = append(out.Regions, status)
	}

	if len(out.Failures()) == len(regions) {
		return out, fmt.Errorf("candidate selection: all %d region queries failed: %w",
			len(regions), joinRegionErrors(out.Failures()))
	}

	// Price only the usable candidates: pricing is a Price List call per type and
	// region, and a type that cannot load the model does not need a rate to be
	// rejected.
	err := s.price(ctx, out)
	sortCandidates(out.Candidates)
	return out, err
}

// evaluate judges one instance type against the model. Each guard returns the
// candidate with its rejection, so the sequence reads as the order in which reasons
// disqualify a type — cheapest and most common test first.
func (s *Selector) evaluate(m model.Model, sizing model.Sizing, it truffle.InstanceTypeResult, observedAt time.Time) Candidate {
	source := fmt.Sprintf("AWS EC2 DescribeInstanceTypes %s %s", it.InstanceType, it.Region)

	c := Candidate{
		InstanceType:    it.InstanceType,
		Region:          it.Region,
		GPUModel:        it.GPUModel,
		GPUManufacturer: it.GPUManufacturer,
		Architecture:    it.Architecture,
		GPUs:            int(it.GPUs),
		VCPUs:           int(it.VCPUs),
		HostMemory: report.Live(float64(it.MemoryMiB)/mibPerGiB, report.UnitGiB,
			source+" MemoryInfo.SizeInMiB", observedAt),
	}

	// Neither GPUs nor GPU memory: not a serving candidate. First because it is most
	// of the catalogue.
	if it.GPUs <= 0 && it.GPUMemoryMiB <= 0 {
		return c.reject(RejectNoGPU, "no GPU")
	}

	// GPU memory without a whole GPU — the fractional types. Answered before any
	// division, since the count is the divisor. See [Candidate.GPUs].
	if it.GPUs <= 0 {
		c.GPUMemory = report.Live(float64(it.GPUMemoryMiB)/mibPerGiB, report.UnitGiB,
			source+" GpuInfo.TotalGpuMemoryInMiB", observedAt)
		return c.reject(RejectFractionalGPU, fmt.Sprintf(
			"a time-sliced fraction of %s with %s of GPU memory and no whole GPU; "+
				"a shared partition cannot host a dedicated inference server",
			displayGPU(it.GPUModel), c.GPUMemory))
	}

	// GPUs with no memory figure: nothing can be checked, so nothing is claimed.
	if it.GPUMemoryMiB <= 0 {
		return c.reject(RejectUnknownGPUMemory, fmt.Sprintf(
			"DescribeInstanceTypes reports %d GPU(s) with no memory size, so this type "+
				"cannot be checked against the model", it.GPUs))
	}

	c.GPUMemory = report.Live(float64(it.GPUMemoryMiB)/mibPerGiB, report.UnitGiB,
		source+" GpuInfo.TotalGpuMemoryInMiB", observedAt)
	c.PerGPUMemory = report.Live(float64(it.GPUMemoryMiB)/mibPerGiB/float64(it.GPUs), report.UnitGiB,
		fmt.Sprintf("%s GpuInfo.TotalGpuMemoryInMiB / %d GPUs", source, it.GPUs), observedAt)

	// The engines cultivar deploys are CUDA-only.
	if !isNVIDIA(it.GPUManufacturer) {
		return c.reject(RejectNotCUDA, fmt.Sprintf(
			"%s %s is not an NVIDIA GPU, and vLLM, SGLang and llama.cpp's server all need CUDA",
			displayGPU(it.GPUManufacturer), displayGPU(it.GPUModel)))
	}

	// An unsizable model rejects every type, and the reason names the model, because
	// no change of hardware fixes it.
	if !sizing.Total.Known() {
		return c.reject(RejectUnknownRequirement,
			"cannot size the model, so no instance can be recommended: "+sizing.Total.Source())
	}

	// How many of the GPUs can actually be combined for this model.
	sizes := tensorParallelSizes(int(it.GPUs), m.Config.NumAttentionHeads, m.Config.NumKeyValueHeads)
	if len(sizes) == 0 {
		return c.reject(RejectShardingUndivisible, fmt.Sprintf(
			"%d GPUs cannot be combined for this model: %s, and tensor parallelism "+
				"requires an even split", it.GPUs, headDescription(m.Config)))
	}
	c.TensorParallel = sizes[len(sizes)-1]

	c.UsableMemory = report.Scale(c.PerGPUMemory, float64(c.TensorParallel), fmt.Sprintf(
		"GPU memory reachable at tensor-parallel %d (%d of %d GPUs)",
		c.TensorParallel, c.TensorParallel, it.GPUs))

	if fits, why := sizing.FitsIn(c.UsableMemory); !fits {
		detail := why
		if c.TensorParallel < int(it.GPUs) {
			detail = fmt.Sprintf("%s; only %d of %d GPUs are usable because %s",
				why, c.TensorParallel, it.GPUs, headDescription(m.Config))
		}
		return c.reject(RejectInsufficientVRAM, detail)
	}

	c.Headroom = sizing.Headroom(c.UsableMemory)
	return c
}

// reject stamps a rejection, and fills every Amount the guards did not reach with an
// unavailable carrying the same reason.
//
// The check is on [report.Provenance.Valid] rather than [report.Amount.Known]: the
// zero Amount has an invalid provenance, so this distinguishes "never set" from
// "deliberately set to unavailable", and does not overwrite a reason a guard already
// wrote. Leaving them zero would produce amounts that fail [report.Amount.Valid] and
// serialize with an empty provenance.
func (c Candidate) reject(r Rejection, reason string) Candidate {
	c.Rejection = r
	c.Reason = reason

	unset := func(a report.Amount) bool { return !a.Provenance().Valid() }

	if unset(c.GPUMemory) {
		c.GPUMemory = report.Unavailable(report.UnitGiB, "no GPU memory: "+reason)
	}
	if unset(c.PerGPUMemory) {
		c.PerGPUMemory = report.Unavailable(report.UnitGiB, "no per-GPU memory: "+reason)
	}
	if unset(c.UsableMemory) {
		c.UsableMemory = report.Unavailable(report.UnitGiB, "no usable GPU memory: "+reason)
	}
	if unset(c.Headroom) {
		c.Headroom = report.Unavailable(report.UnitFraction, "not a usable candidate: "+reason)
	}
	if unset(c.OnDemand) {
		c.OnDemand = report.Unavailable(report.UnitUSDPerHour, fmt.Sprintf(
			"not priced: %s cannot serve this model (%s)", c.InstanceType, r))
	}
	return c
}

// price fills in the on-demand rate for usable candidates.
//
// An absent price — which [ec2.Pricer] reports as an unavailable Amount with a nil
// error — is a fact about the market and travels through as unpriced. A lookup
// *failure* is a fact about this run, and is returned: a ranking assembled while
// credentials were expired would order candidates by which lookups happened to
// survive, which is not a ranking of anything.
func (s *Selector) price(ctx context.Context, sel *Selection) error {
	if s.pricer == nil {
		return nil
	}
	type key struct{ typ, region string }
	seen := map[key]report.Amount{}
	for i, c := range sel.Candidates {
		if !c.Usable() {
			continue
		}
		k := key{c.InstanceType, c.Region}
		amount, ok := seen[k]
		if !ok {
			var err error
			amount, err = s.pricer.OnDemand(ctx, c.InstanceType, c.Region)
			if err != nil {
				return fmt.Errorf("candidate selection: pricing %s in %s: %w",
					c.InstanceType, c.Region, err)
			}
			seen[k] = amount
		}
		sel.Candidates[i].OnDemand = amount
	}
	return nil
}

// tensorParallelSizes returns the tensor-parallel degrees a model with the given
// head counts can use on an instance with gpus GPUs, ascending.
//
// Two constraints, both real and both silent when violated — the server aborts at
// load time, after the instance has started billing:
//
//   - The degree must divide the GPU count, since the GPUs are what the weights are
//     being split across. Using 3 of 4 GPUs is not a configuration any engine offers.
//   - Attention heads must distribute: every engine requires
//     num_attention_heads % tp == 0. KV heads are looser, because they can be
//     replicated — tp must either divide the KV head count or be a multiple of it.
//     So a model with 4 KV heads runs at tp 8 with each head duplicated, but a model
//     with 6 cannot run at tp 4.
//
// Head counts of zero mean config.json did not publish them, which yields tp 1 only.
// A single GPU needs no divisibility, so it is the one degree that is safe without
// the information; claiming more would be guessing about whether the server starts.
// Qwen3-235B-A22B is the live case that makes the KV rule matter — 64 attention heads
// but 4 KV heads, so 8-way parallelism works only through replication.
func tensorParallelSizes(gpus, heads, kvHeads int) []int {
	if gpus <= 0 {
		return nil
	}
	var out []int
	for tp := 1; tp <= gpus; tp++ {
		if gpus%tp != 0 {
			continue
		}
		if tp == 1 {
			out = append(out, tp)
			continue
		}
		if heads <= 0 || heads%tp != 0 {
			continue
		}
		if kvHeads > 0 && kvHeads%tp != 0 && tp%kvHeads != 0 {
			continue
		}
		out = append(out, tp)
	}
	return out
}

// headDescription renders the head counts for a rejection reason, so a reader can
// see which number blocked the sharding rather than just that something did.
func headDescription(c model.Config) string {
	switch {
	case c.NumAttentionHeads <= 0:
		return "config.json publishes no attention-head count, so only one GPU can be used"
	case c.NumKeyValueHeads <= 0:
		return fmt.Sprintf("%d attention heads do not divide evenly", c.NumAttentionHeads)
	default:
		return fmt.Sprintf("%d attention heads and %d KV heads do not divide evenly",
			c.NumAttentionHeads, c.NumKeyValueHeads)
	}
}

// isNVIDIA matches the manufacturer case-insensitively. DescribeInstanceTypes reports
// "NVIDIA" and "AMD" today; casing is not a documented contract and matching loosely
// costs nothing.
func isNVIDIA(manufacturer string) bool {
	return strings.EqualFold(strings.TrimSpace(manufacturer), "nvidia")
}

// displayGPU renders a reported field that may be empty, so a reason never reads
// "  is not an NVIDIA GPU".
func displayGPU(s string) string {
	if strings.TrimSpace(s) == "" {
		return "an unnamed GPU"
	}
	return s
}

// sortCandidates puts usable candidates first, then cheapest, then a stable
// tiebreak. Rejected candidates are grouped by rejection and ordered too, so a diff
// of two runs shows real changes rather than map iteration order.
func sortCandidates(cs []Candidate) {
	sort.SliceStable(cs, func(i, j int) bool {
		a, b := cs[i], cs[j]
		if a.Usable() != b.Usable() {
			return a.Usable()
		}
		if a.Usable() {
			// report.Compare sorts unknown last, so an unpriced candidate cannot
			// arrive at the top of the list and be read as the cheapest.
			if n := report.Compare(a.OnDemand, b.OnDemand); n != 0 {
				return n < 0
			}
		} else if a.Rejection != b.Rejection {
			return a.Rejection < b.Rejection
		}
		if a.InstanceType != b.InstanceType {
			return a.InstanceType < b.InstanceType
		}
		return a.Region < b.Region
	})
}

// joinRegionErrors combines per-region failures into one error naming each region, so
// "all regions failed" says which and why. errors.Join rather than string
// concatenation so errors.Is still reaches the underlying AWS error — a caller
// checking for a throttle should find one.
func joinRegionErrors(failures []RegionStatus) error {
	wrapped := make([]error, 0, len(failures))
	for _, f := range failures {
		wrapped = append(wrapped, fmt.Errorf("%s: %w", f.Region, f.Err))
	}
	return errors.Join(wrapped...)
}
