//go:build live

// Opt-in suite that hits the real EC2 DescribeInstanceTypes API. Run with
// `make test-live` (AWS_PROFILE=aws). Every call here is a free read-only
// describe — no instance is launched and nothing bills.
//
// The point is schema drift, not arithmetic. The offline tests pin the decisions this
// package must make; these check that AWS still reports instances in the shape those
// decisions assume. Candidate selection is built entirely on live catalogue fields,
// so a field that changes meaning — a GPU count that starts arriving as null, a
// manufacturer string that changes case, a fractional family that starts reporting
// whole GPUs — would reach users as a confidently wrong recommendation rather than as
// an error. Each measurement below is dated so a future reader can tell a real change
// from a stale expectation.
package compare

import (
	"context"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	truffle "github.com/spore-host/truffle/pkg/aws"

	"github.com/scttfrdmn/cultivar/internal/ec2"
	"github.com/scttfrdmn/cultivar/internal/model"
)

// liveRegion is the region these tests measure. us-east-1 has the broadest GPU
// lineup, so it is where a new family appears first.
const liveRegion = "us-east-1"

func liveCtx(t *testing.T) (context.Context, func()) {
	t.Helper()
	return context.WithTimeout(context.Background(), 120*time.Second)
}

func liveClient(t *testing.T) *truffle.Client {
	t.Helper()
	ctx, cancel := liveCtx(t)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}
	// NewClientFromConfig rather than NewClient: the config is loaded here anyway, and
	// per-region clients are built inside searchInRegion from it.
	return truffle.NewClientFromConfig(cfg)
}

// liveGPUTypes returns every instance type in a region that reports any GPU.
func liveGPUTypes(t *testing.T, region string) []truffle.InstanceTypeResult {
	t.Helper()
	client := liveClient(t)
	ctx, cancel := liveCtx(t)
	defer cancel()

	all, err := client.SearchInstanceTypes(ctx, []string{region}, anyInstanceType, truffle.FilterOptions{})
	if err != nil {
		t.Fatalf("SearchInstanceTypes in %s: %v", region, err)
	}
	if len(all) < 100 {
		t.Fatalf("only %d instance types in %s; the query is not enumerating the catalogue", len(all), region)
	}
	var gpu []truffle.InstanceTypeResult
	for _, it := range all {
		if it.GPUs > 0 || it.GPUMemoryMiB > 0 {
			gpu = append(gpu, it)
		}
	}
	return gpu
}

// TestLiveTheGPUCatalogueIsStillTheSizeWeThink is the coarsest drift detector: if the
// GPU type count moves a long way, a family was added or withdrawn and the recorded
// measurements in this file need re-checking.
//
// Measured 2026-07-28 in us-east-1: 69 GPU types.
func TestLiveTheGPUCatalogueIsStillTheSizeWeThink(t *testing.T) {
	gpu := liveGPUTypes(t, liveRegion)
	const recorded = 69
	t.Logf("%d GPU types in %s (recorded %d on 2026-07-28)", len(gpu), liveRegion, recorded)

	// A wide band: AWS adds families regularly and that is not a failure. This catches
	// the catalogue collapsing — a filter that stopped working — or doubling.
	if len(gpu) < recorded/2 || len(gpu) > recorded*2 {
		t.Errorf("%d GPU types in %s, recorded %d; the catalogue moved far enough that the "+
			"other measurements in this file should be re-verified", len(gpu), liveRegion, recorded)
	}
}

// TestLiveFractionalGPUTypesStillReportZeroWholeGPUs pins the strangest shape in the
// catalogue, because it is the one an independent reimplementation gets wrong.
//
// g6f and gr6f report GpuInfo.Gpus[0].Count as 0 with a GpuPartitionSize of 0.125 to
// 0.5 and a LogicalGpuCount of 1. So the per-GPU sum (Count x MemoryInfo) is zero
// while TotalGpuMemoryInMiB is not — truffle lands on the right number only because
// buildResultFromEC2 falls back to Total when the sum is zero.
//
// Two things must stay true for [Selector.evaluate] to be correct: these types must
// still report zero whole GPUs (so the fractional guard fires before any division),
// and they must still carry non-zero memory (so they are not silently reclassified as
// no-GPU types). Measured 2026-07-28: g6f.large, g6f.xlarge, g6f.2xlarge, g6f.4xlarge,
// gr6f.4xlarge.
func TestLiveFractionalGPUTypesStillReportZeroWholeGPUs(t *testing.T) {
	gpu := liveGPUTypes(t, liveRegion)

	var fractional []string
	for _, it := range gpu {
		if it.GPUs == 0 {
			fractional = append(fractional, it.InstanceType)
			if it.GPUMemoryMiB <= 0 {
				t.Errorf("%s reports 0 GPUs and 0 GPU memory; it would now be classified "+
					"as a non-GPU type rather than as a fractional one", it.InstanceType)
			}
		}
	}
	sort.Strings(fractional)
	t.Logf("types reporting 0 whole GPUs: %v", fractional)

	if len(fractional) == 0 {
		t.Error("no type reports 0 whole GPUs; either the fractional families were " +
			"withdrawn or EC2 now reports a whole-GPU count for them. The " +
			"RejectFractionalGPU guard is now unreachable and should be re-justified.")
	}
	// Every fractional type today is in the *f families. A fractional type outside them
	// is new information worth surfacing rather than silently absorbing.
	for _, name := range fractional {
		family := strings.SplitN(name, ".", 2)[0]
		if !strings.HasSuffix(family, "f") {
			t.Errorf("%s reports 0 whole GPUs but is not in an f-suffixed family; "+
				"a new kind of shared-GPU offering exists and its serving story is unverified", name)
		}
	}
}

// TestLiveEveryGPUTypeReportsItsMemory: the fit decision is impossible without
// GpuInfo.TotalGpuMemoryInMiB. RejectUnknownGPUMemory exists for the case where it is
// absent, and this measures whether that case is real.
//
// Measured 2026-07-28: all 69 report non-zero memory, so the guard is defensive.
func TestLiveEveryGPUTypeReportsItsMemory(t *testing.T) {
	gpu := liveGPUTypes(t, liveRegion)
	var missing []string
	for _, it := range gpu {
		if it.GPUMemoryMiB <= 0 {
			missing = append(missing, it.InstanceType)
		}
	}
	if len(missing) > 0 {
		// Not a failure — the code handles it — but it changes the guard from
		// defensive to load-bearing, which is worth knowing.
		t.Logf("NOTE: %d GPU types now report no memory size (%v); RejectUnknownGPUMemory "+
			"is now load-bearing rather than defensive", len(missing), missing)
	}
}

// TestLiveTheOnlyNonNVIDIAGPUsAreStillAMD pins the manufacturer values, because
// [isNVIDIA] is a string comparison and RejectNotCUDA turns on it. If AWS started
// spelling it "Nvidia", the case-insensitive match absorbs that; if a third
// manufacturer appears, its engine story is unverified and this says so.
//
// Measured 2026-07-28 in us-east-1: NVIDIA 64, AMD 5 (the g4ad family, Radeon Pro V520).
func TestLiveTheOnlyNonNVIDIAGPUsAreStillAMD(t *testing.T) {
	gpu := liveGPUTypes(t, liveRegion)

	byMfr := map[string][]string{}
	for _, it := range gpu {
		byMfr[it.GPUManufacturer] = append(byMfr[it.GPUManufacturer], it.InstanceType)
	}
	for mfr, types := range byMfr {
		t.Logf("%s: %d types", mfr, len(types))
		switch {
		case isNVIDIA(mfr):
		case strings.EqualFold(strings.TrimSpace(mfr), "amd"):
		default:
			sort.Strings(types)
			t.Errorf("unrecognized GPU manufacturer %q on %d types (%v); cultivar rejects "+
				"everything non-NVIDIA as non-CUDA, so a new accelerator is being turned away "+
				"without anyone having checked whether an engine supports it", mfr, len(types), types)
		}
	}

	// The NVIDIA spelling itself: a casing change is absorbed by isNVIDIA, but an
	// entirely different string would silently reject the whole GPU lineup.
	var nvidia int
	for _, it := range gpu {
		if isNVIDIA(it.GPUManufacturer) {
			nvidia++
		}
	}
	if nvidia == 0 {
		t.Fatal("no GPU type matches the NVIDIA manufacturer string; every candidate " +
			"would now be rejected as non-CUDA and the tool would report that nothing can serve")
	}
	if nvidia < len(gpu)/2 {
		t.Errorf("only %d of %d GPU types match NVIDIA; the manufacturer field's values have moved",
			nvidia, len(gpu))
	}
}

// TestLiveArmGPUTypesStillExist: g5g pairs an NVIDIA T4g with Graviton2, so an
// otherwise-valid candidate can need an arm64 container image. [Candidate.Architecture]
// carries that, and this confirms the case is real rather than hypothetical.
//
// Measured 2026-07-28: 6 arm64 GPU types, all g5g.
func TestLiveArmGPUTypesStillExist(t *testing.T) {
	gpu := liveGPUTypes(t, liveRegion)
	var arm []string
	for _, it := range gpu {
		if it.Architecture == "arm64" {
			arm = append(arm, it.InstanceType)
		}
	}
	sort.Strings(arm)
	t.Logf("arm64 GPU types: %v", arm)

	if len(arm) == 0 {
		t.Log("NOTE: no arm64 GPU types remain; the architecture caveat is now dead weight")
		return
	}
	// An arm64 GPU type must still be reported as usable when it fits — dropping it
	// would hide a working option. The architecture is a caveat, not a rejection.
	for _, it := range gpu {
		if it.Architecture != "arm64" || !isNVIDIA(it.GPUManufacturer) || it.GPUs <= 0 {
			continue
		}
		tiny := tinyModel()
		sizing := tiny.Size(model.SizingRequest{ContextTokens: 512})
		c := NewSelector(nil, nil, func() time.Time { return time.Now().UTC() }).
			evaluate(tiny, sizing, it, time.Now().UTC())
		if !c.Usable() {
			t.Errorf("arm64 GPU type %s was rejected for a tiny model: %s", it.InstanceType, c.Reason)
		}
		break
	}
}

// TestLiveNoTypeReportsMultipleGPUKinds guards an assumption in the per-GPU
// arithmetic: [Candidate.PerGPUMemory] divides total memory by the GPU count, which is
// only meaningful when every GPU on the instance is the same. EC2's schema permits a
// Gpus[] array with several entries; if a mixed-accelerator type ever ships, the
// division silently averages two different cards and the fit check is wrong.
//
// Measured 2026-07-28: no type in us-east-1 has more than one Gpus[] entry. truffle
// flattens the array, so this is checked through the flattened result: a mixed type
// would show a GPUModel from the last entry with memory summed across all of them.
func TestLiveNoTypeReportsMultipleGPUKinds(t *testing.T) {
	gpu := liveGPUTypes(t, liveRegion)

	// truffle discards the array structure, so this test can only verify the
	// consequence: for every whole-GPU type, total memory must divide evenly by the
	// count. Mixed cards would almost certainly leave a remainder.
	for _, it := range gpu {
		if it.GPUs <= 0 {
			continue
		}
		if it.GPUMemoryMiB%int64(it.GPUs) != 0 {
			t.Errorf("%s reports %d MiB across %d GPUs, which does not divide evenly; "+
				"the instance may carry more than one kind of GPU, and PerGPUMemory would "+
				"be an average of unlike cards", it.InstanceType, it.GPUMemoryMiB, it.GPUs)
		}
	}
}

// TestLiveTheThinRegionIsThinButNotEmpty measures us-west-1, and corrects the claim in
// issue #11 that it "offers none of the modern GPU families".
//
// Measured 2026-07-28: 9 GPU types across g4dn, p5 and p5en, of which 3 can serve
// Qwen3-32B (g4dn.metal, p5.48xlarge, p5en.48xlarge). So the region is thin, not
// empty — and what it does carry is the expensive end. p5.48xlarge is ~$55/hr against
// a g7e.2xlarge at $3.36 in us-east-1, so a fit check that ignored price would
// cheerfully recommend a 16x more expensive machine here. That is the reason
// [Selection.Cheapest] exists and why price ranking is not optional.
func TestLiveTheThinRegionIsThinButNotEmpty(t *testing.T) {
	client := liveClient(t)
	m := qwen3ForLive()
	sizing := m.Size(model.SizingRequest{ContextTokens: 4096})

	ctx, cancel := liveCtx(t)
	defer cancel()
	sel, err := NewSelector(client, nil, nil).Select(ctx, m, sizing, []string{"us-west-1"})
	if err != nil {
		t.Fatalf("Select in us-west-1: %v", err)
	}

	status := sel.Regions[0]
	if status.Err != nil {
		t.Fatalf("us-west-1 query failed: %v", status.Err)
	}
	if status.Considered == 0 {
		t.Fatal("us-west-1 considered no instance types at all; the query did not run")
	}
	t.Logf("us-west-1: considered %d types, %d can serve %s (%s)",
		status.Considered, status.Usable, m.ID, sizing.Total)

	// The query ran and produced an answer, which is the state under test: a success
	// with few candidates must stay distinguishable from a failure — see [RegionStatus].
	families := map[string]bool{}
	names := []string{}
	for _, c := range sel.Usable() {
		names = append(names, c.InstanceType)
		families[strings.SplitN(c.InstanceType, ".", 2)[0]] = true
	}
	sort.Strings(names)
	t.Logf("us-west-1 can serve %s on: %v", m.ID, names)

	if status.Usable == 0 {
		t.Logf("NOTE: us-west-1 now serves %s on nothing; it did offer p5/p5en on "+
			"2026-07-28, so a family was withdrawn", m.ID)
	}
	// The thin-region hazard: if the only options here are p-family, the cheapest
	// answer is in another region, and a per-region comparison that stopped at "it
	// fits" would miss that by more than an order of magnitude.
	if len(families) > 0 && !families["g6e"] && !families["g7e"] && !families["g6"] {
		t.Logf("NOTE: us-west-1 still offers no mid-range g-family option for %s, so every "+
			"candidate here is p-family and priced accordingly", m.ID)
	}
}

// TestLiveQwen3_32BStillNeedsMoreThanOneL40S is the end-to-end verdict test, and the
// regression guard for the bug that motivated the package: a 22.4 GiB g5 must never
// come back usable for a 61 GiB model, and the 44.7 GiB L40S must not either.
//
// It also confirms the honest positive: multi-GPU instances do serve this model, so a
// selection that returned nothing would be a bug rather than an answer.
func TestLiveQwen3_32BStillNeedsMoreThanOneL40S(t *testing.T) {
	client := liveClient(t)
	m := qwen3ForLive()
	sizing := m.Size(model.SizingRequest{ContextTokens: 4096})
	need, ok := sizing.Total.Value()
	if !ok {
		t.Fatalf("could not size the fixture: %s", sizing.Total.Source())
	}
	t.Logf("%s needs %s at 4k context", m.ID, sizing.Total)

	ctx, cancel := liveCtx(t)
	defer cancel()
	sel, err := NewSelector(client, nil, nil).Select(ctx, m, sizing, []string{liveRegion})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}

	usable := sel.Usable()
	if len(usable) == 0 {
		t.Fatalf("no instance type in %s can serve %s, which cannot be true", liveRegion, m.ID)
	}
	t.Logf("%d of %d types can serve it", len(usable), sel.Regions[0].Considered)

	// Every usable candidate must genuinely have the memory, at the tensor-parallel
	// degree that was chosen.
	for _, c := range usable {
		have, ok := c.UsableMemory.Value()
		if !ok {
			t.Errorf("%s is usable with unavailable UsableMemory", c.InstanceType)
			continue
		}
		if have < need {
			t.Errorf("%s offered with %.1f GiB usable against a %.1f GiB requirement",
				c.InstanceType, have, need)
		}
		if !isNVIDIA(c.GPUManufacturer) {
			t.Errorf("%s offered with a %s GPU, which cannot run CUDA", c.InstanceType, c.GPUManufacturer)
		}
		if c.GPUs <= 0 {
			t.Errorf("%s offered with %d whole GPUs", c.InstanceType, c.GPUs)
		}
	}

	// The named negative cases, which are the whole point.
	for _, tooSmall := range []string{"g5.2xlarge", "g5.4xlarge", "g6.4xlarge", "g6e.xlarge"} {
		for _, c := range sel.Candidates {
			if c.InstanceType != tooSmall {
				continue
			}
			if c.Usable() {
				t.Errorf("%s (%s usable) was offered for a %s model", tooSmall, c.UsableMemory, sizing.Total)
			}
			if c.Rejection != RejectInsufficientVRAM {
				t.Errorf("%s rejection is %q, want %q; it is being turned away for the wrong reason",
					tooSmall, c.Rejection, RejectInsufficientVRAM)
			}
		}
	}
}

// TestLiveDiscoveryIsNotSilentlyFilteredByTruffle guards the interaction described on
// [anyInstanceType]: truffle's extractSpecificTypes turns a wildcard-free pattern into
// a DescribeInstanceTypes InstanceTypes filter. If ".*" ever stopped being recognized
// as a wildcard, the query would ask EC2 for a type literally named ".*" and return
// nothing — which reads as "no instance can serve this model".
func TestLiveDiscoveryIsNotSilentlyFilteredByTruffle(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := liveCtx(t)
	defer cancel()

	all, err := client.SearchInstanceTypes(ctx, []string{liveRegion}, anyInstanceType, truffle.FilterOptions{})
	if err != nil {
		t.Fatalf("SearchInstanceTypes: %v", err)
	}
	t.Logf("%s enumerated %d instance types with pattern %q", liveRegion, len(all), anyInstanceType)

	// us-east-1 had ~900 types on 2026-07-28. Anything in the low tens means the
	// pattern was pushed down as a filter.
	if len(all) < 100 {
		t.Fatalf("pattern %q returned only %d types; truffle is treating it as a specific "+
			"instance-type filter rather than a wildcard, and discovery is silently scoped",
			anyInstanceType, len(all))
	}

	// The pattern must also survive truffle's own wildcard check, independently of the
	// result count — a change there is the mechanism by which this would break.
	if !regexp.MustCompile(`\.\*|\.\+|\[|\(|\|`).MatchString(anyInstanceType.String()) {
		t.Errorf("pattern %q contains none of the metacharacters truffle recognizes as a "+
			"wildcard; it would be pushed down as an InstanceTypes filter", anyInstanceType)
	}
}

// TestLivePricingReachesTheUsableCandidates joins selection to the pricer, which is
// where a fabricated rate would enter. Every usable candidate must come back either
// with a live price or with an explicit unpriced reason — never with a derived guess.
//
// Deliberately scoped to one small region set: this is a Price List call per usable
// type, and the point is the shape of the answer, not a full sweep.
func TestLivePricingReachesTheUsableCandidates(t *testing.T) {
	client := liveClient(t)
	ctx, cancel := liveCtx(t)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}

	m := qwen3ForLive()
	sizing := m.Size(model.SizingRequest{ContextTokens: 4096})
	sel, err := NewSelector(client, ec2.NewPricer(cfg), nil).
		Select(ctx, m, sizing, []string{liveRegion})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}

	usable := sel.Usable()
	if len(usable) == 0 {
		t.Fatal("no usable candidates to price")
	}

	var priced, unpriced int
	for _, c := range usable {
		if err := c.OnDemand.Valid(); err != nil {
			t.Errorf("%s OnDemand is malformed: %v", c.InstanceType, err)
		}
		switch {
		case c.OnDemand.Known():
			priced++
			// A live rate, never a derived or external one: this package must not be
			// reachable by a fallback that estimates.
			if p := c.OnDemand.Provenance(); p != "live" {
				t.Errorf("%s price has provenance %q, want live; a fabricated rate has "+
					"reached the report", c.InstanceType, p)
			}
		default:
			unpriced++
			if c.OnDemand.Source() == "" {
				t.Errorf("%s is unpriced with no reason given", c.InstanceType)
			}
		}
	}
	t.Logf("%d usable candidates: %d priced, %d unpriced", len(usable), priced, unpriced)

	// The cheapest priced candidate is the headline number, so log it for the record.
	if c, ok := sel.Cheapest(); ok {
		t.Logf("cheapest for %s in %s: %s at %s (%d GPUs, tp %d, %s usable)",
			m.ID, liveRegion, c.InstanceType, c.OnDemand, c.GPUs, c.TensorParallel, c.UsableMemory)
	} else {
		t.Error("no usable candidate carries a price; the comparison would have nothing to rank")
	}

	if priced == 0 {
		t.Error("every usable candidate is unpriced; the Price List join is broken")
	}
}

// TestLiveASecondRegionAgreesOnTheHardware checks that the same model resolves to the
// same instance types in us-east-2, since region-to-region differences should be in
// availability and price, not in what a GPU is. A divergence here means one region's
// catalogue reports something differently.
func TestLiveASecondRegionAgreesOnTheHardware(t *testing.T) {
	client := liveClient(t)
	m := qwen3ForLive()
	sizing := m.Size(model.SizingRequest{ContextTokens: 4096})

	ctx, cancel := liveCtx(t)
	defer cancel()
	sel, err := NewSelector(client, nil, nil).Select(ctx, m, sizing,
		[]string{liveRegion, "us-east-2"})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	if len(sel.Failures()) > 0 {
		t.Fatalf("region query failed: %v", sel.Failures())
	}

	// Per-type usable memory must agree wherever a type is offered in both regions.
	type spec struct {
		gpus, tp int
		usable   float64
	}
	seen := map[string]map[string]spec{}
	for _, c := range sel.Candidates {
		if !c.Usable() {
			continue
		}
		usable, _ := c.UsableMemory.Value()
		if seen[c.InstanceType] == nil {
			seen[c.InstanceType] = map[string]spec{}
		}
		seen[c.InstanceType][c.Region] = spec{c.GPUs, c.TensorParallel, usable}
	}
	for typ, byRegion := range seen {
		if len(byRegion) < 2 {
			continue
		}
		var first string
		var want spec
		for region, got := range byRegion {
			if first == "" {
				first, want = region, got
				continue
			}
			if got != want {
				t.Errorf("%s differs between %s (%+v) and %s (%+v); the same hardware is "+
					"being described differently by region", typ, first, want, region, got)
			}
		}
	}

	for _, r := range sel.Regions {
		t.Logf("%s: considered %d, usable %d", r.Region, r.Considered, r.Usable)
	}
}

// TestLiveTensorParallelDegreesMatchRealGPUCounts checks that the sharding logic
// produces a degree that is actually achievable on each live type: it must divide the
// instance's real GPU count, and it must not exceed it.
func TestLiveTensorParallelDegreesMatchRealGPUCounts(t *testing.T) {
	gpu := liveGPUTypes(t, liveRegion)
	m := qwen3ForLive()
	sizing := m.Size(model.SizingRequest{ContextTokens: 4096})
	sel := NewSelector(nil, nil, nil)
	now := time.Now().UTC()

	var counts []int
	for _, it := range gpu {
		c := sel.evaluate(m, sizing, it, now)
		if c.TensorParallel == 0 {
			continue // rejected before sharding was considered
		}
		if c.TensorParallel > int(it.GPUs) {
			t.Errorf("%s: tensor-parallel %d exceeds its %d GPUs", it.InstanceType, c.TensorParallel, it.GPUs)
		}
		if int(it.GPUs)%c.TensorParallel != 0 {
			t.Errorf("%s: tensor-parallel %d does not divide %d GPUs; no engine can be "+
				"launched in that shape", it.InstanceType, c.TensorParallel, it.GPUs)
		}
		counts = append(counts, int(it.GPUs))
	}

	// The GPU counts AWS ships. A count that is not a power of two would restrict
	// which models can shard at all, so it is worth surfacing.
	uniq := map[int]bool{}
	for _, n := range counts {
		uniq[n] = true
	}
	var shapes []int
	for n := range uniq {
		shapes = append(shapes, n)
	}
	sort.Ints(shapes)
	t.Logf("GPU counts in %s: %v", liveRegion, shapes)
	for _, n := range shapes {
		if n&(n-1) != 0 {
			t.Logf("NOTE: %d GPUs is not a power of two; models whose head counts are "+
				"powers of two cannot use all of them", n)
		}
	}
}

// qwen3ForLive is Qwen/Qwen3-32B's published metadata, which the offline tests also
// use. Hardcoded rather than fetched so this suite tests EC2's schema and not
// Hugging Face's — internal/model has its own live suite for that.
func qwen3ForLive() model.Model {
	return model.Model{
		ID:             "Qwen/Qwen3-32B",
		Gate:           model.GateNone,
		Parameters:     map[string]int64{"BF16": 32_762_123_264},
		HasSafetensors: true,
		Architectures:  []string{"Qwen3ForCausalLM"},
		Config: model.Config{
			MaxPositionEmbeddings: 40960, HiddenSize: 5120, NumHiddenLayers: 64,
			NumAttentionHeads: 64, NumKeyValueHeads: 8, HeadDim: 128,
			TorchDType: "bfloat16",
		},
		ObservedAt: time.Now().UTC(),
	}
}

// tinyModel is small enough to fit any GPU, for tests about classification rather
// than capacity.
func tinyModel() model.Model {
	return model.Model{
		ID:             "tiny/model",
		Parameters:     map[string]int64{"BF16": 200_000_000},
		HasSafetensors: true,
		Config: model.Config{
			MaxPositionEmbeddings: 2048, HiddenSize: 512, NumHiddenLayers: 4,
			NumAttentionHeads: 8, NumKeyValueHeads: 8, HeadDim: 64,
		},
		ObservedAt: time.Now().UTC(),
	}
}
