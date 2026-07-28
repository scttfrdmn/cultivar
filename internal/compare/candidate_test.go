package compare

import (
	"context"
	"errors"
	"fmt"
	"regexp"
	"strings"
	"testing"
	"time"

	truffle "github.com/spore-host/truffle/pkg/aws"

	"github.com/scttfrdmn/cultivar/internal/model"
	"github.com/scttfrdmn/cultivar/internal/report"
)

var observed = time.Date(2026, 7, 28, 12, 0, 0, 0, time.UTC)

func clock() time.Time { return observed }

// The instance shapes below are transcribed from live DescribeInstanceTypes output in
// us-east-1 on 2026-07-28, not invented, and the MiB figures are verbatim. The
// fractional and AMD rows in particular are the reason two of the rejections exist,
// so a fixture that rounded them off would test nothing.
var (
	// g6e.xlarge: one L40S, 45776 MiB (44.7 GiB). The ordinary single-GPU case.
	g6eXlarge = truffle.InstanceTypeResult{
		InstanceType: "g6e.xlarge", Region: "us-east-1", VCPUs: 4, MemoryMiB: 32768,
		Architecture: "x86_64", GPUs: 1, GPUMemoryMiB: 45776,
		GPUModel: "L40S", GPUManufacturer: "NVIDIA",
	}
	// g6e.12xlarge: 4x L40S, 183104 MiB aggregate (178.8 GiB). The sharding case.
	g6e12xlarge = truffle.InstanceTypeResult{
		InstanceType: "g6e.12xlarge", Region: "us-east-1", VCPUs: 48, MemoryMiB: 393216,
		Architecture: "x86_64", GPUs: 4, GPUMemoryMiB: 183104,
		GPUModel: "L40S", GPUManufacturer: "NVIDIA",
	}
	// g5.2xlarge: one A10G, 22888 MiB (22.4 GiB). The instance an early draft of this
	// project paired with Qwen3-32B, which is the bug this package exists to prevent.
	g52xlarge = truffle.InstanceTypeResult{
		InstanceType: "g5.2xlarge", Region: "us-east-1", VCPUs: 8, MemoryMiB: 32768,
		Architecture: "x86_64", GPUs: 1, GPUMemoryMiB: 22888,
		GPUModel: "A10G", GPUManufacturer: "NVIDIA",
	}
	// g6f.large: GpuPartitionSize 0.125, Count 0, LogicalGpuCount 1, 2861 MiB. truffle
	// reports GPUs 0 with GPUMemoryMiB falling back to TotalGpuMemoryInMiB.
	g6fLarge = truffle.InstanceTypeResult{
		InstanceType: "g6f.large", Region: "us-east-1", VCPUs: 2, MemoryMiB: 8192,
		Architecture: "x86_64", GPUs: 0, GPUMemoryMiB: 2861,
		GPUModel: "L4", GPUManufacturer: "NVIDIA",
	}
	// g4ad.xlarge: AMD Radeon Pro V520. Cannot run CUDA.
	g4adXlarge = truffle.InstanceTypeResult{
		InstanceType: "g4ad.xlarge", Region: "us-east-1", VCPUs: 4, MemoryMiB: 16384,
		Architecture: "x86_64", GPUs: 1, GPUMemoryMiB: 8192,
		GPUModel: "Radeon Pro V520", GPUManufacturer: "AMD",
	}
	// g5g.2xlarge: NVIDIA T4g on Graviton2. Usable, but arm64.
	g5g2xlarge = truffle.InstanceTypeResult{
		InstanceType: "g5g.2xlarge", Region: "us-east-1", VCPUs: 8, MemoryMiB: 16384,
		Architecture: "arm64", GPUs: 1, GPUMemoryMiB: 16384,
		GPUModel: "T4g", GPUManufacturer: "NVIDIA",
	}
	// m5.large: no GPU at all. Most of the catalogue.
	m5Large = truffle.InstanceTypeResult{
		InstanceType: "m5.large", Region: "us-east-1", VCPUs: 2, MemoryMiB: 8192,
		Architecture: "x86_64",
	}
)

// fakeSearch returns canned results per region, or a canned error.
type fakeSearch struct {
	byRegion map[string][]truffle.InstanceTypeResult
	errs     map[string]error
	calls    [][]string
	patterns []string
}

func (f *fakeSearch) SearchInstanceTypes(_ context.Context, regions []string, matcher *regexp.Regexp, _ truffle.FilterOptions) ([]truffle.InstanceTypeResult, error) {
	f.calls = append(f.calls, regions)
	f.patterns = append(f.patterns, matcher.String())
	// Mirror truffle's contract: an error only when every requested region fails.
	var out []truffle.InstanceTypeResult
	failed := 0
	for _, r := range regions {
		if err, bad := f.errs[r]; bad {
			failed++
			_ = err
			continue
		}
		for _, it := range f.byRegion[r] {
			it.Region = r
			out = append(out, it)
		}
	}
	if failed == len(regions) && len(regions) > 0 {
		var joined []error
		for _, r := range regions {
			joined = append(joined, f.errs[r])
		}
		return out, fmt.Errorf("all %d region queries failed: %w", len(regions), errors.Join(joined...))
	}
	return out, nil
}

// fakePricer prices from a table. A type absent from the table is unpriced with a nil
// error, which is how ec2.Pricer reports p5e.48xlarge.
type fakePricer struct {
	rates map[string]float64
	err   error
	calls int
}

func (f *fakePricer) OnDemand(_ context.Context, instanceType, region string) (report.Amount, error) {
	f.calls++
	if f.err != nil {
		return report.Unavailable(report.UnitUSDPerHour, "lookup failed"), f.err
	}
	rate, ok := f.rates[instanceType]
	if !ok {
		return report.Unavailable(report.UnitUSDPerHour,
			"no on-demand price for "+instanceType+" in "+region), nil
	}
	return report.Live(rate, report.UnitUSDPerHour, "fixture", observed), nil
}

// qwen3_32B is Qwen/Qwen3-32B as the HF API reports it: 32.8B BF16 parameters, 64
// attention heads, 8 KV heads, explicit head_dim 128.
func qwen3_32B() model.Model {
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
		ObservedAt: observed,
	}
}

// sizeAt sizes a model at a context length with everything else defaulted.
func sizeAt(m model.Model, ctx int) model.Sizing {
	return m.Size(model.SizingRequest{ContextTokens: ctx})
}

func selectAll(t *testing.T, m model.Model, sizing model.Sizing, types []truffle.InstanceTypeResult, pricer onDemandPricer) *Selection {
	t.Helper()
	search := &fakeSearch{byRegion: map[string][]truffle.InstanceTypeResult{"us-east-1": types}}
	sel, err := NewSelector(search, pricer, clock).Select(context.Background(), m, sizing, []string{"us-east-1"})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	return sel
}

// find returns the candidate for an instance type.
func find(t *testing.T, sel *Selection, instanceType string) Candidate {
	t.Helper()
	for _, c := range sel.Candidates {
		if c.InstanceType == instanceType {
			return c
		}
	}
	t.Fatalf("no candidate for %s; got %d candidates", instanceType, len(sel.Candidates))
	return Candidate{}
}

// TestAModelIsNeverPairedWithAnInstanceTooSmallForIt is the regression test for the
// bug that motivated the whole package: Qwen3-32B (61 GiB at 4k context) on a
// g5.2xlarge (22.9 GiB). If this ever passes as usable, the tool recommends hardware
// that cannot load the weights.
func TestAModelIsNeverPairedWithAnInstanceTooSmallForIt(t *testing.T) {
	m := qwen3_32B()
	sizing := sizeAt(m, 4096)
	sel := selectAll(t, m, sizing, []truffle.InstanceTypeResult{g52xlarge, g6eXlarge}, nil)

	small := find(t, sel, "g5.2xlarge")
	if small.Usable() {
		t.Errorf("g5.2xlarge (22.9 GiB) accepted for a %s model", sizing.Total)
	}
	if small.Rejection != RejectInsufficientVRAM {
		t.Errorf("rejection was %q, want %q", small.Rejection, RejectInsufficientVRAM)
	}
	// The reason must carry both figures: a bare "does not fit" gives a reader no way
	// to tell a near miss from an order of magnitude.
	if !strings.Contains(small.Reason, "22.4") || !strings.Contains(small.Reason, "needs") {
		t.Errorf("reason %q does not state what it needs against what it has", small.Reason)
	}

	// The 44.7 GiB L40S does not fit 61 GiB either — which is the honest answer, and
	// the reason the 4-GPU case below matters.
	if c := find(t, sel, "g6e.xlarge"); c.Usable() {
		t.Errorf("g6e.xlarge (44.7 GiB) accepted for a %s model", sizing.Total)
	}
}

// TestShardingAcrossGPUsIsAValidFit is the other half: a model too big for one GPU is
// still servable on an instance whose GPUs can be combined. Dropping these would
// leave large models looking unservable on hardware that serves them.
func TestShardingAcrossGPUsIsAValidFit(t *testing.T) {
	m := qwen3_32B()
	sizing := sizeAt(m, 4096)
	sel := selectAll(t, m, sizing, []truffle.InstanceTypeResult{g6eXlarge, g6e12xlarge}, nil)

	c := find(t, sel, "g6e.12xlarge")
	if !c.Usable() {
		t.Fatalf("g6e.12xlarge (4x L40S, 178.8 GiB) rejected for a %s model: %s", sizing.Total, c.Reason)
	}
	if c.TensorParallel != 4 {
		t.Errorf("TensorParallel = %d, want 4 (64 heads and 8 KV heads both divide by 4)", c.TensorParallel)
	}
	usable, ok := c.UsableMemory.Value()
	if !ok {
		t.Fatalf("UsableMemory is unavailable: %s", c.UsableMemory.Source())
	}
	// Derived from the fixture's own MiB figure, which is the *input* here — the
	// arithmetic is what is under test, so this cannot drift when AWS changes a card.
	if want := float64(g6e12xlarge.GPUMemoryMiB) / mibPerGiB; usable < want-0.1 || usable > want+0.1 {
		t.Errorf("UsableMemory = %.1f GiB, want %.1f (all four GPUs)", usable, want)
	}
	// UsableMemory is computed, not read, so it must not claim to be live.
	if p := c.UsableMemory.Provenance(); p != report.ProvenanceDerived {
		t.Errorf("UsableMemory provenance is %q, want derived", p)
	}
}

// TestPerGPUMemoryBoundsAModelThatCannotShard checks that aggregate VRAM alone does
// not decide the fit. A model whose heads pin it to one GPU sees only that GPU's
// memory, however many the instance has.
func TestPerGPUMemoryBoundsAModelThatCannotShard(t *testing.T) {
	m := qwen3_32B()
	// 7 attention heads: prime, so no tp above 1 divides it. Contrived, but the shape
	// is real — models with head counts not divisible by 8 exist, and on an 8-GPU box
	// they run on one GPU or not at all.
	m.Config.NumAttentionHeads = 7
	m.Config.NumKeyValueHeads = 7
	sizing := sizeAt(m, 4096)

	sel := selectAll(t, m, sizing, []truffle.InstanceTypeResult{g6e12xlarge}, nil)
	c := find(t, sel, "g6e.12xlarge")

	if c.TensorParallel != 1 {
		t.Errorf("TensorParallel = %d, want 1 (7 heads divide by nothing else)", c.TensorParallel)
	}
	usable, ok := c.UsableMemory.Value()
	if !ok {
		t.Fatalf("UsableMemory unavailable: %s", c.UsableMemory.Source())
	}
	if want := float64(g6e12xlarge.GPUMemoryMiB) / mibPerGiB / 4; usable < want-0.1 || usable > want+0.1 {
		t.Errorf("UsableMemory = %.1f GiB, want %.1f (one of four GPUs); "+
			"aggregate VRAM was used where per-GPU was required", usable, want)
	}
	// The rejection must explain that the GPUs went unused, or a reader sees a
	// 178.8 GiB instance refusing a 61 GiB model with no explanation.
	if c.Usable() {
		t.Fatal("a 61 GiB model was accepted on a single 44.7 GiB GPU")
	}
	if !strings.Contains(c.Reason, "1 of 4 GPUs") {
		t.Errorf("reason %q does not say that only one GPU is usable", c.Reason)
	}
}

// TestFractionalGPUTypesAreNotOffered covers the g6f/gr6f family: Count 0 with
// non-zero TotalGpuMemoryInMiB. Two ways to get this wrong — divide by the count, or
// treat the partition as servable VRAM — and this asserts neither happens.
func TestFractionalGPUTypesAreNotOffered(t *testing.T) {
	m := qwen3_32B()
	// Sized so 2.8 GiB is genuinely too small either way; the point is the *reason*.
	sizing := sizeAt(m, 4096)
	sel := selectAll(t, m, sizing, []truffle.InstanceTypeResult{g6fLarge}, nil)

	c := find(t, sel, "g6f.large")
	if c.Usable() {
		t.Fatal("a 0.125-GPU partition was offered as a serving candidate")
	}
	if c.Rejection != RejectFractionalGPU {
		t.Errorf("rejection was %q, want %q; the fractional case was not recognized",
			c.Rejection, RejectFractionalGPU)
	}
	// PerGPUMemory must be unavailable, not +Inf or NaN from a divide by zero.
	if v, ok := c.PerGPUMemory.Value(); ok {
		t.Errorf("PerGPUMemory = %v for a type with 0 GPUs; the count was used as a divisor", v)
	}
	if err := c.PerGPUMemory.Valid(); err != nil {
		t.Errorf("PerGPUMemory is not a well-formed amount: %v", err)
	}
	// The 2.8 GiB is still reported, because it is a real measurement and a reader
	// checking against the console should find it.
	if v, ok := c.GPUMemory.Value(); !ok || v < 2.7 || v > 2.9 {
		t.Errorf("GPUMemory = %v (%v), want ~2.8 GiB from TotalGpuMemoryInMiB", v, c.GPUMemory)
	}
}

// TestFractionalGPUIsRejectedEvenWhenTheModelWouldFit is the sharper version: a tiny
// model *would* fit in 2.8 GiB, so only the fractional rejection keeps it out. If the
// guard were ordered after the VRAM check, this would come back usable and a user
// would be sent to a graphics partition that cannot host a server.
func TestFractionalGPUIsRejectedEvenWhenTheModelWouldFit(t *testing.T) {
	tiny := model.Model{
		ID:             "tiny/model",
		Parameters:     map[string]int64{"BF16": 200_000_000},
		HasSafetensors: true,
		Config: model.Config{
			MaxPositionEmbeddings: 2048, HiddenSize: 512, NumHiddenLayers: 4,
			NumAttentionHeads: 8, NumKeyValueHeads: 8, HeadDim: 64,
		},
		ObservedAt: observed,
	}
	sizing := sizeAt(tiny, 512)
	if need, _ := sizing.Total.Value(); need > 2.8 {
		t.Fatalf("fixture is not small enough to test the ordering: needs %.2f GiB", need)
	}

	sel := selectAll(t, tiny, sizing, []truffle.InstanceTypeResult{g6fLarge}, nil)
	c := find(t, sel, "g6f.large")
	if c.Usable() {
		t.Fatalf("a %s model was placed on a 0.125-GPU partition because it happened to fit",
			sizing.Total)
	}
	if c.Rejection != RejectFractionalGPU {
		t.Errorf("rejection was %q, want %q", c.Rejection, RejectFractionalGPU)
	}
}

// TestNonNVIDIAGPUsAreRejected covers g4ad. Its 8 GiB is real memory on a real GPU;
// only the manufacturer disqualifies it, so a fit check alone would accept it for a
// small model and produce a launch where no engine starts.
func TestNonNVIDIAGPUsAreRejected(t *testing.T) {
	tiny := model.Model{
		ID:             "tiny/model",
		Parameters:     map[string]int64{"BF16": 200_000_000},
		HasSafetensors: true,
		Config: model.Config{
			MaxPositionEmbeddings: 2048, HiddenSize: 512, NumHiddenLayers: 4,
			NumAttentionHeads: 8, NumKeyValueHeads: 8, HeadDim: 64,
		},
		ObservedAt: observed,
	}
	sel := selectAll(t, tiny, sizeAt(tiny, 512),
		[]truffle.InstanceTypeResult{g4adXlarge}, nil)

	c := find(t, sel, "g4ad.xlarge")
	if c.Usable() {
		t.Fatal("an AMD Radeon Pro V520 was offered for a CUDA-only engine")
	}
	if c.Rejection != RejectNotCUDA {
		t.Errorf("rejection was %q, want %q", c.Rejection, RejectNotCUDA)
	}
	if !strings.Contains(c.Reason, "CUDA") {
		t.Errorf("reason %q does not mention CUDA", c.Reason)
	}
}

// TestManufacturerMatchingIsCaseInsensitive guards the other direction: truffle
// reports whatever EC2 sends, and a casing change must not silently reject every
// NVIDIA type and leave the tool concluding no GPU can serve anything.
func TestManufacturerMatchingIsCaseInsensitive(t *testing.T) {
	for _, spelling := range []string{"NVIDIA", "nvidia", "Nvidia", " NVIDIA "} {
		it := g6eXlarge
		it.GPUManufacturer = spelling
		m := qwen3_32B()
		sel := selectAll(t, m, sizeAt(m, 1024), []truffle.InstanceTypeResult{it}, nil)
		if c := find(t, sel, "g6e.xlarge"); c.Rejection == RejectNotCUDA {
			t.Errorf("manufacturer %q was rejected as non-CUDA", spelling)
		}
	}
}

// TestArmGPUTypesAreUsableButMarked: g5g is a genuine NVIDIA GPU on arm64. Rejecting
// it would drop a working option; not recording the architecture would hand a user an
// instance whose container image does not exist.
func TestArmGPUTypesAreUsableButMarked(t *testing.T) {
	tiny := model.Model{
		ID:             "tiny/model",
		Parameters:     map[string]int64{"BF16": 1_000_000_000},
		HasSafetensors: true,
		Config: model.Config{
			MaxPositionEmbeddings: 2048, HiddenSize: 2048, NumHiddenLayers: 16,
			NumAttentionHeads: 16, NumKeyValueHeads: 8, HeadDim: 128,
		},
		ObservedAt: observed,
	}
	sel := selectAll(t, tiny, sizeAt(tiny, 2048),
		[]truffle.InstanceTypeResult{g5g2xlarge}, nil)

	c := find(t, sel, "g5g.2xlarge")
	if !c.Usable() {
		t.Fatalf("g5g.2xlarge (NVIDIA T4g, 16 GiB) rejected for a %s model: %s",
			sizeAt(tiny, 2048).Total, c.Reason)
	}
	if c.Architecture != "arm64" {
		t.Errorf("Architecture = %q, want arm64; a caller cannot warn about the image", c.Architecture)
	}
}

// TestNonGPUTypesAreRejectedNotDropped: the ~900 CPU types must be accounted for, so
// that "considered 900, usable 2" is distinguishable from a query that returned two
// rows.
func TestNonGPUTypesAreRejectedNotDropped(t *testing.T) {
	m := qwen3_32B()
	sel := selectAll(t, m, sizeAt(m, 1024),
		[]truffle.InstanceTypeResult{m5Large, g6e12xlarge}, nil)

	if len(sel.Candidates) != 2 {
		t.Fatalf("got %d candidates, want 2; a type was dropped rather than rejected", len(sel.Candidates))
	}
	if c := find(t, sel, "m5.large"); c.Rejection != RejectNoGPU {
		t.Errorf("m5.large rejection was %q, want %q", c.Rejection, RejectNoGPU)
	}
	if got := sel.Regions[0].Considered; got != 2 {
		t.Errorf("Considered = %d, want 2", got)
	}
	if got := sel.Regions[0].Usable; got != 1 {
		t.Errorf("Usable = %d, want 1", got)
	}
}

// TestAnUnsizableModelRecommendsNothing: gpt-oss-120b with an unrecognized dtype
// cannot be sized, and the correct output is a refusal that names the model. A
// zero requirement would make every GPU look like it fits.
func TestAnUnsizableModelRecommendsNothing(t *testing.T) {
	m := qwen3_32B()
	m.Parameters = map[string]int64{"MX9": 120_000_000_000} // not a dtype this code knows
	sizing := sizeAt(m, 4096)
	if sizing.Total.Known() {
		t.Fatal("fixture sized successfully; the dtype guard did not fire")
	}

	sel := selectAll(t, m, sizing,
		[]truffle.InstanceTypeResult{g6eXlarge, g6e12xlarge}, nil)

	for _, c := range sel.Candidates {
		if c.Usable() {
			t.Errorf("%s was recommended for an unsizable model", c.InstanceType)
		}
		if c.Rejection != RejectUnknownRequirement {
			t.Errorf("%s rejection was %q, want %q", c.InstanceType, c.Rejection, RejectUnknownRequirement)
		}
		if !strings.Contains(c.Reason, m.ID) {
			t.Errorf("%s reason %q does not name the model, so a reader would blame the hardware",
				c.InstanceType, c.Reason)
		}
	}
}

// TestUnknownGPUMemoryIsNotZero: a GPU type reporting no memory size must not be
// treated as having none, which would read as a rejection for being too small rather
// than for being unmeasurable.
func TestUnknownGPUMemoryIsNotZero(t *testing.T) {
	it := g6eXlarge
	it.GPUMemoryMiB = 0
	m := qwen3_32B()
	sel := selectAll(t, m, sizeAt(m, 1024), []truffle.InstanceTypeResult{it}, nil)

	c := find(t, sel, "g6e.xlarge")
	if c.Rejection != RejectUnknownGPUMemory {
		t.Errorf("rejection was %q, want %q", c.Rejection, RejectUnknownGPUMemory)
	}
	if v, ok := c.GPUMemory.Value(); ok {
		t.Errorf("GPUMemory = %v, want unavailable", v)
	}
}

// TestCheapestUsableWins checks the ranking, and that an unpriced candidate is never
// presented as the cheapest — the inversion where "unpriced" at the top of a sorted
// list reads as free.
func TestCheapestUsableWins(t *testing.T) {
	m := qwen3_32B()
	sizing := sizeAt(m, 4096)

	// p5.48xlarge: 8x H100, 640 GiB. Fits, and is by far the most expensive.
	p548 := truffle.InstanceTypeResult{
		InstanceType: "p5.48xlarge", Region: "us-east-1", VCPUs: 192, MemoryMiB: 2097152,
		Architecture: "x86_64", GPUs: 8, GPUMemoryMiB: 655360,
		GPUModel: "H100", GPUManufacturer: "NVIDIA",
	}
	// p5e.48xlarge: 8x H200, and genuinely has no on-demand price.
	p5e48 := truffle.InstanceTypeResult{
		InstanceType: "p5e.48xlarge", Region: "us-east-1", VCPUs: 192, MemoryMiB: 2097152,
		Architecture: "x86_64", GPUs: 8, GPUMemoryMiB: 1146880,
		GPUModel: "H200", GPUManufacturer: "NVIDIA",
	}

	pricer := &fakePricer{rates: map[string]float64{
		"g6e.12xlarge": 10.4901,
		"p5.48xlarge":  55.04,
		// p5e.48xlarge deliberately absent: unpriced with a nil error.
	}}
	sel := selectAll(t, m, sizing,
		[]truffle.InstanceTypeResult{p548, p5e48, g6e12xlarge, g52xlarge}, pricer)

	usable := sel.Usable()
	if len(usable) != 3 {
		t.Fatalf("got %d usable candidates, want 3: %v", len(usable), names(usable))
	}
	if usable[0].InstanceType != "g6e.12xlarge" {
		t.Errorf("cheapest usable is %s, want g6e.12xlarge", usable[0].InstanceType)
	}
	// The unpriced one sorts last among usable candidates, not first.
	if last := usable[len(usable)-1]; last.InstanceType != "p5e.48xlarge" {
		t.Errorf("last usable is %s, want p5e.48xlarge (unpriced sorts last)", last.InstanceType)
	}

	cheapest, ok := sel.Cheapest()
	if !ok {
		t.Fatal("Cheapest reported no priced candidate")
	}
	if cheapest.InstanceType != "g6e.12xlarge" {
		t.Errorf("Cheapest = %s, want g6e.12xlarge", cheapest.InstanceType)
	}
}

// TestCheapestRefusesToNameAnUnpricedCandidate: when the only servable option has no
// price, Cheapest must report that rather than returning it. "Cheapest: p5e.48xlarge
// (unpriced)" would read as free for a machine that costs $47.76/hr through a
// capacity block.
func TestCheapestRefusesToNameAnUnpricedCandidate(t *testing.T) {
	m := qwen3_32B()
	p5e48 := truffle.InstanceTypeResult{
		InstanceType: "p5e.48xlarge", Region: "us-east-1", VCPUs: 192, MemoryMiB: 2097152,
		Architecture: "x86_64", GPUs: 8, GPUMemoryMiB: 1146880,
		GPUModel: "H200", GPUManufacturer: "NVIDIA",
	}
	sel := selectAll(t, m, sizeAt(m, 4096),
		[]truffle.InstanceTypeResult{p5e48}, &fakePricer{rates: map[string]float64{}})

	if len(sel.Usable()) != 1 {
		t.Fatalf("got %d usable, want 1", len(sel.Usable()))
	}
	if c, ok := sel.Cheapest(); ok {
		t.Errorf("Cheapest returned %s with an unavailable price; it would render as free",
			c.InstanceType)
	}
}

// TestRejectedCandidatesAreNotPriced: pricing is an API call per type and region, and
// a type that cannot load the model does not need one. With ~900 types per region,
// pricing them all would make a comparison unusably slow.
func TestRejectedCandidatesAreNotPriced(t *testing.T) {
	m := qwen3_32B()
	pricer := &fakePricer{rates: map[string]float64{"g6e.12xlarge": 10.4901}}
	sel := selectAll(t, m, sizeAt(m, 4096),
		[]truffle.InstanceTypeResult{m5Large, g52xlarge, g4adXlarge, g6fLarge, g6e12xlarge}, pricer)

	if pricer.calls != 1 {
		t.Errorf("pricer called %d times, want 1 (only the usable candidate)", pricer.calls)
	}
	// A rejected candidate still carries a well-formed unpriced amount, not a zero
	// Amount that would fail validation on the way out.
	for _, c := range sel.Candidates {
		if c.Usable() {
			continue
		}
		if err := c.OnDemand.Valid(); err != nil {
			t.Errorf("%s OnDemand is malformed: %v", c.InstanceType, err)
		}
		if c.OnDemand.Known() {
			t.Errorf("%s is rejected but carries a price", c.InstanceType)
		}
	}
}

// TestDuplicateTypeAndRegionIsPricedOnce guards the memo. The same type appearing
// twice in one region should not double the API calls.
func TestDuplicateTypeAndRegionIsPricedOnce(t *testing.T) {
	m := qwen3_32B()
	pricer := &fakePricer{rates: map[string]float64{"g6e.12xlarge": 10.4901}}
	selectAll(t, m, sizeAt(m, 4096),
		[]truffle.InstanceTypeResult{g6e12xlarge, g6e12xlarge}, pricer)

	if pricer.calls != 1 {
		t.Errorf("pricer called %d times for one type in one region, want 1", pricer.calls)
	}
}

// TestAPricingFailureIsNotAnAbsentPrice: a throttle or expired credential must not
// render as "unpriced". A ranking assembled from whichever lookups survived an outage
// is not a ranking.
func TestAPricingFailureIsNotAnAbsentPrice(t *testing.T) {
	m := qwen3_32B()
	search := &fakeSearch{byRegion: map[string][]truffle.InstanceTypeResult{
		"us-east-1": {g6e12xlarge},
	}}
	boom := errors.New("ThrottlingException: rate exceeded")
	sel, err := NewSelector(search, &fakePricer{err: boom}, clock).
		Select(context.Background(), m, sizeAt(m, 4096), []string{"us-east-1"})

	if err == nil {
		t.Fatal("a pricing failure was swallowed; the candidate would read as unpriced")
	}
	if !errors.Is(err, boom) {
		t.Errorf("error %v does not wrap the underlying failure", err)
	}
	// The partial selection still comes back, so a caller can report what was learned.
	if sel == nil {
		t.Fatal("Select returned no selection alongside the error")
	}
}

// TestAFailedRegionIsDistinguishableFromAnEmptyOne is the reason regions are queried
// one at a time. truffle only errors when every region fails, so a multi-region call
// would report AccessDenied in us-east-2 as "no GPUs in us-east-2" — and send the
// user to a pricier region.
func TestAFailedRegionIsDistinguishableFromAnEmptyOne(t *testing.T) {
	m := qwen3_32B()
	denied := errors.New("AccessDenied: not authorized to perform ec2:DescribeInstanceTypes")
	search := &fakeSearch{
		byRegion: map[string][]truffle.InstanceTypeResult{
			"us-east-1": {g6e12xlarge},
			"us-west-1": {m5Large}, // genuinely offers no modern GPU family
		},
		errs: map[string]error{"us-east-2": denied},
	}
	sel, err := NewSelector(search, nil, clock).Select(context.Background(), m,
		sizeAt(m, 4096), []string{"us-east-1", "us-east-2", "us-west-1"})
	if err != nil {
		t.Fatalf("Select errored on a partial failure: %v", err)
	}

	// One call per region, so the failure attributes to us-east-2 rather than to the
	// whole batch.
	if len(search.calls) != 3 {
		t.Errorf("made %d search calls, want 3 (one per region)", len(search.calls))
	}
	for _, call := range search.calls {
		if len(call) != 1 {
			t.Errorf("a search call requested %d regions, want 1; a failure would not be attributable", len(call))
		}
	}

	byRegion := map[string]RegionStatus{}
	for _, r := range sel.Regions {
		byRegion[r.Region] = r
	}
	if got := byRegion["us-east-2"]; got.Err == nil {
		t.Error("us-east-2 recorded no error; a denied region is indistinguishable from an empty one")
	}
	// us-west-1 succeeded and found nothing usable, which is a different claim.
	west := byRegion["us-west-1"]
	if west.Err != nil {
		t.Errorf("us-west-1 recorded an error: %v", west.Err)
	}
	if west.Considered != 1 || west.Usable != 0 {
		t.Errorf("us-west-1 status = considered %d usable %d, want 1 and 0", west.Considered, west.Usable)
	}
	if len(sel.Failures()) != 1 {
		t.Errorf("Failures() has %d entries, want 1", len(sel.Failures()))
	}
}

// TestEveryRegionFailingIsAnError: no information at all is not an answer of "nothing
// fits". The error must name each region, or a user cannot tell which credential or
// permission to fix.
func TestEveryRegionFailingIsAnError(t *testing.T) {
	m := qwen3_32B()
	denied := errors.New("AccessDenied")
	throttled := errors.New("ThrottlingException")
	search := &fakeSearch{errs: map[string]error{"us-east-1": denied, "us-east-2": throttled}}

	sel, err := NewSelector(search, nil, clock).Select(context.Background(), m,
		sizeAt(m, 4096), []string{"us-east-1", "us-east-2"})
	if err == nil {
		t.Fatal("all regions failed but Select succeeded; the result reads as 'nothing fits'")
	}
	if !errors.Is(err, denied) || !errors.Is(err, throttled) {
		t.Errorf("error %v does not wrap both underlying failures", err)
	}
	for _, region := range []string{"us-east-1", "us-east-2"} {
		if !strings.Contains(err.Error(), region) {
			t.Errorf("error %q does not name %s", err, region)
		}
	}
	if len(sel.Failures()) != 2 {
		t.Errorf("Failures() has %d entries, want 2", len(sel.Failures()))
	}
}

// TestNoRegionsIsAnError: an empty region set would otherwise produce an empty
// selection, which reads as "nothing fits anywhere".
func TestNoRegionsIsAnError(t *testing.T) {
	m := qwen3_32B()
	_, err := NewSelector(&fakeSearch{}, nil, clock).
		Select(context.Background(), m, sizeAt(m, 4096), nil)
	if err == nil {
		t.Fatal("an empty region set was accepted")
	}
}

// TestDiscoveryIsNotFilteredByInstanceName is the no-hardcoded-families invariant.
// The search pattern must not encode a family list, and it must be one truffle
// recognizes as a wildcard — otherwise truffle pushes it down as a
// DescribeInstanceTypes InstanceTypes filter and asks EC2 for a type by that literal
// name.
func TestDiscoveryIsNotFilteredByInstanceName(t *testing.T) {
	m := qwen3_32B()
	search := &fakeSearch{byRegion: map[string][]truffle.InstanceTypeResult{
		"us-east-1": {g6e12xlarge},
	}}
	if _, err := NewSelector(search, nil, clock).Select(context.Background(), m,
		sizeAt(m, 4096), []string{"us-east-1"}); err != nil {
		t.Fatalf("Select: %v", err)
	}

	if len(search.patterns) == 0 {
		t.Fatal("no search was made")
	}
	pattern := search.patterns[0]
	if pattern != ".*" {
		t.Errorf("search pattern is %q, want %q; discovery is scoped by name", pattern, ".*")
	}
	// The pattern must match families that do not exist yet, including one whose name
	// starts with neither g nor p. Every GPU type in the catalogue today does start
	// with one of those letters, so a pattern anchored on them would pass a test that
	// only checked current types — and fail silently the day AWS ships an accelerator
	// named otherwise.
	re := regexp.MustCompile(pattern)
	for _, future := range []string{"g9e.4xlarge", "gr6.4xlarge", "zz1.48xlarge"} {
		if !re.MatchString(future) {
			t.Errorf("pattern %q does not match %s; a new GPU family would be invisible", pattern, future)
		}
	}
}

func TestTensorParallelSizes(t *testing.T) {
	tests := []struct {
		name             string
		gpus, heads, kvs int
		want             []int
	}{
		{"single GPU needs no divisibility", 1, 64, 8, []int{1}},
		{"Qwen3-32B on 4 GPUs", 4, 64, 8, []int{1, 2, 4}},
		{"Qwen3-32B on 8 GPUs", 8, 64, 8, []int{1, 2, 4, 8}},
		// Qwen3-235B-A22B: 4 KV heads across 8 GPUs works only by replicating each
		// head twice, which tp % kvHeads == 0 permits.
		{"Qwen3-235B replicates 4 KV heads at tp 8", 8, 64, 4, []int{1, 2, 4, 8}},
		// 6 KV heads at tp 4: 4 neither divides 6 nor is a multiple of it, so there is
		// no valid assignment. tp 2 is fine (3 heads each).
		{"6 KV heads cannot use tp 4", 4, 24, 6, []int{1, 2}},
		// A prime head count pins the model to one GPU however many are present.
		{"prime head count pins to one GPU", 8, 7, 7, []int{1}},
		// tp must divide the GPU count. The absence of 4 is the assertion: 24 heads and
		// 8 KV heads both divide by 4, so only the GPU count rules it out.
		{"tp must divide the GPU count", 6, 24, 6, []int{1, 2, 3, 6}},
		// Unpublished head counts: one GPU is the only safe claim.
		{"no head counts yields tp 1 only", 8, 0, 0, []int{1}},
		{"no attention heads yields tp 1 only", 8, 0, 8, []int{1}},
		// KV heads absent but attention heads present: the attention constraint still
		// applies, and KV replication cannot be checked so it is not blocked on.
		{"kv absent falls back to attention heads", 8, 64, 0, []int{1, 2, 4, 8}},
		{"zero GPUs has no valid degree", 0, 64, 8, nil},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tensorParallelSizes(tc.gpus, tc.heads, tc.kvs)
			if fmt.Sprint(got) != fmt.Sprint(tc.want) {
				t.Errorf("tensorParallelSizes(%d, %d, %d) = %v, want %v",
					tc.gpus, tc.heads, tc.kvs, got, tc.want)
			}
		})
	}
}

// TestQwen3_235BShardsAcrossEightGPUs is the live case behind the KV replication
// rule. If tp were required to divide the KV head count, this model would be capped
// at tp 4 and would not fit on a p5.48xlarge — a false negative on the one instance
// type that serves it.
func TestQwen3_235BShardsAcrossEightGPUs(t *testing.T) {
	m := model.Model{
		ID:             "Qwen/Qwen3-235B-A22B",
		Parameters:     map[string]int64{"BF16": 235_000_000_000},
		HasSafetensors: true,
		Config: model.Config{
			MaxPositionEmbeddings: 40960, HiddenSize: 4096, NumHiddenLayers: 94,
			NumAttentionHeads: 64, NumKeyValueHeads: 4, HeadDim: 128,
			TorchDType: "bfloat16",
		},
		ObservedAt: observed,
	}
	// 8x H200 = 1120 GiB, enough for 470 GiB of BF16 weights plus cache.
	p5e48 := truffle.InstanceTypeResult{
		InstanceType: "p5e.48xlarge", Region: "us-east-1", VCPUs: 192, MemoryMiB: 2097152,
		Architecture: "x86_64", GPUs: 8, GPUMemoryMiB: 1146880,
		GPUModel: "H200", GPUManufacturer: "NVIDIA",
	}
	sel := selectAll(t, m, sizeAt(m, 4096), []truffle.InstanceTypeResult{p5e48}, nil)

	c := find(t, sel, "p5e.48xlarge")
	if !c.Usable() {
		t.Fatalf("p5e.48xlarge rejected for Qwen3-235B: %s", c.Reason)
	}
	if c.TensorParallel != 8 {
		t.Errorf("TensorParallel = %d, want 8; 4 KV heads replicate across 8 GPUs", c.TensorParallel)
	}
}

// TestEveryAmountIsWellFormed: an Amount with no provenance serializes with an empty
// provenance field and fails report validation. Since candidates flow into
// report.v1.json, every one of them must be constructed, on every path.
func TestEveryAmountIsWellFormed(t *testing.T) {
	m := qwen3_32B()
	unsizable := qwen3_32B()
	unsizable.Parameters = map[string]int64{"MX9": 1}

	for _, tc := range []struct {
		name  string
		model model.Model
	}{{"sizable", m}, {"unsizable", unsizable}} {
		t.Run(tc.name, func(t *testing.T) {
			pricer := &fakePricer{rates: map[string]float64{"g6e.12xlarge": 10.4901}}
			sel := selectAll(t, tc.model, sizeAt(tc.model, 4096), []truffle.InstanceTypeResult{
				m5Large, g52xlarge, g4adXlarge, g6fLarge, g5g2xlarge, g6eXlarge, g6e12xlarge,
			}, pricer)

			for _, c := range sel.Candidates {
				amounts := map[string]report.Amount{
					"GPUMemory": c.GPUMemory, "PerGPUMemory": c.PerGPUMemory,
					"UsableMemory": c.UsableMemory, "HostMemory": c.HostMemory,
					"OnDemand": c.OnDemand, "Headroom": c.Headroom,
				}
				for field, a := range amounts {
					if err := a.Valid(); err != nil {
						t.Errorf("%s.%s is malformed: %v", c.InstanceType, field, err)
					}
				}
				if !c.Usable() && c.Reason == "" {
					t.Errorf("%s is rejected as %q with no reason", c.InstanceType, c.Rejection)
				}
				if c.Usable() && c.Reason != "" {
					t.Errorf("%s is usable but carries a reason: %q", c.InstanceType, c.Reason)
				}
			}
		})
	}
}

// TestOrderingDoesNotDependOnDiscoveryOrder: a report is diffed against the previous
// one in the history log, so two runs over the same catalogue must produce the same
// order even when discovery hands the types over differently.
//
// The input is permuted rather than fixed, because a fixed input would be ordered
// deterministically by sort.SliceStable alone and the tiebreak would never be
// exercised. Discovery order genuinely does vary: truffle's SearchInstanceTypes fans
// regions out across goroutines and appends each one's results under a mutex, so the
// order it returns is whatever the scheduler produced.
func TestOrderingDoesNotDependOnDiscoveryOrder(t *testing.T) {
	m := qwen3_32B()
	types := []truffle.InstanceTypeResult{
		m5Large, g52xlarge, g4adXlarge, g6fLarge, g5g2xlarge, g6eXlarge, g6e12xlarge,
	}
	// Two priced usable candidates at an identical rate, so the tiebreak — not the
	// price — is what has to settle their relative order.
	rates := map[string]float64{"g6e.12xlarge": 10.4901, "g6e.xlarge": 10.4901}

	var first []string
	for run := 0; run < len(types); run++ {
		// Rotate the slice: every type takes a turn at being discovered first.
		permuted := append(append([]truffle.InstanceTypeResult{}, types[run:]...), types[:run]...)
		sel := selectAll(t, m, sizeAt(m, 1024), permuted, &fakePricer{rates: rates})
		got := names(sel.Candidates)
		if run == 0 {
			first = got
			continue
		}
		if fmt.Sprint(got) != fmt.Sprint(first) {
			t.Fatalf("discovery order %d produced %v, but the first ordering was %v;\n"+
				"the candidate order depends on the order AWS happened to return types in",
				run, got, first)
		}
	}
}

// TestHeadroomIsAFractionOfWhatIsReachable: headroom must be measured against the
// memory actually usable at the chosen tensor-parallel degree, not the instance's
// aggregate. Otherwise a model pinned to one of eight GPUs looks like it has 87%
// headroom when it has none.
func TestHeadroomIsAFractionOfWhatIsReachable(t *testing.T) {
	m := qwen3_32B()
	sizing := sizeAt(m, 4096)
	need := sizing.Total.MustValue()

	sel := selectAll(t, m, sizing, []truffle.InstanceTypeResult{g6e12xlarge}, nil)
	c := find(t, sel, "g6e.12xlarge")

	got, ok := c.Headroom.Value()
	if !ok {
		t.Fatalf("Headroom unavailable: %s", c.Headroom.Source())
	}
	have := float64(g6e12xlarge.GPUMemoryMiB) / mibPerGiB
	want := (have - need) / have
	if got < want-0.001 || got > want+0.001 {
		t.Errorf("Headroom = %.4f, want %.4f (against %.1f GiB reachable)", got, want, have)
	}
	if c.Headroom.Unit() != report.UnitFraction {
		t.Errorf("Headroom unit is %s, want %s", c.Headroom.Unit(), report.UnitFraction)
	}
}

// TestObservedAtIsRecorded: instance data has a shelf life, and a candidate set with
// no observation time cannot be aged out.
func TestObservedAtIsRecorded(t *testing.T) {
	m := qwen3_32B()
	sel := selectAll(t, m, sizeAt(m, 4096), []truffle.InstanceTypeResult{g6e12xlarge}, nil)
	if !sel.ObservedAt.Equal(observed) {
		t.Errorf("ObservedAt = %v, want %v", sel.ObservedAt, observed)
	}
	c := find(t, sel, "g6e.12xlarge")
	if ts := c.GPUMemory.ObservedAt(); !ts.Equal(observed) {
		t.Errorf("GPUMemory.ObservedAt = %v, want %v", ts, observed)
	}
}

// TestNilSelectionMethodsDoNotPanic: the accessors are called from renderers that may
// hold a nil selection after an error.
func TestNilSelectionMethodsDoNotPanic(t *testing.T) {
	var s *Selection
	if got := s.Usable(); got != nil {
		t.Errorf("Usable() = %v, want nil", got)
	}
	if got := s.Failures(); got != nil {
		t.Errorf("Failures() = %v, want nil", got)
	}
	if _, ok := s.Cheapest(); ok {
		t.Error("Cheapest() reported a candidate on a nil selection")
	}
}

func names(cs []Candidate) []string {
	out := make([]string, 0, len(cs))
	for _, c := range cs {
		out = append(out, c.InstanceType)
	}
	return out
}
