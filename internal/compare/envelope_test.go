package compare

import (
	"errors"
	"strings"
	"testing"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// The conversion that decides whether cultivar says "us-east-2 has nothing that
// fits" or "us-east-2 could not be checked". Said wrongly, the first sends a user to
// a pricier region — us-west-1 carries p5 at $55/hr and no g6e/g7e at $3.
func TestAFailedRegionNeverConvertsToAnEmptyOne(t *testing.T) {
	tests := []struct {
		name  string
		in    RegionStatus
		state report.RegionState
	}{
		{"read, and something fits", RegionStatus{Region: "us-east-2", Considered: 61, Usable: 14}, report.RegionOK},
		{"read, and nothing fits", RegionStatus{Region: "us-west-1", Considered: 9}, report.RegionEmpty},
		{"never read", RegionStatus{Region: "us-east-2", Err: errors.New("AccessDenied")}, report.RegionFailed},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.in.Report()
			if got.State != tc.state {
				t.Errorf("state = %s, want %s", got.State, tc.state)
			}
			if got.Name != tc.in.Region {
				t.Errorf("name = %q, want %q", got.Name, tc.in.Region)
			}
			// Whatever the state, the result must be publishable — report.Region.Validate
			// rejects the inconsistent combinations, and this conversion is the only thing
			// that builds one from a real query.
			if err := got.Validate(); err != nil {
				t.Errorf("the converted region does not validate: %v", err)
			}
		})
	}
}

// A failure must carry its reason. report.Region.Validate rejects a failure without
// one precisely because an unexplained failure is indistinguishable, downstream, from
// an empty region — which is the confusion this whole type exists to prevent.
func TestAFailureCarriesItsReason(t *testing.T) {
	got := RegionStatus{Region: "us-east-2", Err: errors.New("AccessDenied: not authorized to call DescribeInstanceTypes")}.Report()

	if !strings.Contains(got.Error, "AccessDenied") {
		t.Errorf("error text = %q, want the AccessDenied that explains the empty result", got.Error)
	}
	if got.State.Informative() {
		t.Error("a failed region must not be informative; treating it as such states a " +
			"permissions problem as an absence of capacity")
	}
	if err := got.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

// Counts from a query that then failed are dropped rather than reported. truffle's
// SearchInstanceTypes fans out and can fail after partial results, and "we looked at
// 40 of 61 types" reported as a complete count is a fit conclusion drawn from an
// incomplete catalogue.
func TestAFailedRegionReportsNoCounts(t *testing.T) {
	got := RegionStatus{Region: "us-east-2", Considered: 40, Usable: 3, Err: errors.New("throttled")}.Report()

	if got.Considered != 0 || got.Usable != 0 {
		t.Errorf("counts = %d/%d, want 0/0: a partial count from a failed query reads as "+
			"a complete one", got.Considered, got.Usable)
	}
	if got.State != report.RegionFailed {
		t.Errorf("state = %s, want failed", got.State)
	}
	// Without zeroing, this combination — failed with usable candidates — is what
	// Validate would have to reject or silently accept.
	if err := got.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
}

// A whole Selection converts in the order queried, so the envelope's region list
// lines up with its query's, which is what Envelope.Validate cross-checks.
func TestSelectionConvertsEveryRegionInOrder(t *testing.T) {
	sel := &Selection{Regions: []RegionStatus{
		{Region: "us-east-2", Considered: 61, Usable: 14},
		{Region: "us-east-1", Err: errors.New("AccessDenied")},
		{Region: "us-west-1", Considered: 9},
	}}

	got := sel.ReportRegions()
	if len(got) != 3 {
		t.Fatalf("converted %d of 3 regions; a dropped region reads as one with no capacity", len(got))
	}
	var names []string
	for _, r := range got {
		names = append(names, r.Name)
	}
	if want := "us-east-2 us-east-1 us-west-1"; strings.Join(names, " ") != want {
		t.Errorf("order = %q, want the queried order %q", strings.Join(names, " "), want)
	}

	// The states must not all collapse to one value, which a conversion that ignored
	// its input would produce while still passing the order check.
	if got[0].State != report.RegionOK || got[1].State != report.RegionFailed || got[2].State != report.RegionEmpty {
		t.Errorf("states = %s/%s/%s, want ok/failed/empty", got[0].State, got[1].State, got[2].State)
	}

	// A nil Selection converts to nothing rather than panicking: Select returns nil
	// when every region failed, and a caller assembling a report from that path is
	// exactly the case that must not crash.
	var nilSel *Selection
	if got := nilSel.ReportRegions(); got != nil {
		t.Errorf("nil Selection converted to %v, want nil", got)
	}
}

// The conversion output has to satisfy the envelope it is destined for, region set
// and all. This is the join #16 freezes: compare produces the regions, report
// validates that they match the query.
func TestConvertedRegionsSatisfyTheEnvelope(t *testing.T) {
	regions := []string{"us-east-1", "us-east-2"}
	sel := &Selection{Regions: []RegionStatus{
		{Region: "us-east-1", Considered: 69, Usable: 12},
		{Region: "us-east-2", Err: errors.New("AccessDenied")},
	}}

	e := report.NewEnvelope("compare", "0.2.0", clock())
	e.Subject = report.Subject{ModelID: "Qwen/Qwen3-32B", ObservedAt: clock()}
	e.Query = report.Query{Regions: regions}
	e.Regions = sel.ReportRegions()
	e.Assumptions = threeToOne.Record(report.Assumptions{}).
		WithSizing(4096, 1, 2, 0.15).
		WithThroughput(report.External(1200, report.UnitTokensPerSecond, "vLLM benchmark", clock()))

	if err := e.Validate(); err != nil {
		t.Fatalf("an envelope assembled from a real Selection must validate: %v", err)
	}
	if !e.Degraded() {
		t.Error("Degraded() = false with a failed region in the selection")
	}
}

// The break-even assumptions reach the report unchanged. A report stating a 3:1 blend
// while the figure behind it was computed at 1:1 is worse than one stating nothing,
// because it is confidently wrong about its own derivation.
func TestBreakEvenAssumptionsReachTheReport(t *testing.T) {
	a := Assumptions{InputWeight: 1, OutputWeight: 3, Utilization: 0.4}
	got := a.Record(report.Assumptions{})

	if got.InputTokenWeight != 1 || got.OutputTokenWeight != 3 {
		t.Errorf("weights = %g:%g, want 1:3; swapping them prices output as input",
			got.InputTokenWeight, got.OutputTokenWeight)
	}
	if got.Utilization != 0.4 {
		t.Errorf("utilization = %g, want 0.4", got.Utilization)
	}
	// Both types render the ratio, and they must agree: one is printed in the verdict,
	// the other stored in the report, and a reader comparing them is checking exactly
	// this.
	if got.Ratio() != a.Ratio() {
		t.Errorf("report says %q, break-even says %q", got.Ratio(), a.Ratio())
	}

	// Recording must not disturb the sizing fields another owner fills in.
	existing := report.Assumptions{ContextTokens: 40960, Concurrency: 8, KVCacheDTypeBytes: 2}
	merged := a.Record(existing)
	if merged.ContextTokens != 40960 || merged.Concurrency != 8 {
		t.Errorf("recording the blend clobbered the sizing: %+v", merged)
	}
	if existing.Utilization != 0 {
		t.Errorf("Record mutated its argument: utilization = %g", existing.Utilization)
	}
}
