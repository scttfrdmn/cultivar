//go:build live

// Opt-in suite that assembles a real report envelope from real AWS query results. Run
// with `make test-live` (AWS_PROFILE=aws). Every call is a free read-only query.
//
// The offline tests pin the envelope's rules against constructed values. What they
// cannot check is that a report is *assemblable* from what the AWS APIs actually
// return — that a real region set, real per-region outcomes, and a real model's
// metadata satisfy the validation gate together. A schema freeze that only ever
// validates hand-built fixtures freezes a shape nothing produces.
package compare

import (
	"context"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	truffle "github.com/spore-host/truffle/pkg/aws"

	"github.com/scttfrdmn/cultivar/internal/bedrock"
	"github.com/scttfrdmn/cultivar/internal/ec2"
	"github.com/scttfrdmn/cultivar/internal/identity"
	"github.com/scttfrdmn/cultivar/internal/model"
	"github.com/scttfrdmn/cultivar/internal/report"
)

// envelopeRegions spans the range deliberately: us-east-1 has the broadest lineup
// (69 GPU types), us-east-2 the cheapest full one (61), us-west-1 only 9 across 3
// families. That last one is the case worth carrying — thin but expensive, so a fit
// check alone recommends p5 at $55/hr there. Measured 2026-07-29, this yields 26, 24,
// and 3 usable candidates for Qwen3-32B at 4k context.
var envelopeRegions = []string{"us-east-1", "us-east-2", "us-west-1"}

// benchmarkPublished is a stand-in publication date for the external throughput
// figure. Fixed rather than time.Now() because [report.External] records when the
// third party measured a number, and a stale citation stamped with the current time
// is indistinguishable from a fresh one.
var benchmarkPublished = time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)

// TestLiveAnEnvelopeIsAssemblableFromRealQueries is the end-to-end check that the
// frozen shape can actually be produced.
func TestLiveAnEnvelopeIsAssemblableFromRealQueries(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Second)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}

	m := qwen3ForLive()
	sizing := m.Size(model.SizingRequest{ContextTokens: 4096})
	sel, err := NewSelector(truffle.NewClientFromConfig(cfg), ec2.NewPricer(cfg), time.Now).
		Select(ctx, m, sizing, envelopeRegions)
	if err != nil {
		t.Fatalf("Select across %v: %v", envelopeRegions, err)
	}

	price, err := bedrock.NewTokenPricer(cfg).Lookup(ctx, qwen3BedrockID, liveRegion)
	if err != nil {
		t.Fatalf("token price: %v", err)
	}
	parity, err := BreakEven(mustCheapest(t, sel).OnDemand,
		price.Amount(bedrock.TierStandard, bedrock.MeterInput),
		price.Amount(bedrock.TierStandard, bedrock.MeterOutput),
		liveAssumptions)
	if err != nil {
		t.Fatalf("BreakEven: %v", err)
	}

	// The tok/s figure is external until a benchmark has been run on this hardware,
	// and the envelope has to be able to say so rather than pushing a caller to invent
	// a number. That provenance is the most load-bearing one in the report.
	//
	// The date is the figure's publication date, not now: External records when the
	// third party measured it, and stamping it with the current time would make a
	// two-year-old benchmark look freshly observed — the exact staleness the
	// provenance exists to expose.
	throughput := report.External(1200, report.UnitTokensPerSecond,
		"published vLLM benchmark for a 32B model on one L40S-class GPU", benchmarkPublished)

	e := report.NewEnvelope("compare", "0.2.0-live-test", time.Now())
	e.Subject = m.Subject()
	e.Subject.BedrockModelID = qwen3BedrockID
	e.Query = report.Query{Regions: envelopeRegions}
	e.Regions = sel.ReportRegions()
	e.Assumptions = parity.Assumptions.Record(sizing.Record(report.Assumptions{})).
		WithThroughput(throughput)

	// The identity is the one part of the envelope with a producer outside this
	// package, and it is optional: an unresolvable one is recorded as absent rather
	// than failing a run over two informative fields. That tolerance belongs here at
	// the call site, which is why the resolver itself is strict.
	if id, ierr := identity.NewResolver(cfg).Resolve(ctx); ierr != nil {
		t.Logf("identity unresolved, recording neither field: %v", ierr)
	} else {
		e = id.Record(e)
		t.Logf("identity: account %s, partition %s", e.Account, e.Partition)
	}

	if err := e.Validate(); err != nil {
		t.Fatalf("an envelope built from real queries does not validate: %v\n"+
			"regions: %+v\nassumptions: %+v", err, e.Regions, e.Assumptions)
	}

	for _, r := range e.Regions {
		t.Logf("%-12s %-7s %d considered, %d usable %s", r.Name, r.State, r.Considered, r.Usable, r.Error)
	}
	t.Logf("subject %s (bedrock %s, gated=%v) → parity at %s under %s",
		e.Subject.ModelID, e.Subject.BedrockModelID, e.Subject.Gated,
		parity.Throughput, e.Assumptions.Ratio())

	// Every requested region must have produced a result, in order. This is the
	// silent-drop check against a real fan-out rather than a constructed slice.
	if len(e.Regions) != len(envelopeRegions) {
		t.Fatalf("%d results for %d regions", len(e.Regions), len(envelopeRegions))
	}
	for i, want := range envelopeRegions {
		if e.Regions[i].Name != want {
			t.Errorf("result %d is %s, want %s", i, e.Regions[i].Name, want)
		}
	}
	// And at least one region must have found something, or the run says nothing about
	// AWS and the rest of the assertions are vacuous.
	if len(e.Informative()) == 0 {
		t.Fatal("no region was successfully read; the envelope contains no findings")
	}

	// The assumptions must be the ones the arithmetic used, not defaults. Checked
	// against the two objects that computed them rather than against literals.
	if e.Assumptions.Utilization != liveAssumptions.Utilization {
		t.Errorf("recorded utilization %g != the %g break-even used",
			e.Assumptions.Utilization, liveAssumptions.Utilization)
	}
	if e.Assumptions.ContextTokens != sizing.ContextTokens {
		t.Errorf("recorded context %d != the %d sizing used",
			e.Assumptions.ContextTokens, sizing.ContextTokens)
	}
	if e.Assumptions.Ratio() != parity.Assumptions.Ratio() {
		t.Errorf("recorded ratio %q != break-even's %q", e.Assumptions.Ratio(), parity.Assumptions.Ratio())
	}
}

// us-west-1 is the region that makes the empty state a real answer rather than a
// theoretical one: 9 GPU types against us-east-1's 69, and the fit check for a 61 GiB
// model may legitimately find nothing there. Whatever it finds, the state must be
// attributable — and it must not be `failed`, since the query does succeed.
func TestLiveASparseRegionIsEmptyNotFailed(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}

	m := qwen3ForLive()
	sel, err := NewSelector(truffle.NewClientFromConfig(cfg), ec2.NewPricer(cfg), time.Now).
		Select(ctx, m, m.Size(model.SizingRequest{ContextTokens: 4096}), []string{"us-west-1"})
	if err != nil {
		t.Fatalf("Select in us-west-1: %v", err)
	}

	got := sel.ReportRegions()
	if len(got) != 1 {
		t.Fatalf("%d results for one region", len(got))
	}
	r := got[0]
	// Considered is the whole instance catalogue, not the GPU subset: 668 types in
	// us-west-1 on 2026-07-29, of which 3 can hold a 61 GiB model. Those 3 are the
	// finding — us-west-1 carries p5-class hardware at ~$55/hr and no g6e/g7e at $3,
	// so a fit check alone recommends it happily (CLAUDE.md trap 10).
	t.Logf("us-west-1: %s, %d types considered, %d usable (recorded 668/3 on 2026-07-29)",
		r.State, r.Considered, r.Usable)

	if r.State == report.RegionFailed {
		t.Fatalf("us-west-1 reported as failed: %s — a permissions or throttling problem, "+
			"not an inventory finding", r.Error)
	}
	if !r.State.Informative() {
		t.Errorf("state %s is not informative", r.State)
	}
	if r.Considered == 0 {
		t.Error("0 types considered in us-west-1; the region has a catalogue, so an empty " +
			"count means the query returned nothing and the region is being misreported")
	}
	if err := r.Validate(); err != nil {
		t.Errorf("Validate: %v", err)
	}
	// The state must follow from the counts, which is the invariant that keeps an
	// unreadable region from wearing an empty one's clothes.
	if (r.Usable > 0) != (r.State == report.RegionOK) {
		t.Errorf("state %s disagrees with %d usable candidates", r.State, r.Usable)
	}
}

// A bogus region name is the closest free stand-in for the AccessDenied case: the
// query genuinely fails, and the envelope must carry that as `failed` with a reason
// rather than as an absence of capacity. This is truffle#117 seen from cultivar's
// side — the case that reads as "no GPUs there" and steers a user to a pricier region.
func TestLiveAnUnreadableRegionIsNotAnEmptyOne(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}

	const bogus = "us-nowhere-9"
	m := qwen3ForLive()
	sel, err := NewSelector(truffle.NewClientFromConfig(cfg), ec2.NewPricer(cfg), time.Now).
		Select(ctx, m, m.Size(model.SizingRequest{ContextTokens: 4096}), []string{liveRegion, bogus})
	if err != nil {
		t.Fatalf("Select must not fail when one of two regions fails: %v", err)
	}

	got := sel.ReportRegions()
	if len(got) != 2 {
		t.Fatalf("%d results for 2 regions; the failed one was dropped, which reads as "+
			"a region with no capacity", len(got))
	}

	var bad report.Region
	for _, r := range got {
		if r.Name == bogus {
			bad = r
		}
	}
	t.Logf("%s: %s (%s)", bad.Name, bad.State, bad.Error)

	if bad.State != report.RegionFailed {
		t.Fatalf("%s reported as %s; an unreachable region must never read as one with "+
			"no capacity", bogus, bad.State)
	}
	if bad.Error == "" {
		t.Error("no reason recorded, so the failure is indistinguishable from an empty region")
	}
	if bad.Considered != 0 || bad.Usable != 0 {
		t.Errorf("counts %d/%d from a failed query", bad.Considered, bad.Usable)
	}

	// The envelope built from this must validate and must declare itself degraded: a
	// partial answer presented as a complete one is the same class of error as an
	// estimate presented as a price.
	e := report.NewEnvelope("compare", "0.2.0-live-test", time.Now())
	e.Subject = m.Subject()
	e.Query = report.Query{Regions: []string{liveRegion, bogus}}
	e.Regions = got
	e.Assumptions = liveAssumptions.Record(m.Size(model.SizingRequest{ContextTokens: 4096}).
		Record(report.Assumptions{})).
		WithThroughput(report.Unavailable(report.UnitTokensPerSecond, "no benchmark on this hardware"))

	if err := e.Validate(); err != nil {
		t.Fatalf("an envelope with a failed region must validate: %v", err)
	}
	if !e.Degraded() {
		t.Error("Degraded() = false with an unreadable region")
	}
	if len(e.Informative()) != 1 {
		t.Errorf("%d informative regions, want 1", len(e.Informative()))
	}
}

// A real envelope survives serialization and comes back validating. The history log
// stores exactly these bytes, so a field that does not survive the round trip is a
// field that silently vanishes from every stored record.
func TestLiveARealEnvelopeRoundTrips(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}

	m := qwen3ForLive()
	sizing := m.Size(model.SizingRequest{ContextTokens: 4096})
	sel, err := NewSelector(truffle.NewClientFromConfig(cfg), ec2.NewPricer(cfg), time.Now).
		Select(ctx, m, sizing, []string{liveRegion})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}

	e := report.NewEnvelope("compare", "0.2.0-live-test", time.Now())
	e.Subject = m.Subject()
	e.Subject.BedrockModelID = qwen3BedrockID
	e.Query = report.Query{Regions: []string{liveRegion}, InstanceCount: 1}
	e.Regions = sel.ReportRegions()
	e.Assumptions = liveAssumptions.Record(sizing.Record(report.Assumptions{})).
		WithThroughput(report.External(1200, report.UnitTokensPerSecond, "published vLLM benchmark", benchmarkPublished))

	// Recorded here so the round trip covers the identity fields too: they are
	// omitempty, so an envelope without them never exercises their serialization at
	// all, and the history log is exactly these bytes.
	id, err := identity.NewResolver(cfg).Resolve(ctx)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	e = id.Record(e)

	if err := e.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	data, err := json.MarshalIndent(e, "", "  ")
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	t.Logf("envelope (%d bytes):\n%s", len(data), data)

	var got report.Envelope
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if err := got.Validate(); err != nil {
		t.Fatalf("the decoded envelope does not validate: %v", err)
	}
	again, err := json.MarshalIndent(got, "", "  ")
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(data) != string(again) {
		t.Errorf("round trip changed the bytes:\n%s\n---\n%s", data, again)
	}

	// generatedAt must be RFC3339 UTC on the wire whatever the machine's timezone: a
	// history log spanning timezones sorts wrongly otherwise, and it does so while
	// looking like working output.
	if !strings.Contains(string(data), `"generatedAt": "`) {
		t.Error("no generatedAt in the serialized form")
	}
	stamp := got.GeneratedAt
	if stamp.Location() != time.UTC {
		t.Errorf("generatedAt decoded in %v, want UTC", stamp.Location())
	}
	if got.Account != id.Account || got.Partition != id.Partition {
		t.Errorf("identity round-tripped as %q/%q, want %q/%q",
			got.Account, got.Partition, id.Account, id.Partition)
	}
	if age := got.Age(time.Now()); age < 0 || age > 10*time.Minute {
		t.Errorf("age = %v, want a few seconds; the report claims to be from the future "+
			"or from long ago", age)
	}

	// The envelope's age is the floor on staleness, not the whole story. Here a
	// freshly generated report carries a two-month-old benchmark, and both ages
	// survive independently — flattening them to one would present a stale citation
	// as a current measurement, which is the failure the per-amount provenance exists
	// to prevent.
	tp := got.Assumptions.Throughput
	if tp.Provenance() != report.ProvenanceExternal {
		t.Errorf("throughput provenance = %s, want external", tp.Provenance())
	}
	tpAge, ok := tp.Age(time.Now())
	if !ok {
		t.Fatal("the external throughput has no publication date, so its staleness is unknowable")
	}
	if tpAge <= got.Age(time.Now()) {
		t.Errorf("the contained benchmark (%v old) is not older than the report (%v old); "+
			"the two ages have been collapsed", tpAge, got.Age(time.Now()))
	}
	t.Logf("report is %v old and carries a benchmark %v old", got.Age(time.Now()).Round(time.Second), tpAge.Round(time.Hour))
}

// mustCheapest fails the test rather than returning a zero Candidate, so a missing
// price surfaces here instead of as a $0.00 hourly rate downstream.
func mustCheapest(t *testing.T, sel *Selection) Candidate {
	t.Helper()
	c, ok := sel.Cheapest()
	if !ok {
		t.Fatal("no priced instance can serve the model in any queried region")
	}
	return c
}
