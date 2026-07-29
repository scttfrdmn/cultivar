package report

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
)

// stamp is a fixed time so a marshalled envelope is byte-comparable across runs.
var stamp = time.Date(2026, 7, 29, 14, 30, 0, 0, time.UTC)

// complete returns an envelope that passes validation, for tests that break one
// field at a time. Built by a function rather than shared as a package var because
// several tests mutate it, and a shared value would couple them.
func complete() Envelope {
	e := NewEnvelope("compare", "0.2.0", stamp)
	e.Subject = Subject{ModelID: "Qwen/Qwen3-32B", BedrockModelID: "qwen.qwen3-32b-v1:0", ObservedAt: stamp}
	e.Query = Query{Regions: []string{"us-east-1", "us-east-2"}}
	e.Regions = []Region{
		{Name: "us-east-1", State: RegionOK, Considered: 69, Usable: 12},
		{Name: "us-east-2", State: RegionOK, Considered: 61, Usable: 14},
	}
	e.Assumptions = Assumptions{
		InputTokenWeight: 3, OutputTokenWeight: 1, Utilization: 1.0,
		ContextTokens: 4096, Concurrency: 1, KVCacheDTypeBytes: 2, OverheadFraction: 0.15,
		Throughput: External(1200, UnitTokensPerSecond, "published vLLM benchmark", stamp),
	}
	return e
}

func TestACompleteEnvelopeValidates(t *testing.T) {
	if err := complete().Validate(); err != nil {
		t.Fatalf("the fixture every other test mutates must itself be valid: %v", err)
	}
}

// Each of these is a field whose absence makes the report uninterpretable after the
// fact. The run that produced it is gone, so there is no repairing it later — which
// is why validation is a gate on writing rather than a warning on reading.
func TestAnUninterpretableReportIsRefused(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Envelope)
		want   string
	}{
		{"no tool version", func(e *Envelope) { e.ToolVersion = "" }, "tool version"},
		{"whitespace tool version", func(e *Envelope) { e.ToolVersion = "  " }, "tool version"},
		{"no kind", func(e *Envelope) { e.Kind = "" }, "kind"},
		{"no generatedAt", func(e *Envelope) { e.GeneratedAt = time.Time{} }, "generatedAt"},
		{"no model id", func(e *Envelope) { e.Subject.ModelID = "" }, "model id"},
		{"no model observation time", func(e *Envelope) { e.Subject.ObservedAt = time.Time{} }, "observation time"},
		{"no regions queried", func(e *Envelope) { e.Query.Regions = nil }, "no regions"},
		{"no region results", func(e *Envelope) { e.Regions = nil }, "no region results"},
		// The assumption block: each of these silently moves the verdict.
		{"unstated utilization", func(e *Envelope) { e.Assumptions.Utilization = 0 }, "utilization"},
		{"impossible utilization", func(e *Envelope) { e.Assumptions.Utilization = 1.5 }, "utilization"},
		{"unstated traffic mix", func(e *Envelope) {
			e.Assumptions.InputTokenWeight, e.Assumptions.OutputTokenWeight = 0, 0
		}, "traffic mix"},
		{"negative weight", func(e *Envelope) { e.Assumptions.InputTokenWeight = -1 }, "negative token weight"},
		{"no context length", func(e *Envelope) { e.Assumptions.ContextTokens = 0 }, "context length"},
		{"no concurrency", func(e *Envelope) { e.Assumptions.Concurrency = 0 }, "concurrency"},
		{"no cache width", func(e *Envelope) { e.Assumptions.KVCacheDTypeBytes = 0 }, "dtype width"},
		{"negative overhead", func(e *Envelope) { e.Assumptions.OverheadFraction = -0.1 }, "overhead"},
		// A zero-value throughput Amount reads as 0 tok/s with no provenance. The
		// unavailable case is legitimate and tested separately below; this is the
		// forgotten-field case, which must not pass for a field the verdict is this
		// sensitive to.
		{"throughput never set", func(e *Envelope) { e.Assumptions.Throughput = Amount{} }, "throughput"},
		{"throughput in the wrong unit", func(e *Envelope) {
			e.Assumptions.Throughput = Live(4.0, UnitUSDPerHour, "price list", stamp)
		}, "throughput has unit"},
		// The next three are malformed in ways the unit check cannot see, because they
		// carry the right unit. They are why the throughput goes through [Amount.Valid]
		// and is not merely unit-checked: a zero Amount happens to have an empty unit, so
		// without these the Valid call is unreachable in practice.
		{"NaN throughput", func(e *Envelope) {
			e.Assumptions.Throughput = External(math.NaN(), UnitTokensPerSecond, "benchmark", stamp)
		}, "not finite"},
		{"uncitable throughput", func(e *Envelope) {
			e.Assumptions.Throughput = External(1200, UnitTokensPerSecond, "", stamp)
		}, "no source"},
		// A live figure with no observation time cannot be aged, and this is the number
		// the verdict is most sensitive to: a measurement from two vLLM releases ago is
		// not the same claim as one from today.
		{"undateable live throughput", func(e *Envelope) {
			e.Assumptions.Throughput = Live(1200, UnitTokensPerSecond, "benchmark run", time.Time{})
		}, "no observation time"},
		// Schema version: a consumer that trusts field meanings must be able to trust
		// this one field first.
		{"wrong schema version", func(e *Envelope) { e.SchemaVersion = "report.v2" }, "schema version"},
		{"no schema version", func(e *Envelope) { e.SchemaVersion = "" }, "schema version"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := complete()
			tc.mutate(&e)
			err := e.Validate()
			if err == nil {
				t.Fatalf("%s validated; the report would be published uninterpretable", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q, so it does not say what to fix", err, tc.want)
			}
		})
	}
}

// An unavailable throughput is a legitimate state and must validate: until a
// benchmark has been run on this hardware there is no figure, and the report has to
// be able to say so. Refusing it would push callers toward inventing one.
func TestAnUnmeasuredThroughputIsAValidAssumption(t *testing.T) {
	e := complete()
	e.Assumptions.Throughput = Unavailable(UnitTokensPerSecond,
		"no benchmark for Qwen3-32B on g7e.4xlarge")
	if err := e.Validate(); err != nil {
		t.Fatalf("an unmeasured throughput must be recordable, not an error: %v", err)
	}
}

// The distinction the whole per-region type exists for: a region that could not be
// read must never be presentable as one with no capacity. us-west-1 genuinely has 9
// GPU types against us-east-1's 69, so "empty" is a real answer and an AccessDenied
// wearing its clothes is undetectable.
func TestAFailedRegionIsNotAnEmptyOne(t *testing.T) {
	e := complete()
	e.Regions[1] = Region{Name: "us-east-2", State: RegionFailed, Error: "AccessDenied"}
	if err := e.Validate(); err != nil {
		t.Fatalf("a failed region is a valid thing to report: %v", err)
	}

	if !e.Degraded() {
		t.Error("Degraded() = false with a failed region; a partial answer would be presented as a full one")
	}
	if got := len(e.Failures()); got != 1 {
		t.Errorf("Failures() returned %d regions, want 1", got)
	}
	if got := len(e.Informative()); got != 1 {
		t.Errorf("Informative() returned %d regions, want 1 (only us-east-1 was read)", got)
	}
	if e.Informative()[0].Name != "us-east-1" {
		t.Errorf("Informative() = %s, want us-east-1", e.Informative()[0].Name)
	}

	// And the healthy case is not degraded, or the flag means nothing.
	if complete().Degraded() {
		t.Error("Degraded() = true with every region ok")
	}
	// An empty region is informative: it was asked, and the answer was nothing.
	empty := complete()
	empty.Regions[1] = Region{Name: "us-east-2", State: RegionEmpty, Considered: 61}
	if empty.Degraded() {
		t.Error("a genuinely empty region must not read as degraded; that is a real answer")
	}
	if got := len(empty.Informative()); got != 2 {
		t.Errorf("an empty region is informative: got %d, want 2", got)
	}
}

// A failure with no explanation is exactly as uninterpretable as no failure record
// at all, so it is rejected rather than stored.
func TestARegionStatusMustBeSelfConsistent(t *testing.T) {
	tests := []struct {
		name   string
		region Region
		want   string
	}{
		{"failed with no reason", Region{Name: "us-east-2", State: RegionFailed}, "no error text"},
		{"ok but carrying an error", Region{Name: "us-east-2", State: RegionOK, Considered: 5, Usable: 1, Error: "AccessDenied"}, "carries an error"},
		{"empty but carrying an error", Region{Name: "us-east-2", State: RegionEmpty, Error: "AccessDenied"}, "carries an error"},
		{"ok with nothing usable", Region{Name: "us-east-2", State: RegionOK, Considered: 61}, "should be"},
		{"empty with usable candidates", Region{Name: "us-east-2", State: RegionEmpty, Considered: 61, Usable: 3}, "usable candidates"},
		{"more usable than considered", Region{Name: "us-east-2", State: RegionOK, Considered: 2, Usable: 3}, "exceeds"},
		{"negative count", Region{Name: "us-east-2", State: RegionEmpty, Considered: -1}, "negative"},
		{"no name", Region{State: RegionEmpty}, "no name"},
		{"undefined state", Region{Name: "us-east-2", State: "denied"}, "not one of"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.region.Validate()
			if err == nil {
				t.Fatalf("%s validated", tc.name)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("error %q does not mention %q", err, tc.want)
			}
			// And it must fail through the envelope too, which is the only path a real
			// report takes.
			e := complete()
			e.Query.Regions = []string{tc.region.Name}
			e.Regions = []Region{tc.region}
			if err := e.Validate(); err == nil {
				t.Errorf("%s passed envelope validation", tc.name)
			}
		})
	}
}

// A region asked for and missing from the results is the silent-drop failure. It
// reads as a region with no capacity, which is the wrong answer in the expensive
// direction: it steers a user away from us-east-2, the cheapest full-lineup region.
func TestADroppedRegionIsCaught(t *testing.T) {
	e := complete()
	e.Regions = e.Regions[:1] // us-east-2's result vanishes

	err := e.Validate()
	if err == nil {
		t.Fatal("a dropped region validated; it would read as a region with no capacity")
	}
	for _, want := range []string{"us-east-2", "no capacity"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention %q", err, want)
		}
	}
}

// The other direction: a result for a region nobody asked about means the region set
// in the report is not the one that produced it, so the report's own header is
// wrong about itself.
func TestAnUnrequestedRegionResultIsCaught(t *testing.T) {
	e := complete()
	e.Regions = append(e.Regions, Region{Name: "ap-northeast-1", State: RegionEmpty, Considered: 4})

	err := e.Validate()
	if err == nil {
		t.Fatal("a result for an unqueried region validated")
	}
	if !strings.Contains(err.Error(), "not queried") {
		t.Errorf("error %q does not say the region was not queried", err)
	}
}

func TestDuplicateRegionsAreCaught(t *testing.T) {
	dupQuery := complete()
	dupQuery.Query.Regions = []string{"us-east-1", "us-east-2", "us-east-1"}
	if err := dupQuery.Validate(); err == nil || !strings.Contains(err.Error(), "twice") {
		t.Errorf("a region queried twice validated with error %v", err)
	}

	dupResult := complete()
	dupResult.Regions = append(dupResult.Regions, dupResult.Regions[0])
	if err := dupResult.Validate(); err == nil || !strings.Contains(err.Error(), "twice") {
		t.Errorf("a region reported twice validated with error %v", err)
	}

	blank := complete()
	blank.Query.Regions = []string{"us-east-1", " "}
	if err := blank.Validate(); err == nil || !strings.Contains(err.Error(), "empty region") {
		t.Errorf("a blank region name validated with error %v", err)
	}
}

// The gating parameters. truffle#109: asking for 8 instances where 1 was requested
// flipped "no capacity exists" to "offerings available", so a capacity report that
// does not state its count is not interpretable — and two runs at different counts
// look like AWS changed its mind.
func TestGatingParametersSurviveSerialization(t *testing.T) {
	e := complete()
	e.Query.InstanceCount = 8
	e.Query.DurationHours = 24
	e.Query.InstanceTypes = []string{"p5.48xlarge", "p5e.48xlarge"}

	if err := e.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	data, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Envelope
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Query.InstanceCount != 8 {
		t.Errorf("instance count = %d, want 8; a capacity finding without it is not interpretable",
			got.Query.InstanceCount)
	}
	if got.Query.DurationHours != 24 {
		t.Errorf("duration = %d, want 24", got.Query.DurationHours)
	}
	if len(got.Query.InstanceTypes) != 2 {
		t.Errorf("instance types = %v, want 2 entries; 'nothing fits' and 'nothing fits among "+
			"the types you named' are different findings", got.Query.InstanceTypes)
	}
	if err := got.Validate(); err != nil {
		t.Errorf("the decoded envelope no longer validates: %v", err)
	}
}

// Negative gating values would be silently ignored by the AWS APIs, producing a
// report whose stated query is not the one that ran.
func TestNegativeGatingValuesAreRefused(t *testing.T) {
	count := complete()
	count.Query.InstanceCount = -1
	if err := count.Validate(); err == nil || !strings.Contains(err.Error(), "instance count") {
		t.Errorf("negative instance count validated with error %v", err)
	}
	dur := complete()
	dur.Query.DurationHours = -4
	if err := dur.Validate(); err == nil || !strings.Contains(err.Error(), "duration") {
		t.Errorf("negative duration validated with error %v", err)
	}
}

// Every assumption that moves the verdict has to survive a round trip, because the
// history log's whole purpose is comparing today's run against a stored one. An
// assumption that serializes and comes back different makes a price change and an
// assumption change indistinguishable.
func TestAssumptionsRoundTrip(t *testing.T) {
	e := complete()
	e.Assumptions = Assumptions{
		InputTokenWeight: 1, OutputTokenWeight: 1, Utilization: 0.4,
		ContextTokens: 40960, Concurrency: 8, KVCacheDTypeBytes: 1, OverheadFraction: 0.2,
		Throughput: External(1200, UnitTokensPerSecond, "vLLM 0.6.3 on 1xL40S", stamp),
	}
	data, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Envelope
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Assumptions != e.Assumptions {
		t.Errorf("assumptions changed across a round trip:\n got %+v\nwant %+v", got.Assumptions, e.Assumptions)
	}
	// The throughput's provenance matters as much as its value: 1200 tok/s measured
	// and 1200 tok/s cited from someone else's blog post support different decisions.
	if got := got.Assumptions.Throughput.Provenance(); got != ProvenanceExternal {
		t.Errorf("throughput provenance = %s, want external", got)
	}
	if src := got.Assumptions.Throughput.Source(); src != "vLLM 0.6.3 on 1xL40S" {
		t.Errorf("throughput citation = %q, want the engine and hardware it was measured on", src)
	}
}

// The ratio is printed in the verdict, echoed by the CLI flag, and written into the
// provenance of every derived token price. A separator or ordering that disagrees
// between those is a silently misstated assumption.
func TestBlendRatioRendersAsStated(t *testing.T) {
	tests := []struct {
		in, out float64
		want    string
	}{
		{3, 1, "3:1"},
		{1, 1, "1:1"},
		{1, 3, "1:3"}, // not symmetric: reversing it prices output as input
		{2.5, 1, "2.5:1"},
		{10, 0, "10:0"},
	}
	for _, tc := range tests {
		if got := BlendRatio(tc.in, tc.out); got != tc.want {
			t.Errorf("BlendRatio(%g, %g) = %q, want %q", tc.in, tc.out, got, tc.want)
		}
		a := Assumptions{InputTokenWeight: tc.in, OutputTokenWeight: tc.out}
		if got := a.Ratio(); got != tc.want {
			t.Errorf("Assumptions.Ratio() = %q, want %q", got, tc.want)
		}
	}
}

// generatedAt is normalized to UTC on both construction and marshalling. A history
// log spanning timezones sorts wrongly otherwise, and it sorts wrongly in a way that
// looks like working output.
func TestGeneratedAtIsAlwaysUTC(t *testing.T) {
	tokyo, err := time.LoadLocation("Asia/Tokyo")
	if err != nil {
		t.Skipf("no tzdata: %v", err)
	}
	local := stamp.In(tokyo)
	if local.Hour() == stamp.Hour() {
		t.Fatal("the fixture time is not actually offset in Tokyo; the test proves nothing")
	}

	e := NewEnvelope("compare", "0.2.0", local)
	if e.GeneratedAt.Location() != time.UTC {
		t.Errorf("NewEnvelope stored a %v time, want UTC", e.GeneratedAt.Location())
	}
	if !e.GeneratedAt.Equal(stamp) {
		t.Errorf("normalizing to UTC changed the instant: %v vs %v", e.GeneratedAt, stamp)
	}

	// And a hand-built envelope carrying a non-UTC time is normalized on the way out,
	// so the wire format never depends on where the tool ran.
	full := complete()
	full.GeneratedAt = local
	data, err := json.Marshal(full)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if !strings.Contains(string(data), `"generatedAt":"2026-07-29T14:30:00Z"`) {
		t.Errorf("marshalled generatedAt is not UTC RFC3339:\n%s", data)
	}
}

// Regions serialize in the order requested, not the order they completed. Discovery
// fans out across goroutines, so completion order varies run to run and a history
// log full of reordered-but-identical reports is one where a real change cannot be
// spotted in a diff.
func TestRegionsSerializeInTheOrderRequested(t *testing.T) {
	e := complete()
	e.Query.Regions = []string{"us-east-2", "us-east-1", "us-west-2"}
	// Arrived in a different order, as concurrent queries do.
	e.Regions = []Region{
		{Name: "us-west-2", State: RegionOK, Considered: 40, Usable: 8},
		{Name: "us-east-1", State: RegionOK, Considered: 69, Usable: 12},
		{Name: "us-east-2", State: RegionOK, Considered: 61, Usable: 14},
	}
	if err := e.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}

	data, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Envelope
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	var names []string
	for _, r := range got.Regions {
		names = append(names, r.Name)
	}
	want := "us-east-2 us-east-1 us-west-2"
	if strings.Join(names, " ") != want {
		t.Errorf("regions serialized as %q, want the requested order %q", strings.Join(names, " "), want)
	}

	// Two envelopes differing only in arrival order must serialize identically, which
	// is the property that makes a history diff meaningful.
	shuffled := e
	shuffled.Regions = []Region{e.Regions[2], e.Regions[0], e.Regions[1]}
	other, err := json.Marshal(shuffled)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(data) != string(other) {
		t.Errorf("arrival order changed the bytes:\n%s\n%s", data, other)
	}
}

// Ordering must not drop anything, even a region that should not be there. Silently
// discarding an unrequested result would hide the inconsistency Validate reports.
func TestOrderingNeverDropsARegion(t *testing.T) {
	got := orderRegions([]Region{
		{Name: "ap-northeast-1", State: RegionEmpty},
		{Name: "us-east-2", State: RegionOK, Usable: 1},
		{Name: "eu-central-1", State: RegionEmpty},
	}, []string{"us-east-2"})

	if len(got) != 3 {
		t.Fatalf("orderRegions returned %d of 3 regions; a dropped result hides a bug", len(got))
	}
	// Requested first, then the strays alphabetically for a stable diff.
	want := "us-east-2 ap-northeast-1 eu-central-1"
	var names []string
	for _, r := range got {
		names = append(names, r.Name)
	}
	if strings.Join(names, " ") != want {
		t.Errorf("order = %q, want %q", strings.Join(names, " "), want)
	}
}

// Age is the outer bound on staleness — nothing in the report is newer — which is
// what lets a page degrade its own claims rather than presenting a cached
// availability figure as current.
func TestAgeIsTheOuterBoundOnStaleness(t *testing.T) {
	e := complete()
	now := stamp.Add(14 * time.Hour)
	if got := e.Age(now); got != 14*time.Hour {
		t.Errorf("Age = %v, want 14h", got)
	}

	// An amount inside can legitimately be older than the envelope, since a price read
	// from a cache predates the report containing it. The envelope's age must not be
	// mistaken for its contents'.
	older := Live(4.0, UnitUSDPerHour, "Price List", stamp.Add(-3*time.Hour))
	amountAge, ok := older.Age(now)
	if !ok {
		t.Fatal("a live amount must have an age")
	}
	if amountAge <= e.Age(now) {
		t.Errorf("contained amount age %v is not older than the envelope's %v; the fixture "+
			"does not exercise the case", amountAge, e.Age(now))
	}
}

// The optional AWS identity fields are omitted rather than emitted empty: prices are
// not account-specific, so most reports have nothing to say here, and an empty
// string would read as an account named "".
func TestAWSIdentityIsOptional(t *testing.T) {
	e := complete()
	data, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if strings.Contains(string(data), "account") {
		t.Errorf("an unset account was serialized:\n%s", data)
	}
	if err := e.Validate(); err != nil {
		t.Errorf("an envelope without an account must validate; prices are not "+
			"account-specific: %v", err)
	}

	// When set — as it must be for a quota finding, which is account-specific — it
	// round-trips.
	e.Account = "942542972736"
	e.Partition = "aws"
	data, err = json.Marshal(e)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Envelope
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if got.Account != "942542972736" || got.Partition != "aws" {
		t.Errorf("identity = %q/%q, want 942542972736/aws", got.Account, got.Partition)
	}
}

// Optional means absence is allowed, not that anything goes. Neither field has a unit
// or a provenance, so nothing downstream would notice a profile name in Account or a
// region in Partition — and the partition is the one field a reader would use to
// decide whether the report's prices apply to them at all.
func TestAMalformedIdentityIsRefused(t *testing.T) {
	for _, tc := range []struct {
		name          string
		account, part string
		wantErr       string
	}{
		{"an ARN in the account field", "arn:aws:iam::942542972736:user/x", "aws", "12-digit"},
		{"a profile name", "aws", "aws", "12-digit"},
		{"a hyphenated account", "9425-4297-2736", "aws", "12-digit"},
		{"eleven digits", "94254297273", "aws", "12-digit"},
		{"thirteen digits", "9425429727360", "aws", "12-digit"},
		{"twelve non-digits", "94254297273x", "aws", "not all digits"},
		{"a region as the partition", "942542972736", "us-east-1", "not an AWS partition"},
		{"a service namespace", "942542972736", "ec2", "not an AWS partition"},
		{"a partition-ish typo", "942542972736", "awsgov", "not an AWS partition"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			e := complete()
			e.Account, e.Partition = tc.account, tc.part
			err := e.Validate()
			if err == nil {
				t.Fatalf("Validate accepted account=%q partition=%q", tc.account, tc.part)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("error %q does not mention %q", err, tc.wantErr)
			}
		})
	}

	// The partitions that are real must pass, including ones this tool will probably
	// never see: an unrecognized aws-iso member is still a partition, and rejecting it
	// would fail a legitimate run over a field that is only informative. Each is
	// checked with the account both set and empty, since the two are independent.
	for _, part := range []string{"aws", "aws-us-gov", "aws-cn", "aws-iso", "aws-iso-b", "aws-iso-f"} {
		for _, acct := range []string{"942542972736", ""} {
			e := complete()
			e.Account, e.Partition = acct, part
			if err := e.Validate(); err != nil {
				t.Errorf("partition %q with account %q rejected: %v", part, acct, err)
			}
		}
	}
	// An account with no partition is also legitimate: the ARN may not have parsed,
	// and the account is still the fact a quota finding needs.
	e := complete()
	e.Account = "942542972736"
	if err := e.Validate(); err != nil {
		t.Errorf("an account with no partition rejected: %v", err)
	}
}

// The whole envelope survives a round trip unchanged, including the unexported
// fields of every contained Amount. This is the schema-freeze property #16 depends
// on: a consumer reading a stored report must see what the producer wrote.
func TestTheWholeEnvelopeRoundTrips(t *testing.T) {
	e := complete()
	e.Account, e.Partition = "942542972736", "aws"
	e.Query.InstanceCount, e.Query.DurationHours = 2, 19
	e.Subject.Quantization = "mxfp4"
	e.Regions[1] = Region{Name: "us-east-2", State: RegionFailed, Error: "AccessDenied: not authorized"}

	if err := e.Validate(); err != nil {
		t.Fatalf("Validate: %v", err)
	}
	data, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var got Envelope
	if err := json.Unmarshal(data, &got); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if err := got.Validate(); err != nil {
		t.Fatalf("the decoded envelope does not validate: %v", err)
	}

	// Marshalling the decode must reproduce the bytes. Comparing serialized forms
	// rather than structs catches a field that decodes into the wrong place as well as
	// one that is dropped, and it covers Amount's unexported fields without reaching
	// into them.
	again, err := json.Marshal(got)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	if string(data) != string(again) {
		t.Errorf("round trip changed the bytes:\n%s\n%s", data, again)
	}
	if got.Regions[1].Error != "AccessDenied: not authorized" {
		t.Errorf("the failure reason was lost: %q", got.Regions[1].Error)
	}
	if got.Subject.BedrockModelID != "qwen.qwen3-32b-v1:0" {
		t.Errorf("bedrock id = %q", got.Subject.BedrockModelID)
	}
}

// The field names are the wire contract #16 freezes. A rename is invisible to Go
// tests that go through the struct, and breaks every stored report and the page at
// once — so the names are asserted as strings.
func TestTheWireContractIsStable(t *testing.T) {
	e := complete()
	e.Query.InstanceCount = 8
	data, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	for _, key := range []string{
		"schemaVersion", "kind", "toolVersion", "generatedAt",
		"subject", "query", "regions", "assumptions",
	} {
		if _, ok := raw[key]; !ok {
			t.Errorf("top-level key %q is missing; renaming it breaks every stored report", key)
		}
	}

	var assumptions map[string]json.RawMessage
	if err := json.Unmarshal(raw["assumptions"], &assumptions); err != nil {
		t.Fatalf("Unmarshal assumptions: %v", err)
	}
	for _, key := range []string{
		"inputTokenWeight", "outputTokenWeight", "utilization",
		"contextTokens", "concurrency", "kvCacheDTypeBytes", "overheadFraction", "throughput",
	} {
		if _, ok := assumptions[key]; !ok {
			t.Errorf("assumptions key %q is missing", key)
		}
	}

	if got := string(raw["schemaVersion"]); got != `"report.v1"` {
		t.Errorf("schemaVersion = %s, want \"report.v1\"; bumping it is a deliberate act, "+
			"not something to discover from a failing test elsewhere", got)
	}
}

func TestRegionStateHelpers(t *testing.T) {
	for _, s := range []RegionState{RegionOK, RegionEmpty, RegionFailed} {
		if !s.Valid() {
			t.Errorf("%s is not Valid()", s)
		}
	}
	for _, s := range []RegionState{"", "denied", "throttled", "OK"} {
		if RegionState(s).Valid() {
			t.Errorf("%q is Valid(); an undefined state must not pass, or a typo becomes a state", s)
		}
	}
	// Informative is the load-bearing one: it decides whether a conclusion about a
	// region is a conclusion about AWS or about the query.
	if !RegionOK.Informative() || !RegionEmpty.Informative() {
		t.Error("a completed query is informative whether or not it found anything")
	}
	if RegionFailed.Informative() {
		t.Error("a failed query is not informative; treating it as such states an " +
			"AccessDenied as an absence of capacity")
	}
	// An unrecognized state — a future member this build predates — must not be
	// treated as informative. Guessing yes would turn a state this code does not
	// understand into a claim about capacity.
	if RegionState("throttled").Informative() {
		t.Error("an unrecognized state must not be informative")
	}
}

// The compatibility policy is the thing consumers rely on, so it is asserted rather
// than left as prose that can quietly stop being true.
func TestNewEnvelopeStampsTheCurrentSchema(t *testing.T) {
	e := NewEnvelope("price", "1.2.3", stamp)
	if e.SchemaVersion != SchemaVersion {
		t.Errorf("SchemaVersion = %q, want %q", e.SchemaVersion, SchemaVersion)
	}
	if e.Kind != "price" || e.ToolVersion != "1.2.3" {
		t.Errorf("kind/version = %q/%q", e.Kind, e.ToolVersion)
	}
	// A fresh envelope must NOT validate: it has no subject, no regions, no
	// assumptions. Making the constructor produce a publishable-looking value would
	// defeat the point of validating at all.
	if err := e.Validate(); err == nil {
		t.Error("a bare envelope validated; the constructor must not fill in defaults for " +
			"fields whose absence makes a report uninterpretable")
	}
}

// Unknown fields are ignored, not rejected — the additive half of the compatibility
// policy. A v0.3 report read by a v0.2 binary must still parse, or every schema
// addition is a breaking change in practice.
func TestAnUnknownFieldIsIgnored(t *testing.T) {
	data, err := json.Marshal(complete())
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	// Splice in a field a future version might add.
	injected := strings.Replace(string(data), `{"schemaVersion"`,
		`{"obtainability":{"score":7},"schemaVersion"`, 1)
	if injected == string(data) {
		t.Fatal("failed to inject an unknown field; the test proves nothing")
	}

	var got Envelope
	if err := json.Unmarshal([]byte(injected), &got); err != nil {
		t.Fatalf("an unknown field must be ignored, not rejected: %v", err)
	}
	if err := got.Validate(); err != nil {
		t.Errorf("Validate after ignoring an unknown field: %v", err)
	}
	if got.Subject.ModelID != "Qwen/Qwen3-32B" {
		t.Errorf("model id = %q; the known fields must survive", got.Subject.ModelID)
	}
}
