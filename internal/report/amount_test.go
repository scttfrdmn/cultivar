package report

import (
	"encoding/json"
	"math"
	"strings"
	"testing"
	"time"
)

// Real values measured against live AWS APIs on 2026-07-27, us-east-1 unless
// noted. They appear here so the arithmetic is exercised on the numbers the tool
// actually reports rather than on round test constants.
var (
	observed = time.Date(2026, 7, 27, 12, 0, 0, 0, time.UTC)

	// g7e.4xlarge on-demand. spore-host/libs' static table says $0.80 (5x low).
	g7e4xlarge = Live(4.00, UnitUSDPerHour, "PriceList OnDemand g7e.4xlarge us-east-1", observed)
	// p6-b200.48xlarge on-demand. The static table says $9.60 (11.9x low).
	p6b200 = Live(113.93, UnitUSDPerHour, "PriceList OnDemand p6-b200.48xlarge us-east-1", observed)
	// p5e.48xlarge has NO on-demand row of any kind. The static table invents $9.60.
	p5e48xlarge = Unavailable(UnitUSDPerHour, "no on-demand price exists; capacity-block only")

	// Bedrock Qwen3-32B standard tier, separate meters.
	qwenInput  = Live(0.15, UnitUSDPerMillionTokens, "PriceList Bedrock Qwen3 32B input standard", observed)
	qwenOutput = Live(0.60, UnitUSDPerMillionTokens, "PriceList Bedrock Qwen3 32B output standard", observed)
)

func TestZeroAmountIsNotAPrice(t *testing.T) {
	// The failure this package exists to prevent: a struct literal that forgot to
	// set a price must not read as free. The zero Amount has an invalid provenance,
	// no value, and fails validation.
	var a Amount
	if _, ok := a.Value(); ok {
		t.Error("zero Amount reports a known value")
	}
	if a.Provenance().Valid() {
		t.Errorf("zero Amount provenance %q claims to be valid", a.Provenance())
	}
	if err := a.Valid(); err == nil {
		t.Error("zero Amount passed validation")
	}
}

func TestUnavailableIsNotZero(t *testing.T) {
	// p5e.48xlarge has no on-demand price. Reporting $0.00 would make it look like
	// the cheapest 8xH200 box on AWS; reporting a guess makes it look 12x cheap.
	if _, ok := p5e48xlarge.Value(); ok {
		t.Fatal("unavailable amount reports a known value")
	}
	if got := p5e48xlarge.String(); got != "unpriced" {
		t.Errorf("String() = %q, want %q", got, "unpriced")
	}
	if !strings.Contains(p5e48xlarge.Source(), "capacity-block only") {
		t.Errorf("Source() = %q, does not say what would make the price obtainable", p5e48xlarge.Source())
	}
}

func TestUnknownNonMoneyReadsAsUnknown(t *testing.T) {
	// Throughput is commonly unknown before any benchmark has run, and "unpriced"
	// would be nonsense for it.
	a := Unavailable(UnitTokensPerSecond, "no measurement for this engine version")
	if got := a.String(); got != "unknown" {
		t.Errorf("String() = %q, want %q", got, "unknown")
	}
}

func TestValidCatchesMalformedAmounts(t *testing.T) {
	cases := []struct {
		name string
		a    Amount
		want string
	}{
		{"no unit", Amount{value: 1, provenance: ProvenanceLive, source: "x", observedAt: observed}, "no unit"},
		{"no source", Amount{value: 1, unit: UnitUSDPerHour, provenance: ProvenanceLive, observedAt: observed}, "no source"},
		{"bad provenance", Amount{value: 1, unit: UnitUSDPerHour, provenance: "guessed", source: "x"}, "not one of"},
		{"live without time", Amount{value: 1, unit: UnitUSDPerHour, provenance: ProvenanceLive, source: "x"}, "no observation time"},
		{"NaN", Amount{value: math.NaN(), unit: UnitUSDPerHour, provenance: ProvenanceDerived, source: "x"}, "not finite"},
		{"Inf", Amount{value: math.Inf(1), unit: UnitUSDPerHour, provenance: ProvenanceDerived, source: "x"}, "not finite"},
		{"negative money", Amount{value: -1, unit: UnitUSDPerHour, provenance: ProvenanceDerived, source: "x"}, "negative"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.a.Valid()
			if err == nil {
				t.Fatalf("Valid() = nil, want an error containing %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("Valid() = %v, want an error containing %q", err, tc.want)
			}
		})
	}
	if err := g7e4xlarge.Valid(); err != nil {
		t.Errorf("a well-formed live amount failed validation: %v", err)
	}
}

func TestLiveAmountRequiresObservationTime(t *testing.T) {
	// Availability data has a shelf life of hours. A live amount that cannot say
	// when it was read cannot be aged out, so it is invalid.
	a := Live(4.00, UnitUSDPerHour, "PriceList", time.Time{})
	if err := a.Valid(); err == nil {
		t.Error("live amount with no observation time passed validation")
	}
}

func TestAge(t *testing.T) {
	now := observed.Add(14 * time.Hour)
	got, ok := g7e4xlarge.Age(now)
	if !ok {
		t.Fatal("live amount has no age")
	}
	if got != 14*time.Hour {
		t.Errorf("Age() = %v, want 14h", got)
	}
	// A derived amount has no observation time of its own; its age is that of its
	// inputs, which the caller must track.
	if _, ok := Derived(1, UnitUSDPerHour, "x").Age(now); ok {
		t.Error("derived amount reports an age")
	}
}

func TestStringFormatting(t *testing.T) {
	cases := []struct {
		a    Amount
		want string
	}{
		{g7e4xlarge, "$4/hr"},
		{p6b200, "$113.93/hr"},
		{Live(6.8752, UnitUSDPerHour, "p5.4xlarge", observed), "$6.8752/hr"},
		{Live(1146.24, UnitUSD, "capacity block upfront fee", observed), "$1146.24"},
		{qwenInput, "$0.15/1M tok"},
		{Live(4223, UnitTokensPerSecond, "break-even", observed), "4223 tok/s"},
		{Live(63.5, UnitGiB, "Qwen3-32B BF16", observed), "63.5 GiB"},
		{Live(19, UnitHours, "partial capacity block", observed), "19h"},
		{Live(0.245, UnitFraction, "p5 CB discount", observed), "24.5%"},
	}
	for _, tc := range cases {
		if got := tc.a.String(); got != tc.want {
			t.Errorf("String() = %q, want %q", got, tc.want)
		}
	}
}

func TestJSONRoundTrip(t *testing.T) {
	for _, a := range []Amount{g7e4xlarge, p5e48xlarge, qwenOutput,
		External(2400, UnitTokensPerSecond, "vLLM 0.9 blog post", observed),
		Derived(4223, UnitTokensPerSecond, "break-even = rate / token price")} {
		data, err := json.Marshal(a)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var back Amount
		if err := json.Unmarshal(data, &back); err != nil {
			t.Fatalf("unmarshal %s: %v", data, err)
		}
		if back.provenance != a.provenance || back.unit != a.unit || back.source != a.source {
			t.Errorf("round trip changed metadata: %+v -> %+v", a, back)
		}
		av, aok := a.Value()
		bv, bok := back.Value()
		if aok != bok || av != bv {
			t.Errorf("round trip changed value: (%v,%v) -> (%v,%v)", av, aok, bv, bok)
		}
	}
}

func TestUnavailableMarshalsAsNullNotZero(t *testing.T) {
	// A JSON consumer that ignores provenance entirely must still be unable to
	// mistake "no price" for "free".
	data, err := json.Marshal(p5e48xlarge)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		t.Fatalf("unmarshal to map: %v", err)
	}
	if v, ok := raw["value"]; !ok || v != nil {
		t.Errorf("value = %#v, want null", v)
	}
	if raw["provenance"] != string(ProvenanceUnavailable) {
		t.Errorf("provenance = %#v, want %q", raw["provenance"], ProvenanceUnavailable)
	}
}

func TestEveryMarshalledAmountCarriesProvenance(t *testing.T) {
	// The invariant CI enforces: no numeric in a report travels bare.
	for _, a := range []Amount{g7e4xlarge, p5e48xlarge, qwenInput} {
		data, err := json.Marshal(a)
		if err != nil {
			t.Fatalf("marshal: %v", err)
		}
		var raw map[string]any
		if err := json.Unmarshal(data, &raw); err != nil {
			t.Fatal(err)
		}
		for _, field := range []string{"provenance", "unit", "source"} {
			if v, ok := raw[field]; !ok || v == "" {
				t.Errorf("marshalled amount is missing %s: %s", field, data)
			}
		}
	}
}

func TestUnmarshalRejectsProvenanceWithoutValue(t *testing.T) {
	// A truncated or hand-edited report must not decode into a zero price.
	const payload = `{"value":null,"unit":"USD/hour","provenance":"live","source":"PriceList"}`
	var a Amount
	if err := json.Unmarshal([]byte(payload), &a); err == nil {
		t.Error("unmarshal accepted a live amount with a null value")
	}
}

// And the converse, which is the direction that costs money.
//
// The number in this payload is the fabricated one: truffle's fallback pricer returns
// $9.60/hr for p5e.48xlarge, which has no on-demand rate at all. A report that decoded
// this silently would carry a price the tool considers unresolved *and* be unable to show
// it — [Amount.Value] reports an unavailable amount as unknown — so the only consumer that
// ever sees the 9.60 is one reading the raw JSON, which is the careless one.
func TestUnmarshalRejectsAValueItCallsUnavailable(t *testing.T) {
	const payload = `{"value":9.60,"unit":"USD/hour","provenance":"unavailable","source":"no on-demand price exists"}`
	var a Amount
	err := json.Unmarshal([]byte(payload), &a)
	if err == nil {
		v, known := a.Value()
		t.Errorf("unmarshal accepted an unavailable amount carrying 9.60; it decoded to "+
			"value %v known=%v, so the asserted price is invisible to every accessor", v, known)
	}
	if err != nil && !strings.Contains(err.Error(), "9.6") {
		t.Errorf("the error does not name the offending value, so a reader cannot tell what "+
			"was rejected: %v", err)
	}
}

func TestUnmarshalRejectsUnknownProvenance(t *testing.T) {
	const payload = `{"value":9.60,"unit":"USD/hour","provenance":"estimated","source":"family heuristic"}`
	var a Amount
	if err := json.Unmarshal([]byte(payload), &a); err == nil {
		t.Error(`unmarshal accepted provenance "estimated"; there is no estimated provenance by design`)
	}
}

func TestMustValuePanicsOnUnavailable(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Error("MustValue did not panic on an unavailable amount")
		}
	}()
	_ = p5e48xlarge.MustValue()
}

func TestProvenanceKnown(t *testing.T) {
	for _, p := range []Provenance{ProvenanceLive, ProvenanceDerived, ProvenanceExternal} {
		if !p.Known() {
			t.Errorf("%s.Known() = false", p)
		}
	}
	if ProvenanceUnavailable.Known() {
		t.Error("unavailable.Known() = true")
	}
	if Provenance("").Valid() {
		t.Error(`empty provenance is valid; a forgotten field would default silently`)
	}
}

func TestUnitIsMoney(t *testing.T) {
	money := []Unit{UnitUSD, UnitUSDPerHour, UnitUSDPerMillionTokens, UnitUSDPerGBMonth}
	notMoney := []Unit{UnitTokensPerSecond, UnitGiB, UnitHours, UnitCount, UnitFraction}
	for _, u := range money {
		if !u.IsMoney() {
			t.Errorf("%s.IsMoney() = false", u)
		}
	}
	for _, u := range notMoney {
		if u.IsMoney() {
			t.Errorf("%s.IsMoney() = true", u)
		}
	}
}
