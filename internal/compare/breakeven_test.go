package compare

import (
	"math"
	"strings"
	"testing"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// Qwen3-32B in us-east-1, standard tier, read from Price List 2026-07-27. These two
// meters and the ratio between them are what every figure in this file derives from.
var (
	qwenInput  = report.Live(0.15, report.UnitUSDPerMillionTokens, "Price List Qwen3 32B input", observed)
	qwenOutput = report.Live(0.60, report.UnitUSDPerMillionTokens, "Price List Qwen3 32B output", observed)
)

// threeToOne is the mix the plan's worked example assumes, at full utilization —
// the baseline the issue's table was computed under.
var threeToOne = Assumptions{InputWeight: 3, OutputWeight: 1, Utilization: 1.0}

func rate(v float64, name string) report.Amount {
	return report.Live(v, report.UnitUSDPerHour, name, observed)
}

func tps(v float64, source string) report.Amount {
	return report.External(v, report.UnitTokensPerSecond, source, observed)
}

func parity(t *testing.T, hourly report.Amount, a Assumptions) Parity {
	t.Helper()
	p, err := BreakEven(hourly, qwenInput, qwenOutput, a)
	if err != nil {
		t.Fatalf("BreakEven: %v", err)
	}
	return p
}

// TestBreakEvenReproducesTheWorkedExample is the spec. The figures are issue #12's
// corrected table, which internal/report's TestBreakEvenThroughput pins with raw
// arithmetic; this asserts the engine agrees with it, so the two cannot drift apart
// silently.
func TestBreakEvenReproducesTheWorkedExample(t *testing.T) {
	cases := []struct {
		instance string
		hourly   float64
		want     float64 // tok/s
	}{
		{"g7e.4xlarge", 4.00, 4233},
		{"p5.4xlarge", 6.88, 7280},
		{"g6e.12xlarge", 10.49, 11100},
		{"p5.48xlarge", 55.04, 58243},
		{"p6-b200.48xlarge", 113.93, 120561},
	}
	for _, tc := range cases {
		t.Run(tc.instance, func(t *testing.T) {
			p := parity(t, rate(tc.hourly, tc.instance), threeToOne)

			if got := round(p.TokenPrice.MustValue(), 4); got != 0.2625 {
				t.Errorf("3:1 blend = %v, want 0.2625", got)
			}
			if got := p.Throughput.MustValue(); math.Abs(got-tc.want) > 1 {
				t.Errorf("break-even = %.0f tok/s, want ~%.0f", got, tc.want)
			}
			if p.Throughput.Unit() != report.UnitTokensPerSecond {
				t.Errorf("unit = %s, want tok/s", p.Throughput.Unit())
			}
			if p.Throughput.Provenance() != report.ProvenanceDerived {
				t.Errorf("provenance = %s, want derived", p.Throughput.Provenance())
			}
		})
	}
}

// The verdict this whole tool exists to deliver. Qwen3-32B on one g7e.4xlarge does
// not come close, and the useful output is the multiple, not the boolean.
func TestTheHonestAnswerForQwen3IsBedrock(t *testing.T) {
	p := parity(t, rate(4.00, "g7e.4xlarge"), threeToOne)
	// A generous single-GPU figure for a 32B model at bf16; external because nothing
	// here has been measured on the hardware.
	c := p.At(tps(1200, "vLLM published benchmark"))

	if c.Outcome != OutcomeBedrock {
		t.Fatalf("outcome = %s, want bedrock", c.Outcome)
	}
	if got := round(c.Shortfall.MustValue(), 1); got != 3.5 {
		t.Errorf("shortfall = %vx, want 3.5x (4233 / 1200)", got)
	}
	// $4.00/hr over 1200 tok/s = 4.32M tokens/hr => $0.926/1M against Bedrock's
	// $0.2625/1M.
	if got := round(c.SelfHostCost.MustValue(), 4); got != 0.9259 {
		t.Errorf("self-hosted cost = $%v/1M, want $0.9259/1M", got)
	}
	if report.Compare(c.SelfHostCost, p.TokenPrice) != 1 {
		t.Error("expected self-hosting to cost more per token than Bedrock")
	}
}

// Beating parity is what flips the verdict, and the shortfall has to invert into a
// margin rather than saturating at 1.
func TestSelfHostingWinsAboveParity(t *testing.T) {
	p := parity(t, rate(4.00, "g7e.4xlarge"), threeToOne)
	c := p.At(tps(8466, "hypothetical 2x parity"))

	if c.Outcome != OutcomeSelfHost {
		t.Fatalf("outcome = %s, want self-host", c.Outcome)
	}
	if got := round(c.Shortfall.MustValue(), 2); got != 0.5 {
		t.Errorf("shortfall = %v, want 0.5 (i.e. 2x margin)", got)
	}
	if report.Compare(c.SelfHostCost, p.TokenPrice) != -1 {
		t.Error("expected self-hosting to cost less per token above parity")
	}
}

// A tie is not a reason to self-host: at equal cost the serverless option still
// avoids the quota request, the capacity hunt, and the instance left running.
func TestATieGoesToBedrock(t *testing.T) {
	p := parity(t, rate(4.00, "g7e.4xlarge"), threeToOne)
	c := p.At(report.Derived(p.Throughput.MustValue(), report.UnitTokensPerSecond, "exactly parity"))
	if c.Outcome != OutcomeBedrock {
		t.Errorf("outcome at exact parity = %s, want bedrock", c.Outcome)
	}
}

// TestUtilizationIsNotAppliedToBothSides is the point of the utilization term. An
// idle GPU bills and an idle Bedrock endpoint does not, so the penalty must land on
// one side only. Cancelling it out is the bug: it would make a GPU used 8 hours a
// day look exactly as economical as a saturated one.
func TestUtilizationIsNotAppliedToBothSides(t *testing.T) {
	full := parity(t, rate(4.00, "g7e.4xlarge"), threeToOne)

	third := threeToOne
	third.Utilization = 1.0 / 3.0 // eight hours a day
	partial := parity(t, rate(4.00, "g7e.4xlarge"), third)

	// Required throughput triples: the same hourly bill must be recovered from a
	// third as many tokens.
	if got, want := partial.Throughput.MustValue(), full.Throughput.MustValue()*3; math.Abs(got-want) > 1 {
		t.Errorf("break-even at 33%% utilization = %.0f tok/s, want ~%.0f (3x the saturated figure)", got, want)
	}
	// And the same physical throughput costs 3x per token produced.
	measured := tps(1200, "vLLM published benchmark")
	fullCost := full.CostPerMillion(measured).MustValue()
	partialCost := partial.CostPerMillion(measured).MustValue()
	if got, want := partialCost/fullCost, 3.0; math.Abs(got-want) > 0.001 {
		t.Errorf("cost ratio at 33%% vs 100%% utilization = %.3fx, want 3x", got)
	}
	// The Bedrock rate is untouched by how busy someone keeps their own hardware.
	if full.TokenPrice.MustValue() != partial.TokenPrice.MustValue() {
		t.Error("utilization changed the Bedrock token price; it bills per token, not per hour")
	}
}

// Utilization has no default, because 100% is the single most flattering assumption
// available to a self-hosting recommendation and is almost never true.
func TestUtilizationMustBeStated(t *testing.T) {
	cases := []struct {
		name string
		u    float64
	}{
		{"zero value is not a default", 0},
		{"negative", -0.5},
		{"above one", 1.5},
		{"NaN", math.NaN()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := Assumptions{InputWeight: 3, OutputWeight: 1, Utilization: tc.u}
			if _, err := BreakEven(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput, a); err == nil {
				t.Fatalf("utilization %v was accepted", tc.u)
			}
		})
	}
	// 1.0 is a legitimate claim — just one that has to be made explicitly.
	if _, err := BreakEven(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput, threeToOne); err != nil {
		t.Errorf("100%% utilization rejected: %v", err)
	}
}

// The ratio moves the answer by 43%, which is enough to flip a recommendation, so
// it can never be a buried constant.
func TestTheBlendRatioMovesTheAnswer(t *testing.T) {
	three := parity(t, rate(4.00, "g7e.4xlarge"), threeToOne)
	one := parity(t, rate(4.00, "g7e.4xlarge"),
		Assumptions{InputWeight: 1, OutputWeight: 1, Utilization: 1.0})

	if got := round(one.TokenPrice.MustValue(), 4); got != 0.375 {
		t.Errorf("1:1 blend = %v, want 0.375", got)
	}
	if got := round(one.Throughput.MustValue(), 0); got != 2963 {
		t.Errorf("1:1 break-even = %v tok/s, want 2963", got)
	}
	// Note the direction: a pricier Bedrock token makes self-hosting *easier* to
	// justify, so the 1:1 requirement is lower, not higher.
	if report.Compare(one.Throughput, three.Throughput) != -1 {
		t.Error("a more expensive Bedrock blend should lower the break-even bar")
	}
	if got := round((one.TokenPrice.MustValue()-three.TokenPrice.MustValue())/three.TokenPrice.MustValue()*100, 0); got != 43 {
		t.Errorf("1:1 vs 3:1 price swing = %v%%, want 43%%", got)
	}
}

// Weights of zero are meaningful (all-input or all-output traffic); weights that
// sum to zero are not a mix at all.
func TestDegenerateWeights(t *testing.T) {
	allOutput := parity(t, rate(4.00, "g7e.4xlarge"),
		Assumptions{InputWeight: 0, OutputWeight: 1, Utilization: 1.0})
	if got := round(allOutput.TokenPrice.MustValue(), 4); got != 0.60 {
		t.Errorf("0:1 blend = %v, want the output rate 0.60", got)
	}
	for _, a := range []Assumptions{
		{InputWeight: 0, OutputWeight: 0, Utilization: 1.0},
		{InputWeight: -1, OutputWeight: 2, Utilization: 1.0},
		{InputWeight: math.NaN(), OutputWeight: 1, Utilization: 1.0},
	} {
		if _, err := BreakEven(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput, a); err == nil {
			t.Errorf("weights %g:%g were accepted", a.InputWeight, a.OutputWeight)
		}
		// Also asserted on Validate directly. BreakEven would reject the negative case
		// anyway via report.Blend, so testing only through BreakEven would leave
		// Validate's own check unexercised — and Validate is what a CLI calls to
		// reject a bad --input-output-ratio before doing any work.
		if err := a.Validate(); err == nil {
			t.Errorf("Validate accepted weights %g:%g", a.InputWeight, a.OutputWeight)
		}
	}
}

// TestAnInstanceWithNoOnDemandPriceIsUndetermined covers p5e.48xlarge, which has no
// on-demand rate at all. truffle's default pricer invents $9.60 for it; the whole
// point of the Amount type is that a missing rate stays missing, and a comparison
// built on one must not name a winner.
func TestAnInstanceWithNoOnDemandPriceIsUndetermined(t *testing.T) {
	unpriced := report.Unavailable(report.UnitUSDPerHour,
		"no on-demand price exists for p5e.48xlarge; capacity-block only")
	p, err := BreakEven(unpriced, qwenInput, qwenOutput, threeToOne)
	if err != nil {
		t.Fatalf("an unpriced instance is a state to report, not an error: %v", err)
	}
	if p.Throughput.Known() {
		t.Errorf("break-even = %v from an unpriced instance", p.Throughput)
	}
	if !strings.Contains(p.Throughput.Source(), "capacity-block only") {
		t.Errorf("reason lost the cause: %q", p.Throughput.Source())
	}

	c := p.At(tps(1200, "vLLM published benchmark"))
	if c.Outcome != OutcomeUndetermined {
		t.Errorf("outcome = %s, want undetermined", c.Outcome)
	}
	if c.SelfHostCost.Known() || c.Shortfall.Known() {
		t.Errorf("derived a cost (%v) or shortfall (%v) with no instance price",
			c.SelfHostCost, c.Shortfall)
	}
}

// 94 of 132 mappable HF models are Bedrock marketplace-only: no token meter exists,
// so there is nothing to be cheaper than. That is not the same claim as
// "self-hosting wins", and conflating them would have the tool recommend hardware
// on the strength of a comparison it never made.
func TestNoTokenPriceIsItsOwnOutcome(t *testing.T) {
	noRate := report.Unavailable(report.UnitUSDPerMillionTokens,
		"no standard-tier output token rate published for meta.llama3-1-405b in us-east-1")
	p, err := BreakEven(rate(4.00, "g7e.4xlarge"), qwenInput, noRate, threeToOne)
	if err != nil {
		t.Fatalf("a marketplace-only model is a state to report, not an error: %v", err)
	}
	if p.TokenPrice.Known() || p.Throughput.Known() {
		t.Errorf("blended a price (%v) from a missing meter, break-even %v", p.TokenPrice, p.Throughput)
	}

	c := p.At(tps(1200, "vLLM published benchmark"))
	if c.Outcome != OutcomeNoTokenPrice {
		t.Fatalf("outcome = %s, want no-token-price", c.Outcome)
	}
	if c.Outcome.Decided() {
		t.Error("no-token-price must not read as a winner")
	}
	// The self-hosted cost is still the answer to "what would this cost me?", and it
	// is computable without any Bedrock rate.
	if got := round(c.SelfHostCost.MustValue(), 4); got != 0.9259 {
		t.Errorf("self-hosted cost = $%v/1M, want $0.9259/1M even with no Bedrock rate", got)
	}
}

// Until a benchmark runs there is often no throughput figure at all. The required
// throughput is still computable and still useful — that is the whole reason the
// engine is split at this seam — but the comparison cannot be decided.
func TestNoThroughputStillYieldsARequiredThroughput(t *testing.T) {
	p := parity(t, rate(4.00, "g7e.4xlarge"), threeToOne)
	if !p.Throughput.Known() {
		t.Fatal("required throughput needs no measurement")
	}
	for _, unknown := range []report.Amount{
		report.Unavailable(report.UnitTokensPerSecond, "no benchmark for Qwen3-32B on g7e.4xlarge"),
		report.External(0, report.UnitTokensPerSecond, "a zero measurement is a missing one", observed),
		report.Derived(-5, report.UnitTokensPerSecond, "nonsense"),
	} {
		c := p.At(unknown)
		if c.Outcome != OutcomeUndetermined {
			t.Errorf("throughput %v gave outcome %s, want undetermined", unknown, c.Outcome)
		}
		if c.Shortfall.Known() || c.SelfHostCost.Known() || c.UtilizationForParity.Known() {
			t.Errorf("throughput %v produced numbers anyway", unknown)
		}
	}
}

// Wrong units are a programming error, not a data gap: an hourly rate passed as a
// token price would blend into a plausible-looking number.
func TestUnitsAreChecked(t *testing.T) {
	perHour := rate(4.00, "g7e.4xlarge")
	perToken := qwenInput
	cases := []struct {
		name                  string
		hourly, input, output report.Amount
	}{
		{"token price as hourly cost", perToken, qwenInput, qwenOutput},
		{"hourly rate as input meter", perHour, perHour, qwenOutput},
		{"hourly rate as output meter", perHour, qwenInput, perHour},
		{"GiB as hourly cost", report.Live(80, report.UnitGiB, "vram", observed), qwenInput, qwenOutput},
		// Both meters wrong in the same way is the case Blend cannot see: it compares
		// its two operands against each other, and two hourly rates agree. Only
		// checking each meter against the unit it is supposed to be in catches this,
		// and without it the blend yields a plausible $/hr figure that goes on to be
		// divided as though it were a token price.
		{"both meters are hourly rates", perHour, perHour, perHour},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := BreakEven(tc.hourly, tc.input, tc.output, threeToOne); err == nil {
				t.Error("mismatched units were accepted")
			}
		})
	}
	// Demonstrating that gap directly: Blend is happy to average two hourly rates.
	if _, err := report.Blend(perHour, 3, perHour, 1, "two hourly rates"); err != nil {
		t.Errorf("expected Blend to accept two same-unit operands (that is the gap): %v", err)
	}

	// A throughput figure in the wrong unit is likewise refused rather than divided.
	p := parity(t, perHour, threeToOne)
	c := p.At(report.Live(4.00, report.UnitUSDPerHour, "not a throughput", observed))
	if c.Shortfall.Known() || c.SelfHostCost.Known() {
		t.Error("a throughput in USD/hour produced numbers")
	}
	if !strings.Contains(c.Shortfall.Source(), "want tok/s") {
		t.Errorf("reason does not name the expected unit: %q", c.Shortfall.Source())
	}
}

// The other unknown in the same equation, and the more actionable one when the
// hardware is already chosen: a faster GPU is a purchase, a busier one is a
// scheduling change.
func TestUtilizationForParityIsTheSameEquationSolvedTheOtherWay(t *testing.T) {
	p := parity(t, rate(4.00, "g7e.4xlarge"), threeToOne)

	// At exactly the break-even throughput, 100% utilization is what it takes.
	atParity := p.UtilizationForParity(
		report.Derived(p.Throughput.MustValue(), report.UnitTokensPerSecond, "parity"))
	if got := round(atParity.MustValue(), 4); got != 1.0 {
		t.Errorf("utilization for parity at the break-even rate = %v, want 1.0", got)
	}

	// Deliberately uncapped: above 100% is the answer, and it says the duty cycle is
	// the wrong lever.
	over := p.UtilizationForParity(tps(1200, "vLLM published benchmark"))
	if got := round(over.MustValue(), 2); got != 3.53 {
		t.Errorf("utilization for parity at 1200 tok/s = %v, want 3.53 (impossible, as intended)", got)
	}
	if over.MustValue() <= 1 {
		t.Error("a figure above 1 was clamped; that hides the finding")
	}

	// Twice the parity throughput needs half the duty cycle.
	half := p.UtilizationForParity(
		report.Derived(p.Throughput.MustValue()*2, report.UnitTokensPerSecond, "2x parity"))
	if got := round(half.MustValue(), 3); got != 0.5 {
		t.Errorf("utilization for parity at 2x = %v, want 0.5", got)
	}
}

// Utilization asymmetry, checked from the other side: the duty cycle needed for
// parity is a property of hardware speed and price, so it must not shift when the
// assumed utilization changes.
func TestUtilizationForParityDoesNotDependOnAssumedUtilization(t *testing.T) {
	measured := tps(1200, "vLLM published benchmark")
	full := parity(t, rate(4.00, "g7e.4xlarge"), threeToOne).UtilizationForParity(measured)

	third := threeToOne
	third.Utilization = 1.0 / 3.0
	partial := parity(t, rate(4.00, "g7e.4xlarge"), third).UtilizationForParity(measured)

	if math.Abs(full.MustValue()-partial.MustValue()) > 1e-9 {
		t.Errorf("required utilization moved with the assumed one: %v vs %v", full, partial)
	}
}

// Every amount that reaches a report has to be well-formed, including the ones
// carrying a refusal: a zero-value Amount renders as $0.00 and validates as
// nothing.
func TestEveryBreakEvenAmountIsWellFormed(t *testing.T) {
	unpricedHour := report.Unavailable(report.UnitUSDPerHour, "capacity-block only")
	noMeter := report.Unavailable(report.UnitUSDPerMillionTokens, "no output rate")
	throughputs := []report.Amount{
		tps(1200, "vLLM published benchmark"),
		report.Unavailable(report.UnitTokensPerSecond, "no benchmark"),
		report.Live(4.00, report.UnitUSDPerHour, "wrong unit entirely", observed),
	}

	for _, hourly := range []report.Amount{rate(4.00, "g7e.4xlarge"), unpricedHour} {
		for _, out := range []report.Amount{qwenOutput, noMeter} {
			p, err := BreakEven(hourly, qwenInput, out, threeToOne)
			if err != nil {
				t.Fatalf("BreakEven: %v", err)
			}
			for _, tc := range throughputs {
				c := p.At(tc)
				for name, a := range map[string]report.Amount{
					"Hourly":               c.Parity.Hourly,
					"TokenPrice":           c.Parity.TokenPrice,
					"Throughput":           c.Parity.Throughput,
					"SelfHostCost":         c.SelfHostCost,
					"Shortfall":            c.Shortfall,
					"UtilizationForParity": c.UtilizationForParity,
				} {
					if err := a.Valid(); err != nil {
						t.Errorf("%s invalid (hourly %v, output %v, tput %v): %v", name, hourly, out, tc, err)
					}
				}
				if !Outcome(c.Outcome).valid() {
					t.Errorf("outcome %q is not one of the defined values", c.Outcome)
				}
			}
		}
	}
}

// The assumptions have to survive into the output, or the recommendation is
// unfalsifiable and two runs under silently different inputs look identical.
func TestAssumptionsAreRecordedInEveryDerivation(t *testing.T) {
	a := Assumptions{InputWeight: 3, OutputWeight: 1, Utilization: 0.4}
	p := parity(t, rate(4.00, "g7e.4xlarge"), a)
	c := p.At(tps(1200, "vLLM published benchmark"))

	if p.Assumptions != a {
		t.Errorf("assumptions = %+v, want %+v", p.Assumptions, a)
	}
	for name, src := range map[string]string{
		"TokenPrice":   p.TokenPrice.Source(),
		"Throughput":   p.Throughput.Source(),
		"SelfHostCost": c.SelfHostCost.Source(),
	} {
		if !strings.Contains(src, "3:1") {
			t.Errorf("%s source omits the blend ratio: %q", name, src)
		}
	}
	for name, src := range map[string]string{
		"Throughput":   p.Throughput.Source(),
		"SelfHostCost": c.SelfHostCost.Source(),
	} {
		if !strings.Contains(src, "40%") {
			t.Errorf("%s source omits the utilization: %q", name, src)
		}
	}
	// The instance rate's own provenance survives into the derived figures, so a
	// reader can trace a break-even number back to the Price List row behind it.
	if !strings.Contains(p.Throughput.Source(), "g7e.4xlarge") {
		t.Errorf("throughput source lost the instance: %q", p.Throughput.Source())
	}
}

// Ratio() renders what the user typed, so the report and the flag agree.
func TestRatioRendersAsStated(t *testing.T) {
	cases := []struct {
		in, out float64
		want    string
	}{
		{3, 1, "3:1"},
		{1, 1, "1:1"},
		{2.5, 1, "2.5:1"},
		{0, 1, "0:1"},
	}
	for _, tc := range cases {
		got := Assumptions{InputWeight: tc.in, OutputWeight: tc.out, Utilization: 1}.Ratio()
		if got != tc.want {
			t.Errorf("Ratio(%g, %g) = %q, want %q", tc.in, tc.out, got, tc.want)
		}
	}
}

// The shortfall is deliberately a count, not a fraction: report.UnitFraction
// renders as a percentage, and a 3.5x shortfall printed as "353%" is a number
// readers halve or double by mistake.
func TestShortfallReadsAsAMultipleNotAPercentage(t *testing.T) {
	p := parity(t, rate(4.00, "g7e.4xlarge"), threeToOne)
	s := p.At(tps(1200, "vLLM published benchmark")).Shortfall
	if s.Unit() != report.UnitCount {
		t.Errorf("shortfall unit = %s, want count", s.Unit())
	}
	if got := s.String(); strings.Contains(got, "%") {
		t.Errorf("shortfall renders as %q; a percentage invites misreading", got)
	}
}

func (o Outcome) valid() bool {
	switch o {
	case OutcomeBedrock, OutcomeSelfHost, OutcomeNoTokenPrice, OutcomeUndetermined:
		return true
	}
	return false
}

func round(v float64, places int) float64 {
	shift := math.Pow(10, float64(places))
	return math.Round(v*shift) / shift
}
