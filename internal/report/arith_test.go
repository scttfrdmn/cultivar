package report

import (
	"math"
	"strings"
	"testing"
)

func TestSumPropagatesUnavailability(t *testing.T) {
	// The core rule. A total that silently drops a missing component understates
	// the cost, which is the direction that makes a bad recommendation look good.
	ebs := Live(0.08, UnitUSDPerHour, "EBS gp3 150GB amortized", observed)
	got, err := Sum(UnitUSDPerHour, "total hourly", p5e48xlarge, ebs)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}
	if _, ok := got.Value(); ok {
		t.Error("sum with an unavailable operand produced a value")
	}
	if !strings.Contains(got.Source(), "capacity-block only") {
		t.Errorf("source %q does not name the missing input", got.Source())
	}
}

func TestSumOfKnownAmounts(t *testing.T) {
	ebs := Live(0.08, UnitUSDPerHour, "EBS gp3 150GB amortized", observed)
	got, err := Sum(UnitUSDPerHour, "total hourly", g7e4xlarge, ebs)
	if err != nil {
		t.Fatalf("Sum: %v", err)
	}
	if v := got.MustValue(); v != 4.08 {
		t.Errorf("sum = %v, want 4.08", v)
	}
	if got.Provenance() != ProvenanceDerived {
		t.Errorf("provenance = %s, want derived", got.Provenance())
	}
	// The source must show the arithmetic, since that is what makes the number
	// checkable against the console.
	if !strings.Contains(got.Source(), "+") {
		t.Errorf("source %q does not show the derivation", got.Source())
	}
}

func TestSumRejectsUnitMismatch(t *testing.T) {
	// The realistic mistake: a capacity block's upfront fee is USD, not USD/hour.
	// Adding it to an hourly rate yields a number that is wrong by ~1000x.
	fee := Live(1146.24, UnitUSD, "capacity block upfront fee", observed)
	_, err := Sum(UnitUSDPerHour, "total hourly", g7e4xlarge, fee)
	if err == nil {
		t.Fatal("Sum accepted USD added to USD/hour")
	}
	if !strings.Contains(err.Error(), "unit") {
		t.Errorf("error %v does not mention the unit mismatch", err)
	}
}

func TestSumOfNothingIsAnError(t *testing.T) {
	if _, err := Sum(UnitUSDPerHour, "empty"); err == nil {
		t.Error("Sum of no amounts returned no error; an empty sum would read as $0.00")
	}
}

func TestScale(t *testing.T) {
	// The SageMaker premium: derived live from two rates, never hardcoded. g-family
	// measured at +25% on 2026-07-27 (g7e.4xlarge $4.00 -> $5.00).
	got := Scale(g7e4xlarge, 1.25, "SageMaker hosting rate")
	if v := got.MustValue(); v != 5.00 {
		t.Errorf("scaled = %v, want 5.00", v)
	}
	if got.Unit() != UnitUSDPerHour {
		t.Errorf("unit = %s, want unchanged", got.Unit())
	}
	if _, ok := Scale(p5e48xlarge, 1.25, "x").Value(); ok {
		t.Error("scaling an unavailable amount produced a value")
	}
	if _, ok := Scale(g7e4xlarge, math.NaN(), "x").Value(); ok {
		t.Error("scaling by NaN produced a value")
	}
}

func TestConvertUpfrontFeeToHourlyRate(t *testing.T) {
	// Verified 2026-07-27, us-west-2 p5e.48xlarge: a 24h block at $1146.24 and a
	// 19h block at $935.30. Each must be divided by ITS OWN duration.
	full := Live(1146.24, UnitUSD, "cb-0630487666239ed35 upfront fee", observed)
	partial := Live(935.30, UnitUSD, "cb-093d91a1f5abd75fb upfront fee", observed)

	fullRate := Convert(full, 1.0/24.0, UnitUSDPerHour, "per instance-hour over 24h")
	if got := round(fullRate.MustValue(), 2); got != 47.76 {
		t.Errorf("24h block rate = %v, want 47.76", got)
	}
	partialRate := Convert(partial, 1.0/19.0, UnitUSDPerHour, "per instance-hour over 19h")
	if got := round(partialRate.MustValue(), 2); got != 49.23 {
		t.Errorf("19h block rate = %v, want 49.23", got)
	}
	// The partial block is PRICIER per hour, not cheaper. Its advantages are lower
	// total outlay and a near-immediate start.
	if Compare(partialRate, fullRate) != 1 {
		t.Error("the 19h partial block did not sort as pricier per hour than the 24h block")
	}
	// Dividing the 19h block by the REQUESTED 24h instead of its own duration
	// understates it by 21%, which is the bug this test pins.
	wrong := Convert(partial, 1.0/24.0, UnitUSDPerHour, "wrong: requested duration")
	if round(wrong.MustValue(), 2) != 38.97 {
		t.Errorf("sanity check: wrong-duration rate = %v, want 38.97", round(wrong.MustValue(), 2))
	}
	if Compare(wrong, partialRate) != -1 {
		t.Error("expected the wrong-duration figure to look cheaper; the trap is that it does")
	}
	if fullRate.Unit() != UnitUSDPerHour {
		t.Errorf("unit = %s, want USD/hour", fullRate.Unit())
	}
}

func TestBreakEvenThroughput(t *testing.T) {
	// The number the whole product turns on, reproducing the plan's worked example:
	// Qwen3-32B at $0.2625/1M blended 3:1, us-east-1, 2026-07-27.
	blended, err := Blend(qwenInput, 3, qwenOutput, 1, "blended token price")
	if err != nil {
		t.Fatalf("Blend: %v", err)
	}
	if got := round(blended.MustValue(), 4); got != 0.2625 {
		t.Errorf("3:1 blend = %v, want 0.2625", got)
	}

	cases := []struct {
		name string
		rate Amount
		want float64 // tok/s, rounded to the nearest whole token
	}{
		{"g7e.4xlarge", g7e4xlarge, 4233},
		{"p5.4xlarge", Live(6.88, UnitUSDPerHour, "p5.4xlarge", observed), 7280},
		{"g6e.12xlarge", Live(10.49, UnitUSDPerHour, "g6e.12xlarge", observed), 11100},
		{"p6-b200.48xlarge", p6b200, 120561},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// required tok/s = (rate/hr / $ per token) / 3600
			perToken := Convert(blended, 1e-6, "USD/token", "per token")
			tokensPerHour := Ratio(tc.rate, perToken, UnitCount, "tokens per hour to break even")
			got := Convert(tokensPerHour, 1.0/3600.0, UnitTokensPerSecond, "sustained throughput to break even")
			if diff := math.Abs(got.MustValue() - tc.want); diff > 1 {
				t.Errorf("break-even = %.0f tok/s, want ~%.0f", got.MustValue(), tc.want)
			}
			if got.Provenance() != ProvenanceDerived {
				t.Errorf("provenance = %s, want derived", got.Provenance())
			}
		})
	}
}

func TestBlendRatioSensitivity(t *testing.T) {
	// At 1:1 the blend is $0.375/1M — 43% above the 3:1 figure — which moves
	// g7e.4xlarge's break-even from ~4,233 to ~2,963 tok/s. Large enough to flip a
	// recommendation, so the ratio can never be a buried constant.
	//
	// The direction is downward, and worth stating because it is easy to get
	// backwards: a pricier Bedrock token makes self-hosting easier to justify, so
	// output-heavy traffic lowers the throughput bar rather than raising it.
	three, err := Blend(qwenInput, 3, qwenOutput, 1, "3:1")
	if err != nil {
		t.Fatal(err)
	}
	one, err := Blend(qwenInput, 1, qwenOutput, 1, "1:1")
	if err != nil {
		t.Fatal(err)
	}
	if got := round(one.MustValue(), 4); got != 0.375 {
		t.Errorf("1:1 blend = %v, want 0.375", got)
	}
	swing := (one.MustValue() - three.MustValue()) / three.MustValue()
	if got := round(swing*100, 0); got != 43 {
		t.Errorf("1:1 vs 3:1 swing = %v%%, want 43%%", got)
	}
	// The blend's source must record the ratio, or the report is unfalsifiable.
	if !strings.Contains(three.Source(), "3:1") && !strings.Contains(three.Source(), "3 :1") {
		t.Errorf("blend source %q does not record the ratio used: %s", three.Source(), three.Source())
	}
}

func TestBlendPropagatesUnavailability(t *testing.T) {
	// ~71% of hf-bedrock-map models are marketplace-only and have no token price.
	noPrice := Unavailable(UnitUSDPerMillionTokens, "marketplace catalog: no per-token price")
	got, err := Blend(qwenInput, 3, noPrice, 1, "blended")
	if err != nil {
		t.Fatalf("Blend: %v", err)
	}
	if _, ok := got.Value(); ok {
		t.Error("blend with a missing meter produced a value")
	}
}

func TestBlendRejectsBadWeights(t *testing.T) {
	for _, tc := range []struct{ a, b float64 }{{0, 0}, {-1, 1}, {1, -1}} {
		if _, err := Blend(qwenInput, tc.a, qwenOutput, tc.b, "x"); err == nil {
			t.Errorf("Blend accepted weights (%g, %g)", tc.a, tc.b)
		}
	}
	if _, err := Blend(qwenInput, 1, g7e4xlarge, 1, "x"); err == nil {
		t.Error("Blend accepted mismatched units")
	}
}

func TestRatioByZeroIsUnavailableNotInfinity(t *testing.T) {
	// "You would need infinite throughput to break even" is not worth rendering,
	// and an Inf would fail validation downstream anyway.
	zero := Live(0, UnitUSDPerMillionTokens, "free tier", observed)
	got := Ratio(g7e4xlarge, zero, UnitTokensPerSecond, "break-even")
	if _, ok := got.Value(); ok {
		t.Error("division by zero produced a value")
	}
	if !strings.Contains(got.Source(), "zero") {
		t.Errorf("source %q does not explain the zero denominator", got.Source())
	}
}

func TestRatioPropagatesEitherSide(t *testing.T) {
	known := Live(1, UnitUSDPerMillionTokens, "x", observed)
	if _, ok := Ratio(p5e48xlarge, known, UnitCount, "x").Value(); ok {
		t.Error("unavailable numerator produced a value")
	}
	noPrice := Unavailable(UnitUSDPerMillionTokens, "marketplace only")
	if _, ok := Ratio(g7e4xlarge, noPrice, UnitCount, "x").Value(); ok {
		t.Error("unavailable denominator produced a value")
	}
}

func TestUnpricedNeverSortsFirst(t *testing.T) {
	// A ranking that puts "unpriced" at the top reads as "cheapest", which is the
	// inverse of the truth. p5e.48xlarge is unpriced on-demand AND expensive.
	if Compare(p5e48xlarge, p6b200) != 1 {
		t.Error("an unpriced amount sorted before a priced one")
	}
	if Compare(p6b200, p5e48xlarge) != -1 {
		t.Error("a priced amount did not sort before an unpriced one")
	}
	if Compare(p5e48xlarge, Unavailable(UnitUSDPerHour, "other")) != 0 {
		t.Error("two unpriced amounts did not compare equal")
	}
	if Compare(g7e4xlarge, p6b200) != -1 {
		t.Error("$4.00/hr did not sort before $113.93/hr")
	}
}

func TestSageMakerPremiumDerivedNotHardcoded(t *testing.T) {
	// Measured 2026-07-27: +25% on g-family, +15% on p-family. A hardcoded 25%
	// would overstate p-family cost by 8.7%, so the premium is a Ratio of two live
	// rates and carries derived provenance.
	cases := []struct {
		name        string
		ec2, sm     float64
		wantPremium float64
	}{
		{"g7e.4xlarge", 4.00, 5.00, 0.25},
		{"p5.48xlarge", 55.04, 63.30, 0.15},
		{"p4d.24xlarge", 21.96, 25.251, 0.15},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			ec2 := Live(tc.ec2, UnitUSDPerHour, "PriceList EC2 "+tc.name, observed)
			sm := Live(tc.sm, UnitUSDPerHour, "PriceList SageMaker Hosting "+tc.name, observed)
			ratio := Ratio(sm, ec2, UnitFraction, "SageMaker premium")
			premium := ratio.MustValue() - 1
			if diff := math.Abs(premium - tc.wantPremium); diff > 0.01 {
				t.Errorf("premium = %.3f, want ~%.2f", premium, tc.wantPremium)
			}
			if ratio.Provenance() != ProvenanceDerived {
				t.Errorf("provenance = %s, want derived", ratio.Provenance())
			}
		})
	}
}

func TestCapacityBlockUtilizationCliff(t *testing.T) {
	// p5.48xlarge: CB at $41.53/inst-hr vs $55.04 on-demand is a 25% headline
	// discount, but the block is prepaid. Effective cost = fee / hours ACTUALLY
	// used, so break-even utilization is 41.53/55.04 = 75.4% of the 24h block.
	cbRate := Live(41.53, UnitUSDPerHour, "capacity block p5.48xlarge", observed)
	onDemand := Live(55.04, UnitUSDPerHour, "PriceList OnDemand p5.48xlarge", observed)

	breakEven := Ratio(cbRate, onDemand, UnitFraction, "break-even utilization")
	if got := round(breakEven.MustValue()*100, 1); got != 75.5 {
		t.Errorf("break-even utilization = %v%%, want 75.5%%", got)
	}
	hours := Convert(breakEven, 24, UnitHours, "break-even hours of a 24h block")
	if got := round(hours.MustValue(), 1); got != 18.1 {
		t.Errorf("break-even hours = %v, want 18.1", got)
	}

	// Below that, the "discount" is a penalty. A 2h benchmark on a prepaid 24h
	// block costs 9x the on-demand rate per used hour.
	fee := Scale(cbRate, 24, "block total")
	for _, tc := range []struct{ used, wantEffective float64 }{
		{24, 41.53}, {18, 55.37}, {12, 83.06}, {8, 124.59}, {2, 498.36},
	} {
		effective := Convert(fee, 1/tc.used, UnitUSDPerHour, "per used hour")
		if diff := math.Abs(effective.MustValue() - tc.wantEffective); diff > 0.01 {
			t.Errorf("%vh used: effective = %.2f/hr, want %.2f", tc.used, effective.MustValue(), tc.wantEffective)
		}
		if tc.used < 18 && Compare(effective, onDemand) != 1 {
			t.Errorf("%vh used: effective rate did not sort as pricier than on-demand", tc.used)
		}
	}
}

// round is a test helper; production code never rounds before comparing.
func round(v float64, places int) float64 {
	shift := math.Pow(10, float64(places))
	return math.Round(v*shift) / shift
}
