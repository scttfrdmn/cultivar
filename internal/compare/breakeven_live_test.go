//go:build live

// Opt-in suite that computes real break-even figures from the real Price List. Run
// with `make test-live` (AWS_PROFILE=aws). Every call is a free read-only query — no
// instance is launched and nothing bills.
//
// The offline tests pin the arithmetic against fixed rates. These check the thing
// fixtures cannot: that the *inputs* still arrive in the shape the arithmetic
// assumes, and that the conclusion the whole product rests on still holds against
// today's prices. A Bedrock rate cut or a new GPU family could in principle flip
// "don't self-host" for a 32B model, and if it ever does, this suite is where that
// shows up — as a failure to investigate rather than a stale claim in a README.
package compare

import (
	"context"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
	truffle "github.com/spore-host/truffle/pkg/aws"

	"github.com/scttfrdmn/cultivar/internal/bedrock"
	"github.com/scttfrdmn/cultivar/internal/ec2"
	"github.com/scttfrdmn/cultivar/internal/model"
	"github.com/scttfrdmn/cultivar/internal/report"
)

// qwen3BedrockID is the Bedrock foundation-model id for Qwen3-32B — the model this
// project's headline comparison is built on, and one of the 38 with a token price at
// all.
const qwen3BedrockID = "qwen.qwen3-32b-v1:0"

// liveAssumptions is the 3:1 mix at full utilization: the most favorable honest case
// for self-hosting, which is the right baseline for a "don't self-host" claim. Any
// real utilization only makes the gap wider.
var liveAssumptions = Assumptions{InputWeight: 3, OutputWeight: 1, Utilization: 1.0}

func liveTokenPrice(t *testing.T) *bedrock.TokenPrice {
	t.Helper()
	ctx, cancel := liveCtx(t)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}
	price, err := bedrock.NewTokenPricer(cfg).Lookup(ctx, qwen3BedrockID, liveRegion)
	if err != nil {
		t.Fatalf("token price for %s in %s: %v", qwen3BedrockID, liveRegion, err)
	}
	return price
}

func liveHourly(t *testing.T, instanceType string) report.Amount {
	t.Helper()
	ctx, cancel := liveCtx(t)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}
	amount, err := ec2.NewPricer(cfg).OnDemand(ctx, instanceType, liveRegion)
	if err != nil {
		t.Fatalf("on-demand price for %s in %s: %v", instanceType, liveRegion, err)
	}
	return amount
}

func liveParity(t *testing.T, instanceType string) Parity {
	t.Helper()
	price := liveTokenPrice(t)
	p, err := BreakEven(liveHourly(t, instanceType),
		price.Amount(bedrock.TierStandard, bedrock.MeterInput),
		price.Amount(bedrock.TierStandard, bedrock.MeterOutput),
		liveAssumptions)
	if err != nil {
		t.Fatalf("BreakEven for %s: %v", instanceType, err)
	}
	return p
}

// TestLiveTheBreakEvenFigureIsStillFourFigures is the product's central claim,
// checked against today's rates rather than a fixture.
//
// Measured 2026-07-28: g7e.4xlarge at $4.00/hr against a $0.2625/1M blend needs
// ~4,233 tok/s. The band is wide because both inputs legitimately move; what it
// catches is the figure changing by an order of magnitude, which would mean a unit
// slipped somewhere — a per-1K rate read as per-1M, or a per-second figure read as
// per-hour — not that AWS changed a price.
func TestLiveTheBreakEvenFigureIsStillFourFigures(t *testing.T) {
	p := liveParity(t, "g7e.4xlarge")
	tps := p.Throughput.MustValue()
	t.Logf("g7e.4xlarge at %s vs %s → %.0f tok/s for parity (recorded ~4233 on 2026-07-28)",
		p.Hourly, p.TokenPrice, tps)

	if tps < 500 || tps > 40000 {
		t.Errorf("break-even = %.0f tok/s, expected the low thousands; a figure this far off "+
			"means a unit conversion is wrong, not that a price changed", tps)
	}
	// Both inputs must be live. A derived or external provenance on either is the
	// signature of a fallback estimate reaching the comparison — precisely the
	// libs#29 failure, where a fabricated GPU rate arrives with err == nil.
	if got := p.Hourly.Provenance(); got != report.ProvenanceLive {
		t.Errorf("hourly rate provenance = %s, want live (%s)", got, p.Hourly.Source())
	}
	if got := p.TokenPrice.Provenance(); got != report.ProvenanceDerived {
		t.Errorf("token price provenance = %s, want derived from live meters", got)
	}
}

// The trap from CLAUDE.md item 1, checked end to end: truffle's default pricer
// invents $0.80/hr for g7e.4xlarge against a real $4.00. That 5x error would make
// break-even look like ~850 tok/s — a figure a single GPU can plausibly hit, which
// is how a fabricated rate turns into a recommendation to self-host.
func TestLiveTheRealRateDefeatsTheFabricatedOne(t *testing.T) {
	p := liveParity(t, "g7e.4xlarge")
	const fabricated = 0.80

	real := p.Hourly.MustValue()
	if real < fabricated*2 {
		t.Errorf("g7e.4xlarge on-demand = $%.4f/hr, which is close to the static table's "+
			"$%.2f; either AWS cut the price 5x or the no-fallback pricer is not wired in",
			real, fabricated)
	}

	fake, err := BreakEven(report.Live(fabricated, report.UnitUSDPerHour, "static table", time.Now().UTC()),
		liveTokenPrice(t).Amount(bedrock.TierStandard, bedrock.MeterInput),
		liveTokenPrice(t).Amount(bedrock.TierStandard, bedrock.MeterOutput),
		liveAssumptions)
	if err != nil {
		t.Fatalf("BreakEven: %v", err)
	}
	t.Logf("the fabricated $%.2f/hr rate would put break-even at %.0f tok/s instead of %.0f",
		fabricated, fake.Throughput.MustValue(), p.Throughput.MustValue())

	// A plausible single-GPU throughput for a 32B model. The point is that the
	// fabricated rate lands under it and the real one does not, so the two rates give
	// opposite verdicts on the same hardware.
	achievable := report.External(1200, report.UnitTokensPerSecond, "published vLLM benchmark", time.Now().UTC())
	if fake.At(achievable).Outcome != OutcomeSelfHost {
		t.Log("note: the fabricated rate no longer flips the verdict at 1200 tok/s; " +
			"rates moved, but the wiring check above is what matters")
	}
	if got := p.At(achievable).Outcome; got != OutcomeBedrock {
		t.Errorf("at the real rate the outcome is %s, want bedrock at 1200 tok/s", got)
	}
}

// TestLiveEvenTheCheapestFittingInstanceLosesToBedrock closes the loop between the
// two halves of this package: selection picks the cheapest instance that can
// actually hold Qwen3-32B, and the break-even engine prices it against Bedrock.
//
// This is the verdict a user gets, computed the way they will get it. Measured
// 2026-07-28: g7e.2xlarge at $3.3631/hr needs ~3,556 tok/s, roughly 3x what one
// L40S-class GPU delivers for a 32B model.
func TestLiveEvenTheCheapestFittingInstanceLosesToBedrock(t *testing.T) {
	m := qwen3ForLive()
	sizing := m.Size(model.SizingRequest{ContextTokens: 4096})

	ctx, cancel := context.WithTimeout(context.Background(), 180*time.Second)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}
	sel, err := NewSelector(truffle.NewClientFromConfig(cfg), ec2.NewPricer(cfg), time.Now).
		Select(ctx, m, sizing, []string{liveRegion})
	if err != nil {
		t.Fatalf("Select: %v", err)
	}
	cheapest, ok := sel.Cheapest()
	if !ok {
		t.Fatalf("no priced instance in %s can serve %s at 4k context", liveRegion, m.ID)
	}

	price := liveTokenPrice(t)
	p, err := BreakEven(cheapest.OnDemand,
		price.Amount(bedrock.TierStandard, bedrock.MeterInput),
		price.Amount(bedrock.TierStandard, bedrock.MeterOutput),
		liveAssumptions)
	if err != nil {
		t.Fatalf("BreakEven: %v", err)
	}

	achievable := report.External(1200, report.UnitTokensPerSecond, "published vLLM benchmark", time.Now().UTC())
	c := p.At(achievable)
	t.Logf("cheapest fit: %s at %s (%d GPU, tp %d, %s usable) → parity at %.0f tok/s, "+
		"%.1fx short at 1200 tok/s, self-hosted $%.4f/1M vs Bedrock %s",
		cheapest.InstanceType, cheapest.OnDemand, cheapest.GPUs, cheapest.TensorParallel,
		cheapest.UsableMemory, p.Throughput.MustValue(), c.Shortfall.MustValue(),
		c.SelfHostCost.MustValue(), p.TokenPrice)

	if c.Outcome != OutcomeBedrock {
		t.Errorf("outcome = %s at 1200 tok/s on the cheapest fitting instance; if this is "+
			"genuinely self-host now, the project's headline claim changed and the README "+
			"and CLAUDE.md need updating", c.Outcome)
	}
	if c.Shortfall.MustValue() <= 1 {
		t.Errorf("shortfall = %.2fx, expected > 1 when the outcome is bedrock", c.Shortfall.MustValue())
	}
	// The utilization figure has to be impossible here, which is the useful statement:
	// no duty cycle fixes a throughput deficit.
	if u := c.UtilizationForParity.MustValue(); u <= 1 {
		t.Errorf("utilization for parity = %.2f; expected > 1 (unreachable) at this throughput", u)
	}
}

// Every tier Bedrock publishes yields a usable break-even figure, and they order the
// way the rates do. flex and batch are half price, so they raise the bar for
// self-hosting; priority is 1.75x and lowers it. A tool that silently compared
// against priority would overstate self-hosting's case by 75%.
func TestLiveEveryPublishedTierYieldsAComparison(t *testing.T) {
	price := liveTokenPrice(t)
	hourly := liveHourly(t, "g7e.4xlarge")

	tiers := price.Tiers()
	if len(tiers) < 2 {
		t.Fatalf("only %d complete tiers for %s; expected standard plus at least one of "+
			"priority/flex/batch", len(tiers), qwen3BedrockID)
	}

	var prev float64
	for i, tier := range tiers {
		p, err := BreakEven(hourly,
			price.Amount(tier, bedrock.MeterInput),
			price.Amount(tier, bedrock.MeterOutput),
			liveAssumptions)
		if err != nil {
			t.Fatalf("BreakEven for tier %s: %v", tier, err)
		}
		tps := p.Throughput.MustValue()
		t.Logf("tier %-8s %s → parity at %.0f tok/s", tier, p.TokenPrice, tps)

		if tps <= 0 {
			t.Errorf("tier %s gave a non-positive break-even: %.0f tok/s", tier, tps)
		}
		// Tiers() sorts by ascending input cost, so a cheaper token price must mean a
		// higher throughput requirement. An inversion means the tier attribution is
		// wrong — a batch rate filed as standard, the CLAUDE.md item 2 trap.
		if i > 0 && tps > prev {
			t.Errorf("tier %s needs %.0f tok/s, more than the cheaper tier before it (%.0f); "+
				"tier ordering and break-even disagree", tier, tps, prev)
		}
		prev = tps
	}
}

// An instance with no on-demand price at all must produce an unpriced break-even
// rather than a number. p5e.48xlarge is the case: truffle's static table invents
// $9.60/hr for it, which would yield a confident ~10,150 tok/s figure for hardware
// that cannot be bought on demand at any price.
func TestLiveACapacityBlockOnlyTypeHasNoBreakEven(t *testing.T) {
	ctx, cancel := liveCtx(t)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(liveRegion))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}

	hourly, err := ec2.NewPricer(cfg).OnDemand(ctx, "p5e.48xlarge", liveRegion)
	if err != nil {
		t.Fatalf("p5e.48xlarge: a missing price must be an unavailable amount, not an error: %v", err)
	}
	if hourly.Known() {
		t.Skipf("p5e.48xlarge now has an on-demand price (%s); this test's premise is gone", hourly)
	}
	t.Logf("p5e.48xlarge on-demand: %s (%s)", hourly, hourly.Source())

	price := liveTokenPrice(t)
	p, err := BreakEven(hourly,
		price.Amount(bedrock.TierStandard, bedrock.MeterInput),
		price.Amount(bedrock.TierStandard, bedrock.MeterOutput),
		liveAssumptions)
	if err != nil {
		t.Fatalf("BreakEven: %v", err)
	}
	if p.Throughput.Known() {
		t.Errorf("break-even = %s for an instance with no on-demand price", p.Throughput)
	}
	c := p.At(report.External(1200, report.UnitTokensPerSecond, "published vLLM benchmark", time.Now().UTC()))
	if c.Outcome != OutcomeUndetermined {
		t.Errorf("outcome = %s, want undetermined", c.Outcome)
	}
	if c.Outcome.Decided() {
		t.Error("an unpriced instance must never name a winner")
	}
}

// The break-even figure scales with the hourly rate and nothing else, so the ratio
// between two instances' requirements must equal the ratio of their prices. This is
// the one relationship in the engine that holds regardless of what any price is, so
// it is checkable against live data without recording any expected number.
func TestLiveBreakEvenScalesWithThePrice(t *testing.T) {
	cheap := liveParity(t, "g6e.xlarge")
	dear := liveParity(t, "g6e.12xlarge")

	priceRatio := dear.Hourly.MustValue() / cheap.Hourly.MustValue()
	tpsRatio := dear.Throughput.MustValue() / cheap.Throughput.MustValue()
	t.Logf("g6e.12xlarge/%s price ratio %.4f, break-even ratio %.4f",
		"g6e.xlarge", priceRatio, tpsRatio)

	if diff := tpsRatio - priceRatio; diff > 1e-6 || diff < -1e-6 {
		t.Errorf("break-even ratio %.6f != price ratio %.6f; the figure depends on something "+
			"other than the hourly rate and the token price", tpsRatio, priceRatio)
	}
}
