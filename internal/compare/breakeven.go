package compare

import (
	"fmt"
	"math"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// Break-even arithmetic: the number the whole product turns on.
//
// Self-hosting buys time, Bedrock buys tokens, and the two are only comparable at
// a stated throughput. That makes the comparison a function of the one number this
// tool cannot get for free — sustained tok/s — so the engine is split along that
// seam. [BreakEven] answers "how fast would this have to be?" and needs no
// benchmark, no launched instance, and no money. [Parity.At] takes a throughput
// figure from wherever one can be had and says who wins and by how much.
//
// The split is what makes `compare` useful before a single GPU is rented: a
// required-throughput figure alone settles most questions, because 4,233 tok/s
// from one 32B model on one GPU is not close.

const (
	secondsPerHour   = 3600.0
	tokensPerMillion = 1e6
)

// Assumptions are the inputs a break-even figure is meaningless without.
//
// They travel with the result and into the report because the same model on the
// same instance yields recommendations that differ by 40% under different
// assumptions, and a recommendation whose assumptions are not written down cannot
// be checked, reproduced, or argued with. Every derived amount in this file
// records these values in its source string for the same reason.
type Assumptions struct {
	// InputWeight and OutputWeight are the input:output traffic ratio used to blend
	// Bedrock's two meters into one price. Both are required and neither has a
	// default, because output tokens cost 4x input for Qwen3-32B: a 3:1 mix prices
	// at $0.2625/1M and a 1:1 mix at $0.375, moving g7e.4xlarge's break-even from
	// ~4,233 to ~6,047 tok/s. That is enough to flip a recommendation, so the ratio
	// is an argument, never a constant.
	InputWeight  float64
	OutputWeight float64

	// Utilization is the fraction of *billed* hours during which the instance is
	// actually serving at the throughput being compared. It is required, and there
	// is deliberately no default: a silent 1.0 is the single most flattering
	// assumption available to a self-hosting recommendation, and it is almost never
	// true of a real endpoint.
	//
	// This is the term that decides most real cases. An idle GPU bills; an idle
	// Bedrock endpoint does not. A model served eight hours a day on an instance
	// left running costs three times as much per token as the raw rate suggests, and
	// no faster GPU fixes that — only shutting it down does.
	Utilization float64
}

// Validate reports whether these assumptions can produce a number.
//
// NaN is checked explicitly rather than relying on the range comparisons: NaN is
// neither <= 0 nor > 1, so a range check alone admits it and every downstream
// amount silently becomes NaN.
func (a Assumptions) Validate() error {
	if math.IsNaN(a.InputWeight) || math.IsNaN(a.OutputWeight) {
		return fmt.Errorf("break-even: input:output weights are not numbers (%g:%g)", a.InputWeight, a.OutputWeight)
	}
	if a.InputWeight < 0 || a.OutputWeight < 0 {
		return fmt.Errorf("break-even: negative input:output weight (%g:%g)", a.InputWeight, a.OutputWeight)
	}
	if a.InputWeight+a.OutputWeight == 0 {
		return fmt.Errorf("break-even: input:output weights sum to zero; state the traffic mix")
	}
	if math.IsNaN(a.Utilization) {
		return fmt.Errorf("break-even: utilization is not a number")
	}
	if a.Utilization <= 0 || a.Utilization > 1 {
		return fmt.Errorf("break-even: utilization must be in (0, 1], got %g "+
			"(it is the fraction of billed hours actually serving, and has no default)", a.Utilization)
	}
	return nil
}

// Ratio renders the blend ratio the way a user states it: "3:1".
func (a Assumptions) Ratio() string {
	return report.BlendRatio(a.InputWeight, a.OutputWeight)
}

// Record writes these assumptions into a report envelope's assumption block.
//
// The mapping lives here, beside the fields it reads, rather than in the report
// package: [report.Assumptions] is a wire shape with no arithmetic in it, and
// report cannot import this package anyway. What matters is that there is exactly
// one such mapping, because a report that states a 3:1 blend while the break-even
// figure was computed at 1:1 is worse than one that states nothing.
func (a Assumptions) Record(into report.Assumptions) report.Assumptions {
	return into.WithBlend(a.InputWeight, a.OutputWeight, a.Utilization)
}

// Parity is the point at which self-hosting and Bedrock cost the same.
type Parity struct {
	// Hourly is everything billed per hour for the self-hosted option. The caller
	// composes it — instance rate plus EBS, plus whatever else is metered by time —
	// with [report.Sum], so this engine never needs to know what is in it and a new
	// cost component does not change this arithmetic.
	Hourly report.Amount

	// TokenPrice is the blended Bedrock rate in USD per 1M tokens, computed from the
	// two meters and the ratio in [Assumptions]. Unavailable when either meter is
	// missing, which is a real state: Llama 3.1 405B publishes batch rates only.
	TokenPrice report.Amount

	// Throughput is the sustained tok/s at which the two options cost the same, at
	// this utilization. Unavailable when either input is.
	Throughput report.Amount

	// Assumptions are the inputs the two figures above were computed under.
	Assumptions Assumptions
}

// BreakEven computes the sustained throughput at which self-hosting at hourly
// reaches parity with Bedrock's per-token price.
//
// The two token meters are passed separately and blended here, so the traffic mix
// is recorded in the result's provenance rather than pre-collapsed by the caller
// into a single figure whose ratio nobody can recover.
//
// An error means the question was malformed — wrong units, unstated assumptions.
// A missing *price* is not an error: it comes back as an unavailable throughput,
// because "no on-demand rate exists for p5e.48xlarge" and "no per-token rate
// exists for this model" are both facts the report has to state rather than
// failures to abort on.
func BreakEven(hourly, inputPrice, outputPrice report.Amount, a Assumptions) (Parity, error) {
	if err := a.Validate(); err != nil {
		return Parity{}, err
	}
	if hourly.Unit() != report.UnitUSDPerHour {
		return Parity{}, fmt.Errorf("break-even: hourly cost has unit %s, want %s",
			hourly.Unit(), report.UnitUSDPerHour)
	}
	// Checked before Blend, which only verifies that the two operands agree with each
	// other — two hourly rates would blend happily into a nonsense token price.
	for name, m := range map[string]report.Amount{"input": inputPrice, "output": outputPrice} {
		if m.Unit() != report.UnitUSDPerMillionTokens {
			return Parity{}, fmt.Errorf("break-even: %s token price has unit %s, want %s",
				name, m.Unit(), report.UnitUSDPerMillionTokens)
		}
	}

	// The ratio is not repeated in the label: [report.Blend] writes it into the
	// source string itself, which is where that invariant is tested.
	blended, err := report.Blend(inputPrice, a.InputWeight, outputPrice, a.OutputWeight,
		"blended token price")
	if err != nil {
		return Parity{}, fmt.Errorf("break-even: %w", err)
	}

	p := Parity{Hourly: hourly, TokenPrice: blended, Assumptions: a}
	p.Throughput = report.Ratio(hourly, p.bedrockHourlyAt(1), report.UnitTokensPerSecond,
		fmt.Sprintf("sustained throughput for parity at %s", p.utilizationNote()))
	return p, nil
}

// bedrockHourlyAt returns what Bedrock would bill per hour to serve tps tokens per
// second, at this utilization — the figure the instance's hourly cost is compared
// against.
//
// Utilization divides the token count here, on the Bedrock side only, and that
// asymmetry is the whole point. Bedrock bills per token, so an idle hour costs
// nothing; a running GPU bills whether or not anyone is asking it for tokens. So
// the honest comparison holds the instance's hourly rate fixed and shrinks the
// tokens it manages to produce in that hour. Applying utilization to both sides
// cancels it out, and an endpoint used eight hours a day then looks exactly as
// economical as a saturated one.
func (p Parity) bedrockHourlyAt(tps float64) report.Amount {
	return report.Convert(p.TokenPrice, tps*secondsPerHour*p.Assumptions.Utilization/tokensPerMillion,
		report.UnitUSDPerHour, fmt.Sprintf("Bedrock cost per hour at %g tok/s sustained, %s",
			tps, p.utilizationNote()))
}

func (p Parity) utilizationNote() string {
	return fmt.Sprintf("%s blend and %.0f%% utilization",
		p.Assumptions.Ratio(), p.Assumptions.Utilization*100)
}

// CostPerMillion returns what self-hosting actually costs per 1M tokens at a
// sustained throughput — the figure directly comparable to [Parity.TokenPrice],
// and the one to print when the user's question is "so what would this cost me?"
//
// Note that it is per 1M tokens *produced*, not per 1M tokens of capacity: the
// utilization penalty is already in it, which is why it can exceed the Bedrock
// rate on an instance whose raw throughput is more than sufficient.
func (p Parity) CostPerMillion(achievable report.Amount) report.Amount {
	tps, why, ok := throughputValue(achievable, report.UnitUSDPerMillionTokens)
	if !ok {
		return why
	}
	millions := report.Convert(achievable, secondsPerHour*p.Assumptions.Utilization/tokensPerMillion,
		report.UnitCount, fmt.Sprintf("millions of tokens per billed hour at %g tok/s and %s",
			tps, p.utilizationNote()))
	return report.Ratio(p.Hourly, millions, report.UnitUSDPerMillionTokens,
		"self-hosted cost per 1M tokens produced")
}

// Shortfall returns how many times the given throughput has to improve to reach
// parity: 20 means "you need 20x more throughput than this". Below 1 means
// self-hosting already wins and the reciprocal is the margin.
//
// This is the statement the verdict is built on. "Bedrock is cheaper" invites an
// argument about assumptions; "you would need 20x the throughput this GPU can
// produce" ends it, because no assumption in the report moves a number by 20x.
//
// The unit is [report.UnitCount] rather than [report.UnitFraction] specifically
// because a fraction renders as a percentage, and a 20x shortfall printed as
// "2000%" is a number readers halve, double, or otherwise misread.
func (p Parity) Shortfall(achievable report.Amount) report.Amount {
	if _, why, ok := throughputValue(achievable, report.UnitCount); !ok {
		return why
	}
	return report.Ratio(p.Throughput, achievable, report.UnitCount,
		"throughput shortfall against parity")
}

// UtilizationForParity returns the fraction of billed hours the instance would
// have to spend serving at the given throughput to match Bedrock's cost.
//
// It is the same equation as [Parity.Throughput] solved for the other unknown, and
// it is the more actionable of the two when the hardware is already chosen: buying
// a faster GPU is a purchase, keeping the one you have busier is a scheduling
// decision.
//
// The result is deliberately not capped at 1. A figure above 100% is the answer —
// it says no duty cycle reaches parity at this throughput, so the lever is the
// wrong one.
func (p Parity) UtilizationForParity(achievable report.Amount) report.Amount {
	tps, why, ok := throughputValue(achievable, report.UnitFraction)
	if !ok {
		return why
	}
	// Compared against the full-utilization cost: the ratio of the instance's hourly
	// bill to what Bedrock would charge for an hour of uninterrupted output at this
	// rate *is* the duty cycle that equalizes them.
	full := report.Convert(p.TokenPrice, tps*secondsPerHour/tokensPerMillion, report.UnitUSDPerHour,
		fmt.Sprintf("Bedrock cost per hour of uninterrupted output at %g tok/s, %s blend",
			tps, p.Assumptions.Ratio()))
	return report.Ratio(p.Hourly, full, report.UnitFraction,
		"utilization needed for parity at this throughput")
}

// Outcome is who wins, or why nobody does.
type Outcome string

const (
	// OutcomeBedrock means Bedrock's per-token price is cheaper at the throughput
	// compared. This is the usual answer.
	OutcomeBedrock Outcome = "bedrock"

	// OutcomeSelfHost means the instance produces tokens more cheaply than Bedrock
	// sells them, under the stated assumptions.
	OutcomeSelfHost Outcome = "self-host"

	// OutcomeNoTokenPrice means the model has no per-token option, so the break-even
	// question does not apply — 94 of 132 mappable Hugging Face models are Bedrock
	// marketplace-only, where you rent an instance and there is no token meter at
	// all.
	//
	// Kept distinct from [OutcomeSelfHost] because "self-hosting is cheaper" and
	// "there is nothing to be cheaper than" support very different decisions, and
	// from [OutcomeUndetermined] because this one is a settled fact about the model
	// rather than a number that failed to resolve.
	OutcomeNoTokenPrice Outcome = "no-token-price"

	// OutcomeUndetermined means a number the comparison needs did not resolve: an
	// instance with no on-demand rate (p5e.48xlarge has none), or no throughput
	// figure for this model on this hardware. Never rendered as a winner — an
	// unpriced option that ranks first reads as the cheapest one.
	OutcomeUndetermined Outcome = "undetermined"
)

// Decided reports whether o names a winner.
func (o Outcome) Decided() bool { return o == OutcomeBedrock || o == OutcomeSelfHost }

// Comparison is a [Parity] evaluated against a throughput the hardware can
// actually reach.
type Comparison struct {
	Parity Parity

	// Achievable is the sustained tok/s used for the comparison, carrying its own
	// provenance. It is rarely live: until a benchmark has been run it is external
	// (a published measurement, with its citation and date) or unavailable. The
	// provenance matters more here than anywhere else in a report, because this is
	// the number the verdict is most sensitive to and the one least likely to have
	// been measured on the hardware under discussion.
	Achievable report.Amount

	// SelfHostCost is USD per 1M tokens produced, at Achievable and the parity's
	// utilization. Computed even when there is no Bedrock rate to compare it to,
	// since "this costs $3.40/1M and there is no serverless alternative" is still
	// the answer to the user's question.
	SelfHostCost report.Amount

	// Shortfall is how many times Achievable must improve to reach parity.
	Shortfall report.Amount

	// UtilizationForParity is the duty cycle that would reach parity at Achievable.
	UtilizationForParity report.Amount

	// Outcome is the verdict input.
	Outcome Outcome
}

// At evaluates this parity against a throughput figure.
//
// A tie goes to [OutcomeBedrock]. At equal cost the serverless option still wins
// on everything that is not cost — no quota request, no capacity to obtain, no
// instance to leave running by accident — so parity is not a reason to self-host.
func (p Parity) At(achievable report.Amount) Comparison {
	c := Comparison{
		Parity:               p,
		Achievable:           achievable,
		SelfHostCost:         p.CostPerMillion(achievable),
		Shortfall:            p.Shortfall(achievable),
		UtilizationForParity: p.UtilizationForParity(achievable),
	}

	// Ordered: the absence of a token price is checked first because it is a fact
	// about the model that no amount of further data would change, whereas an
	// unresolved instance rate or throughput is a gap that could be filled.
	parity, parityKnown := p.Throughput.Value()
	got, gotKnown := achievable.Value()
	switch {
	case !p.TokenPrice.Known():
		c.Outcome = OutcomeNoTokenPrice
	case !parityKnown || !gotKnown || got <= 0:
		c.Outcome = OutcomeUndetermined
	case got > parity:
		c.Outcome = OutcomeSelfHost
	default:
		c.Outcome = OutcomeBedrock
	}
	return c
}

// throughputValue extracts a usable tok/s figure, or an unavailable amount in unit
// explaining why there isn't one. A zero or negative throughput is rejected rather
// than divided by: it is not a slow endpoint, it is a missing measurement.
func throughputValue(a report.Amount, unit report.Unit) (float64, report.Amount, bool) {
	if a.Unit() != report.UnitTokensPerSecond {
		return 0, report.Unavailable(unit, fmt.Sprintf("throughput has unit %s, want %s",
			a.Unit(), report.UnitTokensPerSecond)), false
	}
	v, ok := a.Value()
	if !ok {
		return 0, report.Unavailable(unit, "unknown throughput: "+a.Source()), false
	}
	if v <= 0 {
		return 0, report.Unavailable(unit, fmt.Sprintf(
			"throughput is not positive (%g tok/s): %s", v, a.Source())), false
	}
	return v, report.Amount{}, true
}
