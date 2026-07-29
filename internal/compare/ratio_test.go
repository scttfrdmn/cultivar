package compare

import (
	"math"
	"strings"
	"testing"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// The forms a user actually writes, and the one deliberately refused.
func TestARatioParsesAsWritten(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want Ratio
	}{
		{"3:1", Ratio{3, 1}},
		{"10:1", Ratio{10, 1}},
		{"1:3", Ratio{1, 3}},
		{"2.5:1", Ratio{2.5, 1}},
		{"1:0.5", Ratio{1, 0.5}},
		{"  3 : 1  ", Ratio{3, 1}},
		// A bare number is N:1, since "my traffic is 3 to 1" is how the assumption is
		// usually held. That makes "0" the all-output mix rather than an error — the
		// same value as "0:1", and the only mix a bare zero could mean.
		{"3", Ratio{3, 1}},
		{"0.5", Ratio{0.5, 1}},
		{"0", Ratio{0, 1}},
		// A zero on one side is a real workload: embeddings are all-input, and an
		// agent loop's generation leg is all-output.
		{"1:0", Ratio{1, 0}},
		{"0:1", Ratio{0, 1}},
		// Preset names, case- and space-insensitive.
		{"chat", Ratio{3, 1}},
		{"CHAT", Ratio{3, 1}},
		{" summarization ", Ratio{10, 1}},
		{"generation", Ratio{1, 3}},
	} {
		t.Run(tc.in, func(t *testing.T) {
			got, err := ParseRatio(tc.in)
			if err != nil {
				t.Fatalf("ParseRatio(%q): %v", tc.in, err)
			}
			if got != tc.want {
				t.Errorf("ParseRatio(%q) = %v, want %v", tc.in, got, tc.want)
			}
		})
	}
}

// A malformed mix is refused rather than defaulted. Silently falling back to 3:1 on a
// typo is the failure this parser exists to prevent: the report would print an
// assumption the user did not make and had no way to notice.
func TestAMalformedRatioIsRefused(t *testing.T) {
	for _, tc := range []struct {
		in, wantErr string
	}{
		{"", "no traffic mix"},
		{"   ", "no traffic mix"},
		{"abc", "not a number"},
		{"3:x", "not a number"},
		{"x:1", "not a number"},
		{"3:", "not a number"},
		{":1", "not a number"},
		{"3:1:1", "not a number"},
		{"-3:1", "negative weight"},
		{"3:-1", "negative weight"},
		{"0:0", "no mix at all"},
		{"NaN:1", "not numeric"},
		{"Inf:1", "infinite"},
		// A slash is refused with a message naming the accepted form: "3/1" is
		// ambiguous between a ratio and a quotient, and guessing wrong inverts the
		// assumption silently.
		{"3/1", "not a slash"},
		{"3\\1", "not a slash"},
	} {
		t.Run(tc.in, func(t *testing.T) {
			got, err := ParseRatio(tc.in)
			if err == nil {
				t.Fatalf("ParseRatio(%q) = %v, want an error", tc.in, got)
			}
			if !strings.Contains(err.Error(), tc.wantErr) {
				t.Errorf("ParseRatio(%q) error %q does not mention %q", tc.in, err, tc.wantErr)
			}
			if got != (Ratio{}) {
				t.Errorf("ParseRatio(%q) returned %v alongside its error; a partial mix "+
					"could be used as if it had parsed", tc.in, got)
			}
		})
	}
}

// Every ratio renders through report.BlendRatio, so the separator and ordering match
// the verdict line, the flag echo, and the provenance string of every blended price.
// A ratio that reads "1:3" in one place and "3:1" in another is a misstated assumption
// that no test of either place alone would catch.
func TestARatioRendersConsistently(t *testing.T) {
	for _, tc := range []struct {
		r    Ratio
		want string
	}{
		{Ratio{3, 1}, "3:1"},
		{Ratio{1, 3}, "1:3"},
		{Ratio{2.5, 1}, "2.5:1"},
		{Ratio{10, 1}, "10:1"},
	} {
		if got := tc.r.String(); got != tc.want {
			t.Errorf("%+v renders as %q, want %q", tc.r, got, tc.want)
		}
		if got := report.BlendRatio(tc.r.Input, tc.r.Output); got != tc.r.String() {
			t.Errorf("Ratio.String() = %q but report.BlendRatio = %q", tc.r.String(), got)
		}
	}

	// Every parseable form round-trips through String and back to the same mix, which
	// is what lets a report echo the assumption in a form the user can re-run.
	for _, in := range []string{"3:1", "1:3", "2.5:1", "10:1", "1:0", "0:1", "chat"} {
		r, err := ParseRatio(in)
		if err != nil {
			t.Fatalf("ParseRatio(%q): %v", in, err)
		}
		again, err := ParseRatio(r.String())
		if err != nil {
			t.Fatalf("ParseRatio(%q) round trip: %v", r, err)
		}
		if again != r {
			t.Errorf("%q -> %v -> %v", in, r, again)
		}
	}
}

// The label names the matching preset without substituting for the numbers. A stored
// report saying only "chat" would mean something different if a later release adjusted
// what chat means.
func TestALabelNamesThePresetWithoutReplacingTheNumbers(t *testing.T) {
	for _, tc := range []struct {
		r    Ratio
		want string
	}{
		{Ratio{3, 1}, "3:1 (chat)"},
		{Ratio{10, 1}, "10:1 (summarization)"},
		{Ratio{1, 1}, "1:1 (balanced)"},
		{Ratio{1, 3}, "1:3 (generation)"},
		// An equivalent mix written differently still finds its preset.
		{Ratio{6, 2}, "6:2 (chat)"},
		// And a mix with no preset is just the numbers.
		{Ratio{7, 2}, "7:2"},
		{Ratio{1, 0}, "1:0"},
		// The zero value is the absence of an assumption, so it must not come back
		// wearing a preset's name. It cross-multiplies equal to every mix, which makes
		// this the one label that could be invented rather than looked up.
		{Ratio{}, "0:0"},
	} {
		if got := tc.r.Label(); got != tc.want {
			t.Errorf("%v.Label() = %q, want %q", tc.r, got, tc.want)
		}
	}
}

// 6:2 and 3:1 are the same assumption written differently. Treating them as distinct
// would show a spurious assumption change in the history log, which is the one place
// a real change has to stand out.
func TestEquivalentMixesAreRecognized(t *testing.T) {
	for _, tc := range []struct {
		a, b Ratio
		want bool
	}{
		{Ratio{3, 1}, Ratio{3, 1}, true},
		{Ratio{3, 1}, Ratio{6, 2}, true},
		{Ratio{3, 1}, Ratio{30, 10}, true},
		{Ratio{1, 3}, Ratio{2, 6}, true},
		{Ratio{3, 1}, Ratio{1, 3}, false},
		{Ratio{3, 1}, Ratio{4, 1}, false},
		// All-input and all-output are opposite assumptions, not the same one.
		{Ratio{1, 0}, Ratio{0, 1}, false},
		{Ratio{1, 0}, Ratio{5, 0}, true},
		{Ratio{0, 1}, Ratio{0, 7}, true},
		{Ratio{1, 0}, Ratio{3, 1}, false},
		// 0:0 is the case cross-multiplication alone gets wrong: it comes out 0 == 0
		// against every mix. It is also the zero value, so it is what an unset field or
		// a failed parse holds — and equating it with everything would make Label print
		// the absence of an assumption as a named one.
		{Ratio{}, Ratio{10, 1}, false},
		{Ratio{}, Ratio{1, 3}, false},
		{Ratio{}, Ratio{1, 0}, false},
		{Ratio{}, Ratio{0, 1}, false},
		{Ratio{}, Ratio{}, true},
	} {
		if got := tc.a.Equivalent(tc.b); got != tc.want {
			t.Errorf("%v.Equivalent(%v) = %v, want %v", tc.a, tc.b, got, tc.want)
		}
		// Symmetric, or the answer depends on argument order.
		if got := tc.b.Equivalent(tc.a); got != tc.want {
			t.Errorf("%v.Equivalent(%v) = %v, want %v (asymmetric)", tc.b, tc.a, got, tc.want)
		}
	}
}

// The default is 3:1 and it is chat-shaped. Pinned as a test rather than left to the
// var, because every unstated assumption in the product flows from this one value and a
// change to it silently reprices every report that did not pass a flag.
func TestTheDefaultIsChatShaped(t *testing.T) {
	if DefaultRatio != (Ratio{3, 1}) {
		t.Errorf("DefaultRatio = %v, want 3:1", DefaultRatio)
	}
	if got := DefaultRatio.Label(); got != "3:1 (chat)" {
		t.Errorf("the default does not name its preset: %q", got)
	}
	chat, ok := PresetNamed("chat")
	if !ok {
		t.Fatal("no chat preset")
	}
	if chat.Ratio != DefaultRatio {
		t.Errorf("chat is %v but the default is %v; the default must be a named shape "+
			"so a report can say what it assumed", chat.Ratio, DefaultRatio)
	}
}

// The presets are ordered input-heavy to output-heavy, which is also the order from
// "Bedrock wins hardest" to "self-hosting is easiest to justify". The ordering is
// load-bearing: the crossover search walks adjacent pairs and assumes monotonicity.
func TestPresetsAreOrderedInputHeavyFirst(t *testing.T) {
	if len(Presets) < 2 {
		t.Fatal("not enough presets to have an order")
	}
	prev := math.Inf(1)
	seen := map[string]bool{}
	for _, p := range Presets {
		if seen[p.Name] {
			t.Errorf("duplicate preset %q", p.Name)
		}
		seen[p.Name] = true
		if err := p.Ratio.Validate(); err != nil {
			t.Errorf("preset %q is not a valid mix: %v", p.Name, err)
		}
		if strings.TrimSpace(p.Why) == "" {
			t.Errorf("preset %q has no justification, so --explain cannot say why it was offered", p.Name)
		}
		per, ok := p.Ratio.perOutput()
		if !ok {
			t.Fatalf("preset %q is all-input, which has no position on the sweep axis", p.Name)
		}
		if per >= prev {
			t.Errorf("preset %q at %g is not more output-heavy than the one before (%g)",
				p.Name, per, prev)
		}
		prev = per
	}

	// And the names resolve, which is what the flag depends on.
	for _, name := range PresetNames() {
		if _, ok := PresetNamed(name); !ok {
			t.Errorf("PresetNames lists %q but PresetNamed does not resolve it", name)
		}
	}
	if _, ok := PresetNamed("nonexistent"); ok {
		t.Error("PresetNamed resolved a name that does not exist")
	}
}

// The direction, pinned against real Qwen3-32B meters. This is the claim an earlier
// draft of this repo's own notes got backwards, so it is asserted rather than
// commented: output-heavy traffic raises Bedrock's blend, which *lowers* the
// throughput a self-hosted instance needs to reach parity.
func TestAnOutputHeavyMixLowersTheBar(t *testing.T) {
	hourly := rate(4.00, "g7e.4xlarge")
	prevPrice, prevTPS := 0.0, math.Inf(1)
	for _, p := range Presets {
		parity, err := BreakEven(hourly, qwenInput, qwenOutput, p.Ratio.Assumptions(1.0))
		if err != nil {
			t.Fatalf("%s: %v", p.Name, err)
		}
		price, tps := parity.TokenPrice.MustValue(), parity.Throughput.MustValue()
		t.Logf("%-14s %-5s $%.4f/1M -> parity %.0f tok/s", p.Name, p.Ratio, price, tps)

		if price <= prevPrice {
			t.Errorf("%s prices at $%.4f, not above the previous $%.4f; the presets are "+
				"ordered input-heavy first, so the blend must rise", p.Name, price, prevPrice)
		}
		if tps >= prevTPS {
			t.Errorf("%s needs %.0f tok/s, not below the previous %.0f; a pricier Bedrock "+
				"blend makes self-hosting easier to justify, not harder", p.Name, tps, prevTPS)
		}
		prevPrice, prevTPS = price, tps
	}

	// The two figures the plan and CLAUDE.md quote, so a drift in either shows up here
	// rather than in prose nobody re-checks.
	three := parity(t, hourly, DefaultRatio.Assumptions(1.0))
	one := parity(t, hourly, Ratio{1, 1}.Assumptions(1.0))
	if got := round(three.TokenPrice.MustValue(), 4); got != 0.2625 {
		t.Errorf("3:1 blend = %v, want 0.2625", got)
	}
	if got := round(one.TokenPrice.MustValue(), 4); got != 0.375 {
		t.Errorf("1:1 blend = %v, want 0.375", got)
	}
	if got := round(three.Throughput.MustValue(), 0); got != 4233 {
		t.Errorf("3:1 parity = %v tok/s, want 4233", got)
	}
	if got := round(one.Throughput.MustValue(), 0); got != 2963 {
		t.Errorf("1:1 parity = %v tok/s, want 2963", got)
	}
	if got := round((one.TokenPrice.MustValue()-three.TokenPrice.MustValue())/
		three.TokenPrice.MustValue()*100, 0); got != 43 {
		t.Errorf("1:1 vs 3:1 price swing = %v%%, want 43%%", got)
	}
}

// A verdict that holds across every plausible mix is settled. This is the usual case
// for Qwen3-32B — 1200 tok/s is 2.5x short even at the most favourable mix — and the
// sweep has to say so rather than implying the answer is delicate.
func TestASettledVerdictDoesNotFlip(t *testing.T) {
	s := Sweep(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput,
		tps(1200, "published vLLM benchmark"), 1.0)

	if len(s.Points) != len(Presets) {
		t.Fatalf("%d points for %d presets", len(s.Points), len(Presets))
	}
	for i, pt := range s.Points {
		if pt.Preset.Name != Presets[i].Name {
			t.Errorf("point %d is %s, want %s", i, pt.Preset.Name, Presets[i].Name)
		}
		if pt.Outcome != OutcomeBedrock {
			t.Errorf("%s: outcome %s at 1200 tok/s, want bedrock", pt.Preset.Name, pt.Outcome)
		}
	}
	if s.Flips {
		t.Error("Flips = true where every mix says Bedrock; that reads as a marginal call")
	}
	if s.HasCrossover {
		t.Errorf("a crossover at %v where nothing crosses", s.Crossover)
	}
}

// A marginal verdict must announce itself. The user's real traffic mix is something
// they can measure and this tool cannot, so "your answer depends on which of these you
// are" is the most useful thing --explain can say.
func TestAMarginalVerdictFlipsAndLocatesTheCrossover(t *testing.T) {
	// A throughput chosen to sit between the parity figures of two adjacent presets:
	// 3800 tok/s beats 3:1's 4233 requirement at no mix below chat, and clears
	// balanced's 2963.
	s := Sweep(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput,
		tps(3800, "hypothetical benchmark"), 1.0)

	if !s.Flips {
		t.Fatalf("Flips = false; outcomes were %s", outcomes(s))
	}
	if !s.HasCrossover {
		t.Fatal("a flipping sweep with no crossover; --explain cannot say where the answer changes")
	}
	t.Logf("outcomes %s, crossover at %v", outcomes(s), s.Crossover)

	// The crossover must sit strictly between the two presets whose outcomes differ,
	// or it points at a mix that is not where the answer changes.
	var lo, hi float64
	var found bool
	for i := 1; i < len(s.Points); i++ {
		if s.Points[i-1].Outcome != s.Points[i].Outcome {
			lo, _ = s.Points[i].Preset.Ratio.perOutput()
			hi, _ = s.Points[i-1].Preset.Ratio.perOutput()
			found = true
			break
		}
	}
	if !found {
		t.Fatal("no adjacent pair of points disagrees, yet Flips is true")
	}
	got, ok := s.Crossover.perOutput()
	if !ok {
		t.Fatalf("crossover %v has no position on the sweep axis", s.Crossover)
	}
	if got <= lo || got >= hi {
		t.Errorf("crossover at %g is not between %g and %g", got, lo, hi)
	}
	// And it must be re-runnable: the figure is only useful if the user can pass it
	// back in to see the boundary case for themselves.
	if _, err := ParseRatio(s.Crossover.String()); err != nil {
		t.Errorf("the crossover %q does not parse as a mix: %v", s.Crossover, err)
	}
}

// An unpriced instance is a missing number, not a third verdict the mix moved the
// answer to. Counting undetermined as a flip would report every capacity-block-only
// type as a marginal call, when in fact nothing about the mix was in question.
func TestAnUndeterminedSweepIsNotAFlip(t *testing.T) {
	unpriced := report.Unavailable(report.UnitUSDPerHour, "p5e.48xlarge has no on-demand price")
	s := Sweep(unpriced, qwenInput, qwenOutput, tps(1200, "benchmark"), 1.0)

	if len(s.Points) != len(Presets) {
		t.Fatalf("%d points for %d presets; an unpriced instance still has a mix-dependent "+
			"Bedrock price to show", len(s.Points), len(Presets))
	}
	for _, pt := range s.Points {
		if pt.Outcome != OutcomeUndetermined {
			t.Errorf("%s: outcome %s with no hourly rate", pt.Preset.Name, pt.Outcome)
		}
		// The Bedrock side is still known and still moves with the mix, which is the
		// part worth printing even when the instance cannot be priced.
		if !pt.TokenPrice.Known() {
			t.Errorf("%s: no token price, though both meters were given", pt.Preset.Name)
		}
	}
	if s.Flips {
		t.Error("Flips = true across five undetermined outcomes; that reads as an " +
			"assumption-sensitive verdict where there is no verdict at all")
	}
}

// The flip rule at the point Sweep cannot reach. A real sweep's points all agree on
// whether they are decided — an unpriced instance is unpriced at every mix — so the
// mixed case has to be stated directly against the rule.
//
// It is worth stating because the wrong reading is tempting: a run where one mix has a
// verdict and another does not looks like a mix-sensitive answer. It isn't. Nothing
// about the traffic shape changed the recommendation; a number went missing.
func TestAGapBesideAVerdictIsNotAFlip(t *testing.T) {
	at := func(i int, o Outcome) SensitivityPoint {
		return SensitivityPoint{Preset: Presets[i], Outcome: o}
	}
	for _, tc := range []struct {
		name   string
		points []SensitivityPoint
		want   bool
	}{
		{"one verdict beside four gaps", []SensitivityPoint{
			at(0, OutcomeUndetermined), at(1, OutcomeUndetermined),
			at(2, OutcomeBedrock),
			at(3, OutcomeUndetermined), at(4, OutcomeUndetermined),
		}, false},
		{"a gap between two agreeing verdicts", []SensitivityPoint{
			at(0, OutcomeBedrock), at(1, OutcomeUndetermined), at(2, OutcomeBedrock),
		}, false},
		// The gap must not mask a real disagreement on either side of it.
		{"a gap between two disagreeing verdicts", []SensitivityPoint{
			at(0, OutcomeBedrock), at(1, OutcomeUndetermined), at(2, OutcomeSelfHost),
		}, true},
		{"no token meter beside a verdict", []SensitivityPoint{
			at(0, OutcomeNoTokenPrice), at(1, OutcomeSelfHost),
		}, false},
		{"nothing decided at all", []SensitivityPoint{
			at(0, OutcomeUndetermined), at(1, OutcomeNoTokenPrice),
		}, false},
		{"no points", nil, false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := flips(tc.points); got != tc.want {
				t.Errorf("flips = %v, want %v", got, tc.want)
			}
			// And a sweep built from these points reports the same thing, since Decided
			// and Flips are read together by --explain.
			s := Sensitivity{Points: tc.points, Flips: flips(tc.points)}
			if s.Flips && !s.Decided() {
				t.Error("a sweep with no verdict reports a flip")
			}
		})
	}
}

// The two crossover rules Sweep cannot reach, for the same reason as the flip rule: a
// real sweep's points either all have verdicts or none do, and no preset is all-input.
// Both guards decide where --explain tells the user their answer changes, so a wrong one
// points at the wrong traffic shape.
func TestTheCrossoverBracketsOnlyDecidedPresets(t *testing.T) {
	at := func(i int, o Outcome) SensitivityPoint {
		return SensitivityPoint{Preset: Presets[i], Outcome: o}
	}

	// A gap sitting between the two disagreeing verdicts tells us nothing about where
	// the answer turns, so the bracket is the whole span across it: 10:1 to 3:1 gives
	// 6.5:1, not 7.5:1 from pairing the gap with its neighbour.
	s := Sensitivity{Points: []SensitivityPoint{
		at(0, OutcomeBedrock), at(1, OutcomeUndetermined), at(2, OutcomeSelfHost),
	}}
	got, ok := s.crossover()
	if !ok {
		t.Fatal("no crossover across a gap between two disagreeing verdicts")
	}
	if want := (Ratio{Input: 6.5, Output: 1}); got != want {
		t.Errorf("crossover = %v, want %v; an undetermined point was used as a bracket", got, want)
	}

	// An all-input mix has no position on the sweep axis — the ratio is unbounded, not
	// large — so a pair that includes one yields no crossover rather than a fabricated
	// midpoint. Reported as "it depends on your mix" with no number, which is honest.
	allInput := Sensitivity{Points: []SensitivityPoint{
		{Preset: Preset{Name: "embedding", Ratio: Ratio{1, 0}}, Outcome: OutcomeBedrock},
		at(2, OutcomeSelfHost),
	}}
	if got, ok := allInput.crossover(); ok {
		t.Errorf("crossover %v placed against an all-input mix, which has no axis position", got)
	}
	// The flip itself still stands: the outcomes do differ.
	if !flips(allInput.Points) {
		t.Error("flips = false where bedrock and self-host disagree")
	}
}

// perOutput is the sweep axis, and an all-input mix has no place on it. Stated directly
// because its false branch is reachable only through the crossover pair above.
func TestAnAllInputMixHasNoPositionOnTheSweepAxis(t *testing.T) {
	for _, tc := range []struct {
		r    Ratio
		want float64
		ok   bool
	}{
		{Ratio{3, 1}, 3, true},
		{Ratio{10, 1}, 10, true},
		{Ratio{1, 3}, 1.0 / 3.0, true},
		{Ratio{6, 2}, 3, true},
		{Ratio{0, 1}, 0, true},
		// Not 1, and not "very large": there is no such figure.
		{Ratio{1, 0}, 0, false},
		{Ratio{}, 0, false},
	} {
		got, ok := tc.r.perOutput()
		if ok != tc.ok {
			t.Errorf("%v.perOutput() ok = %v, want %v", tc.r, ok, tc.ok)
		}
		if ok && got != tc.want {
			t.Errorf("%v.perOutput() = %g, want %g", tc.r, got, tc.want)
		}
	}
}

// A model with no token meter has nothing to be cheaper than, at any mix. 94 of 132
// mappable models are in this state, so it is the common case rather than an edge.
func TestAModelWithNoTokenPriceNeverFlips(t *testing.T) {
	none := report.Unavailable(report.UnitUSDPerMillionTokens, "marketplace-only model")
	s := Sweep(rate(4.00, "g7e.4xlarge"), none, none, tps(1200, "benchmark"), 1.0)
	for _, pt := range s.Points {
		if pt.Outcome != OutcomeNoTokenPrice {
			t.Errorf("%s: outcome %s for a marketplace-only model", pt.Preset.Name, pt.Outcome)
		}
	}
	if s.Flips {
		t.Error("Flips = true where there is no token meter at any mix")
	}
}

// A sweep whose every point fails to compute has no points and claims nothing. The
// honest rendering of "this says nothing" is an empty result, not a flip.
func TestASweepThatCannotComputeClaimsNothing(t *testing.T) {
	// Utilization out of range makes every BreakEven call fail, which is the one input
	// that is malformed rather than merely unknown.
	s := Sweep(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput, tps(1200, "benchmark"), 0)
	if len(s.Points) != 0 {
		t.Errorf("%d points from a sweep where every break-even was malformed", len(s.Points))
	}
	if s.Flips || s.HasCrossover {
		t.Errorf("Flips=%v HasCrossover=%v from an empty sweep", s.Flips, s.HasCrossover)
	}
}

// The mix must reach the report envelope as the pair, not as a rendered string. A
// report the history log cannot re-derive the arithmetic from is one where an
// assumption change and a price change look the same.
func TestTheMixReachesTheEnvelope(t *testing.T) {
	r := Ratio{10, 1}
	a := r.Assumptions(0.6)
	if a.InputWeight != 10 || a.OutputWeight != 1 || a.Utilization != 0.6 {
		t.Fatalf("Assumptions() = %+v", a)
	}
	if err := a.Validate(); err != nil {
		t.Fatalf("a preset mix does not validate: %v", err)
	}

	rec := a.Record(report.Assumptions{})
	if rec.InputTokenWeight != 10 || rec.OutputTokenWeight != 1 {
		t.Errorf("recorded %g:%g, want 10:1", rec.InputTokenWeight, rec.OutputTokenWeight)
	}
	if rec.Ratio() != r.String() {
		t.Errorf("the envelope renders %q where the mix renders %q", rec.Ratio(), r.String())
	}
	if rec.Utilization != 0.6 {
		t.Errorf("recorded utilization %g, want 0.6", rec.Utilization)
	}
}

// outcomes renders a sweep's verdicts for a failure message.
func outcomes(s Sensitivity) string {
	parts := make([]string, 0, len(s.Points))
	for _, pt := range s.Points {
		parts = append(parts, pt.Preset.Name+"="+string(pt.Outcome))
	}
	return strings.Join(parts, " ")
}
