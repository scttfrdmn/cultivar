// The traffic mix, as a thing a user can state and a report has to print.
//
// Bedrock meters input and output separately and output costs 4x input for
// Qwen3-32B, so any single $/1M figure has already assumed a traffic shape. The
// assumption is worth 43% at the extremes of ordinary usage, which is enough to flip
// a verdict — so it is a parsed, echoed, named argument rather than a constant
// somewhere in the arithmetic.
//
// The direction is the part people get backwards, including an earlier draft of this
// repo's own notes. Output-heavy traffic makes Bedrock's blend *pricier*, which makes
// self-hosting easier to justify and lowers the throughput bar. So summarization —
// long inputs, short outputs — is the case where Bedrock wins hardest, and the
// input-heavy end of the range is the conservative end for a self-hosting
// recommendation, not the flattering one.

package compare

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// DefaultRatio is the traffic mix assumed when the user states none: three input
// tokens per output token.
//
// It is an assumption and is labelled as one everywhere it appears. 3:1 is roughly
// chat-shaped — a prompt plus a few turns of history against a paragraph of reply —
// and it sits between the two realistic extremes rather than at either. What it is
// not is a measurement of anyone's traffic, which is why every report prints it and
// why --explain shows what happens on both sides of it.
var DefaultRatio = Ratio{Input: 3, Output: 1}

// Ratio is an input:output traffic mix.
//
// A pair rather than a single quotient, because the pair is what a user states and
// what the report has to echo back: "3:1" round-trips, while 3.0 has already thrown
// away how it was written. The arithmetic only ever needs the pair anyway — see
// [report.Blend], which weights the two meters directly.
type Ratio struct {
	Input  float64
	Output float64
}

// Preset is a named traffic shape.
//
// Named presets exist because the numbers are not the hard part — knowing that
// summarization and chat sit in very different places is. A user who has never
// thought about their input:output mix can pick a workload and get an honest label
// in the report instead of silently inheriting the default.
type Preset struct {
	// Name is the flag value, e.g. "summarization".
	Name string

	// Ratio is the mix.
	Ratio Ratio

	// Why is the one-line justification, printed by --explain. Each is a shape
	// argument, not a measurement, and says so.
	Why string
}

// Presets are the named traffic shapes, in order from most input-heavy to most
// output-heavy — which is also the order from "Bedrock wins hardest" to "self-hosting
// is easiest to justify".
//
// Deliberately few. A long menu implies a precision none of these have; the point is
// to get a user into the right part of the range and to make the label visible in the
// report, not to pretend the difference between 8:1 and 10:1 is meaningful.
var Presets = []Preset{
	{"summarization", Ratio{10, 1},
		"long documents in, a paragraph out; the most input-heavy realistic shape, " +
			"and the one where Bedrock's blend is cheapest"},
	{"rag", Ratio{5, 1},
		"retrieved context dominates the prompt, answers are short"},
	{"chat", Ratio{3, 1},
		"a prompt plus conversation history against a paragraph of reply; the default"},
	{"balanced", Ratio{1, 1},
		"equal input and output, as in translation or rewriting"},
	{"generation", Ratio{1, 3},
		"a short instruction produces long output — code, drafts, agent loops; the " +
			"shape most favourable to self-hosting"},
}

// PresetNamed returns the preset with this name.
func PresetNamed(name string) (Preset, bool) {
	for _, p := range Presets {
		if p.Name == strings.ToLower(strings.TrimSpace(name)) {
			return p, true
		}
	}
	return Preset{}, false
}

// PresetNames returns the preset names in order, for a flag's usage text.
func PresetNames() []string {
	out := make([]string, 0, len(Presets))
	for _, p := range Presets {
		out = append(out, p.Name)
	}
	return out
}

// ParseRatio parses an input:output mix as a user writes it.
//
// Accepts "3:1", "3", a preset name, and decimals ("2.5:1"). A bare number is read as
// N:1, since "my traffic is 3 to 1" is how the assumption is usually held.
//
// A slash is deliberately not accepted. "3/1" is ambiguous between a ratio and a
// quotient in a way "3:1" is not, and the cost of guessing wrong is a silently
// inverted assumption — the same class of error as reading a batch rate as standard.
func ParseRatio(s string) (Ratio, error) {
	raw := strings.TrimSpace(s)
	if raw == "" {
		return Ratio{}, fmt.Errorf("no traffic mix given; state one as input:output "+
			"(e.g. 3:1) or by name (%s)", strings.Join(PresetNames(), ", "))
	}
	if p, ok := PresetNamed(raw); ok {
		return p.Ratio, nil
	}

	in, out := raw, "1"
	if i := strings.Index(raw, ":"); i >= 0 {
		in, out = strings.TrimSpace(raw[:i]), strings.TrimSpace(raw[i+1:])
	} else if strings.ContainsAny(raw, "/\\") {
		// Caught explicitly so the message names the accepted form rather than
		// failing as an unparseable float.
		return Ratio{}, fmt.Errorf("%q: write a traffic mix with a colon (3:1), not a slash; "+
			"a slash reads as a quotient and an inverted mix is a silently wrong verdict", raw)
	}

	r := Ratio{}
	for _, f := range []struct {
		text  string
		into  *float64
		label string
	}{{in, &r.Input, "input"}, {out, &r.Output, "output"}} {
		v, err := strconv.ParseFloat(f.text, 64)
		if err != nil {
			return Ratio{}, fmt.Errorf("%q: %s weight %q is not a number; expected a mix like "+
				"3:1 or a name (%s)", raw, f.label, f.text, strings.Join(PresetNames(), ", "))
		}
		*f.into = v
	}
	if err := r.Validate(); err != nil {
		return Ratio{}, fmt.Errorf("%q: %w", raw, err)
	}
	return r, nil
}

// Validate reports whether r is a usable mix.
//
// A zero on one side is allowed and meaningful: embedding workloads really are
// all-input, and an agent loop's output-only leg really is all-output. What is not
// allowed is both zero, which is not a mix at all, or a negative weight, which would
// silently subtract one meter from the other and produce a blend below the cheaper
// rate.
func (r Ratio) Validate() error {
	if math.IsNaN(r.Input) || math.IsNaN(r.Output) {
		return fmt.Errorf("traffic mix %s is not numeric", r)
	}
	if math.IsInf(r.Input, 0) || math.IsInf(r.Output, 0) {
		return fmt.Errorf("traffic mix %s is infinite", r)
	}
	if r.Input < 0 || r.Output < 0 {
		return fmt.Errorf("negative weight in traffic mix %s; a negative weight subtracts "+
			"one meter from the other and blends below the cheaper rate", r)
	}
	if r.Input+r.Output == 0 {
		return fmt.Errorf("traffic mix 0:0 states no mix at all")
	}
	return nil
}

// String renders the mix the way it is written, via [report.BlendRatio] so the
// separator and ordering match every other place the ratio appears.
func (r Ratio) String() string { return report.BlendRatio(r.Input, r.Output) }

// Assumptions returns break-even assumptions for this mix at the given utilization.
func (r Ratio) Assumptions(utilization float64) Assumptions {
	return Assumptions{InputWeight: r.Input, OutputWeight: r.Output, Utilization: utilization}
}

// Label describes the mix for a report line, naming the matching preset when there is
// one: "3:1 (chat)".
//
// The name is attached rather than substituted. A preset name alone is not
// reproducible — the numbers behind "chat" could be adjusted in a later release, and
// a stored report would then mean something different than when it was written.
func (r Ratio) Label() string {
	for _, p := range Presets {
		if p.Ratio.Equivalent(r) {
			return fmt.Sprintf("%s (%s)", r, p.Name)
		}
	}
	return r.String()
}

// Equivalent reports whether two mixes describe the same traffic. 6:2 and 3:1 are the
// same assumption written differently, and a report that treated them as distinct
// would show a spurious change in the history log.
func (r Ratio) Equivalent(other Ratio) bool {
	if r.Input*other.Output != other.Input*r.Output {
		return false
	}
	// Cross-multiplication alone equates 0:0 with every mix, since both sides come out
	// 0 == 0. That matters because 0:0 is the zero value: a Ratio nobody set, or one a
	// failed parse returned. Without this line [Label] would render it as the first
	// preset it compared against, and "0:0 (summarization)" reads as a stated
	// assumption rather than as the absence of one.
	//
	// The all-input against all-output case is already handled — 1:0 against 0:1 is
	// 1 != 0 — so this guards the zero mix, not those two.
	return (r.Input == 0) == (other.Input == 0) && (r.Output == 0) == (other.Output == 0)
}

// Sensitivity is what the verdict does across the range of plausible traffic mixes.
//
// This is the answer to the question --explain exists for: not "what is the number"
// but "is this answer robust or marginal?" A verdict that holds from 10:1 to 1:3 is
// settled; one that flips between rag and chat is a coin toss the user should know is
// a coin toss, because their actual traffic mix is something they can measure and this
// tool cannot.
type Sensitivity struct {
	// Points is one entry per preset, in the same order as [Presets].
	Points []SensitivityPoint

	// Flips reports whether the outcome is not the same at every point. When true,
	// the recommendation depends on an assumption rather than on a fact about AWS.
	Flips bool

	// Crossover is the mix at which the verdict changes, as a ratio against 1, when
	// there is one within the swept range. Interpolated between adjacent presets
	// rather than solved exactly — it is a "your traffic would have to be about this
	// shape" figure, and false precision would misrepresent it.
	Crossover Ratio

	// HasCrossover reports whether Crossover was found.
	HasCrossover bool
}

// Decided reports whether any swept mix produced a verdict.
//
// False means the sweep says nothing about the recommendation — every point was
// undetermined or had no token meter — and a caller must not describe such a sweep as
// stable. "The answer holds across every traffic mix" over five missing numbers reads
// as a robust finding and is the opposite of one.
func (s Sensitivity) Decided() bool {
	for _, pt := range s.Points {
		if pt.Outcome.Decided() {
			return true
		}
	}
	return false
}

// SensitivityPoint is one mix's outcome.
type SensitivityPoint struct {
	Preset     Preset
	TokenPrice report.Amount
	Throughput report.Amount
	Outcome    Outcome
}

// Sweep evaluates the same hardware and model across every preset traffic mix.
//
// hourly, inputPrice and outputPrice are the same amounts [BreakEven] takes;
// achievable is the throughput figure to judge each point against. Utilization is
// held fixed, because this sweep is about one assumption at a time — varying two at
// once produces a surface nobody can read.
//
// A point whose break-even could not be computed is skipped rather than recorded as
// undetermined: the sweep exists to show how the answer moves with the mix, and a row
// that failed for an unrelated reason is noise in that picture. If every point fails
// the result has no points and Flips is false, which is the honest rendering of "this
// says nothing".
func Sweep(hourly, inputPrice, outputPrice, achievable report.Amount, utilization float64) Sensitivity {
	var s Sensitivity
	for _, p := range Presets {
		parity, err := BreakEven(hourly, inputPrice, outputPrice, p.Ratio.Assumptions(utilization))
		if err != nil {
			continue
		}
		c := parity.At(achievable)
		s.Points = append(s.Points, SensitivityPoint{
			Preset: p, TokenPrice: parity.TokenPrice,
			Throughput: parity.Throughput, Outcome: c.Outcome,
		})
	}

	s.Flips = flips(s.Points)
	if s.Flips {
		s.Crossover, s.HasCrossover = s.crossover()
	}
	return s
}

// flips reports whether the outcome changes across the swept mixes.
//
// Only a change between two *decided* outcomes counts. Undetermined is not a third
// verdict that the mix moved the answer to — it is a missing number, and counting it as
// a flip would report every unpriced instance as a marginal call.
//
// A function rather than a loop inside [Sweep] because the rule is not reachable
// through Sweep with realistic inputs: an unpriced instance is unpriced at every mix,
// and a model with no token meter has none at any mix, so every point in a real sweep
// agrees on whether it is decided. Left inline, the guard would be a rule no test could
// state.
func flips(points []SensitivityPoint) bool {
	var first Outcome
	for _, pt := range points {
		if !pt.Outcome.Decided() {
			continue
		}
		if first == "" {
			first = pt.Outcome
			continue
		}
		if pt.Outcome != first {
			return true
		}
	}
	return false
}

// crossover finds the mix between the two adjacent presets whose outcomes differ.
//
// Linear interpolation on the input weight against a fixed output weight of 1, which
// is the axis the presets vary along. It is approximate on purpose: the point of the
// figure is "your traffic would have to be about 4:1 for this to change", and solving
// it exactly would present an assumption boundary as a measurement.
func (s Sensitivity) crossover() (Ratio, bool) {
	decided := make([]SensitivityPoint, 0, len(s.Points))
	for _, pt := range s.Points {
		if pt.Outcome.Decided() {
			decided = append(decided, pt)
		}
	}
	for i := 1; i < len(decided); i++ {
		lo, hi := decided[i-1], decided[i]
		if lo.Outcome == hi.Outcome {
			continue
		}
		// Normalized so both sides are per one output token, which is what makes the
		// midpoint meaningful: 10:1 and 1:3 are 10 and 0.333 on that axis.
		a, aok := lo.Preset.Ratio.perOutput()
		b, bok := hi.Preset.Ratio.perOutput()
		if !aok || !bok {
			continue
		}
		return Ratio{Input: (a + b) / 2, Output: 1}, true
	}
	return Ratio{}, false
}

// perOutput returns the input weight per single output token. False for an all-input
// mix, where there is no such figure — the ratio is unbounded, not large.
func (r Ratio) perOutput() (float64, bool) {
	if r.Output == 0 {
		return 0, false
	}
	return r.Input / r.Output, true
}
