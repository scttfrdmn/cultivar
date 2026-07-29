package compare

import (
	"fmt"
	"strings"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// The flag surface for the traffic mix, and the --explain block that shows what the
// assumption is worth.
//
// Separate from ratio.go because this is presentation: what a user types and what they
// read back. The arithmetic does not depend on any of it, and a renderer that grew
// arithmetic of its own would be a second place the ratio could be misstated.

// RatioFlag binds a traffic mix to a command-line flag.
//
// It implements [flag.Value], so `flag.Var(&f, "input-output-ratio", f.Usage())` gives
// the flag its parsing, its default, and its error messages from one place. The zero
// value is the default mix, which is what makes an unset flag behave identically to
// `--input-output-ratio chat` rather than producing a 0:0 that fails validation later.
type RatioFlag struct {
	ratio Ratio
	set   bool
}

// Ratio returns the mix, which is [DefaultRatio] when the flag was never set.
func (f *RatioFlag) Ratio() Ratio {
	if !f.set {
		return DefaultRatio
	}
	return f.ratio
}

// Explicit reports whether the user stated a mix.
//
// The report says so either way, and this is the difference between the two phrasings:
// "at your stated 10:1" versus "assuming 3:1 (chat) — pass --input-output-ratio to
// change it". An assumed value presented as a stated one is the failure this whole
// issue is about, and it is the default case, so it is the one that must be labelled.
func (f *RatioFlag) Explicit() bool { return f.set }

// Set implements [flag.Value].
func (f *RatioFlag) Set(s string) error {
	r, err := ParseRatio(s)
	if err != nil {
		return err
	}
	f.ratio, f.set = r, true
	return nil
}

// String implements [flag.Value] and renders the effective mix.
//
// The flag package calls this on the zero value to discover a default for the usage
// text, so it must not panic on a nil receiver.
func (f *RatioFlag) String() string {
	if f == nil {
		return DefaultRatio.String()
	}
	return f.Ratio().String()
}

// Usage returns the flag's help text, listing the presets with their ratios.
//
// Built rather than written out so a new preset cannot be added without appearing in
// the help — a named shape nobody can discover is not an option, it is dead code.
func (f *RatioFlag) Usage() string {
	parts := make([]string, 0, len(Presets))
	for _, p := range Presets {
		parts = append(parts, fmt.Sprintf("%s=%s", p.Name, p.Ratio))
	}
	return fmt.Sprintf("assumed input:output token mix, as a ratio (3:1) or a name "+
		"(%s); Bedrock bills output at 4x input for Qwen3-32B, so this is an assumption "+
		"worth ~43%% and it is printed in every report (default %s)",
		strings.Join(parts, ", "), DefaultRatio.Label())
}

// Note returns the one line every report carries about the mix in force.
//
// Every report, not just --explain. A blended $/1M figure has already assumed a
// traffic shape, and a report that prints the figure without the shape is stating a
// price it cannot support. Which is why this returns a sentence and not a value: there
// is no correct way to print the price without it.
func (f *RatioFlag) Note() string {
	if f.Explicit() {
		return fmt.Sprintf("traffic mix %s, as given", f.Ratio().Label())
	}
	return fmt.Sprintf("traffic mix %s, assumed — pass --input-output-ratio to state your own",
		f.Ratio().Label())
}

// Explain renders the sensitivity sweep as the --explain block.
//
// The question it answers is not "what is the number" but "is this answer robust?"
// So the verdict-stability line comes first and the table second: a user who reads
// only the first line should learn whether their traffic mix could change the
// recommendation, which is the only thing about this sweep that is actionable.
//
// Returns an empty string when there is nothing to say, so a caller can print it
// unconditionally without emitting an empty heading.
func (s Sensitivity) Explain(current Ratio) string {
	if len(s.Points) == 0 {
		return ""
	}
	var b strings.Builder

	// A stability claim requires something to be stable. With every point undetermined
	// there is no recommendation for the mix to hold across, and saying it "holds" would
	// present a missing number as a robust answer — the exact inversion this tool is
	// built to prevent. The table below is still worth printing: the Bedrock side moves
	// with the mix whether or not the instance could be priced.
	switch {
	case !s.Decided():
		b.WriteString("There is no verdict to be sensitive to — see the table for why. " +
			"The Bedrock prices below still move with the mix.\n")
	case s.Flips && s.HasCrossover:
		fmt.Fprintf(&b, "This recommendation depends on your traffic mix. It changes at "+
			"roughly %s — more input-heavy than that and the answer is the other way.\n",
			s.Crossover)
	case s.Flips:
		b.WriteString("This recommendation depends on your traffic mix: it is not the same " +
			"across the range below.\n")
	default:
		fmt.Fprintf(&b, "This recommendation holds across every traffic mix from %s to %s, "+
			"so it does not rest on that assumption.\n",
			Presets[0].Ratio, Presets[len(Presets)-1].Ratio)
	}

	// The name column is sized from the longest preset name rather than a literal, so
	// adding a longer one cannot silently shear the table.
	w := 0
	for _, p := range Presets {
		if len(p.Name) > w {
			w = len(p.Name)
		}
	}
	fmt.Fprintf(&b, "\n  %-*s %-6s %-13s %-14s %s\n",
		w, "workload", "mix", "bedrock", "parity needs", "verdict")
	for _, pt := range s.Points {
		marker := "  "
		if pt.Preset.Ratio.Equivalent(current) {
			marker = "> " // the mix actually in force
		}
		fmt.Fprintf(&b, "%s%-*s %-6s %-13s %-14s %s\n",
			marker, w, pt.Preset.Name, pt.Preset.Ratio,
			renderPerMillion(pt.TokenPrice), renderTokensPerSecond(pt.Throughput), pt.Outcome)
	}

	// The mix in force may be one nobody named, and a table with no marked row leaves a
	// reader guessing which line is theirs.
	if !s.containsEquivalent(current) {
		fmt.Fprintf(&b, "\nYour mix, %s, is not one of the named shapes above; it sits "+
			"between them.\n", current.Label())
	}
	return b.String()
}

// containsEquivalent reports whether the swept presets include this mix.
func (s Sensitivity) containsEquivalent(r Ratio) bool {
	for _, pt := range s.Points {
		if pt.Preset.Ratio.Equivalent(r) {
			return true
		}
	}
	return false
}

// renderPerMillion formats a token price, or says why there isn't one.
//
// Unpriced rather than $0.00, and the same word every unavailable amount in this tool
// renders as: a gap must never be printed in a form arithmetic could be done on.
func renderPerMillion(a report.Amount) string {
	v, ok := a.Value()
	if !ok {
		return "unpriced"
	}
	return fmt.Sprintf("$%.4f/1M", v)
}

// renderTokensPerSecond formats a throughput requirement.
func renderTokensPerSecond(a report.Amount) string {
	v, ok := a.Value()
	if !ok {
		return "unknown"
	}
	return fmt.Sprintf("%.0f tok/s", v)
}
