package compare

import (
	"flag"
	"strings"
	"testing"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// The flag must satisfy flag.Value, or none of the binding below is real.
var _ flag.Value = (*RatioFlag)(nil)

// An unset flag behaves exactly like --input-output-ratio chat. The alternative is a
// zero-valued 0:0 that fails validation somewhere downstream, at which point the error
// names an arithmetic problem rather than a missing flag.
func TestAnUnsetFlagIsTheDefaultMix(t *testing.T) {
	var f RatioFlag
	if got := f.Ratio(); got != DefaultRatio {
		t.Errorf("unset flag = %v, want %v", got, DefaultRatio)
	}
	if f.Explicit() {
		t.Error("Explicit() = true on an unset flag; an assumed mix would be reported as stated")
	}
	if got := f.String(); got != DefaultRatio.String() {
		t.Errorf("String() = %q, want %q", got, DefaultRatio.String())
	}
	// The flag package calls String on a zero or nil value to build its usage text.
	var nilFlag *RatioFlag
	if got := nilFlag.String(); got != DefaultRatio.String() {
		t.Errorf("nil receiver String() = %q, want %q", got, DefaultRatio.String())
	}
}

// Set goes through ParseRatio, so the flag accepts everything the parser does and
// refuses everything it refuses — one grammar, not two.
func TestTheFlagParsesWhatTheParserDoes(t *testing.T) {
	for _, tc := range []struct {
		in   string
		want Ratio
	}{
		{"10:1", Ratio{10, 1}},
		{"generation", Ratio{1, 3}},
		{"2", Ratio{2, 1}},
	} {
		var f RatioFlag
		if err := f.Set(tc.in); err != nil {
			t.Fatalf("Set(%q): %v", tc.in, err)
		}
		if f.Ratio() != tc.want {
			t.Errorf("Set(%q) -> %v, want %v", tc.in, f.Ratio(), tc.want)
		}
		if !f.Explicit() {
			t.Errorf("Set(%q) did not mark the mix as stated", tc.in)
		}
	}

	// A rejected value must leave the flag at its default rather than half-applied: a
	// partially-set mix would be used as if the user had asked for it.
	var f RatioFlag
	if err := f.Set("3/1"); err == nil {
		t.Fatal("Set accepted a slash-separated mix")
	}
	if f.Ratio() != DefaultRatio || f.Explicit() {
		t.Errorf("a failed Set left the flag at %v (explicit=%v)", f.Ratio(), f.Explicit())
	}
}

// Bound to a real FlagSet, which is how the CLI will use it. This catches the wiring
// mistakes a direct Set call cannot: a flag registered with the wrong Value semantics,
// or a usage string the flag package chokes on.
func TestTheFlagBindsToAFlagSet(t *testing.T) {
	var f RatioFlag
	fs := flag.NewFlagSet("compare", flag.ContinueOnError)
	fs.SetOutput(new(strings.Builder))
	fs.Var(&f, "input-output-ratio", f.Usage())

	if err := fs.Parse([]string{"--input-output-ratio", "summarization"}); err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got := f.Ratio(); got != (Ratio{10, 1}) {
		t.Errorf("parsed %v, want 10:1", got)
	}

	// A bad value must fail the parse rather than being silently absorbed.
	var bad RatioFlag
	fs2 := flag.NewFlagSet("compare", flag.ContinueOnError)
	fs2.SetOutput(new(strings.Builder))
	fs2.Var(&bad, "input-output-ratio", bad.Usage())
	if err := fs2.Parse([]string{"--input-output-ratio", "sideways"}); err == nil {
		t.Error("a nonsense mix parsed without error")
	}
	if bad.Ratio() != DefaultRatio {
		t.Errorf("a rejected flag left the mix at %v", bad.Ratio())
	}
}

// Every preset appears in the usage text. A named shape nobody can discover is not an
// option, it is dead code — and building the text from Presets is what keeps a newly
// added one from being invisible.
func TestUsageListsEveryPreset(t *testing.T) {
	var f RatioFlag
	usage := f.Usage()
	for _, p := range Presets {
		if !strings.Contains(usage, p.Name) {
			t.Errorf("usage text does not mention preset %q", p.Name)
		}
		if !strings.Contains(usage, p.Ratio.String()) {
			t.Errorf("usage text does not give %q's ratio %s", p.Name, p.Ratio)
		}
	}
	if !strings.Contains(usage, DefaultRatio.Label()) {
		t.Errorf("usage text does not state the default %s", DefaultRatio.Label())
	}
	// The reason the flag exists has to be in the help, or a user has no basis for
	// choosing a value.
	if !strings.Contains(usage, "4x") {
		t.Error("usage text does not say why the mix matters")
	}
}

// The distinction between an assumed and a stated mix is the whole point of the issue,
// so the note has to make it in words. A default presented as the user's own choice is
// the failure mode; it is also the common case.
func TestTheNoteDistinguishesAssumedFromStated(t *testing.T) {
	var unset RatioFlag
	note := unset.Note()
	if !strings.Contains(note, "assumed") {
		t.Errorf("an unset mix reads as %q, which does not say it was assumed", note)
	}
	if !strings.Contains(note, "--input-output-ratio") {
		t.Errorf("the note does not say how to change the assumption: %q", note)
	}
	if !strings.Contains(note, "3:1") {
		t.Errorf("the note does not state the mix in force: %q", note)
	}

	var set RatioFlag
	if err := set.Set("10:1"); err != nil {
		t.Fatal(err)
	}
	note = set.Note()
	if strings.Contains(note, "assumed") {
		t.Errorf("a stated mix reads as assumed: %q", note)
	}
	if !strings.Contains(note, "10:1") {
		t.Errorf("the note does not echo the stated mix: %q", note)
	}
	// The preset name is attached where there is one, since "10:1 (summarization)" tells
	// a user whether the number matches their intent and a bare 10:1 does not.
	if !strings.Contains(note, "summarization") {
		t.Errorf("the note does not name the matching shape: %q", note)
	}
}

// The stability sentence comes first because it is the only actionable thing in the
// block. A user who reads one line should learn whether their own traffic could change
// the recommendation.
func TestExplainLeadsWithVerdictStability(t *testing.T) {
	settled := Sweep(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput,
		tps(1200, "published benchmark"), 1.0)
	out := settled.Explain(DefaultRatio)
	t.Logf("settled:\n%s", out)

	first := strings.SplitN(out, "\n", 2)[0]
	if !strings.Contains(first, "holds across") {
		t.Errorf("a settled verdict does not lead with its stability: %q", first)
	}
	if strings.Contains(first, "depends on") {
		t.Errorf("a settled verdict reads as marginal: %q", first)
	}

	marginal := Sweep(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput,
		tps(3800, "hypothetical benchmark"), 1.0)
	out = marginal.Explain(DefaultRatio)
	t.Logf("marginal:\n%s", out)

	first = strings.SplitN(out, "\n", 2)[0]
	if !strings.Contains(first, "depends on your traffic mix") {
		t.Errorf("a flipping verdict does not lead by saying so: %q", first)
	}
	if !marginal.HasCrossover {
		t.Fatal("no crossover to print")
	}
	if !strings.Contains(out, marginal.Crossover.String()) {
		t.Errorf("the block does not name the crossover %s:\n%s", marginal.Crossover, out)
	}
}

// The table has a row per preset and marks the mix in force, or a reader cannot tell
// which line is theirs.
func TestExplainMarksTheMixInForce(t *testing.T) {
	s := Sweep(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput, tps(1200, "benchmark"), 1.0)

	out := s.Explain(Ratio{10, 1})
	for _, p := range Presets {
		if !strings.Contains(out, p.Name) {
			t.Errorf("no row for %q:\n%s", p.Name, out)
		}
	}
	var marked []string
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "> ") {
			marked = append(marked, line)
		}
	}
	if len(marked) != 1 {
		t.Fatalf("%d marked rows, want exactly 1:\n%s", len(marked), out)
	}
	if !strings.Contains(marked[0], "summarization") {
		t.Errorf("the marked row is %q, want summarization", marked[0])
	}

	// An equivalent mix written differently still marks its row: 30:10 is 3:1.
	out = s.Explain(Ratio{30, 10})
	for _, line := range strings.Split(out, "\n") {
		if strings.HasPrefix(line, "> ") && !strings.Contains(line, "chat") {
			t.Errorf("30:10 marked %q instead of chat", line)
		}
	}

	// And a mix nobody named says so, rather than leaving the table with no marked row
	// and no explanation.
	out = s.Explain(Ratio{7, 2})
	if strings.Contains(out, "\n> ") {
		t.Errorf("7:2 matched a named preset:\n%s", out)
	}
	if !strings.Contains(out, "not one of the named shapes") {
		t.Errorf("an unnamed mix is not called out:\n%s", out)
	}
	if !strings.Contains(out, "7:2") {
		t.Errorf("the block does not state the unnamed mix:\n%s", out)
	}
}

// A gap renders as "unpriced", never as $0.00. This is the same rule as everywhere else
// in the tool: a missing number must not be printed in a form arithmetic could be done
// on, because a $0.00/1M Bedrock rate makes serverless look free.
func TestExplainNeverPrintsAGapAsZero(t *testing.T) {
	unpriced := report.Unavailable(report.UnitUSDPerHour, "p5e.48xlarge has no on-demand price")
	s := Sweep(unpriced, qwenInput, qwenOutput, tps(1200, "benchmark"), 1.0)
	out := s.Explain(DefaultRatio)
	t.Logf("unpriced instance:\n%s", out)

	if strings.Contains(out, "$0.0000") || strings.Contains(out, "0 tok/s") {
		t.Errorf("a gap rendered as a zero:\n%s", out)
	}
	if !strings.Contains(out, "unknown") {
		t.Errorf("the unresolvable parity figure is not marked unknown:\n%s", out)
	}
	// The Bedrock side is still known and still moves with the mix, which is the part
	// worth printing even when the instance cannot be priced.
	if !strings.Contains(out, "$0.2625/1M") {
		t.Errorf("the known Bedrock prices are missing:\n%s", out)
	}
	// And there is no verdict here, so the block must not claim one is stable. Saying an
	// answer "holds across every mix" over five missing numbers presents a gap as a
	// robust finding.
	if strings.Contains(out, "holds across") {
		t.Errorf("five undetermined outcomes described as a stable recommendation:\n%s", out)
	}
	if !strings.Contains(out, "no verdict") {
		t.Errorf("the block does not say there is no verdict:\n%s", out)
	}

	noMeter := report.Unavailable(report.UnitUSDPerMillionTokens, "marketplace-only model")
	s = Sweep(rate(4.00, "g7e.4xlarge"), noMeter, noMeter, tps(1200, "benchmark"), 1.0)
	out = s.Explain(DefaultRatio)
	if strings.Contains(out, "$0.0000") {
		t.Errorf("a missing token meter rendered as $0.0000:\n%s", out)
	}
	if !strings.Contains(out, "unpriced") {
		t.Errorf("a missing token meter is not marked unpriced:\n%s", out)
	}
	if strings.Contains(out, "holds across") {
		t.Errorf("a model with no token meter described as a stable recommendation:\n%s", out)
	}
}

// The table's columns are sized from the presets, so the longest name cannot shear the
// layout. A sheared table is not a correctness bug, but this block exists to be read.
func TestTheExplainTableColumnsLineUp(t *testing.T) {
	s := Sweep(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput, tps(1200, "benchmark"), 1.0)
	out := s.Explain(DefaultRatio)

	var rows []string
	for _, line := range strings.Split(out, "\n") {
		if strings.Contains(line, "tok/s") || strings.Contains(line, "workload") {
			rows = append(rows, line)
		}
	}
	if len(rows) != len(Presets)+1 {
		t.Fatalf("%d table rows for %d presets plus a header:\n%s", len(rows), len(Presets), out)
	}
	// Every row's verdict must start at the same column, which is the property a
	// too-narrow name field breaks.
	want := strings.Index(rows[0], "verdict")
	for _, row := range rows[1:] {
		for _, o := range []Outcome{OutcomeBedrock, OutcomeSelfHost, OutcomeUndetermined, OutcomeNoTokenPrice} {
			if i := strings.Index(row, string(o)); i >= 0 {
				if i != want {
					t.Errorf("verdict column at %d, header at %d:\n%s", i, want, out)
				}
				break
			}
		}
	}
}

// An empty sweep prints nothing, so a caller can emit the block unconditionally without
// producing a heading over an empty table.
func TestExplainSaysNothingWhenItKnowsNothing(t *testing.T) {
	empty := Sweep(rate(4.00, "g7e.4xlarge"), qwenInput, qwenOutput, tps(1200, "benchmark"), 0)
	if got := empty.Explain(DefaultRatio); got != "" {
		t.Errorf("an empty sweep rendered:\n%s", got)
	}
}
