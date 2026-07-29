//go:build live

// Opt-in checks that the traffic-mix assumption still behaves against today's real
// Bedrock meters. Run with `make test-live` (AWS_PROFILE=aws). Free read-only Price List
// queries only — nothing launches and nothing bills.
//
// The offline tests pin the arithmetic against fixed $0.15/$0.60 rates. What they cannot
// check is the premise underneath: that Bedrock still meters input and output
// separately, and that output still costs enough more than input for the mix to matter.
// If AWS ever flattened those two meters, the whole --input-output-ratio flag would
// become a knob with no effect, and every "assuming 3:1" line in the product would be
// stating a caveat that no longer applies. That should surface here as a failure to
// investigate rather than as a claim nobody re-checked.
package compare

import (
	"strings"
	"testing"

	"github.com/scttfrdmn/cultivar/internal/bedrock"
)

// The reason the flag exists: output is metered well above input, so a blended $/1M
// figure has already assumed a traffic shape.
//
// Measured 2026-07-28 for Qwen3-32B in us-east-1: $0.15/1M input, $0.60/1M output — 4x.
// The assertion is that output is *meaningfully* pricier, not that the multiple is
// exactly 4: the ratio is AWS's to change, and a drift from 4x to 3x is a fact to note,
// not a bug. A multiple at or below 1 is the real finding, because at that point the mix
// stops moving the answer and this whole file's premise is gone.
func TestLiveOutputIsStillMeteredAboveInput(t *testing.T) {
	price := liveTokenPrice(t)
	in := price.Amount(bedrock.TierStandard, bedrock.MeterInput)
	out := price.Amount(bedrock.TierStandard, bedrock.MeterOutput)

	iv, ok := in.Value()
	if !ok {
		t.Fatalf("no live input meter: %s", in.Source())
	}
	ov, ok := out.Value()
	if !ok {
		t.Fatalf("no live output meter: %s", out.Source())
	}
	t.Logf("Qwen3-32B standard: input $%.4f/1M, output $%.4f/1M — %.2fx (recorded 4.00x on 2026-07-28)",
		iv, ov, ov/iv)

	if ov <= iv {
		t.Errorf("output $%.4f/1M is not above input $%.4f/1M; if the two meters have "+
			"flattened, the traffic mix no longer moves the blend and every report's "+
			"assumption line is stating a caveat that no longer applies", ov, iv)
	}
	if ov/iv < 1.5 {
		t.Errorf("output is only %.2fx input; the mix is a %s-to-%s assumption in this tool "+
			"because that multiple was 4x, and below ~1.5x the flag stops earning its "+
			"place in the default output", ov/iv, Presets[0].Ratio, Presets[len(Presets)-1].Ratio)
	}
}

// The direction, against live meters rather than the pinned ones. This is the claim an
// earlier draft of this repo's notes had backwards in three places, so it is checked in
// both the offline suite and here: output-heavy traffic raises Bedrock's blend, which
// *lowers* the throughput a self-hosted instance needs to reach parity.
//
// Which means summarization is where Bedrock wins hardest, and the input-heavy end of
// the preset range is the conservative end for a self-hosting recommendation.
func TestLiveTheMixMovesTheBarInTheDocumentedDirection(t *testing.T) {
	price := liveTokenPrice(t)
	hourly := liveHourly(t, "g7e.4xlarge")
	in := price.Amount(bedrock.TierStandard, bedrock.MeterInput)
	out := price.Amount(bedrock.TierStandard, bedrock.MeterOutput)

	prevPrice, prevTPS := 0.0, 0.0
	for i, p := range Presets {
		parity, err := BreakEven(hourly, in, out, p.Ratio.Assumptions(1.0))
		if err != nil {
			t.Fatalf("%s: %v", p.Name, err)
		}
		blend, tps := parity.TokenPrice.MustValue(), parity.Throughput.MustValue()
		t.Logf("%-14s %-5s $%.4f/1M → parity %.0f tok/s", p.Name, p.Ratio, blend, tps)

		if i > 0 {
			if blend <= prevPrice {
				t.Errorf("%s blends at $%.4f, not above the more input-heavy $%.4f; the "+
					"presets are ordered input-heavy first, so a live blend that does not "+
					"rise means the meters or the ordering changed", p.Name, blend, prevPrice)
			}
			if tps >= prevTPS {
				t.Errorf("%s needs %.0f tok/s, not below the previous %.0f; a pricier Bedrock "+
					"blend makes self-hosting easier to justify, not harder", p.Name, tps, prevTPS)
			}
		}
		prevPrice, prevTPS = blend, tps
	}
}

// The sweep --explain prints, run against live prices end to end.
//
// The check is not the numbers but the shape of the conclusion: a verdict that holds
// across every plausible mix must be reported as settled, and the block must never
// describe a sweep with no verdict as stable. For Qwen3-32B at any realistic throughput
// the answer is Bedrock everywhere, so a flip here means either the rates moved a long
// way or the sweep is reading them wrong.
func TestLiveTheSweepStillSaysBedrockAtEveryMix(t *testing.T) {
	price := liveTokenPrice(t)
	s := Sweep(liveHourly(t, "g7e.4xlarge"),
		price.Amount(bedrock.TierStandard, bedrock.MeterInput),
		price.Amount(bedrock.TierStandard, bedrock.MeterOutput),
		// A published vLLM figure, not a measurement of ours. Deliberately generous:
		// 1200 tok/s is the sort of number a single-node deployment quotes.
		tps(1200, "published vLLM benchmark"), 1.0)

	if len(s.Points) != len(Presets) {
		t.Fatalf("%d points for %d presets against live prices", len(s.Points), len(Presets))
	}
	if !s.Decided() {
		t.Fatal("no swept mix produced a verdict from live prices; both meters and the " +
			"hourly rate resolved, so this is a break in the comparison rather than a gap")
	}
	for _, pt := range s.Points {
		if pt.Outcome != OutcomeBedrock {
			t.Errorf("%s: %s at 1200 tok/s against live rates; the product's central claim "+
				"is that a 32B model loses to Bedrock at every plausible mix",
				pt.Preset.Name, pt.Outcome)
		}
	}
	if s.Flips {
		t.Errorf("the live sweep flips: %s", outcomes(s))
	}

	block := s.Explain(DefaultRatio)
	t.Logf("--explain against live prices:\n%s", block)
	if block == "" {
		t.Fatal("a fully-priced sweep rendered nothing")
	}
	// The verdict is settled, so the block must say so — and must not print a gap in a
	// form arithmetic could be done on.
	for _, want := range []string{"holds across", "3:1"} {
		if !strings.Contains(block, want) {
			t.Errorf("the live --explain block does not mention %q:\n%s", want, block)
		}
	}
	for _, unwanted := range []string{"unpriced", "unknown", "$0.0000", "no verdict"} {
		if strings.Contains(block, unwanted) {
			t.Errorf("the live --explain block contains %q, though every input resolved:\n%s",
				unwanted, block)
		}
	}
}
