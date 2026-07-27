//go:build live

// Opt-in suite that hits the real hf-bedrock-map API. Run with `make test-live`.
// Static JSON over GitHub Pages — no AWS credentials, no cost.
//
// This one doubles as a census: the "38 of 132 have a token price" claim drives the
// product's framing (the Bedrock escape hatch fires ~29% of the time), so it is
// re-measured rather than trusted. Drift here means the README, the comparison
// page's copy, and cultivar#6 need updating.
package bedrock

import (
	"context"
	"testing"
	"time"
)

func liveCtx(t *testing.T) (context.Context, func()) {
	t.Helper()
	return context.WithTimeout(context.Background(), 30*time.Second)
}

// TestLiveCensusStillMatches re-measures the split the product's framing rests on.
func TestLiveCensusStillMatches(t *testing.T) {
	c := NewClient()
	ctx, cancel := liveCtx(t)
	defer cancel()

	idx, err := c.Index(ctx)
	if err != nil {
		t.Fatal(err)
	}

	var tokenBillable, marketplaceOnly, hiddenBehindIndexZero, multiFM, regionDivergent int
	unknown := map[string]int{}
	for _, m := range idx {
		for _, cat := range m.UnknownCatalogs() {
			unknown[cat]++
		}
		fms := m.FoundationModels()
		switch {
		case len(fms) > 0:
			tokenBillable++
			if !m.Entries[0].Catalog.TokenBillable() {
				hiddenBehindIndexZero++
			}
			if len(fms) > 1 {
				multiFM++
			}
		default:
			marketplaceOnly++
		}
		for _, e := range m.Entries {
			if len(e.Regions) != len(m.Regions) {
				regionDivergent++
				break
			}
		}
	}

	t.Logf("%d models: %d token-billable, %d marketplace-only", len(idx), tokenBillable, marketplaceOnly)
	t.Logf("%d hide their foundation-model path behind entry 0; %d publish several; %d have region divergence",
		hiddenBehindIndexZero, multiFM, regionDivergent)

	// Recorded 2026-07-27: 132 / 38 / 94. The mapping grows as AWS adds models, so
	// growth is fine and shrinkage or a ratio swing is what needs a look.
	if len(idx) < 132 {
		t.Errorf("index has %d models, was 132 on 2026-07-27; the mapping should not shrink", len(idx))
	}
	if tokenBillable+marketplaceOnly != len(idx) {
		t.Errorf("census does not partition: %d + %d != %d", tokenBillable, marketplaceOnly, len(idx))
	}
	if tokenBillable < 38 {
		t.Errorf("%d models are token-billable, was 38 on 2026-07-27", tokenBillable)
	}
	if ratio := float64(tokenBillable) / float64(len(idx)); ratio > 0.5 {
		t.Errorf("token-billable share is now %.0f%%, was 29%%. The README and the page copy "+
			"say the Bedrock escape hatch fires for a minority of models — re-check that claim.", ratio*100)
	}
	// The whole reason FoundationModels() scans instead of indexing.
	if hiddenBehindIndexZero == 0 {
		t.Log("note: no model currently hides its foundation-model path behind entry 0 " +
			"(2 did on 2026-07-27). Keep the scan — publication order is not a contract.")
	}
	if len(unknown) > 0 {
		t.Logf("unrecognized catalogs in the mapping: %v — check whether they bill per token "+
			"before Catalog.TokenBillable can treat them as such", unknown)
	}
}

// TestLiveQwen3IsStillTheTrap guards the specific case this project's headline
// verdict depends on. If entry order or the catalog changed, the recorded fixture
// and the comparison example both need revisiting.
func TestLiveQwen3IsStillTheTrap(t *testing.T) {
	c := NewClient()
	ctx, cancel := liveCtx(t)
	defer cancel()

	m, err := c.Lookup(ctx, "Qwen/Qwen3-32B")
	if err != nil {
		t.Fatal(err)
	}
	if !m.OnBedrock {
		t.Fatal("Qwen/Qwen3-32B is no longer mapped to Bedrock; the headline verdict rests on it")
	}
	fms := m.FoundationModels()
	if len(fms) == 0 {
		t.Fatal("Qwen/Qwen3-32B has no foundation-model entry; there is no token price to compare against")
	}
	if fms[0].ModelID != "qwen.qwen3-32b-v1:0" {
		t.Errorf("foundation model id = %q, recorded qwen.qwen3-32b-v1:0. "+
			"The Price List join in the token-price resolver keys off this.", fms[0].ModelID)
	}
	if m.Entries[0].Catalog.TokenBillable() {
		t.Logf("entry 0 is now the foundation model; it was the marketplace listing on 2026-07-27")
	}
	// us-west-1 carries the marketplace listing but not the foundation model.
	if got := m.FoundationModelsIn("us-west-1"); len(got) != 0 {
		t.Errorf("us-west-1 now offers %d foundation-model entries; it offered none on 2026-07-27", len(got))
	}
	for _, r := range []string{"us-east-1", "us-east-2", "us-west-2"} {
		if len(m.FoundationModelsIn(r)) == 0 {
			t.Errorf("%s no longer offers qwen.qwen3-32b-v1:0", r)
		}
	}
	if age, ok := m.Age(time.Now()); !ok {
		t.Error("no freshness signal on a per-model lookup; Last-Modified must be present")
	} else {
		t.Logf("mapping age %s (per-model documents carry no generatedAt)", age.Round(time.Minute))
	}
}

// TestLiveUnmappedRepoIs404 checks the semantics against the real Pages host,
// including that its HTML 404 body does not surface as a decode error.
func TestLiveUnmappedRepoIs404(t *testing.T) {
	c := NewClient()
	ctx, cancel := liveCtx(t)
	defer cancel()

	m, err := c.Lookup(ctx, "cultivar-test/not-a-real-repo-9f3a")
	if err != nil {
		t.Fatalf("a 404 surfaced as an error: %v", err)
	}
	if m.OnBedrock || m.HasTokenPricing() {
		t.Error("an unmapped repo reported a Bedrock path")
	}
}

// TestLivePerModelMatchesIndex checks that the two endpoints agree. They were
// byte-identical for qwen/qwen3-32b on 2026-07-27; if they diverge, code that reads
// one and code that reads the other would disagree about the same model.
func TestLivePerModelMatchesIndex(t *testing.T) {
	c := NewClient()
	ctx, cancel := liveCtx(t)
	defer cancel()

	idx, err := c.Index(ctx)
	if err != nil {
		t.Fatal(err)
	}
	for _, id := range []string{"qwen/qwen3-32b", "meta-llama/llama-3.3-70b-instruct", "01-ai/yi-1.5-34b"} {
		want := idx[id]
		if want == nil {
			t.Errorf("%s missing from the index", id)
			continue
		}
		got, err := c.Lookup(ctx, id)
		if err != nil {
			t.Errorf("%s: %v", id, err)
			continue
		}
		if got.HFID != want.HFID || got.OnBedrock != want.OnBedrock || len(got.Entries) != len(want.Entries) {
			t.Errorf("%s: per-model and index disagree: %+v vs %+v", id, got, want)
			continue
		}
		for i := range got.Entries {
			if got.Entries[i].ModelID != want.Entries[i].ModelID ||
				got.Entries[i].Catalog != want.Entries[i].Catalog {
				t.Errorf("%s entry %d: per-model %+v, index %+v", id, i, got.Entries[i], want.Entries[i])
			}
		}
	}
}
