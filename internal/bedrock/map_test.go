package bedrock

import (
	"testing"
	"time"
)

var observed = time.Date(2026, 7, 27, 12, 0, 0, 0, time.UTC)

// qwen3_32bMapping is the live mapping for Qwen/Qwen3-32B, read 2026-07-27,
// verbatim in published order. Note which entry comes first.
var qwen3_32bMapping = &Mapping{
	HFID:      "Qwen/Qwen3-32B",
	OnBedrock: true,
	Regions:   []string{"us-east-1", "us-east-2", "us-west-1", "us-west-2"},
	Entries: []Entry{
		{
			ModelID:    "huggingface-reasoning-qwen3-32b",
			Catalog:    CatalogMarketplace,
			Confidence: "confirmed",
			Regions:    []string{"us-east-1", "us-east-2", "us-west-1", "us-west-2"},
		},
		{
			ModelID:    "qwen.qwen3-32b-v1:0",
			Catalog:    CatalogFoundationModel,
			Confidence: "confirmed",
			Regions:    []string{"us-east-1", "us-east-2", "us-west-2"},
		},
	},
	ObservedAt: observed,
}

func TestFoundationModelsScansPastIndexZero(t *testing.T) {
	// This project's headline example is itself the trap: Qwen/Qwen3-32B's first
	// bedrock[] entry is the *marketplace* listing, and the foundation model is
	// second. Reading index 0 concludes "no token price, you must self-host" for the
	// one model whose entire verdict is "use Bedrock, it's ~15x cheaper".
	if got := qwen3_32bMapping.Entries[0].Catalog; got != CatalogMarketplace {
		t.Fatalf("fixture drift: entry 0 is %q, the live mapping had marketplace first", got)
	}
	fms := qwen3_32bMapping.FoundationModels()
	if len(fms) != 1 {
		t.Fatalf("found %d foundation models, want 1", len(fms))
	}
	if fms[0].ModelID != "qwen.qwen3-32b-v1:0" {
		t.Errorf("model id = %q, want qwen.qwen3-32b-v1:0", fms[0].ModelID)
	}
	if !qwen3_32bMapping.HasTokenPricing() {
		t.Error("HasTokenPricing = false for a model with a foundation-model entry")
	}
	if qwen3_32bMapping.MarketplaceOnly() {
		t.Error("MarketplaceOnly = true for a model with a foundation-model entry")
	}
}

func TestMarketplaceOnlyHasNoTokenPrice(t *testing.T) {
	// 94 of 132 mapped repos look like this. The Bedrock offer must not fire: there
	// is no per-token rate to compare against, so recommending "use Bedrock instead"
	// would tell the user to rent an endpoint under a different control plane and
	// call it a saving.
	m := &Mapping{
		HFID:      "01-ai/Yi-1.5-34B",
		OnBedrock: true,
		Regions:   []string{"us-east-1", "us-east-2", "us-west-2"},
		Entries: []Entry{{
			ModelID:    "huggingface-llm-yi-1-5-34b",
			Catalog:    CatalogMarketplace,
			Confidence: "confirmed",
			Regions:    []string{"us-east-1", "us-east-2", "us-west-2"},
		}},
		ObservedAt: observed,
	}
	if m.HasTokenPricing() {
		t.Error("a marketplace-only model reported token pricing")
	}
	if !m.MarketplaceOnly() {
		t.Error("MarketplaceOnly = false for a marketplace-only model")
	}
	if len(m.FoundationModels()) != 0 {
		t.Errorf("FoundationModels returned %d entries for a marketplace-only model", len(m.FoundationModels()))
	}
}

func TestNotOnBedrockIsDistinctFromMarketplaceOnly(t *testing.T) {
	// Three different verdicts, and collapsing any two gives wrong advice:
	// unmapped → "self-hosting is your only option"; marketplace-only → "Bedrock can
	// host it but you still pay for hardware"; foundation-model → "compare per-token".
	unmapped := &Mapping{HFID: "some/thing", OnBedrock: false, ObservedAt: observed}
	if unmapped.MarketplaceOnly() {
		t.Error("an unmapped repo reported as marketplace-only")
	}
	if unmapped.HasTokenPricing() {
		t.Error("an unmapped repo reported token pricing")
	}
}

func TestFoundationModelsDedupesAndOrders(t *testing.T) {
	// 6 of the 38 token-billable repos publish multiple foundation-model entries —
	// context-window variants like meta.llama3-3-70b-instruct-v1:0 and
	// ...-v1:0:128k. They are distinct ids and both stay; a duplicated id does not.
	m := &Mapping{
		HFID: "meta-llama/Llama-3.3-70B-Instruct",
		Entries: []Entry{
			{ModelID: "meta.llama3-3-70b-instruct-v1:0:128k", Catalog: CatalogFoundationModel},
			{ModelID: "meta.llama3-3-70b-instruct-v1:0", Catalog: CatalogFoundationModel},
			{ModelID: "meta.llama3-3-70b-instruct-v1:0", Catalog: CatalogFoundationModel},
		},
	}
	got := m.FoundationModels()
	if len(got) != 2 {
		t.Fatalf("got %d entries, want 2 after dedupe", len(got))
	}
	if got[0].ModelID != "meta.llama3-3-70b-instruct-v1:0" ||
		got[1].ModelID != "meta.llama3-3-70b-instruct-v1:0:128k" {
		t.Errorf("order = %q, %q; want deterministic sort by model id",
			got[0].ModelID, got[1].ModelID)
	}
}

func TestRegionScopingUsesTheEntrysOwnList(t *testing.T) {
	// Qwen3-32B's union covers 4 regions but its foundation-model entry covers 3 —
	// us-west-1 has the marketplace listing only. 8 of the 132 mapped repos diverge
	// this way, and using the union would price a model against a Bedrock endpoint
	// that region cannot serve.
	if got := qwen3_32bMapping.FoundationModelsIn("us-west-2"); len(got) != 1 {
		t.Errorf("us-west-2: got %d foundation models, want 1", len(got))
	}
	if got := qwen3_32bMapping.FoundationModelsIn("us-west-1"); len(got) != 0 {
		t.Errorf("us-west-1: got %d foundation models, want 0 — the union lists it, the entry does not", len(got))
	}
	// Region comparison is case-insensitive; AWS region ids are lowercase but user
	// input is not always.
	if got := qwen3_32bMapping.FoundationModelsIn("US-EAST-2"); len(got) != 1 {
		t.Errorf("US-EAST-2: got %d, want 1", len(got))
	}
}

func TestUnknownCatalogIsSurfacedNotBilled(t *testing.T) {
	// A catalog AWS adds later must not be assumed token-billable. Getting this wrong
	// invents a per-token comparison against a price that does not exist; getting it
	// conservatively wrong only withholds an option, and the caveat says why.
	m := &Mapping{
		HFID: "x/y",
		Entries: []Entry{
			{ModelID: "a", Catalog: "serverless-v2"},
			{ModelID: "b", Catalog: CatalogMarketplace},
			{ModelID: "c", Catalog: "serverless-v2"},
		},
	}
	if m.HasTokenPricing() {
		t.Error("an unknown catalog was treated as token-billable")
	}
	if got := m.UnknownCatalogs(); len(got) != 1 || got[0] != "serverless-v2" {
		t.Errorf("UnknownCatalogs = %v, want [serverless-v2]", got)
	}
	if got := qwen3_32bMapping.UnknownCatalogs(); len(got) != 0 {
		t.Errorf("UnknownCatalogs = %v for a mapping with only known catalogs", got)
	}
}

func TestCatalogTokenBillable(t *testing.T) {
	if !CatalogFoundationModel.TokenBillable() {
		t.Error("foundation-model is not token-billable")
	}
	for _, c := range []Catalog{CatalogMarketplace, "", "unknown", "Foundation-Model"} {
		if c.TokenBillable() {
			t.Errorf("catalog %q reported as token-billable", c)
		}
	}
}

func TestAgePrefersGeneratedAtThenLastModified(t *testing.T) {
	now := observed
	gen := &Mapping{
		GeneratedAt: now.Add(-3 * time.Hour),
		PublishedAt: now.Add(-30 * time.Minute),
	}
	if d, ok := gen.Age(now); !ok || d != 3*time.Hour {
		t.Errorf("age = %v (%v), want 3h from generatedAt", d, ok)
	}
	// Per-model documents publish no generatedAt, so Last-Modified is all there is.
	perModel := &Mapping{PublishedAt: now.Add(-90 * time.Minute)}
	if d, ok := perModel.Age(now); !ok || d != 90*time.Minute {
		t.Errorf("age = %v (%v), want 90m from Last-Modified", d, ok)
	}
	// Neither known must report unknown, not zero: zero reads as "generated just
	// now", which is the opposite of the truth.
	if d, ok := (&Mapping{}).Age(now); ok {
		t.Errorf("age = %v, want unknown when no timestamp is available", d)
	}
}

func TestNilMappingIsSafe(t *testing.T) {
	// A lookup that returns nil on an error path must not panic the report renderer.
	var m *Mapping
	if m.HasTokenPricing() || m.MarketplaceOnly() {
		t.Error("a nil mapping reported Bedrock availability")
	}
	if got := m.FoundationModels(); got != nil {
		t.Errorf("FoundationModels on nil = %v", got)
	}
	if got := m.UnknownCatalogs(); got != nil {
		t.Errorf("UnknownCatalogs on nil = %v", got)
	}
	if _, ok := m.Age(observed); ok {
		t.Error("a nil mapping reported an age")
	}
}

func TestNormalizeID(t *testing.T) {
	// The per-model paths are case-sensitive: hf/Qwen/Qwen3-32B.json is a 404 while
	// hf/qwen/qwen3-32b.json is the entry, so skipping this reads as "not on Bedrock".
	for _, in := range []string{"Qwen/Qwen3-32B", "  qwen/qwen3-32b  ", "/Qwen/Qwen3-32B/"} {
		if got := normalizeID(in); got != "qwen/qwen3-32b" {
			t.Errorf("normalizeID(%q) = %q", in, got)
		}
	}
}
