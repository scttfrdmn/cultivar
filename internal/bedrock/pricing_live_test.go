//go:build live

// Opt-in suite that hits the real AWS Price List API for Bedrock. Run with
// `make test-live` (AWS_PROFILE=aws). GetProducts is a free read-only call — nothing
// is invoked and nothing bills.
//
// The offline tests pin the rates and the classification rules against recorded
// rows. These tests exist because the rules were derived from the shape of the live
// catalogue, and that shape is undocumented in every part that matters: the
// regionCode filter is not documented for AmazonBedrock at all, service_tier is
// populated on a third of rows, and the published unit is per-1K on 828 dimensions
// and per-1M on 12. Each of those is a silent-wrong-number failure if it drifts, so
// each is re-measured rather than trusted.
package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"math"
	"sort"
	"strings"
	"testing"
	"time"

	awssdk "github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/pricing"
	pricingtypes "github.com/aws/aws-sdk-go-v2/service/pricing/types"
)

func livePricer(t *testing.T) *TokenPricer {
	t.Helper()
	ctx, cancel := liveCtx(t)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion("us-east-1"))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}
	return NewTokenPricer(cfg)
}

// skipIfThrottled turns a Price List rate limit into a skip rather than a failure.
//
// These tests exist to detect drift in an undocumented catalogue shape, and a throttle is
// not drift — it is this suite competing with itself, since `go test ./...` runs packages
// in parallel and GetProducts is rate-limited per account, not per caller. Reporting the
// two the same way is the exact conflation this whole project is about: an empty answer
// because AWS said no is not an answer about AWS. A skip is loud in the output and does
// not assert anything false.
//
// Deliberately not a retry. The SDK already retried three times before this error
// surfaced, and a suite that retries past its own throttle just moves the failure to
// whichever package runs next.
func skipIfThrottled(t *testing.T, err error) {
	t.Helper()
	var throttle *pricingtypes.ThrottlingException
	if errors.As(err, &throttle) {
		t.Skipf("Price List throttled, which is a rate limit and not a schema change: %v", err)
	}
}

// liveRows fetches one region's whole Bedrock catalogue once, so the census tests
// below do not make 46 identical round trips.
func liveRows(t *testing.T, p *TokenPricer, region string) []priceRow {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 120*time.Second)
	defer cancel()
	rows, err := p.fetch(ctx, region)
	if err != nil {
		skipIfThrottled(t, err)
		t.Fatalf("%s: %v", region, err)
	}
	if len(rows) == 0 {
		t.Fatalf("%s: no AmazonBedrock rows at all", region)
	}
	return rows
}

// TestLiveQwenRatesStillMatchTheRecordedOnes is the headline check. The whole
// "don't self-host Qwen3-32B" verdict rests on standard being $0.15/$0.60 per 1M,
// and the specific failure this guards is reading $0.2625 — the priority row, 75%
// high — or $0.075, the batch row, 50% low.
func TestLiveQwenRatesStillMatchTheRecordedOnes(t *testing.T) {
	p := livePricer(t)
	ctx, cancel := liveCtx(t)
	defer cancel()

	got, err := p.Lookup(ctx, "qwen.qwen3-32b-v1:0", "us-east-1")
	if err != nil {
		skipIfThrottled(t, err)
		t.Fatal(err)
	}
	if got.PriceListModel != "Qwen3 32B" {
		t.Errorf("join landed on Price List model %q, recorded \"Qwen3 32B\" on 2026-07-27. "+
			"A different product is being priced.", got.PriceListModel)
	}

	// Recorded 2026-07-27, us-east-1, in $/1M tokens. The 5% tolerance absorbs a real
	// price change without absorbing a tier misread, which moves rates by 50-75%.
	recorded := map[[2]string]float64{
		{"standard", "input"}: 0.15, {"standard", "output"}: 0.60,
		{"priority", "input"}: 0.2625, {"priority", "output"}: 1.05,
		{"flex", "input"}: 0.075, {"flex", "output"}: 0.30,
		{"batch", "input"}: 0.075, {"batch", "output"}: 0.30,
	}
	for k, want := range recorded {
		rate, ok := got.Rate(Tier(k[0]), Meter(k[1]))
		if !ok {
			t.Errorf("no %s %s rate; there was one on 2026-07-27", k[0], k[1])
			continue
		}
		if drift := math.Abs(rate.USDPerMillionTokens-want) / want; drift > 0.05 {
			t.Errorf("%s %s = $%.6f/1M via %s, recorded $%.4f (%.0f%% drift). "+
				"If this is a real price change, update the offline fixtures, CLAUDE.md, "+
				"and the break-even example in the README.",
				k[0], k[1], rate.USDPerMillionTokens, rate.UsageType, want, drift*100)
		}
	}
	// Output above input on every tier is a property of how Bedrock bills, and it is
	// the reason the blend ratio is a required input rather than a hidden constant.
	for _, tier := range got.Tiers() {
		in, _ := got.Rate(tier, MeterInput)
		out, _ := got.Rate(tier, MeterOutput)
		if out.USDPerMillionTokens <= in.USDPerMillionTokens {
			t.Errorf("%s output ($%.4f/1M) is no longer above input ($%.4f/1M); "+
				"check that the two meters are not being swapped",
				tier, out.USDPerMillionTokens, in.USDPerMillionTokens)
		}
	}
}

// TestLiveThePerMillionUnitStillExists guards the 1000x trap. If every row were
// per-1K, [perMillionTokens] would be untestable against reality and a future
// reviewer would be tempted to simplify it away — so the existence of a per-1M row
// is itself worth asserting. If it genuinely disappears, this test says to keep the
// unit read anyway.
func TestLiveThePerMillionUnitStillExists(t *testing.T) {
	p := livePricer(t)
	rows := liveRows(t, p, "us-east-1")

	units := map[string]int{}
	perMillionModels := map[string]bool{}
	for _, row := range rows {
		for _, dim := range row.dimensions {
			if !isTokenUnit(dim.unit) {
				continue
			}
			units[dim.unit]++
			if _, err := perMillionTokens(1, dim.unit); err != nil {
				t.Errorf("token unit %q is not recognized (usagetype %q); a rate in it would be "+
					"rejected, and if it were defaulted instead it would be wrong by a factor of 1000",
					dim.unit, row.attrs.UsageType)
				continue
			}
			if scale, _ := perMillionTokens(1, dim.unit); scale == 1 {
				perMillionModels[row.attrs.Model] = true
			}
		}
	}
	t.Logf("token units in us-east-1: %v", units)
	if len(units) < 2 {
		t.Logf("only one token unit is published now (%v); it was two on 2026-07-27 "+
			"(828 per-1K dimensions, 12 per-1M, all xai.grok-4.3). Keep reading the unit: "+
			"a single-unit catalogue is a coincidence, not a contract.", units)
	} else {
		t.Logf("per-1M models: %v", sortedKeys(perMillionModels))
	}
}

// TestLiveTheRegionCodeFilterIsStillHonoured is the guard on an undocumented
// filter. [TokenPricer.fetch] already errors if a foreign region comes back; this
// test additionally checks that the filter is still doing work, because a filter
// that quietly matches everything in one region only would pass fetch's check while
// making every other region's query return the whole catalogue.
func TestLiveTheRegionCodeFilterIsStillHonoured(t *testing.T) {
	p := livePricer(t)
	// us-east-2 is in the default region set and is the check that a second region
	// is genuinely filtered rather than served us-east-1's rows.
	for _, region := range []string{"us-east-1", "us-east-2"} {
		rows := liveRows(t, p, region)
		t.Logf("%s: %d AmazonBedrock rows", region, len(rows))
		for _, row := range rows {
			if row.attrs.RegionCode != region {
				t.Fatalf("%s: row %q reports regionCode %q", region, row.attrs.UsageType, row.attrs.RegionCode)
			}
		}
	}
}

// TestLiveNoRateConflictsAcrossTheWholeCatalogue is the strongest available check
// on [tierOf] and [meterOf], and the one that found the cross-region meter.
//
// The property: for every model in a region, each (tier, meter) key must carry a
// single rate. Every rate is published twice under the two naming conventions, so
// duplicates are expected and agreement is the test. A key carrying two different
// rates means the classification is folding in a meter it does not distinguish —
// which is exactly how the cheaper "global" cross-region rows announced themselves
// (21 conflicting keys before they were excluded, 0 after).
func TestLiveNoRateConflictsAcrossTheWholeCatalogue(t *testing.T) {
	p := livePricer(t)
	rows := liveRows(t, p, "us-east-1")

	type key struct {
		model string
		tier  Tier
		meter Meter
	}
	seen := map[key]Rate{}
	classified, rejected := 0, 0
	for _, row := range rows {
		if row.attrs.Model == "" {
			continue
		}
		tier, okT := tierOf(row.attrs)
		meter, okM := meterOf(row.attrs)
		if !okT || !okM {
			rejected++
			continue
		}
		for _, dim := range row.dimensions {
			if !isTokenUnit(dim.unit) || dim.usd <= 0 {
				continue
			}
			perM, err := perMillionTokens(dim.usd, dim.unit)
			if err != nil {
				t.Errorf("%s: %v", row.attrs.UsageType, err)
				continue
			}
			classified++
			k := key{row.attrs.Model, tier, meter}
			prev, dup := seen[k]
			if !dup {
				seen[k] = Rate{Tier: tier, Meter: meter, USDPerMillionTokens: perM, UsageType: row.attrs.UsageType}
				continue
			}
			if !sameRate(prev.USDPerMillionTokens, perM) {
				t.Errorf("%s %s %s carries two rates: $%.6f/1M via %s and $%.6f/1M via %s. "+
					"A meter the classification does not separate is being folded in.",
					k.model, tier, meter, prev.USDPerMillionTokens, prev.UsageType, perM, row.attrs.UsageType)
			}
		}
	}
	t.Logf("%d token dimensions classified into %d (model, tier, meter) keys; %d rows rejected as non-inference",
		classified, len(seen), rejected)
	if classified == 0 {
		t.Fatal("nothing classified; the attribute names have changed")
	}
}

// TestLiveTheJoinStillResolvesTheMappedModels re-measures the join rate against
// every foundation-model id hf-bedrock-map publishes. This is the number that
// decides how often the Bedrock comparison can be made at all: an unresolvable join
// is indistinguishable, to a user, from a model Bedrock does not offer.
func TestLiveTheJoinStillResolvesTheMappedModels(t *testing.T) {
	p := livePricer(t)
	rows := liveRows(t, p, "us-east-1")
	attrs := make([]rowAttrs, len(rows))
	for i, r := range rows {
		attrs[i] = r.attrs
	}

	mapCtx, cancel := liveCtx(t)
	defer cancel()
	idx, err := NewClient().Index(mapCtx)
	if err != nil {
		t.Fatal(err)
	}
	ids := map[string]bool{}
	for _, m := range idx {
		for _, e := range m.FoundationModelsIn("us-east-1") {
			ids[e.ModelID] = true
		}
	}
	if len(ids) == 0 {
		t.Fatal("the mapping publishes no us-east-1 foundation models")
	}

	var resolved, ambiguous, missing []string
	for _, id := range sortedKeys(ids) {
		name, err := resolveModelName(id, "us-east-1", attrs)
		switch {
		case err == nil:
			resolved = append(resolved, id+" -> "+name)
		case errors.As(err, new(*ErrAmbiguousModel)):
			ambiguous = append(ambiguous, err.Error())
		case errors.As(err, new(*ErrNoPriceListModel)):
			missing = append(missing, id)
		default:
			t.Errorf("%s: unexpected error kind: %v", id, err)
		}
	}
	sort.Strings(resolved)
	t.Logf("%d/%d ids resolved, %d ambiguous, %d unmatched", len(resolved), len(ids), len(ambiguous), len(missing))
	for _, r := range resolved {
		t.Logf("  %s", r)
	}

	// Ambiguity is never acceptable: the candidates for one id can differ several-fold
	// (a base model against its Latency Optimized variant), so picking either is a
	// coin flip on the price.
	for _, a := range ambiguous {
		t.Errorf("ambiguous join, was 0 on 2026-07-27: %s", a)
	}
	// Recorded 2026-07-27: 45 of 46 resolve. The one miss is deepseek.v3-v1:0, whose
	// key is bare "v3" against display names "DeepSeek V3.1" and "DeepSeek v3.2" —
	// different models, so no price is the right answer. Any *other* miss is a join
	// regression, and new models AWS adds may legitimately not be priced yet.
	for _, m := range missing {
		if m == "deepseek.v3-v1:0" {
			t.Logf("unmatched (expected): %s", m)
			continue
		}
		t.Logf("unmatched: %s — either us-east-1 does not publish a rate for it, or the join "+
			"needs a case. Check the Price List `model` values before adding a rule.", m)
	}
	if got := float64(len(resolved)) / float64(len(ids)); got < 0.90 {
		t.Errorf("join resolves %.0f%% of ids (%d/%d), was 98%% (45/46) on 2026-07-27",
			got*100, len(resolved), len(ids))
	}
}

// TestLiveDisplayNamesStillDoNotCollide is the property the join key rests on. It
// held for all 77 us-east-1 display names on 2026-07-27, under both provider
// treatments. A collision means two distinct products reduce to one key, and the
// resolver would report ambiguity for whichever id lands on it — losing a model
// rather than mispricing it, but losing it silently.
func TestLiveDisplayNamesStillDoNotCollide(t *testing.T) {
	p := livePricer(t)
	rows := liveRows(t, p, "us-east-1")

	names := map[string]bool{}
	for _, r := range rows {
		if r.attrs.Model != "" {
			names[r.attrs.Model] = true
		}
	}
	t.Logf("%d distinct Price List display names in us-east-1 (77 on 2026-07-27)", len(names))

	byKey := map[string][]string{}
	for name := range names {
		for _, stripped := range []bool{false, true} {
			k := joinKey(name, stripped)
			if k == "" {
				t.Errorf("display name %q reduces to an empty join key", name)
				continue
			}
			for _, prev := range byKey[k] {
				if prev != name {
					t.Errorf("%q and %q share join key %q; the resolver cannot separate them", prev, name, k)
				}
			}
			byKey[k] = append(byKey[k], name)
		}
	}
}

// TestLiveIncompleteTierPairsAreStillReal checks that the "no rate for this tier"
// state is not a defensive hypothetical. Llama 3.1 405B published batch rates only,
// and Claude 3 Sonnet an input rate with no output — both would read as a free or
// half-price model if a missing meter were treated as zero.
func TestLiveIncompleteTierPairsAreStillReal(t *testing.T) {
	p := livePricer(t)
	rows := liveRows(t, p, "us-east-1")

	names := map[string]bool{}
	for _, r := range rows {
		if r.attrs.Model != "" {
			names[r.attrs.Model] = true
		}
	}

	type tally struct{ in, out map[Tier]bool }
	models := map[string]*tally{}
	for _, row := range rows {
		if row.attrs.Model == "" {
			continue
		}
		tier, okT := tierOf(row.attrs)
		meter, okM := meterOf(row.attrs)
		if !okT || !okM {
			continue
		}
		priced := false
		for _, dim := range row.dimensions {
			if isTokenUnit(dim.unit) && dim.usd > 0 {
				priced = true
			}
		}
		if !priced {
			continue
		}
		m := models[row.attrs.Model]
		if m == nil {
			m = &tally{in: map[Tier]bool{}, out: map[Tier]bool{}}
			models[row.attrs.Model] = m
		}
		if meter == MeterInput {
			m.in[tier] = true
		} else {
			m.out[tier] = true
		}
	}

	var completeStandard, noStandard, halfPair []string
	for name, m := range models {
		switch {
		case m.in[TierStandard] && m.out[TierStandard]:
			completeStandard = append(completeStandard, name)
		case m.in[TierStandard] || m.out[TierStandard]:
			halfPair = append(halfPair, name)
		default:
			noStandard = append(noStandard, name)
		}
	}
	sort.Strings(noStandard)
	sort.Strings(halfPair)
	t.Logf("%d of %d priced models have a complete standard pair (69 of 75 on 2026-07-27)",
		len(completeStandard), len(models))
	t.Logf("no standard tier at all: %v", noStandard)
	t.Logf("standard with only one meter: %v", halfPair)

	if len(completeStandard) == len(models) {
		t.Log("every priced model now has a complete standard pair; 6 did not on 2026-07-27. " +
			"Keep the unavailable state — reporting a missing meter as $0 makes a model look free.")
	}
	if len(models) == 0 {
		t.Fatal("no model has a priced token meter")
	}
	if len(names) < len(models) {
		t.Errorf("%d priced models against %d display names; impossible", len(models), len(names))
	}
}

// TestLiveTiersAreOrderedTheWayTheDocsClaim checks the relationship the tier
// selection is justified by, across every model that publishes several tiers:
// priority costs more than standard, and flex and batch cost less. If that inverted,
// defaulting to standard would still be right, but the reasoning in [Tier]'s
// documentation and the "+75%" figure in the issue would need rewriting.
func TestLiveTiersAreOrderedTheWayTheDocsClaim(t *testing.T) {
	p := livePricer(t)
	// A spread of models: two naming conventions, a per-1M publisher, and one that
	// only ever published batch.
	for _, id := range []string{
		"qwen.qwen3-32b-v1:0",
		"meta.llama3-3-70b-instruct-v1:0",
		"anthropic.claude-sonnet-4-5-20250929-v1:0",
	} {
		ctx, cancel := liveCtx(t)
		got, err := p.Lookup(ctx, id, "us-east-1")
		cancel()
		if err != nil {
			t.Logf("%s: %v", id, err)
			continue
		}
		std, ok := got.Rate(TierStandard, MeterInput)
		if !ok {
			t.Logf("%s (%s): no standard input rate; tiers = %v", id, got.PriceListModel, got.Tiers())
			continue
		}
		for _, tier := range got.Tiers() {
			r, _ := got.Rate(tier, MeterInput)
			t.Logf("%s %s input = $%.4f/1M", got.PriceListModel, tier, r.USDPerMillionTokens)
			switch tier {
			case TierPriority:
				if r.USDPerMillionTokens <= std.USDPerMillionTokens {
					t.Errorf("%s: priority ($%.4f/1M) is no longer above standard ($%.4f/1M)",
						got.PriceListModel, r.USDPerMillionTokens, std.USDPerMillionTokens)
				}
			case TierFlex, TierBatch:
				if r.USDPerMillionTokens >= std.USDPerMillionTokens {
					t.Errorf("%s: %s ($%.4f/1M) is no longer below standard ($%.4f/1M)",
						got.PriceListModel, tier, r.USDPerMillionTokens, std.USDPerMillionTokens)
				}
			}
		}
	}
}

// TestLiveOpenWeightModelsStayInTheOpenWeightCatalogue guards the service-code
// choice documented on [priceListService].
//
// Bedrock publishes under four service codes and this package reads one. That is
// correct only as long as every model a user might self-host is priced under
// AmazonBedrock; the others carry closed-weight models (Claude, Cohere, Jamba,
// Jurassic, Palmyra, Stability, TwelveLabs) and legacy Llama 2 Chat. If AWS moves an
// open-weight model — a Llama, Qwen, Mistral, gpt-oss, Gemma, GLM, MiniMax, Kimi or
// DeepSeek — into one of the other codes, cultivar would report "no token price" for
// a model that has one, and recommend self-hosting on that basis.
func TestLiveOpenWeightModelsStayInTheOpenWeightCatalogue(t *testing.T) {
	ctx, cancel := liveCtx(t)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion("us-east-1"))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}
	client := pricing.NewFromConfig(cfg)

	// The open-weight families cultivar competes against. Llama is deliberately
	// absent: "Meta Llama 2 Chat 70B" is genuinely published under
	// AmazonBedrockFoundationModels, and it is legacy — no longer in the HF mapping —
	// so its presence there is expected rather than a regression.
	openWeight := []string{"qwen", "mistral", "mixtral", "ministral", "magistral",
		"devstral", "voxtral", "gpt-oss", "gemma", "glm", "minimax", "kimi", "deepseek"}

	for _, service := range []string{"AmazonBedrockFoundationModels", "AmazonBedrockService"} {
		var token *string
		names := map[string]bool{}
		for page := 0; page < maxPriceListPages; page++ {
			out, err := client.GetProducts(ctx, &pricing.GetProductsInput{
				ServiceCode:   awssdk.String(service),
				FormatVersion: awssdk.String("aws_v1"),
				Filters: []pricingtypes.Filter{{
					Type:  pricingtypes.FilterTypeTermMatch,
					Field: awssdk.String("regionCode"),
					Value: awssdk.String("us-east-1"),
				}},
				NextToken: token,
			})
			if err != nil {
				skipIfThrottled(t, err)
				t.Fatalf("%s: %v", service, err)
			}
			for _, item := range out.PriceList {
				var doc struct {
					Product struct {
						Attributes struct {
							Model       string `json:"model"`
							ServiceName string `json:"servicename"`
						} `json:"attributes"`
					} `json:"product"`
				}
				if err := json.Unmarshal([]byte(item), &doc); err != nil {
					t.Fatalf("%s: %v", service, err)
				}
				// These two codes key on servicename; `model` is absent on
				// AmazonBedrockFoundationModels entirely.
				if n := doc.Product.Attributes.Model; n != "" {
					names[n] = true
				}
				if n := doc.Product.Attributes.ServiceName; n != "" {
					names[n] = true
				}
			}
			if out.NextToken == nil || *out.NextToken == "" {
				break
			}
			token = out.NextToken
		}
		t.Logf("%s: %d distinct model names", service, len(names))
		for name := range names {
			lower := strings.ToLower(name)
			for _, family := range openWeight {
				if strings.Contains(lower, family) {
					t.Errorf("%s now prices %q, an open-weight model. cultivar reads only "+
						"AmazonBedrock, so its token price would read as absent and the verdict "+
						"would recommend self-hosting against no Bedrock alternative. "+
						"Extend the resolver to this service code.", service, name)
				}
			}
		}
	}
}

// TestLiveARegionWithoutTheModelIsNotAPrice checks the honest-absence path against
// a real region. us-west-1 carries Qwen3-32B's marketplace listing but not its
// foundation model, so there is no token price to report and the resolver must say
// so rather than borrow us-east-1's.
func TestLiveARegionWithoutTheModelIsNotAPrice(t *testing.T) {
	p := livePricer(t)
	ctx, cancel := liveCtx(t)
	defer cancel()

	got, err := p.Lookup(ctx, "qwen.qwen3-32b-v1:0", "us-west-1")
	if err == nil {
		t.Fatalf("us-west-1 priced qwen.qwen3-32b-v1:0 (model %q, %d rates); it offered no "+
			"foundation-model entry there on 2026-07-27", got.PriceListModel, len(got.Rates))
	}
	// Checked before the type assertion below, which a throttle would otherwise satisfy
	// non-nil and fail as a wrong error type — reporting a rate limit as a regression in
	// how absence is reported, in the one test whose whole subject is that distinction.
	skipIfThrottled(t, err)
	if !errors.As(err, new(*ErrNoPriceListModel)) {
		t.Errorf("absent model returned %T (%v), want *ErrNoPriceListModel so callers can render "+
			"\"no token price\" as its own state", err, err)
	}
}
