package bedrock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/service/pricing"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// fakeProducts serves recorded Price List pages. It records the filters it was
// called with, because the regionCode filter is the difference between 16 rows and
// 1013, and a defensive pagination loop only proves itself against a NextToken.
type fakeProducts struct {
	pages   [][]string
	calls   int
	filters []string
	err     error
}

func (f *fakeProducts) GetProducts(_ context.Context, in *pricing.GetProductsInput, _ ...func(*pricing.Options)) (*pricing.GetProductsOutput, error) {
	if f.err != nil {
		return nil, f.err
	}
	for _, filter := range in.Filters {
		f.filters = append(f.filters, *filter.Field+"="+*filter.Value)
	}
	page := f.calls
	f.calls++
	if page >= len(f.pages) {
		return &pricing.GetProductsOutput{}, nil
	}
	out := &pricing.GetProductsOutput{PriceList: f.pages[page]}
	if page+1 < len(f.pages) {
		token := fmt.Sprintf("page-%d", page+1)
		out.NextToken = &token
	}
	return out, nil
}

// product renders one Price List document from a row's attributes and a single
// price dimension, in the shape the API returns.
func product(attrs rowAttrs, usd float64, unit string) string {
	doc := map[string]any{
		"product": map[string]any{
			"sku":           "SKU" + attrs.UsageType,
			"productFamily": nil, // null on all 1013 us-east-1 rows
			"attributes": map[string]any{
				"model":         attrs.Model,
				"usagetype":     attrs.UsageType,
				"inferenceType": attrs.InferenceType,
				"feature":       attrs.Feature,
				"service_tier":  attrs.ServiceTier,
				"tokenType":     attrs.TokenType,
				"regionCode":    attrs.RegionCode,
				"servicecode":   "AmazonBedrock",
			},
		},
		"terms": map[string]any{
			"OnDemand": map[string]any{
				"OFFER": map[string]any{
					"priceDimensions": map[string]any{
						"RATE": map[string]any{
							"unit":         unit,
							"pricePerUnit": map[string]string{"USD": fmt.Sprintf("%.10f", usd)},
						},
					},
				},
			},
		},
	}
	b, err := json.Marshal(doc)
	if err != nil {
		panic(err)
	}
	return string(b)
}

// qwenRates are the live us-east-1 rates for Qwen3-32B, 2026-07-27, paired with the
// 16 rows in qwen3_32bRows. Every rate appears twice, once per naming convention.
var qwenRates = map[string]float64{
	"USE1-Qwen3-32B-input-tokens":                       0.00015,
	"USE1-Qwen3-32B-output-tokens":                      0.0006,
	"USE1-Qwen3-32B-input-tokens-flex":                  0.000075,
	"USE1-Qwen3-32B-output-tokens-flex":                 0.0003,
	"USE1-Qwen3-32B-input-tokens-priority":              0.0002625,
	"USE1-Qwen3-32B-output-tokens-priority":             0.00105,
	"USE1-Qwen3-32B-input-tokens-batch":                 0.000075,
	"USE1-Qwen3-32B-output-tokens-batch":                0.0003,
	"USE1-qwen.qwen3-32b-mantle-input-tokens-standard":  0.00015,
	"USE1-qwen.qwen3-32b-mantle-output-tokens-standard": 0.0006,
	"USE1-qwen.qwen3-32b-mantle-input-tokens-flex":      0.000075,
	"USE1-qwen.qwen3-32b-mantle-output-tokens-flex":     0.0003,
	"USE1-qwen.qwen3-32b-mantle-input-tokens-priority":  0.0002625,
	"USE1-qwen.qwen3-32b-mantle-output-tokens-priority": 0.00105,
	"USE1-qwen.qwen3-32b-mantle-input-tokens-batch":     0.000075,
	"USE1-qwen.qwen3-32b-mantle-output-tokens-batch":    0.0003,
}

func qwenPage() []string {
	out := make([]string, 0, len(qwen3_32bRows))
	for _, r := range qwen3_32bRows {
		rate, ok := qwenRates[r.UsageType]
		if !ok {
			panic("no recorded rate for " + r.UsageType)
		}
		out = append(out, product(r, rate, "1K tokens"))
	}
	return out
}

func qwenPricer(t *testing.T, extra ...string) (*TokenPricer, *fakeProducts) {
	t.Helper()
	api := &fakeProducts{pages: [][]string{append(qwenPage(), extra...)}}
	return NewTokenPricerWith(api, func() time.Time { return observed }), api
}

func TestLookupReturnsEveryTierAndBothMeters(t *testing.T) {
	p, _ := qwenPricer(t)
	got, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.PriceListModel != "Qwen3 32B" {
		t.Errorf("PriceListModel = %q, want Qwen3 32B", got.PriceListModel)
	}
	// The live rates, in $/1M. The 16 rows carry 8 distinct rates.
	want := map[[2]string]float64{
		{"standard", "input"}: 0.15, {"standard", "output"}: 0.60,
		{"priority", "input"}: 0.2625, {"priority", "output"}: 1.05,
		{"flex", "input"}: 0.075, {"flex", "output"}: 0.30,
		{"batch", "input"}: 0.075, {"batch", "output"}: 0.30,
	}
	if len(got.Rates) != len(want) {
		t.Errorf("got %d rates, want %d", len(got.Rates), len(want))
	}
	for k, v := range want {
		rate, ok := got.Rate(Tier(k[0]), Meter(k[1]))
		if !ok {
			t.Errorf("no %s %s rate", k[0], k[1])
			continue
		}
		if !sameRate(rate.USDPerMillionTokens, v) {
			t.Errorf("%s %s = $%.6f/1M, want $%.4f/1M", k[0], k[1], rate.USDPerMillionTokens, v)
		}
	}
}

func TestLookupDoesNotReturnThePriorityRateForAPlainRequest(t *testing.T) {
	// The trap this resolver exists for. Priority is $0.2625/1M input against
	// standard's $0.15 — 75% higher — and it is what taking the first row returns.
	p, _ := qwenPricer(t)
	got, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	in, ok := got.Rate(TierStandard, MeterInput)
	if !ok {
		t.Fatal("no standard input rate")
	}
	if !sameRate(in.USDPerMillionTokens, 0.15) {
		t.Errorf("standard input = $%.6f/1M, want $0.15. $0.2625 means the priority row won.",
			in.USDPerMillionTokens)
	}
	// And standard sorts first, so a caller that reads Rates[0] gets the right answer.
	if got.Rates[0].Tier != TierStandard || got.Rates[0].Meter != MeterInput {
		t.Errorf("Rates[0] = %s %s, want standard input so index 0 is not itself a trap",
			got.Rates[0].Tier, got.Rates[0].Meter)
	}
}

func TestLookupNormalizesThePerMillionUnit(t *testing.T) {
	// xai.grok-4.3 is published per "1M tokens" while everything else is per "1K".
	rows := []rowAttrs{
		{Model: "xai.grok-4.3", UsageType: "USE1-xai.grok-4.3-mantle-input-tokens-standard", TokenType: "input_tokens_mantle", ServiceTier: "standard", RegionCode: "us-east-1"},
		{Model: "xai.grok-4.3", UsageType: "USE1-xai.grok-4.3-mantle-output-tokens-standard", TokenType: "output_tokens_mantle", ServiceTier: "standard", RegionCode: "us-east-1"},
	}
	api := &fakeProducts{pages: [][]string{{
		product(rows[0], 1.25, "1M tokens"),
		product(rows[1], 2.50, "1M tokens"),
	}}}
	p := NewTokenPricerWith(api, func() time.Time { return observed })
	got, err := p.Lookup(context.Background(), "xai.grok-4.3-v1:0", "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	in, _ := got.Rate(TierStandard, MeterInput)
	out, _ := got.Rate(TierStandard, MeterOutput)
	if !sameRate(in.USDPerMillionTokens, 1.25) || !sameRate(out.USDPerMillionTokens, 2.50) {
		t.Errorf("grok standard = $%.4f in / $%.4f out per 1M, want $1.25 / $2.50. "+
			"$1250 means the 1K assumption was applied to a 1M row.",
			in.USDPerMillionTokens, out.USDPerMillionTokens)
	}
	if in.Unit != "1M tokens" {
		t.Errorf("Unit = %q, want the published unit recorded for provenance", in.Unit)
	}
}

func TestLookupUsesTheRegionCodeFilter(t *testing.T) {
	// Without it, us-east-1 is 1013 rows over 11 pages; with it, one model's rows
	// arrive in a single page.
	p, api := qwenPricer(t)
	if _, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-west-2"); err == nil {
		t.Fatal("a us-west-2 query against us-east-1 rows succeeded")
	}
	if got := strings.Join(api.filters, " "); !strings.Contains(got, "regionCode=us-west-2") {
		t.Errorf("filters = %q, want a regionCode TERM_MATCH", got)
	}
}

func TestLookupRejectsRowsFromAnotherRegion(t *testing.T) {
	// The regionCode filter is undocumented for AmazonBedrock. If AWS stops honouring
	// it, the query becomes the whole catalogue and a model silently gets whichever
	// region's rates are listed first — us-east-1 rates quoted for eu-central-1, where
	// they are higher. Verifying what came back is the only way that announces itself.
	rows := make([]rowAttrs, len(qwen3_32bRows))
	copy(rows, qwen3_32bRows)
	page := make([]string, 0, len(rows))
	for i, r := range rows {
		if i == 3 {
			r.RegionCode = "eu-central-1"
		}
		page = append(page, product(r, qwenRates[r.UsageType], "1K tokens"))
	}
	api := &fakeProducts{pages: [][]string{page}}
	p := NewTokenPricerWith(api, func() time.Time { return observed })
	_, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1")
	if err == nil {
		t.Fatal("a row from another region was accepted")
	}
	if !strings.Contains(err.Error(), "eu-central-1") {
		t.Errorf("error %q should name the region that came back", err)
	}
}

func TestLookupPaginates(t *testing.T) {
	// `aws pricing get-products --no-paginate` returns 100 rows *and* a NextToken, so
	// a caller that ignores the token gets a partial catalogue — which looks exactly
	// like a model that is not offered.
	rows := qwenPage()
	api := &fakeProducts{pages: [][]string{rows[:6], rows[6:12], rows[12:]}}
	p := NewTokenPricerWith(api, func() time.Time { return observed })
	got, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	if api.calls != 3 {
		t.Errorf("made %d calls, want 3 — the NextToken was not followed", api.calls)
	}
	if len(got.Rates) != 8 {
		t.Errorf("got %d rates from a paginated response, want 8", len(got.Rates))
	}
}

func TestLookupExcludesTheNoiseRowsThatShareAModelName(t *testing.T) {
	// The same display name carries cache, modality, and customization meters
	// alongside its text rates. Each would land on a (tier, meter) key that already
	// has a rate, and the duplicate check would turn the whole lookup into an error —
	// so these tests fail loudly if the exclusions regress, rather than subtly.
	noise := []string{
		product(rowAttrs{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-cache-read-input-token-count", InferenceType: "Cache read input tokens", Feature: "On-demand Inference", RegionCode: "us-east-1"}, 0.0000375, "1K tokens"),
		product(rowAttrs{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-input-image-token-count", InferenceType: "Input Image Token Count", Feature: "On-demand Inference", RegionCode: "us-east-1"}, 0.001, "1K tokens"),
		product(rowAttrs{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-input-tokens-custom-model", InferenceType: "Input tokens", Feature: "Model Customization", RegionCode: "us-east-1"}, 0.0004, "1K tokens"),
		product(rowAttrs{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-input-tokens-cross-region-global", InferenceType: "Input tokens", Feature: "On-demand Inference", RegionCode: "us-east-1"}, 0.000135, "1K tokens"),
		// Non-token dimensions on the same model.
		product(rowAttrs{Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-ProvisionedThroughput-NoCommit-ModelUnits", Feature: "Provisioned Throughput Inference - no commit", RegionCode: "us-east-1"}, 24.5, "hour"),
	}
	p, _ := qwenPricer(t, noise...)
	got, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1")
	if err != nil {
		t.Fatalf("noise rows broke the lookup: %v", err)
	}
	if len(got.Rates) != 8 {
		t.Errorf("got %d rates, want 8 — a non-text meter was counted", len(got.Rates))
	}
	in, _ := got.Rate(TierStandard, MeterInput)
	if !sameRate(in.USDPerMillionTokens, 0.15) {
		t.Errorf("standard input = $%.6f/1M, want $0.15; a noise row displaced it",
			in.USDPerMillionTokens)
	}
}

func TestLookupErrorsWhenOneMeterCarriesTwoRates(t *testing.T) {
	// The signal that a meter this code does not distinguish is being folded in. It
	// must not resolve by picking one: on Nova 2.0 Pro the two candidates are $1.25
	// and $1.375 per 1M, and silently taking either is a number nobody can check.
	conflicting := product(rowAttrs{
		Model: "Qwen3 32B", UsageType: "USE1-Qwen3-32B-input-tokens-something-new",
		InferenceType: "Input tokens", Feature: "On-demand Inference", RegionCode: "us-east-1",
	}, 0.000135, "1K tokens")
	p, _ := qwenPricer(t, conflicting)
	_, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1")
	if err == nil {
		t.Fatal("two different rates on one meter were accepted")
	}
	for _, want := range []string{"0.150000", "0.135000", "standard", "input"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q should contain %q so the conflict is diagnosable", err, want)
		}
	}
}

func TestLookupToleratesTheDuplicatePublication(t *testing.T) {
	// Every rate legitimately arrives twice, once per naming convention. Agreement is
	// checked, not assumed, but agreement must not be an error.
	p, _ := qwenPricer(t)
	if _, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1"); err != nil {
		t.Fatalf("the two naming conventions publishing the same rate was rejected: %v", err)
	}
}

func TestLookupSkipsZeroRates(t *testing.T) {
	// AWS publishes $0.0000 rows — 20 in us-east-1. A zero that reaches the rate table
	// would advertise a free model.
	rows := []rowAttrs{
		{Model: "Thing", UsageType: "USE1-vendor.thing-mantle-input-tokens-standard", InferenceType: "Input tokens", Feature: "On-demand Inference", RegionCode: "us-east-1"},
		{Model: "Thing", UsageType: "USE1-vendor.thing-mantle-output-tokens-standard", InferenceType: "Output tokens", Feature: "On-demand Inference", RegionCode: "us-east-1"},
	}
	api := &fakeProducts{pages: [][]string{{
		product(rows[0], 0, "1K tokens"),
		product(rows[1], 0.0006, "1K tokens"),
	}}}
	p := NewTokenPricerWith(api, func() time.Time { return observed })
	got, err := p.Lookup(context.Background(), "vendor.thing-v1:0", "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := got.Rate(TierStandard, MeterInput); ok {
		t.Error("a $0.0000 row produced a rate; the tier should be incomplete instead")
	}
	if got.HasTier(TierStandard) {
		t.Error("standard reported complete with only an output rate")
	}
	if a := got.Amount(TierStandard, MeterInput); a.Known() {
		t.Errorf("Amount = %v, want unavailable", a)
	}
}

func TestLookupRejectsAnUnknownUnitRatherThanScalingIt(t *testing.T) {
	rows := []rowAttrs{
		{Model: "Thing", UsageType: "USE1-vendor.thing-mantle-input-tokens-standard", InferenceType: "Input tokens", Feature: "On-demand Inference", RegionCode: "us-east-1"},
	}
	api := &fakeProducts{pages: [][]string{{product(rows[0], 0.15, "1B tokens")}}}
	p := NewTokenPricerWith(api, func() time.Time { return observed })
	_, err := p.Lookup(context.Background(), "vendor.thing-v1:0", "us-east-1")
	if err == nil {
		t.Fatal("an unrecognized token unit was scaled anyway")
	}
	if !strings.Contains(err.Error(), "1B tokens") {
		t.Errorf("error %q should name the unit", err)
	}
}

func TestTiersRequiresBothMetersAndSortsByInputCost(t *testing.T) {
	p, _ := qwenPricer(t)
	got, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	tiers := got.Tiers()
	if len(tiers) != 4 {
		t.Fatalf("Tiers = %v, want all four", tiers)
	}
	// flex and batch are both $0.075; standard $0.15; priority $0.2625.
	if tiers[len(tiers)-1] != TierPriority {
		t.Errorf("Tiers = %v, want priority last (it is the most expensive)", tiers)
	}
	if tiers[0] == TierStandard || tiers[0] == TierPriority {
		t.Errorf("Tiers = %v, want a $0.075 tier first", tiers)
	}
}

func TestBatchOnlyModelReportsNoStandardTier(t *testing.T) {
	// Llama 3.1 405B in us-east-1 publishes batch rates and nothing else. A comparison
	// that silently used them would price an interactive endpoint against asynchronous
	// bulk inference — half the real rate, for a service that cannot answer a request.
	rows := []rowAttrs{
		{Model: "Llama 3.1 405B", UsageType: "USE1-Llama3-1-405B-input-tokens-batch", InferenceType: "Input tokens", Feature: "Batch Inference", RegionCode: "us-east-1"},
		{Model: "Llama 3.1 405B", UsageType: "USE1-Llama3-1-405B-output-tokens-batch", InferenceType: "Output tokens", Feature: "Batch Inference", RegionCode: "us-east-1"},
	}
	api := &fakeProducts{pages: [][]string{{
		product(rows[0], 0.0012, "1K tokens"),
		product(rows[1], 0.0012, "1K tokens"),
	}}}
	p := NewTokenPricerWith(api, func() time.Time { return observed })
	got, err := p.Lookup(context.Background(), "meta.llama3-1-405b-instruct-v1:0", "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	if got.HasTier(TierStandard) {
		t.Error("a batch-only model reported a standard tier")
	}
	if !got.HasTier(TierBatch) {
		t.Error("the batch tier it does publish was lost")
	}
	a := got.Amount(TierStandard, MeterInput)
	if a.Known() {
		t.Errorf("standard input = %v, want unavailable", a)
	}
	if !strings.Contains(a.Source(), "standard") {
		t.Errorf("reason %q should name the tier that is missing", a.Source())
	}
}

func TestAmountCarriesLiveProvenanceAndTheUsageType(t *testing.T) {
	p, _ := qwenPricer(t)
	got, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	a := got.Amount(TierStandard, MeterInput)
	if err := a.Valid(); err != nil {
		t.Fatalf("invalid amount: %v", err)
	}
	if a.Provenance() != report.ProvenanceLive {
		t.Errorf("provenance = %q, want live", a.Provenance())
	}
	if a.Unit() != report.UnitUSDPerMillionTokens {
		t.Errorf("unit = %q, want %q", a.Unit(), report.UnitUSDPerMillionTokens)
	}
	if a.MustValue() != 0.15 {
		t.Errorf("value = %v, want 0.15", a.MustValue())
	}
	if !a.ObservedAt().Equal(observed) {
		t.Errorf("ObservedAt = %v, want the injected clock", a.ObservedAt())
	}
	// The join is heuristic, so the source has to name the product that was priced
	// and the meter it came from, or a wrong number is uncheckable.
	for _, want := range []string{"Qwen3 32B", "USE1-", "standard", "1K tokens"} {
		if !strings.Contains(a.Source(), want) {
			t.Errorf("source %q should contain %q", a.Source(), want)
		}
	}
}

func TestUnavailableAmountIsUnpricedNotZero(t *testing.T) {
	var p *TokenPrice
	a := p.Amount(TierStandard, MeterInput)
	if a.Known() {
		t.Error("a nil TokenPrice produced a known amount")
	}
	if a.String() != "unpriced" {
		t.Errorf("String = %q, want unpriced", a.String())
	}
	if _, ok := p.Rate(TierStandard, MeterInput); ok {
		t.Error("a nil TokenPrice returned a rate")
	}
	if got := p.Tiers(); got != nil {
		t.Errorf("Tiers on nil = %v", got)
	}
	if p.HasTier(TierStandard) {
		t.Error("HasTier on nil = true")
	}
}

func TestLookupPropagatesTheJoinFailure(t *testing.T) {
	p, _ := qwenPricer(t)
	_, err := p.Lookup(context.Background(), "vendor.nothing-like-this-v1:0", "us-east-1")
	var missing *ErrNoPriceListModel
	if !errors.As(err, &missing) {
		t.Fatalf("err = %v, want ErrNoPriceListModel so the caller can render \"no token price\"", err)
	}
}

func TestLookupRequiresBothArguments(t *testing.T) {
	p, _ := qwenPricer(t)
	for _, tc := range [][2]string{{"", "us-east-1"}, {"qwen.qwen3-32b-v1:0", ""}, {"", ""}} {
		if _, err := p.Lookup(context.Background(), tc[0], tc[1]); err == nil {
			t.Errorf("Lookup(%q, %q) succeeded", tc[0], tc[1])
		}
	}
}

func TestLookupSurfacesAPIErrors(t *testing.T) {
	// A throttle or a missing credential is a fact about this run, not about the
	// price. Swallowing it into "no token price" would make an outage look like a
	// model that Bedrock does not serve — and push the verdict toward self-hosting.
	api := &fakeProducts{err: errors.New("ThrottlingException: Rate exceeded")}
	p := NewTokenPricerWith(api, func() time.Time { return observed })
	_, err := p.Lookup(context.Background(), "qwen.qwen3-32b-v1:0", "us-east-1")
	if err == nil {
		t.Fatal("an API error was swallowed")
	}
	var missing *ErrNoPriceListModel
	if errors.As(err, &missing) {
		t.Error("an API error was reported as an absent price")
	}
	if !strings.Contains(err.Error(), "Throttling") {
		t.Errorf("error %q should carry the cause", err)
	}
}

func TestContextCancellationIsNotAnAbsentPrice(t *testing.T) {
	p, _ := qwenPricer(t)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	api := &fakeProducts{err: context.Canceled}
	p = NewTokenPricerWith(api, func() time.Time { return observed })
	_, err := p.Lookup(ctx, "qwen.qwen3-32b-v1:0", "us-east-1")
	if !errors.Is(err, context.Canceled) {
		t.Errorf("err = %v, want context.Canceled to survive wrapping", err)
	}
}

// endlessProducts always hands back a NextToken, which is what an API that has
// started looping — or a token this code fails to advance — looks like from here.
//
// It refuses past a bound of its own so that a missing guard fails the test in
// milliseconds instead of hanging the suite until the go test timeout. Without that,
// "the guard works" and "the fixture ran out of pages" are indistinguishable.
type endlessProducts struct{ calls, refuseAfter int }

func (e *endlessProducts) GetProducts(_ context.Context, _ *pricing.GetProductsInput, _ ...func(*pricing.Options)) (*pricing.GetProductsOutput, error) {
	e.calls++
	if e.calls > e.refuseAfter {
		return nil, fmt.Errorf("fixture: %d calls with no guard; pagination is unbounded", e.calls)
	}
	token := fmt.Sprintf("page-%d", e.calls)
	return &pricing.GetProductsOutput{
		PriceList: []string{product(rowAttrs{
			Model: "Thing", UsageType: fmt.Sprintf("USE1-vendor.thing-mantle-input-tokens-standard-%d", e.calls),
			InferenceType: "Input tokens", RegionCode: "us-east-1",
		}, 0.00015, "1K tokens")},
		NextToken: &token,
	}, nil
}

func TestPaginationHasARunawayGuard(t *testing.T) {
	// The fixture's own bound is a literal, deliberately not a multiple of
	// maxPriceListPages: deriving it from the constant under test would move it in
	// lockstep whenever that constant is wrong, which is precisely when this test has
	// to fail. It only needs to be comfortably above any real page count — us-east-1
	// is 11 pages unfiltered.
	api := &endlessProducts{refuseAfter: 500}
	p := NewTokenPricerWith(api, func() time.Time { return observed })

	_, err := p.Lookup(context.Background(), "vendor.thing-v1:0", "us-east-1")
	if err == nil {
		t.Fatal("pagination past the guard was accepted")
	}
	// The bound is the assertion, not merely the error: a guard that fired late, or an
	// error that came from somewhere else, would both satisfy "err != nil".
	if api.calls != maxPriceListPages {
		t.Errorf("GetProducts called %d times, want exactly %d; the loop is not bounded by the guard",
			api.calls, maxPriceListPages)
	}
	if !strings.Contains(err.Error(), "still paginating") {
		t.Errorf("error was %q, want the guard's own message; something else stopped the loop", err)
	}
}

func TestParsePriceRowRejectsAnUnparseableRate(t *testing.T) {
	// A malformed price must not become zero.
	bad := `{"product":{"sku":"X","attributes":{"usagetype":"USE1-x","regionCode":"us-east-1"}},
	  "terms":{"OnDemand":{"O":{"priceDimensions":{"R":{"unit":"1K tokens","pricePerUnit":{"USD":"not-a-number"}}}}}}}`
	if _, err := parsePriceRow(bad); err == nil {
		t.Error("an unparseable USD rate was accepted")
	}
	if _, err := parsePriceRow("<html>404</html>"); err == nil {
		t.Error("a non-JSON document was accepted")
	}
}
