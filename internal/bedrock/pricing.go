package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	awssdk "github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/pricing"
	pricingtypes "github.com/aws/aws-sdk-go-v2/service/pricing/types"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// priceListService is the Price List service code this package reads.
//
// Bedrock publishes under four service codes, and picking the wrong one returns
// nothing rather than something wrong — which is why the choice is recorded here:
//
//	AmazonBedrock                  the open-weight catalogue: Qwen, Llama 3/4, Mistral,
//	                               GLM, MiniMax, gpt-oss, Gemma. All 46 foundation-model
//	                               ids hf-bedrock-map publishes live here.
//	AmazonBedrockFoundationModels  Bedrock's own hosted models — Claude, Cohere, Jamba,
//	                               Jurassic, Palmyra, Stable Diffusion, TwelveLabs — plus
//	                               legacy Meta Llama 2 Chat. Keyed on `servicename`
//	                               ("Claude Sonnet 4.6 (Amazon Bedrock Edition)") with no
//	                               `model` attribute at all, and usagetypes in a third
//	                               convention ("USE1-MP:USE1_InputTokenCount-Units").
//	AmazonBedrockService           reserved input/output tokens-per-minute commitments
//	                               for Claude Sonnet 4 and 4.5. A capacity purchase, not
//	                               a per-token rate.
//	AmazonBedrockAgentCore         the agent runtime, not model inference.
//
// Only the first is read, because only the first prices models a user can also
// self-host: the others are closed-weight or legacy. That boundary is an observation
// about today's catalogue, not a guarantee, so [TestLiveOpenWeightModelsStayInTheOpenWeightCatalogue]
// re-measures it — if AWS moves an open-weight model to another code, its rates would
// otherwise vanish into a "no token price" verdict.
const priceListService = "AmazonBedrock"

// maxPriceListPages bounds pagination. A regionCode-filtered query for one model
// returns 16 rows in a single page, and the unfiltered us-east-1 catalogue is 1013
// rows over 11 pages, so this is a runaway guard rather than a real limit.
const maxPriceListPages = 40

// productsAPI is the slice of the Price List client this package uses, so tests can
// supply recorded pages without a network or credentials.
type productsAPI interface {
	GetProducts(ctx context.Context, in *pricing.GetProductsInput, opts ...func(*pricing.Options)) (*pricing.GetProductsOutput, error)
}

// TokenPricer resolves Bedrock per-token rates from the AWS Price List API.
//
// It is deliberately not part of truffle. truffle#111 asks truffle to own Bedrock
// pricing, and this is written to move there cleanly — no cultivar types in the
// resolution path, and the messy parts ([tierOf], [meterOf], [perMillionTokens],
// [resolveModelName]) are pure functions over a row struct. Until that lands,
// cultivar owns it, because the number is load-bearing here and nowhere else yet.
type TokenPricer struct {
	api productsAPI
	now func() time.Time
}

// NewTokenPricer returns a pricer over the real Price List API.
//
// The Price List API is served only from us-east-1 and ap-south-1 regardless of
// which region's prices are being asked about, so the config's region is overridden
// unless it is already one of those. This mirrors what truffle's EC2 pricer does.
func NewTokenPricer(cfg awssdk.Config) *TokenPricer {
	if cfg.Region != "us-east-1" && cfg.Region != "ap-south-1" {
		cfg.Region = "us-east-1"
	}
	return &TokenPricer{api: pricing.NewFromConfig(cfg), now: time.Now}
}

// NewTokenPricerWith returns a pricer over an explicit products API, for fixtures
// and tests.
func NewTokenPricerWith(api productsAPI, now func() time.Time) *TokenPricer {
	if now == nil {
		now = time.Now
	}
	return &TokenPricer{api: api, now: now}
}

// TokenPrice is one model's per-token rates in one region.
type TokenPrice struct {
	// ModelID is the Bedrock foundation-model id the rates were resolved for.
	ModelID string

	// PriceListModel is the Price List `model` attribute the join landed on, e.g.
	// "Qwen3 32B" for "qwen.qwen3-32b-v1:0". Recorded because the join is heuristic:
	// a reader checking a rate against the console needs to know which product was
	// priced, and it is the first thing to look at when a number looks wrong.
	PriceListModel string

	// Region is the AWS region the rates are for.
	Region string

	// Rates are every tier/meter rate found, ordered by tier then meter.
	Rates []Rate

	// ObservedAt is when the Price List was read.
	ObservedAt time.Time
}

// Rate returns the rate for one tier and meter.
func (p *TokenPrice) Rate(tier Tier, meter Meter) (Rate, bool) {
	if p == nil {
		return Rate{}, false
	}
	for _, r := range p.Rates {
		if r.Tier == tier && r.Meter == meter {
			return r, true
		}
	}
	return Rate{}, false
}

// Amount returns one rate as a provenanced [report.Amount].
//
// A tier or meter with no published rate comes back
// [report.ProvenanceUnavailable] rather than zero, and the reason names the tier,
// because "no standard rate" is a real state: Llama 3.1 405B in us-east-1 publishes
// batch rates only, and Claude 3 Sonnet publishes an input rate with no output rate.
// Reporting either as $0.00 would make the model look free.
func (p *TokenPrice) Amount(tier Tier, meter Meter) report.Amount {
	r, ok := p.Rate(tier, meter)
	if !ok {
		var id, region string
		if p != nil {
			id, region = p.ModelID, p.Region
		}
		return report.Unavailable(report.UnitUSDPerMillionTokens, fmt.Sprintf(
			"no %s-tier %s token rate published for %s in %s", tier, meter, id, region))
	}
	return report.Live(r.USDPerMillionTokens, report.UnitUSDPerMillionTokens,
		fmt.Sprintf("AWS Price List %s %s %s (%s per %s)",
			priceListService, p.PriceListModel, r.UsageType, r.Tier, r.Unit),
		p.ObservedAt)
}

// Tiers returns the tiers that have both an input and an output rate, in ascending
// order of input cost.
//
// Both meters are required because a tier with only one is not usable for a
// comparison: any blend of input and output needs both, and substituting the other
// meter's rate for a missing one is a fabrication.
func (p *TokenPrice) Tiers() []Tier {
	if p == nil {
		return nil
	}
	type entry struct {
		tier Tier
		in   float64
	}
	var complete []entry
	for _, t := range []Tier{TierStandard, TierPriority, TierFlex, TierBatch} {
		in, okIn := p.Rate(t, MeterInput)
		if _, okOut := p.Rate(t, MeterOutput); okIn && okOut {
			complete = append(complete, entry{t, in.USDPerMillionTokens})
		}
	}
	sort.SliceStable(complete, func(i, j int) bool { return complete[i].in < complete[j].in })
	out := make([]Tier, 0, len(complete))
	for _, e := range complete {
		out = append(out, e.tier)
	}
	return out
}

// HasTier reports whether tier has a complete input+output pair.
func (p *TokenPrice) HasTier(tier Tier) bool {
	for _, t := range p.Tiers() {
		if t == tier {
			return true
		}
	}
	return false
}

// Lookup resolves the per-token rates for one Bedrock foundation-model id in one
// region.
//
// Rates for every tier and both meters come back together. Callers pick a tier —
// [TierStandard] unless the user asked otherwise — and blend input and output with
// an explicit ratio, which this package deliberately does not do: a single $/1M
// figure hides an assumption that moves the break-even point by 40%.
//
// An unresolvable join or a region that does not publish the model returns an error
// ([ErrNoPriceListModel] or [ErrAmbiguousModel]) rather than a zero price, so the
// caller can render "no token price" as its own state.
func (p *TokenPricer) Lookup(ctx context.Context, modelID, region string) (*TokenPrice, error) {
	modelID = strings.TrimSpace(modelID)
	region = strings.TrimSpace(region)
	if modelID == "" || region == "" {
		return nil, fmt.Errorf("bedrock token price: model id and region are both required")
	}

	rows, err := p.fetch(ctx, region)
	if err != nil {
		return nil, err
	}

	attrs := make([]rowAttrs, len(rows))
	for i, r := range rows {
		attrs[i] = r.attrs
	}
	name, err := resolveModelName(modelID, region, attrs)
	if err != nil {
		return nil, err
	}

	out := &TokenPrice{
		ModelID:        modelID,
		PriceListModel: name,
		Region:         region,
		ObservedAt:     p.now().UTC(),
	}

	// Each rate is published once per naming convention — "USE1-Qwen3-32B-input-tokens"
	// and "USE1-qwen.qwen3-32b-mantle-input-tokens-standard" carry the same $0.15 —
	// so a (tier, meter) key legitimately arrives twice. Keeping the first and
	// verifying the rest agree turns a genuine conflict into an error instead of a
	// coin flip. Excluding cross-region meters (see [crossRegionMeters]) is what makes
	// agreement hold: with them included, 21 keys in us-east-1 carry two rates.
	seen := map[[2]string]Rate{}
	for _, row := range rows {
		if row.attrs.Model != name {
			continue
		}
		tier, ok := tierOf(row.attrs)
		if !ok {
			continue
		}
		meter, ok := meterOf(row.attrs)
		if !ok {
			continue
		}
		for _, dim := range row.dimensions {
			if !isTokenUnit(dim.unit) {
				continue
			}
			// AWS publishes $0.0000 rows — 20 of them in us-east-1, all cache-write
			// meters that [meterOf] already drops. A zero that reaches here is a
			// placeholder for a rate that exists but was not populated, and reporting
			// it verbatim would advertise a free model. Skipping it leaves the tier
			// unavailable, which is the honest state.
			if dim.usd <= 0 {
				continue
			}
			perM, err := perMillionTokens(dim.usd, dim.unit)
			if err != nil {
				return nil, fmt.Errorf("bedrock token price %s in %s (%s): %w",
					modelID, region, row.attrs.UsageType, err)
			}
			key := [2]string{string(tier), string(meter)}
			rate := Rate{
				Tier: tier, Meter: meter, USDPerMillionTokens: perM,
				UsageType: row.attrs.UsageType, Unit: dim.unit,
			}
			prev, dup := seen[key]
			if !dup {
				seen[key] = rate
				out.Rates = append(out.Rates, rate)
				continue
			}
			if !sameRate(prev.USDPerMillionTokens, perM) {
				return nil, fmt.Errorf(
					"bedrock token price %s in %s: %s %s published twice with different rates "+
						"($%.6f/1M via %s, $%.6f/1M via %s); a meter this code does not "+
						"distinguish is being folded in",
					modelID, region, tier, meter,
					prev.USDPerMillionTokens, prev.UsageType, perM, row.attrs.UsageType)
			}
		}
	}

	if len(out.Rates) == 0 {
		return nil, fmt.Errorf("bedrock token price %s in %s: Price List model %q has no "+
			"text token rates", modelID, region, name)
	}
	sortRates(out.Rates)
	return out, nil
}

// sameRate compares two normalized rates with a tolerance that absorbs float
// round-tripping through the Price List's 10-decimal strings without absorbing a
// real difference; the smallest published rate is $0.04/1M.
func sameRate(a, b float64) bool {
	const epsilon = 1e-9
	diff := a - b
	if diff < 0 {
		diff = -diff
	}
	return diff <= epsilon
}

// tierOrder is the display order: standard first because it is the default and the
// one a plain comparison quotes, then the alternatives.
var tierOrder = map[Tier]int{TierStandard: 0, TierPriority: 1, TierFlex: 2, TierBatch: 3}

func sortRates(rates []Rate) {
	sort.SliceStable(rates, func(i, j int) bool {
		if ti, tj := tierOrder[rates[i].Tier], tierOrder[rates[j].Tier]; ti != tj {
			return ti < tj
		}
		return rates[i].Meter == MeterInput && rates[j].Meter != MeterInput
	})
}

// priceRow is one Price List product with its price dimensions extracted.
type priceRow struct {
	attrs      rowAttrs
	dimensions []priceDimension
}

type priceDimension struct {
	usd  float64
	unit string
}

// fetch retrieves every AmazonBedrock product in region.
//
// The regionCode filter is what makes this affordable. Without it, us-east-1 returns
// 1013 rows over 11 pages and every model in the catalogue has to be sifted; with
// it, a query is a fraction of that and one model's rows arrive in a single page.
// The filter is undocumented for AmazonBedrock but works, and the loop still
// paginates: --no-paginate on the CLI silently returns 100 rows with a NextToken,
// which is exactly how a partial catalogue looks like a missing model.
func (p *TokenPricer) fetch(ctx context.Context, region string) ([]priceRow, error) {
	filters := []pricingtypes.Filter{
		{
			Type:  pricingtypes.FilterTypeTermMatch,
			Field: awssdk.String("regionCode"),
			Value: awssdk.String(region),
		},
	}

	var (
		out   []priceRow
		token *string
	)
	for page := 0; ; page++ {
		if page >= maxPriceListPages {
			return nil, fmt.Errorf("price list %s in %s: still paginating after %d pages",
				priceListService, region, maxPriceListPages)
		}
		resp, err := p.api.GetProducts(ctx, &pricing.GetProductsInput{
			ServiceCode:   awssdk.String(priceListService),
			FormatVersion: awssdk.String("aws_v1"),
			Filters:       filters,
			NextToken:     token,
		})
		if err != nil {
			return nil, fmt.Errorf("price list %s in %s: %w", priceListService, region, err)
		}
		for _, item := range resp.PriceList {
			row, err := parsePriceRow(item)
			if err != nil {
				return nil, fmt.Errorf("price list %s in %s: %w", priceListService, region, err)
			}
			// The regionCode filter is undocumented for AmazonBedrock. If AWS ever
			// stops honouring it, the query silently becomes the whole catalogue and
			// every model gets whichever region's rates happen to be listed first —
			// a wrong price that looks completely normal. Verifying what came back is
			// the only way that failure announces itself.
			if !strings.EqualFold(row.attrs.RegionCode, region) {
				return nil, fmt.Errorf(
					"price list %s: asked for %s but a row reports region %q (usagetype %q); "+
						"the regionCode filter is no longer being honoured",
					priceListService, region, row.attrs.RegionCode, row.attrs.UsageType)
			}
			out = append(out, row)
		}
		if resp.NextToken == nil || *resp.NextToken == "" {
			return out, nil
		}
		token = resp.NextToken
	}
}

// productJSON is the Price List product document shape.
type productJSON struct {
	Product struct {
		Attributes rowAttrs `json:"attributes"`
		SKU        string   `json:"sku"`
	} `json:"product"`
	Terms struct {
		OnDemand map[string]struct {
			PriceDimensions map[string]struct {
				Unit         string            `json:"unit"`
				PricePerUnit map[string]string `json:"pricePerUnit"`
			} `json:"priceDimensions"`
		} `json:"OnDemand"`
	} `json:"terms"`
}

// rowAttrs is the subset of a product's attributes this package reads.
//
// productFamily is deliberately absent: it is null on all 1013 us-east-1 rows, so
// it cannot be used to separate token meters from provisioned throughput. The
// `provider` attribute is absent for the reason given on [joinProviders].
type rowAttrs struct {
	// Model is the display name the Price List keys rates on, e.g. "Qwen3 32B".
	// Populated on 875 of 1013 us-east-1 rows.
	Model string `json:"model"`
	// UsageType is the billing meter name, and the most reliable field here: it is
	// always present and carries the tier as a suffix even when nothing else does.
	UsageType string `json:"usagetype"`
	// InferenceType has 38 distinct values across the catalogue and inconsistent
	// casing. Present on 869 rows.
	InferenceType string `json:"inferenceType"`
	// Feature distinguishes "On-demand Inference" from "Batch Inference" and from
	// non-inference products. Present on 597 rows.
	Feature string `json:"feature"`
	// ServiceTier is the explicit tier, but only on 332 rows — never on the rows
	// that use the display-name convention.
	ServiceTier string `json:"service_tier"`
	// TokenType is grok-4.3's only meter signal, e.g. "input_tokens_mantle".
	TokenType string `json:"tokenType"`
	// RegionCode is echoed back and checked, so a filter that stops working shows up
	// as an error rather than as another region's prices.
	RegionCode string `json:"regionCode"`
}

func parsePriceRow(item string) (priceRow, error) {
	var doc productJSON
	if err := json.Unmarshal([]byte(item), &doc); err != nil {
		return priceRow{}, fmt.Errorf("decode product: %w", err)
	}
	row := priceRow{attrs: doc.Product.Attributes}
	for _, offer := range doc.Terms.OnDemand {
		for _, dim := range offer.PriceDimensions {
			usd, ok := dim.PricePerUnit["USD"]
			if !ok {
				continue
			}
			v, err := strconv.ParseFloat(usd, 64)
			if err != nil {
				return priceRow{}, fmt.Errorf("product %s: parse USD %q: %w",
					doc.Product.SKU, usd, err)
			}
			row.dimensions = append(row.dimensions, priceDimension{usd: v, unit: dim.Unit})
		}
	}
	return row, nil
}
