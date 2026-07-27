package ec2

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"
	"time"

	libpricing "github.com/spore-host/libs/pricing"

	"github.com/scttfrdmn/cultivar/internal/report"
)

var observed = time.Date(2026, 7, 27, 12, 0, 0, 0, time.UTC)

func fixedNow() time.Time { return observed }

// stubPricer stands in for truffle's OnDemandPricer.
type stubPricer struct {
	rates map[string]float64
	err   error
}

func (s stubPricer) OnDemandPrice(_ context.Context, instanceType, region string) (float64, error) {
	if s.err != nil {
		return 0, s.err
	}
	rate, ok := s.rates[instanceType]
	if !ok {
		// truffle's wording, which isNoPriceFound matches on.
		return 0, fmt.Errorf("no on-demand price found for %s in %s", instanceType, region)
	}
	return rate, nil
}

// Rates measured live against the Price List on 2026-07-27, us-east-1. These are
// the values the tool must report; the numbers in the table below them are what
// the static fallback would have said instead.
var liveRates = map[string]float64{
	"g7e.4xlarge":      4.00,
	"p5.4xlarge":       6.88,
	"g6e.12xlarge":     10.49,
	"p5.48xlarge":      55.04,
	"p6-b200.48xlarge": 113.93,
	// p5e.48xlarge is deliberately absent: it has no on-demand row at all.
}

func newTestPricer(rates map[string]float64) *Pricer {
	return NewPricerWith(stubPricer{rates: rates}, fixedNow)
}

func TestOnDemandReportsLiveRates(t *testing.T) {
	p := newTestPricer(liveRates)
	for typ, want := range liveRates {
		got, err := p.OnDemand(context.Background(), typ, "us-east-1")
		if err != nil {
			t.Fatalf("%s: %v", typ, err)
		}
		if v := got.MustValue(); v != want {
			t.Errorf("%s = %v, want %v", typ, v, want)
		}
		if got.Provenance() != report.ProvenanceLive {
			t.Errorf("%s provenance = %s, want live", typ, got.Provenance())
		}
		if err := got.Valid(); err != nil {
			t.Errorf("%s: %v", typ, err)
		}
		// The source has to be specific enough to re-run by hand, including the
		// four filters that otherwise return junk minimums.
		for _, want := range []string{"AmazonEC2", "Linux", "Shared", "Used", typ} {
			if !strings.Contains(got.Source(), want) {
				t.Errorf("%s source %q is missing %q", typ, got.Source(), want)
			}
		}
	}
}

// TestStaticFallbackWouldFabricateTheseRates is the reason NewPricer disables the
// fallback. It asserts against spore-host/libs directly, so if libs#29 is ever
// fixed this test fails and tells us the workaround can go.
func TestStaticFallbackWouldFabricateTheseRates(t *testing.T) {
	cases := []struct {
		typ         string
		static      float64
		real        float64 // 0 = no on-demand price exists
		description string
	}{
		{"g7e.4xlarge", 0.80, 4.00, "5.0x low"},
		{"p5.4xlarge", 0.80, 6.88, "8.6x low"},
		{"g6e.12xlarge", 2.40, 10.49, "4.4x low"},
		{"p6-b200.48xlarge", 9.60, 113.93, "11.9x low"},
		{"p5e.48xlarge", 9.60, 0, "fabricated: no on-demand price exists"},
	}
	for _, tc := range cases {
		t.Run(tc.typ, func(t *testing.T) {
			// Compared with a tolerance because these are not table lookups: libs
			// reaches estimatePriceByFamily and multiplies a $0.10 base by a size
			// factor, so g6e.12xlarge comes back as 2.4000000000000004.
			got := libpricing.GetEC2HourlyRate("us-east-1", tc.typ)
			if math.Abs(got-tc.static) > 0.001 {
				t.Fatalf("libs static rate for %s = %v, want ~%v — libs#29 may have changed; "+
					"re-verify against live Price List and update NewPricer's doc comment",
					tc.typ, got, tc.static)
			}
			if tc.real == 0 {
				t.Logf("%s: libs returns $%.2f/hr for a type with no on-demand price (%s)",
					tc.typ, got, tc.description)
				return
			}
			t.Logf("%s: libs $%.2f vs real $%.2f (%s)", tc.typ, got, tc.real, tc.description)
			if got >= tc.real {
				t.Errorf("expected the static rate to understate the real one; got %v vs %v", got, tc.real)
			}
		})
	}
}

func TestNoOnDemandPriceIsUnpricedNotAnError(t *testing.T) {
	// p5e.48xlarge: no Price List row of any kind, yet offered in us-east-2 and
	// us-west-2 and purchasable via capacity blocks at $47.76/instance-hour. That
	// is a fact about the market, not a failure of this run.
	p := newTestPricer(liveRates)
	got, err := p.OnDemand(context.Background(), "p5e.48xlarge", "us-west-2")
	if err != nil {
		t.Fatalf("absent price returned an error: %v", err)
	}
	if _, ok := got.Value(); ok {
		t.Fatal("absent price produced a value")
	}
	if got.Provenance() != report.ProvenanceUnavailable {
		t.Errorf("provenance = %s, want unavailable", got.Provenance())
	}
	if got.String() != "unpriced" {
		t.Errorf("String() = %q, want %q", got.String(), "unpriced")
	}
	// The reason must point the user at what would work.
	if !strings.Contains(got.Source(), "capacity-block") {
		t.Errorf("source %q does not mention the capacity-block path", got.Source())
	}
}

func TestLookupFailureIsAnErrorNotAZeroPrice(t *testing.T) {
	// An outage or a missing credential must not render as a free instance. This is
	// the opposite direction from the test above, and the distinction is the whole
	// point: one is about the market, the other about this run.
	sentinel := errors.New("AccessDeniedException: pricing:GetProducts")
	p := NewPricerWith(stubPricer{err: sentinel}, fixedNow)
	got, err := p.OnDemand(context.Background(), "g7e.4xlarge", "us-east-1")
	if err == nil {
		t.Fatal("a failed lookup returned no error")
	}
	if !errors.Is(err, sentinel) {
		t.Errorf("error %v does not wrap the underlying failure", err)
	}
	if _, ok := got.Value(); ok {
		t.Error("a failed lookup produced a value")
	}
}

func TestContextErrorIsNotTreatedAsAbsentPrice(t *testing.T) {
	// A cancelled or timed-out call says nothing about whether a price exists.
	for _, sentinel := range []error{context.Canceled, context.DeadlineExceeded} {
		p := NewPricerWith(stubPricer{err: sentinel}, fixedNow)
		_, err := p.OnDemand(context.Background(), "g7e.4xlarge", "us-east-1")
		if err == nil {
			t.Errorf("%v was swallowed as an absent price", sentinel)
		}
	}
}

func TestNonPositiveRateIsUnpriced(t *testing.T) {
	// AWS emits $0.0000 placeholder rows — every marketoption=CapacityBlock row is
	// one. Reporting those verbatim would advertise free H200s.
	p := newTestPricer(map[string]float64{"p5.48xlarge": 0})
	got, err := p.OnDemand(context.Background(), "p5.48xlarge", "us-east-1")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if v, ok := got.Value(); ok {
		t.Errorf("a $%v rate was reported as a real price", v)
	}
	if !strings.Contains(got.Source(), "not positive") {
		t.Errorf("source %q does not explain why the rate was rejected", got.Source())
	}
}

func TestRegionalSpreadIsCarriedThrough(t *testing.T) {
	// Measured 2026-07-27: g7e.4xlarge is $4.00 in US regions, $6.80 in
	// eu-central-1, $5.80 in ap-northeast-1, and not offered in eu-west-1 — a ~70%
	// spread that makes region a first-class lever. The pricer must not cache
	// across regions or normalize them away.
	regional := map[string]map[string]float64{
		"us-east-1":      {"g7e.4xlarge": 4.00},
		"us-east-2":      {"g7e.4xlarge": 4.00},
		"eu-central-1":   {"g7e.4xlarge": 6.80},
		"ap-northeast-1": {"g7e.4xlarge": 5.80},
		"eu-west-1":      {}, // not offered
	}
	for region, rates := range regional {
		p := newTestPricer(rates)
		got, err := p.OnDemand(context.Background(), "g7e.4xlarge", region)
		if err != nil {
			t.Fatalf("%s: %v", region, err)
		}
		want, offered := rates["g7e.4xlarge"]
		if !offered {
			if _, ok := got.Value(); ok {
				t.Errorf("%s: produced a price for a type not offered there", region)
			}
			continue
		}
		if v := got.MustValue(); v != want {
			t.Errorf("%s = %v, want %v", region, v, want)
		}
		if !strings.Contains(got.Source(), region) {
			t.Errorf("%s: source %q does not name the region", region, got.Source())
		}
	}
}

func TestLiveAmountCarriesObservationTime(t *testing.T) {
	// Prices are the stable half of the data, but the report still has to be able
	// to age them; a live Amount without a timestamp fails validation.
	p := newTestPricer(liveRates)
	got, err := p.OnDemand(context.Background(), "g7e.4xlarge", "us-east-1")
	if err != nil {
		t.Fatal(err)
	}
	age, ok := got.Age(observed.Add(2 * time.Hour))
	if !ok {
		t.Fatal("live rate has no observation time")
	}
	if age != 2*time.Hour {
		t.Errorf("age = %v, want 2h", age)
	}
}
