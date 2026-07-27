//go:build live

// Opt-in suite that hits the real AWS Price List API. Run with `make test-live`
// (AWS_PROFILE=aws). Everything here is a free read-only call — no instance is
// launched and nothing bills.
//
// The point is schema drift, not arithmetic. The offline tests pin the rates this
// tool must report; these tests check that the API still answers in the shape the
// pricer expects, which is how a change in filter semantics or field naming would
// otherwise reach users as a silently wrong number.
package ec2

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"
)

func liveCtx(t *testing.T) (context.Context, func()) {
	t.Helper()
	return context.WithTimeout(context.Background(), 60*time.Second)
}

func livePricer(t *testing.T) *Pricer {
	t.Helper()
	ctx, cancel := liveCtx(t)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion("us-east-1"))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}
	return NewPricer(cfg)
}

// TestLiveRatesStillMatchTheRecordedOnes is the drift detector. A failure here
// does not necessarily mean a bug: AWS does change prices. It means the recorded
// numbers in the offline tests, CLAUDE.md, and the filed upstream issues need
// re-verifying, and the failure message says so.
func TestLiveRatesStillMatchTheRecordedOnes(t *testing.T) {
	p := livePricer(t)
	// Recorded 2026-07-27, us-east-1. The 5% tolerance absorbs a genuine price
	// change without absorbing a filter regression, which moves rates by
	// multiples rather than percent.
	recorded := map[string]float64{
		"g7e.4xlarge":      4.00,
		"p5.4xlarge":       6.88,
		"g6e.12xlarge":     10.49,
		"p5.48xlarge":      55.04,
		"p6-b200.48xlarge": 113.93,
	}
	for typ, want := range recorded {
		ctx, cancel := liveCtx(t)
		got, err := p.OnDemand(ctx, typ, "us-east-1")
		cancel()
		if err != nil {
			t.Errorf("%s: %v", typ, err)
			continue
		}
		v, ok := got.Value()
		if !ok {
			t.Errorf("%s came back unpriced; it had an on-demand rate on 2026-07-27", typ)
			continue
		}
		if drift := math.Abs(v-want) / want; drift > 0.05 {
			t.Errorf("%s = $%.4f/hr, recorded $%.2f (%.0f%% drift). "+
				"Re-verify and update the offline tests, CLAUDE.md, and spore-host/libs#29 if this is a real price change.",
				typ, v, want, drift*100)
		}
	}
}

// TestLiveCapacityBlockOnlyTypeStaysUnpriced is the trap that matters most here,
// because both failure modes are wrong: reporting p5e.48xlarge as an error hides
// a purchasable option, and reporting a number for it is a fabrication.
func TestLiveCapacityBlockOnlyTypeStaysUnpriced(t *testing.T) {
	p := livePricer(t)
	for _, region := range []string{"us-east-2", "us-west-2"} {
		ctx, cancel := liveCtx(t)
		got, err := p.OnDemand(ctx, "p5e.48xlarge", region)
		cancel()
		if err != nil {
			t.Fatalf("%s: absent price returned an error: %v", region, err)
		}
		if v, ok := got.Value(); ok {
			t.Errorf("%s: p5e.48xlarge priced at $%.4f/hr, but it has no on-demand offering. "+
				"Either AWS started selling it on demand (update the docs) or the fallback is back.",
				region, v)
		}
	}
}

// TestLiveUnknownTypeIsUnpricedNotEstimated guards the fallback specifically. A
// nonexistent instance type is the clearest possible probe: any number at all is
// fabricated.
func TestLiveUnknownTypeIsUnpricedNotEstimated(t *testing.T) {
	p := livePricer(t)
	ctx, cancel := liveCtx(t)
	defer cancel()
	got, err := p.OnDemand(ctx, "g99e.128xlarge", "us-east-1")
	if err != nil {
		t.Fatalf("unexpected error for a nonexistent type: %v", err)
	}
	if v, ok := got.Value(); ok {
		t.Errorf("nonexistent instance type priced at $%.4f/hr — the static fallback is active", v)
	}
}

// TestLiveRegionalSpread checks that region is still a real lever and that a type
// absent from a region reports as unpriced rather than borrowing another region's
// rate. us-west-1 offers none of the modern GPU families.
func TestLiveRegionalSpread(t *testing.T) {
	p := livePricer(t)
	seen := map[string]float64{}
	for _, region := range []string{"us-east-1", "us-east-2", "eu-central-1", "us-west-1"} {
		ctx, cancel := liveCtx(t)
		got, err := p.OnDemand(ctx, "g7e.4xlarge", region)
		cancel()
		if err != nil {
			t.Errorf("%s: %v", region, err)
			continue
		}
		if v, ok := got.Value(); ok {
			seen[region] = v
			t.Logf("g7e.4xlarge %s = $%.4f/hr", region, v)
		} else {
			t.Logf("g7e.4xlarge %s = unpriced (%s)", region, got.Source())
		}
	}
	if v, ok := seen["us-west-1"]; ok {
		t.Errorf("us-west-1 priced g7e.4xlarge at $%.4f/hr; it offered no modern GPU families on 2026-07-27", v)
	}
	if eu, ok := seen["eu-central-1"]; ok {
		if us, ok := seen["us-east-1"]; ok && eu <= us {
			t.Errorf("eu-central-1 ($%.4f) is no longer pricier than us-east-1 ($%.4f); "+
				"the regional spread claim in CLAUDE.md needs re-verifying", eu, us)
		}
	}
}
