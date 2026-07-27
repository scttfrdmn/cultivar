// Package ec2 resolves EC2 acquisition costs and availability.
//
// It wraps truffle, which is the suite's pricing authority, with two changes
// cultivar needs and truffle's defaults do not provide:
//
//   - the static price fallback is disabled, because for GPU instances it
//     fabricates rates rather than admitting it does not know one;
//   - a missing price becomes an [report.Amount] with
//     [report.ProvenanceUnavailable], so it can travel through a report as
//     "unpriced" rather than as an error the caller might paper over with zero.
//
// Pricing bugs belong upstream in truffle (see CLAUDE.md). Nothing here
// reimplements a price lookup; it selects which truffle behavior to use and
// attaches provenance.
package ec2

import (
	"context"
	"errors"
	"strings"
	"time"

	awssdk "github.com/aws/aws-sdk-go-v2/aws"
	truffle "github.com/spore-host/truffle/pkg/aws"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// Pricer resolves on-demand instance rates.
type Pricer struct {
	pricer truffle.OnDemandPricer
	now    func() time.Time
}

// NewPricer returns a Pricer backed solely by the AWS Price List API.
//
// The explicit [truffle.NewAWSOnDemandPricer] is the point of this constructor.
// truffle's client default is [newDefaultOnDemandPricer], a fallbackPricer that
// drops to the embedded spore-host/libs table whenever the Price List errors or
// returns a non-positive rate. That table has no g7/g7e/g6e/p5/p5e/p6 entries, so
// it reaches estimatePriceByFamily, which multiplies a $0.10 base by a size
// factor and returns the result with a nil error. Measured against live rates on
// 2026-07-27 that yields g7e.4xlarge at $0.80 against a real $4.00, and
// p6-b200.48xlarge at $9.60 against a real $113.93 — an 11.9x understatement,
// silently, in the direction that makes self-hosting look affordable.
//
// See spore-host/libs#29. Until it is fixed, cultivar opts out entirely: a rate
// this package cannot look up is reported unpriced.
func NewPricer(cfg awssdk.Config) *Pricer {
	return &Pricer{pricer: truffle.NewAWSOnDemandPricer(cfg), now: time.Now}
}

// NewPricerWith returns a Pricer over an explicit truffle pricer, for fixtures
// and the golden-file tests. Callers are responsible for not injecting a
// fabricating pricer.
func NewPricerWith(p truffle.OnDemandPricer, now func() time.Time) *Pricer {
	if now == nil {
		now = time.Now
	}
	return &Pricer{pricer: p, now: now}
}

// OnDemand returns the Linux/shared-tenancy on-demand rate for instanceType in
// region.
//
// A type with no on-demand offering is not an error: p5e.48xlarge (8xH200) has no
// Price List row of any kind, yet it is offered in us-east-2 and us-west-2 and is
// purchasable through capacity blocks at $47.76/instance-hour. So "no on-demand
// price" is a fact about the market that the report must carry, and the caller
// should go look at capacity blocks. It comes back as an unavailable Amount with
// that reason, and the returned error is nil.
//
// A genuine failure — no credentials, a throttle, a network error — returns a
// non-nil error, because that is a fact about this run and not about the price.
// Conflating the two is what lets an outage render as a free instance.
func (p *Pricer) OnDemand(ctx context.Context, instanceType, region string) (report.Amount, error) {
	rate, err := p.pricer.OnDemandPrice(ctx, instanceType, region)
	if err != nil {
		if isNoPriceFound(err) {
			return report.Unavailable(report.UnitUSDPerHour,
				"no on-demand price for "+instanceType+" in "+region+
					"; the type may be capacity-block only or not offered here"), nil
		}
		return report.Unavailable(report.UnitUSDPerHour,
			"on-demand price lookup failed for "+instanceType+" in "+region), err
	}
	// truffle's OnDemandPricer contract is (0, error) when unresolved, but the
	// fallbackPricer path can return a non-positive rate with a nil error. Treat
	// any non-positive rate as unpriced rather than as a real $0.00: AWS does emit
	// $0.0000 placeholder rows, notably every marketoption=CapacityBlock row, and
	// reporting those verbatim would advertise free H200s.
	if rate <= 0 {
		return report.Unavailable(report.UnitUSDPerHour,
			"on-demand rate for "+instanceType+" in "+region+" is not positive; treating as unpriced"), nil
	}
	return report.Live(rate, report.UnitUSDPerHour,
		"AWS Price List AmazonEC2 OnDemand/Linux/Shared/NA/Used "+instanceType+" "+region,
		p.now().UTC()), nil
}

// isNoPriceFound distinguishes "the API answered and there is no such price" from
// "the call failed". truffle returns a formatted error for the former, so this
// matches on its text; a sentinel error upstream would be better and is worth
// asking for if this proves brittle.
func isNoPriceFound(err error) bool {
	if err == nil {
		return false
	}
	// Never swallow a context error as an absent price.
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return false
	}
	return strings.Contains(err.Error(), "no on-demand price found")
}
