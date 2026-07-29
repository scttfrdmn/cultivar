//go:build live

// Opt-in suite that hits the real STS. Run with `make test-live` (AWS_PROFILE=aws).
// GetCallerIdentity is free, read-only, and unbillable.
//
// The offline tests pin the ARN parsing against constructed strings. What they cannot
// check is that a real principal's ARN parses at all — SSO sessions, assumed roles,
// and IAM users produce structurally different ARNs, and the partition is read from a
// position in that string. A parse that silently fails leaves the partition empty,
// which is safe but means every report is missing the field.
package identity

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go-v2/config"

	"github.com/scttfrdmn/cultivar/internal/report"
)

func TestLiveTheCallerIdentityResolves(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion("us-east-1"))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}

	id, err := NewResolver(cfg).Resolve(ctx)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	t.Logf("account %s, partition %s, principal %s", id.Account, id.Partition, id.ARN)

	// Recorded 2026-07-29 against AWS_PROFILE=aws. A mismatch means the profile
	// changed, not that the code did — the message says so rather than asserting a
	// bug.
	const recorded = "942542972736"
	if id.Account != recorded {
		t.Errorf("account %s, expected %s; a different profile is in use, so any recorded "+
			"quota or offered-type finding in this suite needs re-verifying", id.Account, recorded)
	}
	if len(id.Account) != 12 {
		t.Errorf("account %q is not 12 digits", id.Account)
	}

	// The partition being empty is the failure this suite exists to catch: it means a
	// real principal ARN did not parse, and every report from this build would omit
	// the field while looking complete.
	if id.Partition == "" {
		t.Fatalf("no partition resolved from %q; the ARN did not parse, so every report "+
			"silently omits the field", id.ARN)
	}
	if id.Partition != "aws" {
		t.Errorf("partition %q from a commercial profile", id.Partition)
	}
	if !strings.HasPrefix(id.ARN, "arn:"+id.Partition+":") {
		t.Errorf("partition %q is not the one in %q", id.Partition, id.ARN)
	}
}

// The identity must reach a report envelope, which is the only reason it is resolved.
func TestLiveTheIdentityReachesTheEnvelope(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion("us-east-1"))
	if err != nil {
		t.Skipf("no AWS config: %v", err)
	}

	id, err := NewResolver(cfg).Resolve(ctx)
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}

	e := id.Record(report.NewEnvelope("compare", "0.2.0-live-test", envelopeStamp()))
	if e.Account != id.Account || e.Partition != id.Partition {
		t.Errorf("recorded %q/%q, resolved %q/%q", e.Account, e.Partition, id.Account, id.Partition)
	}
	// The principal names a person or a role. The account id does not, and that
	// asymmetry is the reason the ARN has no envelope field.
	if strings.Contains(e.Account, "/") || strings.Contains(e.Partition, "/") {
		t.Errorf("a principal path reached a report field: %q/%q", e.Account, e.Partition)
	}
}
