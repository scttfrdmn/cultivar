package identity

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	awssdk "github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/service/sts"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// envelopeStamp is a fixed generation time, so nothing here depends on the clock.
func envelopeStamp() time.Time { return time.Date(2026, 7, 29, 12, 0, 0, 0, time.UTC) }

// fakeSTS answers with a fixed identity or a fixed error.
//
// blankAccount is the distinction worth having: a nil Account pointer and a pointer to
// an empty string are different wire shapes, and the second is the one a
// present-but-empty field would produce. Without both, half the account check is
// unreachable from the tests.
type fakeSTS struct {
	account      string
	arn          string
	nilArn       bool
	nilOut       bool
	blankAccount bool
	err          error
	calls        int
}

func (f *fakeSTS) GetCallerIdentity(context.Context, *sts.GetCallerIdentityInput, ...func(*sts.Options)) (*sts.GetCallerIdentityOutput, error) {
	f.calls++
	if f.err != nil {
		return nil, f.err
	}
	if f.nilOut {
		return nil, nil
	}
	out := &sts.GetCallerIdentityOutput{}
	switch {
	case f.blankAccount:
		out.Account = awssdk.String("")
	case f.account != "":
		out.Account = awssdk.String(f.account)
	}
	if !f.nilArn {
		out.Arn = awssdk.String(f.arn)
	}
	return out, nil
}

func resolve(t *testing.T, f *fakeSTS) (Identity, error) {
	t.Helper()
	return NewResolverWith(f).Resolve(context.Background())
}

// The partition comes off the ARN, and the whole reason this package exists is that
// the shortcut — inferring it from a region name — gets the non-commercial cases
// wrong while looking right.
func TestThePartitionIsReadOffTheARN(t *testing.T) {
	for _, tc := range []struct {
		name, arn, want string
	}{
		{"commercial user", "arn:aws:iam::942542972736:user/scttfrdmn", "aws"},
		{"assumed role", "arn:aws:sts::942542972736:assumed-role/AdminRole/session", "aws"},
		{"govcloud", "arn:aws-us-gov:iam::942542972736:user/gov", "aws-us-gov"},
		{"china", "arn:aws-cn:iam::942542972736:user/cn", "aws-cn"},
		{"isolated", "arn:aws-iso-b:iam::942542972736:user/iso", "aws-iso-b"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := resolve(t, &fakeSTS{account: "942542972736", arn: tc.arn})
			if err != nil {
				t.Fatalf("Resolve: %v", err)
			}
			if got.Partition != tc.want {
				t.Errorf("partition = %q, want %q", got.Partition, tc.want)
			}
			if got.Account != "942542972736" {
				t.Errorf("account = %q", got.Account)
			}
			if got.ARN != tc.arn {
				t.Errorf("ARN = %q, want %q", got.ARN, tc.arn)
			}
		})
	}
}

// An unparseable ARN leaves the partition empty rather than assuming "aws". A default
// there would relabel a GovCloud run as commercial and every rate in the report would
// be one that does not apply — with nothing in the output looking wrong.
func TestAnUnreadableARNLeavesThePartitionEmpty(t *testing.T) {
	for _, bad := range []string{"", "not-an-arn", "arn:aws:iam", "arn:aws"} {
		got, err := resolve(t, &fakeSTS{account: "942542972736", arn: bad})
		if err != nil {
			t.Fatalf("ARN %q: an unreadable ARN must not fail the resolve; the account is "+
				"still usable: %v", bad, err)
		}
		if got.Partition != "" {
			t.Errorf("ARN %q yielded partition %q; a guessed partition is a claim the "+
				"report cannot support", bad, got.Partition)
		}
		if got.Account != "942542972736" {
			t.Errorf("ARN %q lost the account: %q", bad, got.Account)
		}
	}
}

// A missing ARN field is the same case as an unparseable one, and must not panic on
// the nil pointer.
func TestAMissingARNIsNotAPanic(t *testing.T) {
	got, err := resolve(t, &fakeSTS{account: "942542972736", nilArn: true})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got.Account != "942542972736" || got.Partition != "" || got.ARN != "" {
		t.Errorf("got %+v", got)
	}
}

// The account is the field a quota finding is interpretable through, so an absent one
// is an error rather than a blank: recording no account is the caller's decision to
// make, not something this resolver should fabricate agreement about.
func TestAnAccountlessIdentityIsRefused(t *testing.T) {
	for _, tc := range []struct {
		name string
		f    *fakeSTS
	}{
		{"no account", &fakeSTS{arn: "arn:aws:iam::942542972736:user/x"}},
		{"nil output", &fakeSTS{nilOut: true}},
		// A present-but-empty Account is a different wire shape from an absent one, and
		// the one that would slip a blank account id into a report if only the nil
		// pointer were checked. The ARN here carries no account of its own, so the
		// refusal has to come from the account check rather than from the contradiction
		// check downstream of it.
		{"empty account string", &fakeSTS{blankAccount: true, arn: "arn:aws:s3:::some-bucket"}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := resolve(t, tc.f); err == nil {
				t.Error("Resolve accepted an identity with no account id")
			} else if !strings.Contains(err.Error(), "no account id") {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

// When STS's account and its own ARN disagree, neither is trustworthy. Picking one
// would put an account id in a report that the report's own evidence contradicts.
func TestContradictoryIdentityIsRefused(t *testing.T) {
	_, err := resolve(t, &fakeSTS{account: "942542972736",
		arn: "arn:aws:iam::111122223333:user/somebody-else"})
	if err == nil {
		t.Fatal("Resolve accepted an account id its own ARN contradicts")
	}
	for _, want := range []string{"942542972736", "111122223333"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error does not name %s, so the contradiction is not diagnosable: %v", want, err)
		}
	}
}

// An ARN with no account section — the s3 style, arn:aws:s3:::bucket — is not a
// contradiction. The partition is still readable and must still be recorded.
func TestAnAccountlessARNStillYieldsAPartition(t *testing.T) {
	got, err := resolve(t, &fakeSTS{account: "942542972736", arn: "arn:aws-us-gov:s3:::some-bucket"})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got.Partition != "aws-us-gov" {
		t.Errorf("partition = %q, want aws-us-gov", got.Partition)
	}
}

// A failed call returns an error and no partial identity. The caller records nothing;
// what it must never do is record a zero-valued account, which reads as a fact.
func TestAFailedCallYieldsNothing(t *testing.T) {
	boom := errors.New("ExpiredToken: the security token included in the request is expired")
	got, err := resolve(t, &fakeSTS{err: boom})
	if err == nil {
		t.Fatal("Resolve hid a failed GetCallerIdentity")
	}
	if !errors.Is(err, boom) {
		t.Errorf("the cause is not unwrappable: %v", err)
	}
	if got != (Identity{}) {
		t.Errorf("a failed resolve returned %+v; a partial identity in a report is a claim", got)
	}
}

// Record is the one place that decides what a report says about whose view it is.
func TestRecordFillsBothEnvelopeFields(t *testing.T) {
	e := report.NewEnvelope("compare", "0.2.0", envelopeStamp())
	got := Identity{Account: "942542972736", Partition: "aws-us-gov",
		ARN: "arn:aws-us-gov:iam::942542972736:user/gov"}.Record(e)

	if got.Account != "942542972736" {
		t.Errorf("account = %q", got.Account)
	}
	if got.Partition != "aws-us-gov" {
		t.Errorf("partition = %q", got.Partition)
	}
	// The ARN names a user or a role; the account id does not. It stays out of the
	// document, and there is no envelope field for it to leak into.
	if strings.Contains(got.Account+got.Partition, "user/") {
		t.Error("the principal ARN reached a report field")
	}
	// Value semantics: Record must not mutate the envelope it was handed, or a
	// caller that resolves an identity for one report has silently changed another.
	if e.Account != "" || e.Partition != "" {
		t.Errorf("Record mutated its argument: %+v", e)
	}
}

// An unresolved identity records as absent, which is what the omitempty tags on both
// envelope fields are for. Recording empty strings explicitly would be the same
// mistake as a $0.00 price: a gap wearing a value's clothes.
func TestAnUnresolvedIdentityRecordsNothing(t *testing.T) {
	e := Identity{}.Record(report.NewEnvelope("compare", "0.2.0", envelopeStamp()))
	if e.Account != "" || e.Partition != "" {
		t.Errorf("a zero Identity recorded %q/%q", e.Account, e.Partition)
	}
}

// The resolver makes exactly one call. GetCallerIdentity is cheap but it is on the
// path of every report, and a resolver that called per-region would multiply it by
// the region set for two fields that do not vary by region.
func TestResolveCallsSTSOnce(t *testing.T) {
	f := &fakeSTS{account: "942542972736", arn: "arn:aws:iam::942542972736:user/x"}
	if _, err := resolve(t, f); err != nil {
		t.Fatal(err)
	}
	if f.calls != 1 {
		t.Errorf("%d STS calls for one Resolve", f.calls)
	}
}
