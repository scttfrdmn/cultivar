// Package identity resolves whose AWS view a report describes.
//
// Two facts, both optional in a report and neither of them a number: the account id
// and the partition. They live in a package rather than inline at the call site
// because getting the partition right means reading it off the caller's ARN, and the
// tempting shortcut — inferring it from a region prefix — is wrong in the expensive
// direction. aws-us-gov and aws-cn have their own rates and their own service sets,
// so a GovCloud run mislabelled as commercial produces a report full of prices that
// do not apply to it, and nothing in the report looks wrong.
//
// Why an account id belongs in a report at all: prices are not account-specific, but
// quotas and offered-instance-type sets are. "g7e is not offered here" and "you
// cannot launch g7e here" are different findings, and only the second needs an
// account to be interpretable — or reproducible by anyone else.
//
// sts:GetCallerIdentity is free, read-only, and cannot be denied by IAM policy, so
// this is one of the few AWS calls that either works or means the credentials
// themselves are unusable.
package identity

import (
	"context"
	"errors"
	"fmt"

	awssdk "github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/aws/arn"
	"github.com/aws/aws-sdk-go-v2/service/sts"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// CallerIdentityAPI is the single STS call this package makes. An interface so the
// ARN parsing and the consistency checks are testable without credentials, which is
// most of the logic here.
type CallerIdentityAPI interface {
	GetCallerIdentity(context.Context, *sts.GetCallerIdentityInput, ...func(*sts.Options)) (*sts.GetCallerIdentityOutput, error)
}

// Identity is who the AWS calls in a run are being made as.
type Identity struct {
	// Account is the 12-digit account id.
	Account string

	// Partition is the ARN partition: "aws", "aws-us-gov", "aws-cn", or one of the
	// aws-iso family.
	//
	// Empty when the ARN could not be read, and never defaulted to "aws" — an
	// absent partition is a gap a reader can see, while a wrong one is a claim.
	Partition string

	// ARN is the calling principal, kept for `--explain` output and because it is
	// where Partition came from, so the derivation stays checkable.
	//
	// Deliberately not carried into a report: it names a user or a role, which the
	// account id on its own does not.
	ARN string
}

// Resolver resolves the caller identity.
type Resolver struct {
	api CallerIdentityAPI
}

// NewResolver returns a Resolver over the given config.
//
// The config's region must be set: aws-sdk-go-v2 requires one even for STS. A
// missing region surfaces as an error from [Resolver.Resolve], which callers treat
// as an unrecorded identity rather than as a failed run — see [Resolver.Resolve].
func NewResolver(cfg awssdk.Config) *Resolver {
	return &Resolver{api: sts.NewFromConfig(cfg)}
}

// NewResolverWith returns a Resolver over an explicit API, for tests and fixtures.
func NewResolverWith(api CallerIdentityAPI) *Resolver {
	return &Resolver{api: api}
}

// Resolve returns the identity behind the current credentials.
//
// Strict on purpose, while the fields it feeds are optional. Anything surprising —
// no account id, an ARN whose account contradicts the one STS reported — is an
// error rather than a partially-filled Identity, because the account is only worth
// recording if it is the right one: a quota finding attributed to the wrong account
// is not a weaker finding, it is a false one.
//
// The tolerance belongs at the call site, not here. A report's account and partition
// are optional, so a caller that cannot resolve an identity records neither field
// and carries on; it must not fail a run over an optional field, and it must not
// substitute a guess.
func (r *Resolver) Resolve(ctx context.Context) (Identity, error) {
	out, err := r.api.GetCallerIdentity(ctx, &sts.GetCallerIdentityInput{})
	if err != nil {
		return Identity{}, fmt.Errorf("sts:GetCallerIdentity: %w", err)
	}
	if out == nil || out.Account == nil || *out.Account == "" {
		return Identity{}, errors.New("sts:GetCallerIdentity returned no account id")
	}

	id := Identity{Account: *out.Account}
	if out.Arn != nil {
		id.ARN = *out.Arn
	}

	parsed, perr := arn.Parse(id.ARN)
	if perr != nil {
		// The partition stays empty rather than defaulting. See the package doc: a
		// default of "aws" would relabel a GovCloud or China run as commercial, and
		// every rate in the resulting report would be one that does not apply.
		return id, nil
	}
	// The two should always agree — GetCallerIdentity's Account is the account of
	// the principal the ARN names. If they ever do not, neither is trustworthy, and
	// picking one silently would put an account id in a report that the report's own
	// evidence contradicts.
	if parsed.AccountID != "" && parsed.AccountID != id.Account {
		return Identity{}, fmt.Errorf("sts:GetCallerIdentity reported account %s but its ARN %s "+
			"names account %s; neither can be recorded", id.Account, id.ARN, parsed.AccountID)
	}
	id.Partition = parsed.Partition
	return id, nil
}

// Record writes the identity into a report envelope.
//
// The mapping lives here, beside the ARN parsing that produced the partition, so
// exactly one place decides what a report says about whose view it is. An unresolved
// identity is simply never recorded: both envelope fields are optional and absent is
// the honest rendering.
func (i Identity) Record(into report.Envelope) report.Envelope {
	into.Account, into.Partition = i.Account, i.Partition
	return into
}
