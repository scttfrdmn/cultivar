// Package bedrock resolves a Hugging Face repo id to its Bedrock equivalent, and
// decides whether that equivalent can actually be billed per token.
//
// The distinction the package exists to enforce: a Bedrock entry whose catalog is
// "foundation-model" is serverless per-token billing, and comparing it against a
// self-hosted GPU is the whole product. An entry whose catalog is "marketplace"
// means renting an instance through Bedrock — there is no token price at all, so
// offering "just use Bedrock instead" for one is wrong advice, not merely imprecise.
//
// Measured against the live mapping on 2026-07-27: of 132 reverse-lookupable HF
// ids, 38 have a foundation-model path and 94 are marketplace-only. The escape
// hatch fires about 29% of the time, so the tool has to be genuinely useful for the
// other 71%.
package bedrock

import (
	"sort"
	"strings"
	"time"
)

// Catalog is which Bedrock catalog an entry belongs to. This single field decides
// whether a token price exists.
type Catalog string

const (
	// CatalogFoundationModel is serverless, billed per input and output token.
	CatalogFoundationModel Catalog = "foundation-model"
	// CatalogMarketplace is a model you deploy to an endpoint you rent by the hour.
	// It has no token price, so it is not an alternative to self-hosting — it *is*
	// self-hosting, with a different control plane.
	CatalogMarketplace Catalog = "marketplace"
)

// TokenBillable reports whether this catalog bills per token.
//
// An unrecognized catalog returns false. Unlike the gated-repo case, where both
// defaults cost something, the safe direction here is unambiguous: withholding the
// Bedrock offer for a catalog we do not understand loses an option, while offering
// it produces a per-token comparison against a price that does not exist.
func (c Catalog) TokenBillable() bool { return c == CatalogFoundationModel }

// Confidence is the mapping's own assessment of a link. Observed values are
// "confirmed" (130 entries) and "validated" (17). It is propagated rather than
// interpreted: cultivar reports what the mapping claims about itself instead of
// silently treating every link as certain.
type Confidence string

// Entry is one Bedrock model linked to an HF repo.
type Entry struct {
	// ModelID is the Bedrock model id, e.g. "qwen.qwen3-32b-v1:0" for a foundation
	// model or "huggingface-reasoning-qwen3-32b" for a marketplace listing.
	ModelID string

	// Catalog decides whether ModelID has a token price. See [Catalog].
	Catalog Catalog

	// Confidence is the mapping's self-assessment.
	Confidence Confidence

	// Regions are the regions this entry is offered in. These are not always the
	// same as the parent's: Qwen3-32B's marketplace listing covers four regions
	// while its foundation-model entry covers three, so a comparison scoped to
	// us-west-1 must use the entry's own list.
	Regions []string
}

// OfferedIn reports whether this entry is available in region.
func (e Entry) OfferedIn(region string) bool {
	for _, r := range e.Regions {
		if strings.EqualFold(r, region) {
			return true
		}
	}
	return false
}

// Mapping is the resolved Bedrock relationship for one HF repo.
type Mapping struct {
	// HFID is the canonical (case-preserved) HF id the mapping reports.
	HFID string

	// OnBedrock is the mapping's own flag. A repo absent from the mapping produces
	// a Mapping with OnBedrock false and no entries — see [Client.Lookup].
	OnBedrock bool

	// Regions is the union across entries, as published.
	Regions []string

	// Entries is every Bedrock link, in published order. Read it through
	// [Mapping.FoundationModels]; index 0 is not the interesting one.
	Entries []Entry

	// GeneratedAt is when the mapping was built. Only the bulk index publishes it,
	// so it is zero for a per-model lookup — see [Mapping.PublishedAt].
	GeneratedAt time.Time

	// PublishedAt is the Last-Modified time of the fetched document. It is the only
	// freshness signal available on a per-model lookup, and it is what the report
	// ages when GeneratedAt is zero.
	PublishedAt time.Time

	// ObservedAt is when cultivar read this.
	ObservedAt time.Time

	// Source is the URL the data came from, for the report's provenance.
	Source string
}

// FoundationModels returns the token-billable entries, deduped by model id and
// ordered deterministically.
//
// Scanning is mandatory, not stylistic: for `Qwen/Qwen3-32B` — this project's own
// headline example — `Entries[0]` is the *marketplace* listing and the
// foundation-model entry is second. Reading index 0 would conclude "no token price
// available, you must self-host" for a model whose whole point is that Bedrock is
// cheaper. Two of the 38 token-billable models hide their foundation-model path
// behind index 0 that way.
func (m *Mapping) FoundationModels() []Entry {
	if m == nil {
		return nil
	}
	seen := make(map[string]bool, len(m.Entries))
	out := make([]Entry, 0, len(m.Entries))
	for _, e := range m.Entries {
		if !e.Catalog.TokenBillable() || seen[e.ModelID] {
			continue
		}
		seen[e.ModelID] = true
		out = append(out, e)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ModelID < out[j].ModelID })
	return out
}

// HasTokenPricing reports whether a per-token comparison is even meaningful for
// this repo.
func (m *Mapping) HasTokenPricing() bool { return len(m.FoundationModels()) > 0 }

// MarketplaceOnly reports that the repo is on Bedrock but only as a rented
// endpoint. This is a distinct answer from "not on Bedrock" and must not be
// collapsed into it: the user can still deploy it there, they just cannot escape
// paying for hardware, so self-hosting comparison is the only honest framing.
func (m *Mapping) MarketplaceOnly() bool {
	return m != nil && len(m.Entries) > 0 && !m.HasTokenPricing()
}

// FoundationModelsIn returns the token-billable entries offered in region.
//
// Region matters because the entry lists diverge from the parent's: 8 of the 132
// mapped repos have at least one entry whose regions differ from the union. A
// comparison run for us-west-1 that used the union would price a model against a
// Bedrock endpoint that region cannot serve.
func (m *Mapping) FoundationModelsIn(region string) []Entry {
	var out []Entry
	for _, e := range m.FoundationModels() {
		if e.OfferedIn(region) {
			out = append(out, e)
		}
	}
	return out
}

// UnknownCatalogs returns catalog values that are neither foundation-model nor
// marketplace.
//
// A new catalog is a signal that the token-billable question has an answer this
// code does not know, so callers surface it as a caveat rather than reporting a
// confident "no token pricing". Parsing does not fail on one: erroring would break
// every lookup for a repo that merely gained an extra listing.
func (m *Mapping) UnknownCatalogs() []string {
	if m == nil {
		return nil
	}
	seen := map[string]bool{}
	var out []string
	for _, e := range m.Entries {
		switch e.Catalog {
		case CatalogFoundationModel, CatalogMarketplace:
			continue
		}
		if !seen[string(e.Catalog)] {
			seen[string(e.Catalog)] = true
			out = append(out, string(e.Catalog))
		}
	}
	sort.Strings(out)
	return out
}

// Age reports how stale the mapping is, preferring the index's generatedAt and
// falling back to the document's Last-Modified. Reports false when neither is
// known, so a report says "unknown age" rather than implying freshness.
func (m *Mapping) Age(now time.Time) (time.Duration, bool) {
	switch {
	case m == nil:
		return 0, false
	case !m.GeneratedAt.IsZero():
		return now.Sub(m.GeneratedAt), true
	case !m.PublishedAt.IsZero():
		return now.Sub(m.PublishedAt), true
	}
	return 0, false
}

// normalizeID lowercases an HF id the way the mapping keys it. The published keys
// are all lowercase, and the per-model paths are case-sensitive: requesting
// hf/Qwen/Qwen3-32B.json returns 404 while hf/qwen/qwen3-32b.json returns the
// entry, so skipping this step reads as "not on Bedrock".
func normalizeID(id string) string {
	return strings.ToLower(strings.Trim(strings.TrimSpace(id), "/"))
}
