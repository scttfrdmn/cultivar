// Package report defines the wire format of a cultivar report.
//
// The central rule: a number never travels without its origin. [Amount] pairs a
// value with a [Provenance], and its fields are unexported, so the only way to
// produce one is through a constructor that demands both. A price nobody looked
// up cannot be spelled.
//
// This is structural rather than conventional because the convention has already
// failed four times in this ecosystem: stale hardcoded rates in spawn's pkg/slurm
// (p4d off 49%, p5 off 79%), a hardcoded SageMaker premium, a hardcoded
// capacity-block discount, and spore-host/libs' pricing table, which fabricates a
// p6-b200.48xlarge rate 11.9x too low and returns it with a nil error.
package report

// Provenance records where a number came from. Every numeric field in a report
// carries one.
type Provenance string

const (
	// ProvenanceLive was read from an AWS or Hugging Face API during this run.
	ProvenanceLive Provenance = "live"

	// ProvenanceDerived was computed from other amounts in this report. Its
	// trustworthiness is that of its worst input, which is why the arithmetic in
	// this package propagates ProvenanceUnavailable instead of substituting zero.
	ProvenanceDerived Provenance = "derived"

	// ProvenanceExternal is a published third-party measurement — a throughput
	// number from a benchmark this project did not run, for example. It carries a
	// citation and a date because it can go stale without anything failing.
	ProvenanceExternal Provenance = "external"

	// ProvenanceUnavailable means no value could be resolved. It is a value-less
	// state, deliberately distinct from zero: p5e.48xlarge has no on-demand price
	// at all, and reporting that as $0.00 (or as a family-based guess) is the
	// specific bug this package exists to prevent.
	ProvenanceUnavailable Provenance = "unavailable"
)

// Valid reports whether p is one of the four defined provenances. The zero
// Provenance ("") is not valid, so a struct literal that forgot to set one is
// caught rather than defaulted.
func (p Provenance) Valid() bool {
	switch p {
	case ProvenanceLive, ProvenanceDerived, ProvenanceExternal, ProvenanceUnavailable:
		return true
	}
	return false
}

// Known reports whether p accompanies a usable numeric value.
func (p Provenance) Known() bool {
	switch p {
	case ProvenanceLive, ProvenanceDerived, ProvenanceExternal:
		return true
	}
	return false
}

func (p Provenance) String() string { return string(p) }

// Unit is the dimension of an [Amount]. Units are not merely labels: [Sum]
// refuses to add amounts of differing units, which catches the realistic mistake
// of adding a capacity block's upfront fee (USD) to an hourly rate (USD/hour).
//
// There is deliberately no unit algebra. Operations that change dimension take
// the result unit from the caller, who states what they meant.
type Unit string

const (
	// UnitUSD is a one-off amount, such as a capacity block's upfront fee.
	UnitUSD Unit = "USD"
	// UnitUSDPerHour is an instance-hour rate.
	UnitUSDPerHour Unit = "USD/hour"
	// UnitUSDPerMillionTokens is a Bedrock token rate. Input and output are
	// separate meters and must not be collapsed without an explicit blend ratio.
	UnitUSDPerMillionTokens Unit = "USD/1Mtok"
	// UnitUSDPerGBMonth is EBS storage.
	UnitUSDPerGBMonth Unit = "USD/GB-month"
	// UnitTokensPerSecond is throughput, measured or required.
	UnitTokensPerSecond Unit = "tok/s"
	// UnitGiB sizes model weights and GPU memory.
	UnitGiB Unit = "GiB"
	// UnitHours is a duration, such as a capacity block's actual length.
	UnitHours Unit = "hour"
	// UnitCount is a dimensionless count: GPUs, instances, availability zones.
	UnitCount Unit = "count"
	// UnitFraction is a dimensionless ratio: utilization, the SageMaker premium.
	UnitFraction Unit = "fraction"
)

// IsMoney reports whether u denominates currency, which is what lets an
// unresolved amount render as "unpriced" rather than a bare "unavailable".
func (u Unit) IsMoney() bool {
	switch u {
	case UnitUSD, UnitUSDPerHour, UnitUSDPerMillionTokens, UnitUSDPerGBMonth:
		return true
	}
	return false
}

func (u Unit) String() string { return string(u) }
