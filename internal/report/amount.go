package report

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"time"
)

// Amount is a number that knows where it came from.
//
// The fields are unexported so that every Amount passes through a constructor:
// [Live], [Derived], [External], or [Unavailable]. There is no way to write an
// Amount without stating its provenance, and no zero value that reads as a real
// price — the zero Amount has an invalid provenance and [Amount.Valid] rejects it.
type Amount struct {
	value      float64
	unit       Unit
	provenance Provenance

	// source names the origin: an AWS API and filter set, an HF endpoint, a
	// citation for external data, or the derivation for computed values. Free-form
	// because its audience is a human checking a number against the console.
	source string

	// observedAt is when the value was read. Availability data has a shelf life of
	// hours, so an amount that outlives its usefulness can say so.
	observedAt time.Time
}

// Live returns an amount read directly from an API this run.
func Live(value float64, unit Unit, source string, observedAt time.Time) Amount {
	return Amount{value: value, unit: unit, provenance: ProvenanceLive, source: source, observedAt: observedAt}
}

// Derived returns an amount computed from other amounts. Callers should prefer
// the arithmetic helpers ([Sum], [Scale], [Ratio], [Convert]), which propagate
// unavailability instead of silently producing a number from a missing input.
func Derived(value float64, unit Unit, source string) Amount {
	return Amount{value: value, unit: unit, provenance: ProvenanceDerived, source: source}
}

// External returns a published third-party measurement. Both a citation and the
// date it was published are required: an external throughput figure is only
// interpretable alongside the engine version and hardware it was measured on.
func External(value float64, unit Unit, citation string, publishedAt time.Time) Amount {
	return Amount{value: value, unit: unit, provenance: ProvenanceExternal, source: citation, observedAt: publishedAt}
}

// Unavailable returns a value-less amount, recording why. The reason is shown to
// users, so it should say what would make the value obtainable — "no on-demand
// price exists; capacity-block only" rather than "lookup failed".
func Unavailable(unit Unit, reason string) Amount {
	return Amount{unit: unit, provenance: ProvenanceUnavailable, source: reason}
}

// Value returns the numeric value and whether it is known. Callers must check ok;
// the float is zero when it is not, which is exactly the confusion this signature
// prevents from happening silently.
func (a Amount) Value() (float64, bool) {
	if !a.provenance.Known() {
		return 0, false
	}
	return a.value, true
}

// MustValue returns the value, panicking if it is unknown. For use after an
// explicit [Amount.Known] check, and in tests.
func (a Amount) MustValue() float64 {
	v, ok := a.Value()
	if !ok {
		panic(fmt.Sprintf("report: value of unavailable amount (%s): %s", a.unit, a.source))
	}
	return v
}

// Known reports whether this amount carries a usable value.
func (a Amount) Known() bool { return a.provenance.Known() }

// Provenance returns where the value came from.
func (a Amount) Provenance() Provenance { return a.provenance }

// Unit returns the dimension of the value.
func (a Amount) Unit() Unit { return a.unit }

// Source returns the origin description, or the reason it is unavailable.
func (a Amount) Source() string { return a.source }

// ObservedAt returns when the value was read, or the zero time for derived
// amounts (whose age is that of their inputs).
func (a Amount) ObservedAt() time.Time { return a.observedAt }

// Valid reports whether a is well-formed: a real provenance, a unit, a source,
// and — for known values — a finite number. The zero Amount is not valid, so a
// forgotten field surfaces as a validation error rather than as $0.00.
func (a Amount) Valid() error {
	if !a.provenance.Valid() {
		return fmt.Errorf("provenance %q is not one of live, derived, external, unavailable", a.provenance)
	}
	if a.unit == "" {
		return fmt.Errorf("amount has no unit")
	}
	if a.source == "" {
		return fmt.Errorf("amount (%s, %s) has no source", a.unit, a.provenance)
	}
	if a.provenance.Known() {
		if math.IsNaN(a.value) || math.IsInf(a.value, 0) {
			return fmt.Errorf("amount (%s) value is not finite: %v", a.unit, a.value)
		}
		if a.unit.IsMoney() && a.value < 0 {
			return fmt.Errorf("amount (%s) is negative: %v", a.unit, a.value)
		}
	}
	if a.provenance == ProvenanceLive && a.observedAt.IsZero() {
		return fmt.Errorf("live amount (%s) has no observation time", a.unit)
	}
	return nil
}

// Age returns how long ago the value was observed. Reports zero duration and
// false when the amount has no observation time.
func (a Amount) Age(now time.Time) (time.Duration, bool) {
	if a.observedAt.IsZero() {
		return 0, false
	}
	return now.Sub(a.observedAt), true
}

// String renders the amount for humans. Unavailable money reads as "unpriced",
// which is a statement a reader can act on; "$0.00" is not.
func (a Amount) String() string {
	if !a.provenance.Known() {
		if a.unit.IsMoney() {
			return "unpriced"
		}
		return "unknown"
	}
	switch a.unit {
	case UnitUSD:
		return "$" + trimFloat(a.value, 2)
	case UnitUSDPerHour:
		return "$" + trimFloat(a.value, 4) + "/hr"
	case UnitUSDPerMillionTokens:
		return "$" + trimFloat(a.value, 4) + "/1M tok"
	case UnitUSDPerGBMonth:
		return "$" + trimFloat(a.value, 4) + "/GB-mo"
	case UnitTokensPerSecond:
		return trimFloat(a.value, 0) + " tok/s"
	case UnitGiB:
		return trimFloat(a.value, 1) + " GiB"
	case UnitHours:
		return trimFloat(a.value, 2) + "h"
	case UnitFraction:
		return trimFloat(a.value*100, 1) + "%"
	default:
		return trimFloat(a.value, 2)
	}
}

// trimFloat formats v with at most prec decimals, dropping trailing zeros so
// $4.00/hr does not read as $4.0000/hr.
func trimFloat(v float64, prec int) string {
	s := strconv.FormatFloat(v, 'f', prec, 64)
	if prec == 0 {
		return s
	}
	for len(s) > 0 && s[len(s)-1] == '0' {
		s = s[:len(s)-1]
	}
	if len(s) > 0 && s[len(s)-1] == '.' {
		s = s[:len(s)-1]
	}
	return s
}

// amountJSON is the wire shape. The value is a pointer so an unavailable amount
// serializes as null rather than 0 — a JSON consumer that ignores provenance
// still cannot mistake "no price" for "free".
type amountJSON struct {
	Value      *float64   `json:"value"`
	Unit       Unit       `json:"unit"`
	Provenance Provenance `json:"provenance"`
	Source     string     `json:"source"`
	ObservedAt *time.Time `json:"observedAt,omitempty"`
}

// MarshalJSON implements [json.Marshaler].
func (a Amount) MarshalJSON() ([]byte, error) {
	out := amountJSON{Unit: a.unit, Provenance: a.provenance, Source: a.source}
	if a.provenance.Known() {
		v := a.value
		out.Value = &v
	}
	if !a.observedAt.IsZero() {
		t := a.observedAt.UTC()
		out.ObservedAt = &t
	}
	return json.Marshal(out)
}

// UnmarshalJSON implements [json.Unmarshaler]. A payload whose provenance claims
// a value but omits it is rejected, so a truncated or hand-edited report does not
// decode into a zero price.
func (a *Amount) UnmarshalJSON(data []byte) error {
	var in amountJSON
	if err := json.Unmarshal(data, &in); err != nil {
		return err
	}
	if !in.Provenance.Valid() {
		return fmt.Errorf("report: invalid provenance %q", in.Provenance)
	}
	if in.Provenance.Known() && in.Value == nil {
		return fmt.Errorf("report: provenance %q with no value", in.Provenance)
	}
	*a = Amount{unit: in.Unit, provenance: in.Provenance, source: in.Source}
	if in.Value != nil && in.Provenance.Known() {
		a.value = *in.Value
	}
	if in.ObservedAt != nil {
		a.observedAt = *in.ObservedAt
	}
	return nil
}
