package report

import (
	"fmt"
	"math"
	"strings"
)

// Arithmetic over amounts. Every operation here obeys one rule: an unknown input
// produces an unknown result. Nothing substitutes zero, and nothing estimates.
//
// That rule is the whole point. The failure mode this package exists to prevent
// is not a wrong lookup — it is a missing lookup that becomes a number somewhere
// downstream, so that a report claims a p5e.48xlarge costs $9.60/hr when in truth
// it has no on-demand price at all. Propagating unavailability through the
// arithmetic means a missing input surfaces as "unpriced" in the output instead of
// being laundered into a confident total.

// Sum adds amounts of the same unit. It returns unavailable if any input is
// unavailable, naming the first such input, and errors on a unit mismatch —
// which catches adding a capacity block's upfront fee to an hourly rate.
func Sum(unit Unit, what string, amounts ...Amount) (Amount, error) {
	if len(amounts) == 0 {
		return Amount{}, fmt.Errorf("sum %s: no amounts", what)
	}
	var total float64
	var sources []string
	for i, a := range amounts {
		if a.unit != unit {
			return Amount{}, fmt.Errorf("sum %s: operand %d has unit %s, want %s", what, i, a.unit, unit)
		}
		v, ok := a.Value()
		if !ok {
			return Unavailable(unit, fmt.Sprintf("%s: %s", what, a.source)), nil
		}
		total += v
		sources = append(sources, a.source)
	}
	return Derived(total, unit, fmt.Sprintf("%s = sum of %s", what, strings.Join(sources, " + "))), nil
}

// Scale multiplies an amount by a dimensionless factor, keeping its unit: an
// hourly rate times a count of hours is *not* this operation (see [Convert]).
func Scale(a Amount, factor float64, what string) Amount {
	v, ok := a.Value()
	if !ok {
		return Unavailable(a.unit, fmt.Sprintf("%s: %s", what, a.source))
	}
	if math.IsNaN(factor) || math.IsInf(factor, 0) {
		return Unavailable(a.unit, fmt.Sprintf("%s: scale factor is not finite", what))
	}
	return Derived(v*factor, a.unit, fmt.Sprintf("%s = %s x %g", what, a.source, factor))
}

// Convert changes an amount's unit by multiplying it, for operations that alter
// dimension: an upfront fee (USD) divided by a block's actual duration (hours)
// becomes a rate (USD/hour). The caller states the resulting unit, because this
// package deliberately has no unit algebra to infer it from.
func Convert(a Amount, factor float64, unit Unit, what string) Amount {
	v, ok := a.Value()
	if !ok {
		return Unavailable(unit, fmt.Sprintf("%s: %s", what, a.source))
	}
	if math.IsNaN(factor) || math.IsInf(factor, 0) {
		return Unavailable(unit, fmt.Sprintf("%s: conversion factor is not finite", what))
	}
	return Derived(v*factor, unit, fmt.Sprintf("%s (from %s)", what, a.source))
}

// Ratio divides one amount by another, producing a value in the given unit. A
// zero denominator yields unavailable rather than an infinity: "you would need
// infinite throughput to break even" is not a claim worth rendering, and
// arithmetic like the break-even calculation divides by a token price that may
// legitimately be missing.
func Ratio(num, den Amount, unit Unit, what string) Amount {
	nv, nok := num.Value()
	if !nok {
		return Unavailable(unit, fmt.Sprintf("%s: %s", what, num.source))
	}
	dv, dok := den.Value()
	if !dok {
		return Unavailable(unit, fmt.Sprintf("%s: %s", what, den.source))
	}
	if dv == 0 {
		return Unavailable(unit, fmt.Sprintf("%s: %s is zero", what, den.source))
	}
	return Derived(nv/dv, unit, fmt.Sprintf("%s = %s / %s", what, num.source, den.source))
}

// Blend combines two amounts of the same unit by weight, used for the Bedrock
// input:output token price. The weights need not sum to 1; they are normalized.
//
// Bedrock meters input and output separately and output costs 4x input for
// Qwen3-32B, so any single $/1M figure encodes an assumption about traffic shape.
// Requiring both weights here means that assumption is always written down, and
// the resulting source string records the ratio used.
func Blend(a Amount, aWeight float64, b Amount, bWeight float64, what string) (Amount, error) {
	if a.unit != b.unit {
		return Amount{}, fmt.Errorf("blend %s: units differ (%s vs %s)", what, a.unit, b.unit)
	}
	if aWeight < 0 || bWeight < 0 {
		return Amount{}, fmt.Errorf("blend %s: negative weight (%g, %g)", what, aWeight, bWeight)
	}
	total := aWeight + bWeight
	if total == 0 {
		return Amount{}, fmt.Errorf("blend %s: weights sum to zero", what)
	}
	av, aok := a.Value()
	if !aok {
		return Unavailable(a.unit, fmt.Sprintf("%s: %s", what, a.source)), nil
	}
	bv, bok := b.Value()
	if !bok {
		return Unavailable(b.unit, fmt.Sprintf("%s: %s", what, b.source)), nil
	}
	v := (av*aWeight + bv*bWeight) / total
	return Derived(v, a.unit, fmt.Sprintf("%s = %g:%g blend of %s and %s",
		what, aWeight, bWeight, a.source, b.source)), nil
}

// Compare orders two amounts of the same unit. Unknown values sort last, so an
// unpriced option can never rank first — a ranking that puts "unpriced" at the
// top reads as "cheapest", which is the inverse of the truth.
//
// Returns -1 if a sorts before b, +1 if after, 0 if equivalent.
func Compare(a, b Amount) int {
	av, aok := a.Value()
	bv, bok := b.Value()
	switch {
	case !aok && !bok:
		return 0
	case !aok:
		return 1
	case !bok:
		return -1
	case av < bv:
		return -1
	case av > bv:
		return 1
	default:
		return 0
	}
}
