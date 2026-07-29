package report

import (
	"encoding/json"
	"fmt"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"
)

// The schema is a contract, so these tests guard it from both sides.
//
// A golden file alone catches a deliberate edit and nothing else. The failure that
// actually happens is a field added to a Go struct and forgotten in the schema: the code
// emits it, every test passes, and a consumer validating strictly rejects a report the
// tool considers correct. So the drift test walks the structs by reflection and requires
// the two to agree, which means adding a field to the envelope *cannot* compile-and-pass
// without the schema being updated in the same change.

// The frozen version. Changing this constant is changing the contract, which is the
// point of asserting it: report.v2 is a decision, not a diff.
func TestTheSchemaVersionIsFrozen(t *testing.T) {
	if SchemaVersion != "report.v1" {
		t.Errorf("SchemaVersion = %q; a version bump is a new contract, not a refactor — "+
			"history records and any consumer parser are keyed on this", SchemaVersion)
	}
	m, err := SchemaMap()
	if err != nil {
		t.Fatal(err)
	}
	// The schema declares the same version it describes, or a consumer validating a
	// report against it is checking the wrong contract.
	props, _ := m["properties"].(map[string]any)
	sv, _ := props["schemaVersion"].(map[string]any)
	if got := sv["const"]; got != SchemaVersion {
		t.Errorf("schema pins schemaVersion const = %v, but the code writes %q", got, SchemaVersion)
	}
	if got, ok := m["$id"].(string); !ok || !strings.Contains(got, "report.v1") {
		t.Errorf("schema $id = %v, which does not name the version it describes", m["$id"])
	}
}

// Every JSON field on the envelope and its members appears in the schema, and vice
// versa. This is the test that makes the freeze real: the schema cannot silently fall
// behind the structs, and a property cannot be invented in the schema with no producer.
//
// A field with no producer is worse than no field, which is why the direction matters in
// both senses — account and partition were nearly frozen into v1 with nothing filling
// them, and the fix was to write the producer, not to document the gap.
func TestEveryFieldIsInTheSchema(t *testing.T) {
	m, err := SchemaMap()
	if err != nil {
		t.Fatal(err)
	}
	defs, _ := m["$defs"].(map[string]any)

	for _, tc := range []struct {
		// where is the schema object holding the properties: "" for the root, else a
		// key under $defs.
		where string
		typ   reflect.Type
	}{
		{"", reflect.TypeOf(Envelope{})},
		{"subject", reflect.TypeOf(Subject{})},
		{"query", reflect.TypeOf(Query{})},
		{"region", reflect.TypeOf(Region{})},
		{"assumptions", reflect.TypeOf(Assumptions{})},
		{"amount", reflect.TypeOf(amountJSON{})},
	} {
		t.Run(tc.where+tc.typ.Name(), func(t *testing.T) {
			node := m
			if tc.where != "" {
				node, _ = defs[tc.where].(map[string]any)
				if node == nil {
					t.Fatalf("$defs has no %q", tc.where)
				}
			}
			props, _ := node["properties"].(map[string]any)
			if len(props) == 0 {
				t.Fatalf("%s has no properties", tc.where)
			}

			inSchema := make(map[string]bool, len(props))
			for k := range props {
				inSchema[k] = true
			}

			inGo := map[string]bool{}
			for _, name := range jsonFields(tc.typ) {
				inGo[name] = true
				if !inSchema[name] {
					t.Errorf("%s.%s is emitted by the code and absent from the schema; a "+
						"consumer validating strictly would reject a report this tool "+
						"considers correct", tc.typ.Name(), name)
				}
			}
			for name := range inSchema {
				if !inGo[name] {
					t.Errorf("the schema documents %s.%s, which nothing produces; a frozen "+
						"field with no producer is worse than no field", tc.where, name)
				}
			}
		})
	}
}

// Required means required: a field the schema demands must be one the code always emits,
// and an optional one must actually be omittable. Getting this backwards is how a
// consumer ends up defending against an absence that never occurs, or crashing on one it
// was told could not.
func TestRequiredMatchesOmitempty(t *testing.T) {
	m, err := SchemaMap()
	if err != nil {
		t.Fatal(err)
	}
	defs, _ := m["$defs"].(map[string]any)

	for _, tc := range []struct {
		where string
		typ   reflect.Type
	}{
		{"", reflect.TypeOf(Envelope{})},
		{"subject", reflect.TypeOf(Subject{})},
		{"query", reflect.TypeOf(Query{})},
		{"region", reflect.TypeOf(Region{})},
		{"assumptions", reflect.TypeOf(Assumptions{})},
		{"amount", reflect.TypeOf(amountJSON{})},
	} {
		t.Run(tc.where+tc.typ.Name(), func(t *testing.T) {
			node := m
			if tc.where != "" {
				node, _ = defs[tc.where].(map[string]any)
			}
			required := map[string]bool{}
			for _, r := range node["required"].([]any) {
				required[r.(string)] = true
			}
			for _, f := range reflect.VisibleFields(tc.typ) {
				name, opts, ok := jsonTag(f)
				if !ok {
					continue
				}
				optional := strings.Contains(opts, "omitempty")
				if optional && required[name] {
					t.Errorf("%s.%s is omitempty but the schema requires it", tc.typ.Name(), name)
				}
				if !optional && !required[name] {
					t.Errorf("%s.%s is always emitted but the schema does not require it; a "+
						"consumer would defend against an absence that cannot happen",
						tc.typ.Name(), name)
				}
			}
		})
	}
}

// Every enum in the code is listed in the schema, and nothing is listed that the code
// cannot produce.
//
// The compatibility policy allows these to *gain* members, so the check is not that they
// are closed forever — it is that they agree today. A state the code emits and the schema
// omits is a validation failure on a legitimate report.
func TestTheEnumsAgree(t *testing.T) {
	m, err := SchemaMap()
	if err != nil {
		t.Fatal(err)
	}
	defs, _ := m["$defs"].(map[string]any)
	amount, _ := defs["amount"].(map[string]any)
	amountProps, _ := amount["properties"].(map[string]any)
	region, _ := defs["region"].(map[string]any)
	regionProps, _ := region["properties"].(map[string]any)

	for _, tc := range []struct {
		name string
		node map[string]any
		want []string
	}{
		{"provenance", asMap(amountProps["provenance"]), []string{
			string(ProvenanceLive), string(ProvenanceDerived),
			string(ProvenanceExternal), string(ProvenanceUnavailable)}},
		{"unit", asMap(amountProps["unit"]), []string{
			string(UnitUSD), string(UnitUSDPerHour), string(UnitUSDPerMillionTokens),
			string(UnitUSDPerGBMonth), string(UnitTokensPerSecond), string(UnitGiB),
			string(UnitHours), string(UnitCount), string(UnitFraction)}},
		{"regionState", asMap(regionProps["state"]), []string{
			string(RegionOK), string(RegionEmpty), string(RegionFailed)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			raw, ok := tc.node["enum"].([]any)
			if !ok {
				t.Fatalf("%s has no enum in the schema", tc.name)
			}
			got := make([]string, 0, len(raw))
			for _, v := range raw {
				got = append(got, v.(string))
			}
			sort.Strings(got)
			want := append([]string(nil), tc.want...)
			sort.Strings(want)
			if !reflect.DeepEqual(got, want) {
				t.Errorf("%s enum:\n schema %v\n code   %v", tc.name, got, want)
			}
		})
	}

	// And each listed member is one the type itself accepts, which catches a typo in the
	// schema that the slices above would not: both would have to be wrong the same way.
	for _, p := range []string{"live", "derived", "external", "unavailable"} {
		if !Provenance(p).Valid() {
			t.Errorf("schema lists provenance %q, which Provenance.Valid rejects", p)
		}
	}
	for _, s := range []string{"ok", "empty", "failed"} {
		if !RegionState(s).Valid() {
			t.Errorf("schema lists region state %q, which RegionState.Valid rejects", s)
		}
	}
}

// The golden report: a full envelope serialized, byte for byte.
//
// This is the crude half of the freeze and it earns its place — it catches what
// reflection cannot, which is a change in how a value is *rendered*. A time format that
// drops its timezone, a float that starts serializing as an integer, a nested amount
// that loses its null: all of those keep every field name intact.
func TestTheGoldenReportIsUnchanged(t *testing.T) {
	got, err := json.MarshalIndent(goldenEnvelope(), "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if string(got) != strings.TrimSpace(goldenJSON) {
		t.Errorf("the wire format changed.\n\n--- got ---\n%s\n\n--- want ---\n%s",
			got, strings.TrimSpace(goldenJSON))
	}

	// The golden bytes must also survive a round trip, or the format is writable and not
	// readable — which is the state a history log cannot be queried from.
	var back Envelope
	if err := json.Unmarshal([]byte(goldenJSON), &back); err != nil {
		t.Fatalf("the golden report does not parse: %v", err)
	}
	if err := back.Validate(); err != nil {
		t.Errorf("the golden report does not validate after a round trip: %v", err)
	}
	again, err := json.MarshalIndent(back, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if string(again) != strings.TrimSpace(goldenJSON) {
		t.Errorf("a round trip changed the bytes:\n%s", again)
	}
}

// The schema's own constraints, checked by walking them rather than by trusting that a
// hand-written JSON Schema says what it looks like it says.
//
// This is not a JSON Schema implementation and does not try to be. It asserts the
// specific keywords that carry the rules this format depends on, because each of the
// three was missing on the first draft and each was found by running a real validator:
// the value/provenance dependency, the UTC timestamp pattern, and the absence of
// additionalProperties. A description is not a constraint, and all three read fine.
func TestTheSchemaConstrainsWhatItClaims(t *testing.T) {
	m := mustSchemaMap(t)
	defs, _ := m["$defs"].(map[string]any)

	// An unavailable amount must be null and a known one must be a number. Expressed as
	// if/then/else, since neither a plain type union nor a description can say it.
	amount := asMap(defs["amount"])
	allOf, _ := amount["allOf"].([]any)
	if len(allOf) == 0 {
		t.Fatal("the amount schema has no conditional; the value/provenance dependency is " +
			"described in prose but not constrained, so a payload asserting a price with " +
			"provenance \"unavailable\" would validate")
	}
	cond := asMap(allOf[0])
	ifPart := asMap(asMap(asMap(cond["if"])["properties"])["provenance"])
	if got := ifPart["const"]; got != string(ProvenanceUnavailable) {
		t.Errorf("the conditional keys on provenance = %v, want %q", got, ProvenanceUnavailable)
	}
	thenType := asMap(asMap(asMap(cond["then"])["properties"])["value"])["type"]
	if thenType != "null" {
		t.Errorf("an unavailable amount's value is constrained to %v, want null", thenType)
	}
	elseType := asMap(asMap(asMap(cond["else"])["properties"])["value"])["type"]
	if elseType != "number" {
		t.Errorf("a known amount's value is constrained to %v, want number", elseType)
	}

	// Timestamps carry a pattern, not only a format. "format" is annotation-only by
	// default in Draft 2020-12, so a consumer validating with defaults accepts any string
	// at all — "yesterday" included — and a mixed-timezone report sorts wrongly in the
	// history log.
	ts := asMap(defs["timestamp"])
	pattern, _ := ts["pattern"].(string)
	if pattern == "" {
		t.Fatal("the timestamp definition has no pattern; format alone is annotation-only, " +
			"so any string would validate as an instant")
	}
	if !strings.HasSuffix(pattern, `Z$`) {
		t.Errorf("the timestamp pattern %q does not require UTC; an offset timestamp beside "+
			"a Z one makes two ages incomparable", pattern)
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		t.Fatalf("the timestamp pattern does not compile: %v", err)
	}
	for _, tc := range []struct {
		in string
		ok bool
	}{
		{"2026-07-28T14:30:00Z", true},
		{"2026-07-28T14:30:00.123Z", true},
		{"2026-07-28T16:30:00+02:00", false},
		{"2026-07-28T14:30:00", false},
		{"yesterday", false},
		{"", false},
	} {
		if got := re.MatchString(tc.in); got != tc.ok {
			t.Errorf("the timestamp pattern matches %q = %v, want %v", tc.in, got, tc.ok)
		}
	}
	// And what Go actually emits must match it, which is the half a pattern alone cannot
	// check: the constraint is only worth having if the producer satisfies it.
	for _, s := range []string{
		string(mustJSON(t, time.Date(2026, 7, 28, 14, 30, 0, 0, time.UTC))),
		string(mustJSON(t, time.Date(2026, 7, 28, 16, 30, 0, 0, time.FixedZone("CEST", 2*60*60)).UTC())),
	} {
		if unquoted := strings.Trim(s, `"`); !re.MatchString(unquoted) {
			t.Errorf("Go emits %q, which the schema's own timestamp pattern rejects", unquoted)
		}
	}
}

// A report generated outside UTC still emits UTC everywhere.
//
// Found by generating one in CEST: generatedAt normalized and subject.observedAt did not,
// so the two timestamps a reader compares to judge staleness came out in different
// frames. Nothing failed, and the report looked fine.
func TestEveryTimestampIsUTCWhateverTheLocalZone(t *testing.T) {
	berlin := time.FixedZone("CEST", 2*60*60)
	stamp := time.Date(2026, 7, 28, 16, 30, 0, 0, berlin)

	e := goldenEnvelope()
	e.GeneratedAt = stamp
	e.Subject.ObservedAt = stamp
	e.Assumptions.Throughput = External(1200, UnitTokensPerSecond, "benchmark", stamp)

	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(b), "+02:00") {
		t.Errorf("a report generated in CEST carries a non-UTC timestamp:\n%s", b)
	}
	// Every instant in the document, wherever it sits, against the schema's own pattern.
	ts := asMap(mustSchemaMap(t)["$defs"].(map[string]any)["timestamp"])
	re := regexp.MustCompile(ts["pattern"].(string))
	stamps := timestampsIn(t, b)
	if len(stamps) < 3 {
		t.Fatalf("found %d timestamps in a report that has at least 3", len(stamps))
	}
	for path, v := range stamps {
		if !re.MatchString(v) {
			t.Errorf("%s = %q, which is not the UTC form the schema requires", path, v)
		}
	}
}

// timestampsIn returns every string in a JSON document that looks like an instant, keyed
// by path. Shape-based rather than field-name-based: a new timestamp field added later is
// covered without anyone remembering to list it here.
func timestampsIn(t *testing.T, data []byte) map[string]string {
	t.Helper()
	var tree any
	if err := json.Unmarshal(data, &tree); err != nil {
		t.Fatal(err)
	}
	looksLikeTime := regexp.MustCompile(`^\d{4}-\d{2}-\d{2}T\d{2}:`)
	out := map[string]string{}
	var walk func(string, any)
	walk = func(path string, v any) {
		switch n := v.(type) {
		case map[string]any:
			for k, child := range n {
				walk(path+"/"+k, child)
			}
		case []any:
			for i, child := range n {
				walk(fmt.Sprintf("%s/%d", path, i), child)
			}
		case string:
			if looksLikeTime.MatchString(n) {
				out[path] = n
			}
		}
	}
	walk("", tree)
	return out
}

func mustJSON(t *testing.T, v any) []byte {
	t.Helper()
	b, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	return b
}

// The rule the format exists for, asserted on the wire rather than on the Go value: an
// unresolvable price is null, and its provenance says why. A consumer that ignores
// provenance entirely still cannot read it as free.
func TestAnUnavailableAmountIsNullOnTheWire(t *testing.T) {
	b, err := json.Marshal(Unavailable(UnitUSDPerHour, "p5e.48xlarge has no on-demand price"))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(b, &m); err != nil {
		t.Fatal(err)
	}
	// Present and null. Omitting the key would be worse than a zero: a consumer reading
	// a missing field gets Go's zero value or JavaScript's undefined, and one of those
	// is 0.
	v, present := m["value"]
	if !present {
		t.Errorf("an unavailable amount omits value entirely: %s", b)
	}
	if v != nil {
		t.Errorf("an unavailable amount serialized value = %v, want null: %s", v, b)
	}
	if m["provenance"] != "unavailable" {
		t.Errorf("provenance = %v, want unavailable", m["provenance"])
	}
	if m["source"] == "" || m["source"] == nil {
		t.Error("an unavailable amount carries no reason, so a reader cannot tell what " +
			"would make the value obtainable")
	}
	// A known amount does carry its number, or the null above would be meaningless.
	b, err = json.Marshal(Live(4.00, UnitUSDPerHour, "Price List", time.Date(2026, 7, 28, 0, 0, 0, 0, time.UTC)))
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"value":4`) {
		t.Errorf("a live amount lost its value: %s", b)
	}
}

// Every numeric bound in the schema refuses what [Envelope.Validate] refuses.
//
// The bounds are hand-written JSON, and the off-by-one that matters is minimum where
// exclusiveMinimum was meant: utilization 0 then validates, and a consumer computing a
// per-hour cost from it divides by zero — while this tool, which rejects it, looks like the
// one being strict for no reason. Driven from the Go side so the numbers are not restated:
// each case names a value Validate refuses and requires the schema to refuse it too.
func TestTheNumericBoundsMatchWhatValidateEnforces(t *testing.T) {
	defs := asMap(mustSchemaMap(t)["$defs"])

	for _, tc := range []struct {
		def, field string
		reject     float64
		set        func(*Envelope, float64)
	}{
		// The zero here is the whole reason this test exists: (0, 1] is not [0, 1].
		{"assumptions", "utilization", 0, func(e *Envelope, v float64) { e.Assumptions.Utilization = v }},
		{"assumptions", "utilization", 1.5, func(e *Envelope, v float64) { e.Assumptions.Utilization = v }},
		{"assumptions", "contextTokens", 0, func(e *Envelope, v float64) { e.Assumptions.ContextTokens = int(v) }},
		{"assumptions", "concurrency", 0, func(e *Envelope, v float64) { e.Assumptions.Concurrency = int(v) }},
		{"assumptions", "kvCacheDTypeBytes", 0, func(e *Envelope, v float64) { e.Assumptions.KVCacheDTypeBytes = v }},
		{"assumptions", "overheadFraction", -1, func(e *Envelope, v float64) { e.Assumptions.OverheadFraction = v }},
		{"assumptions", "inputTokenWeight", -1, func(e *Envelope, v float64) { e.Assumptions.InputTokenWeight = v }},
		{"assumptions", "outputTokenWeight", -1, func(e *Envelope, v float64) { e.Assumptions.OutputTokenWeight = v }},
		{"query", "instanceCount", -1, func(e *Envelope, v float64) { e.Query.InstanceCount = int(v) }},
		{"query", "durationHours", -1, func(e *Envelope, v float64) { e.Query.DurationHours = int(v) }},
		{"region", "considered", -1, func(e *Envelope, v float64) { e.Regions[0].Considered = int(v) }},
		{"region", "usable", -1, func(e *Envelope, v float64) { e.Regions[0].Usable = int(v) }},
	} {
		t.Run(fmt.Sprintf("%s/%s=%g", tc.def, tc.field, tc.reject), func(t *testing.T) {
			e := goldenEnvelope()
			tc.set(&e, tc.reject)
			if err := e.Validate(); err == nil {
				t.Fatalf("Validate accepted %s = %g, so this case is not testing what it "+
					"claims — fix the case or the validation", tc.field, tc.reject)
			}
			prop := asMap(asMap(asMap(defs[tc.def])["properties"])[tc.field])
			if len(prop) == 0 {
				t.Fatalf("$defs/%s has no %s", tc.def, tc.field)
			}
			if boundsAdmit(prop, tc.reject) {
				t.Errorf("the schema admits %s/%s = %g, which Validate refuses; a consumer "+
					"validating a hand-written report would accept an assumption this tool "+
					"would never have produced (bounds: %v)", tc.def, tc.field, tc.reject,
					boundKeywords(prop))
			}
		})
	}
}

// boundsAdmit reports whether v satisfies a schema object's numeric bounds. Only the four
// range keywords, because that is all this format uses; anything else here would be a JSON
// Schema implementation, which [TestTheSchemaConstrainsWhatItClaims] explains is not the job.
func boundsAdmit(prop map[string]any, v float64) bool {
	if n, ok := prop["minimum"].(float64); ok && v < n {
		return false
	}
	if n, ok := prop["exclusiveMinimum"].(float64); ok && v <= n {
		return false
	}
	if n, ok := prop["maximum"].(float64); ok && v > n {
		return false
	}
	if n, ok := prop["exclusiveMaximum"].(float64); ok && v >= n {
		return false
	}
	return true
}

// boundKeywords renders the bounds actually present, so a failure says what the schema
// does say and not only what it should.
func boundKeywords(prop map[string]any) map[string]any {
	out := map[string]any{}
	for _, k := range []string{"minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"} {
		if v, ok := prop[k]; ok {
			out[k] = v
		}
	}
	return out
}

// The policy is part of the contract, so it is asserted rather than left in a comment.
// Consumers are told to ignore unknown fields and unknown enum members; a schema that
// closed its objects would contradict that on the first additive change.
func TestTheSchemaPermitsTheAdditiveChangesThePolicyAllows(t *testing.T) {
	if !strings.Contains(CompatibilityPolicy, "additive") {
		t.Errorf("CompatibilityPolicy = %q, which does not state the additive rule", CompatibilityPolicy)
	}
	// additionalProperties: false anywhere would reject a report from a newer build that
	// added a field — the exact change report.v1 promises is safe.
	if n := countKey(mustSchemaMap(t), "additionalProperties"); n != 0 {
		t.Errorf("the schema sets additionalProperties %d time(s); within report.v1 a "+
			"consumer must ignore unknown fields, so closing an object contradicts the "+
			"compatibility policy", n)
	}
	// A newer build's report — an unknown envelope field, an unknown region state — must
	// still parse. Go's decoder ignores unknown fields, which is the behavior consumers
	// are asked to match; what this pins is that the *unknown enum member* does not blow
	// up on the way in.
	future := strings.Replace(goldenJSON, `"state": "empty"`, `"state": "throttled"`, 1)
	future = strings.Replace(future, `"kind": "compare"`, `"kind": "compare", "futureField": {"nested": true}`, 1)
	var e Envelope
	if err := json.Unmarshal([]byte(future), &e); err != nil {
		t.Fatalf("a report from a newer build does not parse: %v", err)
	}
	var found bool
	for _, r := range e.Regions {
		if r.State == "throttled" {
			found = true
			// An unrecognized state must not be treated as informative: a region this
			// build cannot interpret has told it nothing about AWS, and guessing "ok"
			// would turn a throttle into a claim about inventory.
			if r.State.Informative() {
				t.Error("an unrecognized region state reads as informative")
			}
			if r.State.Valid() {
				t.Error("an unrecognized region state reads as valid")
			}
		}
	}
	if !found {
		t.Error("the unknown state did not survive parsing")
	}
	// Validate does reject it, which is correct and is the asymmetry worth stating: this
	// build must not *write* a state it does not define, but it must still read one.
	if err := e.Validate(); err == nil {
		t.Error("Validate accepted a region state this build does not define; writing one " +
			"would emit a report no consumer of report.v1 can interpret")
	}
}

// jsonFields returns the JSON names a type emits, in declaration order.
func jsonFields(t reflect.Type) []string {
	var out []string
	for _, f := range reflect.VisibleFields(t) {
		if name, _, ok := jsonTag(f); ok {
			out = append(out, name)
		}
	}
	return out
}

// jsonTag splits a field's json tag. Reports false for unexported or skipped fields.
func jsonTag(f reflect.StructField) (name, opts string, ok bool) {
	if !f.IsExported() {
		return "", "", false
	}
	tag, has := f.Tag.Lookup("json")
	if !has || tag == "-" {
		return "", "", false
	}
	name, opts, _ = strings.Cut(tag, ",")
	if name == "" {
		name = f.Name
	}
	return name, opts, true
}

func asMap(v any) map[string]any {
	m, _ := v.(map[string]any)
	return m
}

func mustSchemaMap(t *testing.T) map[string]any {
	t.Helper()
	m, err := SchemaMap()
	if err != nil {
		t.Fatal(err)
	}
	return m
}

// countKey counts occurrences of a key anywhere in a decoded JSON tree.
func countKey(v any, key string) int {
	switch t := v.(type) {
	case map[string]any:
		n := 0
		for k, child := range t {
			if k == key {
				n++
			}
			n += countKey(child, key)
		}
		return n
	case []any:
		n := 0
		for _, child := range t {
			n += countKey(child, key)
		}
		return n
	}
	return 0
}

// goldenEnvelope is a complete report: every optional field set, a failed region beside
// an empty one, and an external throughput figure. Deliberately not a minimal example —
// the fields most likely to break are the ones a minimal fixture omits.
func goldenEnvelope() Envelope {
	stamp := time.Date(2026, 7, 28, 14, 30, 0, 0, time.UTC)
	e := NewEnvelope("compare", "0.2.0", stamp)
	e.Account, e.Partition = "942542972736", "aws"
	e.Subject = Subject{
		ModelID:        "Qwen/Qwen3-32B",
		BedrockModelID: "qwen.qwen3-32b-v1:0",
		ObservedAt:     stamp,
	}
	e.Query = Query{
		Regions:       []string{"us-east-1", "us-east-2", "us-west-1"},
		InstanceCount: 1,
	}
	e.Regions = []Region{
		{Name: "us-east-1", State: RegionOK, Considered: 1354, Usable: 26},
		{Name: "us-east-2", State: RegionOK, Considered: 1205, Usable: 24},
		{Name: "us-west-1", State: RegionEmpty, Considered: 668, Usable: 0},
	}
	e.Assumptions = Assumptions{
		InputTokenWeight:  3,
		OutputTokenWeight: 1,
		Utilization:       1,
		ContextTokens:     8192,
		Concurrency:       1,
		KVCacheDTypeBytes: 2,
		OverheadFraction:  0.1,
		Throughput: External(1200, UnitTokensPerSecond,
			"published vLLM benchmark, g7e.4xlarge", time.Date(2026, 6, 1, 0, 0, 0, 0, time.UTC)),
	}
	return e
}

// goldenJSON is the frozen wire form of [goldenEnvelope].
//
// Written out rather than generated so that a change to the format shows up as a diff in
// this file, in a code review, next to the change that caused it. A regenerate-on-fail
// golden file catches nothing: the regeneration is the same keystroke as accepting the
// break.
const goldenJSON = `
{
  "schemaVersion": "report.v1",
  "kind": "compare",
  "toolVersion": "0.2.0",
  "generatedAt": "2026-07-28T14:30:00Z",
  "account": "942542972736",
  "partition": "aws",
  "subject": {
    "modelId": "Qwen/Qwen3-32B",
    "bedrockModelId": "qwen.qwen3-32b-v1:0",
    "gated": false,
    "observedAt": "2026-07-28T14:30:00Z"
  },
  "query": {
    "regions": [
      "us-east-1",
      "us-east-2",
      "us-west-1"
    ],
    "instanceCount": 1
  },
  "regions": [
    {
      "name": "us-east-1",
      "state": "ok",
      "considered": 1354,
      "usable": 26
    },
    {
      "name": "us-east-2",
      "state": "ok",
      "considered": 1205,
      "usable": 24
    },
    {
      "name": "us-west-1",
      "state": "empty",
      "considered": 668,
      "usable": 0
    }
  ],
  "assumptions": {
    "inputTokenWeight": 3,
    "outputTokenWeight": 1,
    "utilization": 1,
    "contextTokens": 8192,
    "concurrency": 1,
    "kvCacheDTypeBytes": 2,
    "overheadFraction": 0.1,
    "throughput": {
      "value": 1200,
      "unit": "tok/s",
      "provenance": "external",
      "source": "published vLLM benchmark, g7e.4xlarge",
      "observedAt": "2026-06-01T00:00:00Z"
    }
  }
}`
