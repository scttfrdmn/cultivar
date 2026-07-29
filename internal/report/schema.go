package report

import (
	"embed"
	"encoding/json"
	"fmt"
)

// The frozen schema, shipped in the binary.
//
// Embedded rather than published only to the docs site so that the bytes a consumer
// validates against are the bytes this build produces reports with. A schema that lives
// only on a web page drifts from the code silently, and the drift is invisible in
// exactly the case that matters: a field added to the Go struct and not to the contract.

//go:embed report.v1.json
var schemaFS embed.FS

// Schema returns the JSON Schema for [SchemaVersion] as raw bytes.
//
// Exported so `cultivar schema` can print it and cmd/refresh can publish it without a
// second copy existing anywhere. Consumers validate against this; [Envelope.Validate]
// enforces the same contract plus the cross-field rules a JSON Schema cannot express —
// that every queried region has a result, that a failed region carries its error text.
func Schema() []byte {
	b, err := schemaFS.ReadFile("report.v1.json")
	if err != nil {
		// Unreachable: the file is embedded at build time, so a failure here means the
		// binary was built without it, which the test suite would not have permitted.
		panic(fmt.Sprintf("report: embedded schema unreadable: %v", err))
	}
	return b
}

// SchemaMap returns the schema decoded, for callers that inspect it — the drift test
// walks it against the Go types, and cmd/refresh reads its $id when publishing.
func SchemaMap() (map[string]any, error) {
	var m map[string]any
	if err := json.Unmarshal(Schema(), &m); err != nil {
		return nil, fmt.Errorf("report: embedded schema is not valid JSON: %w", err)
	}
	return m, nil
}
