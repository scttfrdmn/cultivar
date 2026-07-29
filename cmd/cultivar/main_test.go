package main

import (
	"bytes"
	"encoding/json"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// `cultivar schema` is a wire-format surface, so it is tested as one: built and run, with
// its stdout parsed.
//
// Calling report.Schema() from a test would prove nothing about the command — the failure
// mode here is the plumbing, not the schema. A log line on stdout, a trailing banner, or a
// non-zero exit each break `cultivar schema | jq` while leaving the bytes correct, and any
// of those makes the schema unobtainable by the one route a consumer would use.
func TestSchemaCommandPrintsThePipeableContract(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "cultivar")
	build := exec.Command("go", "build", "-o", bin, ".")
	if out, err := build.CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}

	var stdout, stderr bytes.Buffer
	cmd := exec.Command(bin, "schema")
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("cultivar schema exited %v; stderr:\n%s", err, stderr.String())
	}

	// Byte-identical to the embedded copy. Anything else means there are two schemas.
	if !bytes.Equal(stdout.Bytes(), report.Schema()) {
		t.Errorf("stdout is not the embedded schema (%d bytes out, %d embedded)",
			stdout.Len(), len(report.Schema()))
	}
	// And parseable on its own, which is what "pipeable" means in practice: nothing else
	// may share stdout with it.
	var m map[string]any
	if err := json.Unmarshal(stdout.Bytes(), &m); err != nil {
		t.Fatalf("stdout does not parse as JSON, so it cannot be piped: %v\n%s", err, stdout.String())
	}
	if m["$id"] == nil || m["$schema"] == nil {
		t.Errorf("printed document has no $id or $schema; a consumer cannot tell what it is")
	}
	if got := m["title"]; got != "cultivar "+report.SchemaVersion {
		t.Errorf("title = %v, want %q", got, "cultivar "+report.SchemaVersion)
	}
}

// The no-command path stays on stderr and exits non-zero, which is what makes the schema
// path above pipeable at all: if the usage banner went to stdout, a consumer who mistyped
// the subcommand would get a banner where they expected JSON, with a zero exit status.
func TestUsageGoesToStderrAndFails(t *testing.T) {
	bin := filepath.Join(t.TempDir(), "cultivar")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		t.Fatalf("build: %v\n%s", err, out)
	}
	var stdout, stderr bytes.Buffer
	cmd := exec.Command(bin)
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	err := cmd.Run()
	if err == nil {
		t.Error("cultivar with no command exited 0, so a script cannot tell it did nothing")
	}
	if stdout.Len() != 0 {
		t.Errorf("usage went to stdout: %q", stdout.String())
	}
	if stderr.Len() == 0 {
		t.Error("no usage text at all")
	}
}
