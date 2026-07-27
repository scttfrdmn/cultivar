// Package plugin_test validates the hf-inference-server plugin manifest offline.
//
// The spec types are re-declared here rather than imported from
// github.com/spore-host/spawn on purpose: importing spawn would pull the AWS SDK
// and sigstore into this module's dependency graph for a YAML parse. This mirrors
// what the standalone spore-host plugin repos do (see
// spore-host/spore-host-plugin-tailscale/validate_test.go).
//
// spawn's own validator is the authority on schema validity — run
// `spawn plugin validate --strict ./hf-inference-server/plugin.yaml` for that.
// These tests cover the invariants spawn cannot know: that the manifest agrees
// with itself and with the engines cultivar claims to support.
package plugin_test

import (
	"os"
	"regexp"
	"strings"
	"testing"

	"gopkg.in/yaml.v3"
)

const manifestPath = "plugin.yaml"

type pluginSpec struct {
	Name        string                 `yaml:"name"`
	Version     string                 `yaml:"version"`
	Description string                 `yaml:"description"`
	Config      map[string]configParam `yaml:"config"`
	Conditions  struct {
		Local  []condition `yaml:"local"`
		Remote []condition `yaml:"remote"`
	} `yaml:"conditions"`
	Permissions *struct {
		Controller struct {
			Env      []string `yaml:"env"`
			Network  bool     `yaml:"network"`
			Commands []string `yaml:"commands"`
		} `yaml:"controller"`
		Instance struct {
			Root    bool     `yaml:"root"`
			Network bool     `yaml:"network"`
			Ports   []int    `yaml:"ports"`
			Files   []string `yaml:"files"`
		} `yaml:"instance"`
	} `yaml:"permissions"`
	Remote struct {
		Install   []step `yaml:"install"`
		Configure []step `yaml:"configure"`
		Start     []step `yaml:"start"`
		Stop      []step `yaml:"stop"`
		Health    struct {
			Interval string `yaml:"interval"`
			Steps    []step `yaml:"steps"`
		} `yaml:"health"`
	} `yaml:"remote"`
	Outputs map[string]struct {
		Description string `yaml:"description"`
		Source      string `yaml:"source"`
	} `yaml:"outputs"`
}

type configParam struct {
	Required    bool        `yaml:"required"`
	Default     interface{} `yaml:"default"`
	Type        string      `yaml:"type"`
	Description string      `yaml:"description"`
}

type condition struct {
	Type    string `yaml:"type"`
	Run     string `yaml:"run"`
	OS      string `yaml:"os"`
	Message string `yaml:"message"`
}

type step struct {
	Type string `yaml:"type"`
	Run  string `yaml:"run"`
}

func load(t *testing.T) *pluginSpec {
	t.Helper()
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		t.Fatalf("read %s: %v", manifestPath, err)
	}
	var spec pluginSpec
	if err := yaml.Unmarshal(data, &spec); err != nil {
		t.Fatalf("parse %s: %v", manifestPath, err)
	}
	return &spec
}

// allSteps returns every step in the manifest, so template and shell checks
// cover the whole surface rather than one phase.
func allSteps(s *pluginSpec) []step {
	var out []step
	out = append(out, s.Remote.Install...)
	out = append(out, s.Remote.Configure...)
	out = append(out, s.Remote.Start...)
	out = append(out, s.Remote.Stop...)
	out = append(out, s.Remote.Health.Steps...)
	return out
}

func TestManifestNameMatchesDirectory(t *testing.T) {
	// spawn resolves github:owner/repo/<name> to <name>/plugin.yaml and requires
	// the directory name to equal the plugin name. If these drift, every install
	// 404s or fails validation.
	spec := load(t)
	if spec.Name != "hf-inference-server" {
		t.Errorf("name = %q, want hf-inference-server (must match the directory)", spec.Name)
	}
}

func TestManifestFitsSpawnSpecCap(t *testing.T) {
	// spawn refuses specs over 64 KiB (maxSpecBytes in pkg/plugin/registry.go).
	info, err := os.Stat(manifestPath)
	if err != nil {
		t.Fatalf("stat: %v", err)
	}
	const cap = 64 * 1024
	if info.Size() > cap {
		t.Errorf("manifest is %d bytes, over spawn's %d-byte cap", info.Size(), cap)
	}
}

func TestRequiredParamsHaveNoDefault(t *testing.T) {
	// spawn rejects "required and has a default" as a contradiction.
	spec := load(t)
	for name, p := range spec.Config {
		if p.Required && p.Default != nil {
			t.Errorf("config %q: required with a default %v — pick one", name, p.Default)
		}
	}
}

func TestModelIsTheOnlyRequiredParam(t *testing.T) {
	// The north star: a user supplies a model id and nothing else. Any new
	// required parameter is a UX regression and should be deliberate.
	spec := load(t)
	var required []string
	for name, p := range spec.Config {
		if p.Required {
			required = append(required, name)
		}
	}
	if len(required) != 1 || required[0] != "model" {
		t.Errorf("required params = %v, want exactly [model]", required)
	}
}

func TestEveryConfigParamIsDocumented(t *testing.T) {
	spec := load(t)
	for name, p := range spec.Config {
		if strings.TrimSpace(p.Description) == "" {
			t.Errorf("config %q has no description", name)
		}
	}
}

var configRefRe = regexp.MustCompile(`{{-?\s*config\.([A-Za-z0-9_]+)`)

func TestTemplateReferencesAreDeclared(t *testing.T) {
	// A {{ config.x }} pointing at an undeclared param renders as an empty string
	// at install time rather than failing loudly, so catch it here.
	spec := load(t)
	for _, st := range allSteps(spec) {
		for _, m := range configRefRe.FindAllStringSubmatch(st.Run, -1) {
			if _, ok := spec.Config[m[1]]; !ok {
				t.Errorf("template references undeclared config %q", m[1])
			}
		}
	}
	for _, c := range spec.Conditions.Remote {
		for _, m := range configRefRe.FindAllStringSubmatch(c.Run, -1) {
			if _, ok := spec.Config[m[1]]; !ok {
				t.Errorf("condition references undeclared config %q", m[1])
			}
		}
	}
}

func TestEveryDeclaredParamIsUsed(t *testing.T) {
	// A declared-but-unreferenced param is dead config that silently does
	// nothing when a user sets it.
	spec := load(t)
	var body strings.Builder
	for _, st := range allSteps(spec) {
		body.WriteString(st.Run)
	}
	for _, c := range spec.Conditions.Remote {
		body.WriteString(c.Run)
	}
	text := body.String()
	for name := range spec.Config {
		if !strings.Contains(text, "config."+name) {
			t.Errorf("config %q is declared but never referenced", name)
		}
	}
}

func TestAllThreeEnginesAreHandled(t *testing.T) {
	// cultivar advertises vLLM, SGLang, and llama.cpp. Each needs its own case in
	// the configure step: they differ in image, flag names, and (for llama.cpp)
	// which HF repo the weights come from.
	spec := load(t)
	var configure strings.Builder
	for _, st := range spec.Remote.Configure {
		configure.WriteString(st.Run)
	}
	text := configure.String()
	for _, engine := range []string{"vllm", "sglang", "llamacpp"} {
		if !strings.Contains(text, engine+")") {
			t.Errorf("configure has no case for engine %q", engine)
		}
	}
	// An unknown engine must fail rather than silently start the default.
	if !strings.Contains(text, "unsupported engine") {
		t.Error("configure does not reject an unknown engine")
	}
}

func TestHealthCheckVerifiesModelIsLoaded(t *testing.T) {
	// "systemctl is-active" would pass while a 64 GB model is still downloading.
	// The endpoint is only usable once /v1/models answers, which can be many
	// minutes after the process starts.
	spec := load(t)
	if len(spec.Remote.Health.Steps) == 0 {
		t.Fatal("no health steps declared")
	}
	var health strings.Builder
	for _, st := range spec.Remote.Health.Steps {
		health.WriteString(st.Run)
	}
	if !strings.Contains(health.String(), "/v1/models") {
		t.Error("health check does not probe /v1/models, so it can pass before the model is loaded")
	}
}

func TestShellStepsAreStrict(t *testing.T) {
	// Without pipefail, a failed download inside a pipeline is reported as a
	// successful install and the endpoint never comes up.
	spec := load(t)
	for i, st := range spec.Remote.Install {
		if !strings.Contains(st.Run, "set -euo pipefail") {
			t.Errorf("remote.install[%d] is missing 'set -euo pipefail'", i)
		}
	}
	for i, st := range spec.Remote.Configure {
		if !strings.Contains(st.Run, "set -euo pipefail") {
			t.Errorf("remote.configure[%d] is missing 'set -euo pipefail'", i)
		}
	}
}

func TestTokenFileIsNotWorldReadable(t *testing.T) {
	// inference.env holds HUGGING_FACE_HUB_TOKEN for gated repos.
	spec := load(t)
	var configure strings.Builder
	for _, st := range spec.Remote.Configure {
		configure.WriteString(st.Run)
	}
	text := configure.String()
	if strings.Contains(text, "inference.env") && !strings.Contains(text, "umask 077") {
		t.Error("inference.env may contain an HF token but is written without a restrictive umask")
	}
}

func TestDeclaredPortMatchesDefault(t *testing.T) {
	// permissions.instance.ports is what `spawn plugin inspect` shows a user
	// before they install, so it must match the port actually opened.
	spec := load(t)
	if spec.Permissions == nil {
		t.Fatal("no permissions block (required by spawn plugin validate --strict)")
	}
	def, ok := spec.Config["port"]
	if !ok {
		t.Fatal("no port config param")
	}
	want, ok := def.Default.(int)
	if !ok {
		t.Fatalf("port default %v is not an int", def.Default)
	}
	var found bool
	for _, p := range spec.Permissions.Instance.Ports {
		if p == want {
			found = true
		}
	}
	if !found {
		t.Errorf("permissions.instance.ports = %v, missing the default port %d",
			spec.Permissions.Instance.Ports, want)
	}
}

func TestNoControllerSecretsRequested(t *testing.T) {
	// This plugin runs entirely on the instance. Requesting controller env
	// passthrough would give plugin shell access to the caller's credentials.
	spec := load(t)
	if spec.Permissions == nil {
		t.Fatal("no permissions block")
	}
	if len(spec.Permissions.Controller.Env) != 0 {
		t.Errorf("controller.env = %v, want empty (this plugin needs no controller secrets)",
			spec.Permissions.Controller.Env)
	}
	if spec.Permissions.Controller.Network {
		t.Error("controller.network is true, but this plugin has no local steps")
	}
}

func TestWeightCacheIsHostMounted(t *testing.T) {
	// Model weights are tens of GB. If the cache lives inside the container, a
	// restart re-downloads everything — expensive in time and, on a metered
	// instance, in money.
	spec := load(t)
	var text strings.Builder
	for _, st := range allSteps(spec) {
		text.WriteString(st.Run)
	}
	if !strings.Contains(text.String(), "/var/lib/cultivar/hf-cache") {
		t.Error("no host-mounted HF cache; a restart would re-download the weights")
	}
}
