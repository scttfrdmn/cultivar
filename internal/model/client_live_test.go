//go:build live

// Opt-in suite that hits the real Hugging Face Hub. Run with `make test-live`.
// Every call is an anonymous read of public metadata — nothing is downloaded and
// nothing bills.
//
// The point is schema drift. The offline tests pin the sizing arithmetic against
// recorded metadata; these check that the Hub still publishes that metadata in the
// shape the client parses. A renamed field or a newly-nested config would
// otherwise reach users as "cannot size this model" — or worse, as a plausible
// number computed from a zero.
package model

import (
	"context"
	"errors"
	"math"
	"os"
	"testing"
	"time"
)

func liveClient(t *testing.T) *Client {
	t.Helper()
	opts := []Option{}
	if tok := os.Getenv("HF_TOKEN"); tok != "" {
		opts = append(opts, WithToken(tok))
	}
	return NewClient(opts...)
}

func liveCtx(t *testing.T) (context.Context, func()) {
	t.Helper()
	return context.WithTimeout(context.Background(), 30*time.Second)
}

// TestLiveMetadataStillMatchesTheRecordedFixtures is the drift detector. A failure
// is not necessarily a bug — repos do get re-uploaded and configs do change — but
// it means the fixtures in model_test.go and the numbers in CLAUDE.md need
// re-verifying before they are trusted again.
func TestLiveMetadataStillMatchesTheRecordedFixtures(t *testing.T) {
	c := liveClient(t)
	for _, want := range []Model{qwen3_32b, gptOSS120b} {
		ctx, cancel := liveCtx(t)
		got, err := c.Get(ctx, want.ID)
		cancel()
		if err != nil {
			t.Errorf("%s: %v", want.ID, err)
			continue
		}
		if got.Gate != want.Gate {
			t.Errorf("%s gate = %q, recorded %q", want.ID, got.Gate, want.Gate)
		}
		if got.Config != want.Config {
			t.Errorf("%s config = %+v, recorded %+v (re-verify the sizing fixtures)",
				want.ID, got.Config, want.Config)
		}
		for dtype, n := range want.Parameters {
			if got.Parameters[dtype] != n {
				t.Errorf("%s %s params = %d, recorded %d",
					want.ID, dtype, got.Parameters[dtype], n)
			}
		}
		if len(got.Parameters) != len(want.Parameters) {
			t.Errorf("%s reports dtypes %v, recorded %v (a new dtype changes the sizing)",
				want.ID, keysOf(got.Parameters), keysOf(want.Parameters))
		}
		// The assertion that matters: same inputs, same requirement.
		liveTotal := got.Size(SizingRequest{}).Total
		fixtureTotal := want.Size(SizingRequest{}).Total
		lv, lok := liveTotal.Value()
		fv, fok := fixtureTotal.Value()
		if !lok || !fok {
			t.Errorf("%s: live sizable=%v, fixture sizable=%v (%s)", want.ID, lok, fok, liveTotal.Source())
			continue
		}
		if math.Abs(lv-fv) > 0.05 {
			t.Errorf("%s needs %.2f GiB live, %.2f GiB from the fixture", want.ID, lv, fv)
		}
		t.Logf("%s: %.2f GiB at %d tokens", want.ID, lv, got.Config.MaxPositionEmbeddings)
	}
}

func keysOf(m map[string]int64) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// TestLiveGatedModelStillGated guards the deploy precondition. If this repo ever
// reports ungated, the gating check silently stops protecting anyone — and the
// failure it prevents (a GPU that comes up and cannot download) costs money.
func TestLiveGatedModelStillGated(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := liveCtx(t)
	defer cancel()

	const id = "meta-llama/Llama-3.3-70B-Instruct"
	m, err := c.Get(ctx, id)
	if err != nil {
		// Anonymous reads of this repo's metadata succeeded on 2026-07-27. A
		// 401 here would mean the Hub tightened metadata access too, which the
		// client already maps to ErrNotFound — worth reporting, not failing on.
		if errors.Is(err, ErrNotFound) {
			t.Skipf("%s metadata is no longer anonymously readable: %v", id, err)
		}
		t.Fatalf("%s: %v", id, err)
	}
	if !m.Gate.RequiresToken() {
		t.Errorf("%s reports gate %q; it was gated \"manual\" on 2026-07-27. "+
			"If Meta ungated it, update the fixture — but verify, because this check is "+
			"what stops a paid instance launching for an undownloadable model.", id, m.Gate)
	}
	if got := m.Parameters["BF16"]; got != 70553706496 {
		t.Errorf("%s BF16 params = %d, recorded 70553706496", id, got)
	}

	// Without an approved token, the download path must refuse. With one, it must
	// pass — both are real answers, so assert whichever applies.
	err = c.CanRead(ctx, m)
	if os.Getenv("HF_TOKEN") == "" {
		if err == nil {
			t.Error("CanRead passed for a gated repo with no token configured")
		} else {
			t.Logf("no token: %v", err)
		}
		return
	}
	if err != nil {
		t.Logf("HF_TOKEN is set but cannot read %s: %v", id, err)
	}
}

// TestLiveConfigJSONIsStillASeparateCall records why Get makes two requests: the
// API's own config object omits every sizing field. If the Hub ever starts
// including them, the second call becomes removable — this test says so.
func TestLiveConfigJSONIsStillASeparateCall(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := liveCtx(t)
	defer cancel()

	var info modelInfo
	if err := c.getJSON(ctx, "/api/models/Qwen/Qwen3-32B", &info); err != nil {
		t.Fatal(err)
	}
	if info.Config == nil {
		t.Fatal("no config object in the model API response")
	}
	if len(info.Config.Architectures) == 0 {
		t.Error("the API's config object no longer carries architectures")
	}
	// Decode the raw API config into the sizing struct: if any field lands, the
	// second call may be redundant.
	var probe configJSON
	if err := c.getJSON(ctx, "/api/models/Qwen/Qwen3-32B?full=true", &probe); err == nil {
		if probe.NumHiddenLayers != 0 || probe.NumKeyValueHeads != 0 {
			t.Logf("the model API now exposes sizing fields (layers=%d kv_heads=%d); "+
				"the raw config.json call may be droppable",
				probe.NumHiddenLayers, probe.NumKeyValueHeads)
		}
	}
}

// TestLiveUnknownRepoIsNotFound checks the sentinel against the real Hub, since
// "not found" and "not visible to you" are the same response and the CLI's error
// message depends on getting that mapping right.
func TestLiveUnknownRepoIsNotFound(t *testing.T) {
	c := liveClient(t)
	ctx, cancel := liveCtx(t)
	defer cancel()
	if _, err := c.Get(ctx, "cultivar-test/definitely-not-a-real-repo-9f3a"); !errors.Is(err, ErrNotFound) {
		t.Errorf("err = %v, want ErrNotFound", err)
	}
}
