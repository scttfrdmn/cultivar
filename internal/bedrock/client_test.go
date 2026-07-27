package bedrock

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// pagesServer is a fake GitHub Pages host. It serves an HTML 404 body for unknown
// paths, which is what the real one does — a JSON decoder pointed at it produces a
// confusing error, so the client must check status before decoding.
type pagesServer struct {
	t     *testing.T
	files map[string]string
	// lastModified is served on every hit; per-model documents carry no
	// generatedAt, so this header is their only freshness signal.
	lastModified string
	paths        []string
}

func newPages(t *testing.T) *pagesServer {
	return &pagesServer{
		t:            t,
		files:        map[string]string{},
		lastModified: "Mon, 27 Jul 2026 07:23:47 GMT",
	}
}

func (p *pagesServer) file(path, body string) *pagesServer {
	p.files[path] = body
	return p
}

func (p *pagesServer) client(opts ...Option) *Client {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		p.paths = append(p.paths, r.URL.Path)
		body, ok := p.files[r.URL.Path]
		if !ok {
			w.Header().Set("Content-Type", "text/html; charset=utf-8")
			w.WriteHeader(http.StatusNotFound)
			_, _ = w.Write([]byte("<!DOCTYPE html><html><head><title>Site not found</title></head></html>"))
			return
		}
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		if p.lastModified != "" {
			w.Header().Set("Last-Modified", p.lastModified)
		}
		_, _ = w.Write([]byte(body))
	}))
	p.t.Cleanup(srv.Close)
	base := []Option{WithEndpoint(srv.URL), WithClock(func() time.Time { return observed })}
	return NewClient(append(base, opts...)...)
}

// The live document for Qwen/Qwen3-32B, 2026-07-27. Marketplace first.
const qwenMapJSON = `{
  "hfId": "Qwen/Qwen3-32B",
  "onBedrock": true,
  "regions": ["us-east-1", "us-east-2", "us-west-1", "us-west-2"],
  "bedrock": [
    {"modelId": "huggingface-reasoning-qwen3-32b", "catalog": "marketplace",
     "confidence": "confirmed", "regions": ["us-east-1","us-east-2","us-west-1","us-west-2"]},
    {"modelId": "qwen.qwen3-32b-v1:0", "catalog": "foundation-model",
     "confidence": "confirmed", "regions": ["us-east-1","us-east-2","us-west-2"]}
  ]
}`

func TestLookupLowercasesThePath(t *testing.T) {
	// The published keys and paths are all lowercase and GitHub Pages is
	// case-sensitive, so a request for the canonical HF id 404s. That failure is
	// silent — it reads as "not on Bedrock" — which is why it gets its own test.
	p := newPages(t).file("/hf/qwen/qwen3-32b.json", qwenMapJSON)
	c := p.client()
	m, err := c.Lookup(context.Background(), "Qwen/Qwen3-32B")
	if err != nil {
		t.Fatal(err)
	}
	if !m.OnBedrock {
		t.Fatal("Qwen/Qwen3-32B reported as not on Bedrock")
	}
	if len(p.paths) != 1 || p.paths[0] != "/hf/qwen/qwen3-32b.json" {
		t.Errorf("requested %v, want /hf/qwen/qwen3-32b.json", p.paths)
	}
}

func TestLookupParsesLiveShape(t *testing.T) {
	c := newPages(t).file("/hf/qwen/qwen3-32b.json", qwenMapJSON).client()
	m, err := c.Lookup(context.Background(), "Qwen/Qwen3-32B")
	if err != nil {
		t.Fatal(err)
	}
	if m.HFID != "Qwen/Qwen3-32B" {
		t.Errorf("HFID = %q, want the case-preserved id from the document", m.HFID)
	}
	if len(m.Entries) != 2 {
		t.Fatalf("got %d entries, want 2", len(m.Entries))
	}
	if m.Entries[0].Catalog != CatalogMarketplace || m.Entries[1].Catalog != CatalogFoundationModel {
		t.Errorf("catalogs = %q, %q; want marketplace then foundation-model (published order)",
			m.Entries[0].Catalog, m.Entries[1].Catalog)
	}
	fms := m.FoundationModels()
	if len(fms) != 1 || fms[0].ModelID != "qwen.qwen3-32b-v1:0" {
		t.Fatalf("foundation models = %+v", fms)
	}
	if fms[0].Confidence != "confirmed" {
		t.Errorf("confidence = %q, want it propagated verbatim", fms[0].Confidence)
	}
	if m.ObservedAt != observed {
		t.Errorf("ObservedAt = %v, want the injected clock", m.ObservedAt)
	}
	if m.Source == "" || !strings.HasSuffix(m.Source, "/hf/qwen/qwen3-32b.json") {
		t.Errorf("Source = %q, want the fetched URL for provenance", m.Source)
	}
	// Per-model documents have no generatedAt, so Last-Modified carries the age.
	want := time.Date(2026, 7, 27, 7, 23, 47, 0, time.UTC)
	if !m.PublishedAt.Equal(want) {
		t.Errorf("PublishedAt = %v, want %v from Last-Modified", m.PublishedAt, want)
	}
	if m.GeneratedAt != (time.Time{}) {
		t.Errorf("GeneratedAt = %v, but per-model documents publish none", m.GeneratedAt)
	}
	if _, ok := m.Age(observed); !ok {
		t.Error("age unknown despite a Last-Modified header")
	}
}

func TestLookup404IsAnAnswerNotAnError(t *testing.T) {
	// Most HF repos are not on Bedrock. If that returned an error, callers would
	// either treat "no Bedrock option" as a failure or start ignoring real failures.
	c := newPages(t).client()
	m, err := c.Lookup(context.Background(), "some-org/some-model")
	if err != nil {
		t.Fatalf("a 404 returned an error: %v", err)
	}
	if m.OnBedrock {
		t.Error("an unmapped repo reported as on Bedrock")
	}
	if m.HasTokenPricing() || m.MarketplaceOnly() {
		t.Error("an unmapped repo reported a Bedrock path")
	}
	if m.HFID != "some-org/some-model" {
		t.Errorf("HFID = %q, want the requested id echoed back", m.HFID)
	}
	if m.Source == "" {
		t.Error("Source is empty; the report should still say what was checked")
	}
}

func TestLookupHTMLErrorPageIsNotDecodedAsJSON(t *testing.T) {
	// GitHub Pages serves HTML for a missing path. A 500 with an HTML body must be a
	// clear error naming the URL, not a JSON decode failure.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
		_, _ = w.Write([]byte("<html>500</html>"))
	}))
	defer srv.Close()
	c := NewClient(WithEndpoint(srv.URL), WithClock(func() time.Time { return observed }))
	_, err := c.Lookup(context.Background(), "a/b")
	if err == nil {
		t.Fatal("a 500 was treated as a valid answer")
	}
	if !strings.Contains(err.Error(), "500") || !strings.Contains(err.Error(), "/hf/a/b.json") {
		t.Errorf("error %q should name the status and the URL", err)
	}
}

func TestLookupRejectsEmptyID(t *testing.T) {
	c := newPages(t).client()
	if _, err := c.Lookup(context.Background(), "  /  "); err == nil {
		t.Error("an empty id was accepted")
	}
}

const indexBodyJSON = `{
  "version": "v1",
  "generatedAt": "2026-07-27T07:23:08Z",
  "regions": ["us-east-1","us-east-2","us-west-1","us-west-2"],
  "note": "Reverse Bedrock<->Hugging Face lookup, US regions only.",
  "count": 2,
  "models": {
    "qwen/qwen3-32b": ` + qwenMapJSON + `,
    "01-ai/yi-1.5-34b": {
      "hfId": "01-ai/Yi-1.5-34B", "onBedrock": true,
      "regions": ["us-east-1","us-east-2","us-west-2"],
      "bedrock": [{"modelId":"huggingface-llm-yi-1-5-34b","catalog":"marketplace",
        "confidence":"confirmed","regions":["us-east-1","us-east-2","us-west-2"]}]
    }
  }
}`

func TestIndexCarriesGeneratedAt(t *testing.T) {
	// The bulk index is the only document with a real generation timestamp, which is
	// what the report ages. One 70 KB fetch also beats 132 round trips.
	c := newPages(t).file("/index.json", indexBodyJSON).client()
	idx, err := c.Index(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(idx) != 2 {
		t.Fatalf("got %d models, want 2", len(idx))
	}
	m := idx["qwen/qwen3-32b"]
	if m == nil {
		t.Fatal("qwen/qwen3-32b missing from the index")
	}
	want := time.Date(2026, 7, 27, 7, 23, 8, 0, time.UTC)
	if !m.GeneratedAt.Equal(want) {
		t.Errorf("GeneratedAt = %v, want %v", m.GeneratedAt, want)
	}
	if d, ok := m.Age(observed); !ok || d != observed.Sub(want) {
		t.Errorf("age = %v (%v), want %v", d, ok, observed.Sub(want))
	}
	// Same traps apply to index entries — they are byte-identical to the per-model
	// documents, verified live.
	if fms := m.FoundationModels(); len(fms) != 1 || fms[0].ModelID != "qwen.qwen3-32b-v1:0" {
		t.Errorf("index entry lost the foundation-model path: %+v", fms)
	}
	if !idx["01-ai/yi-1.5-34b"].MarketplaceOnly() {
		t.Error("01-ai/yi-1.5-34b should be marketplace-only")
	}
}

func TestIndexKeysAreNormalized(t *testing.T) {
	body := strings.Replace(indexBodyJSON, `"qwen/qwen3-32b":`, `"Qwen/Qwen3-32B":`, 1)
	c := newPages(t).file("/index.json", body).client()
	idx, err := c.Index(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if idx["qwen/qwen3-32b"] == nil {
		t.Error("a mixed-case index key was not normalized, so lookups by id would miss it")
	}
}

func TestIndexCountMismatchIsAnError(t *testing.T) {
	// A truncated index makes every missing model read as "not on Bedrock" — a wrong
	// verdict that looks like a normal one. count is the document's own claim about
	// its size, so a mismatch is checkable.
	body := strings.Replace(indexBodyJSON, `"count": 2`, `"count": 132`, 1)
	c := newPages(t).file("/index.json", body).client()
	if _, err := c.Index(context.Background()); err == nil {
		t.Error("an index claiming 132 models while carrying 2 was accepted")
	} else if !strings.Contains(err.Error(), "132") {
		t.Errorf("error %q should state the claimed count", err)
	}
}

func TestIndexMissingIsAnErrorUnlikeAModel(t *testing.T) {
	// A missing per-model file is an answer; a missing index is a broken deployment.
	c := newPages(t).client()
	if _, err := c.Index(context.Background()); err == nil {
		t.Error("a 404 on index.json was treated as an empty mapping")
	}
}

func TestIndexBadGeneratedAtIsAnError(t *testing.T) {
	// Silently dropping an unparseable timestamp would leave the report claiming an
	// unknown age for data that does have one, or worse, aging it from Last-Modified
	// while looking authoritative.
	body := strings.Replace(indexBodyJSON, `"2026-07-27T07:23:08Z"`, `"27 July 2026"`, 1)
	c := newPages(t).file("/index.json", body).client()
	if _, err := c.Index(context.Background()); err == nil {
		t.Error("an unparseable generatedAt was accepted")
	}
}

func TestEntryWithNoRegionsInheritsTheParents(t *testing.T) {
	body := `{"hfId":"a/b","onBedrock":true,"regions":["us-east-1","us-west-2"],
	  "bedrock":[{"modelId":"x.y-v1:0","catalog":"foundation-model","confidence":"validated"}]}`
	c := newPages(t).file("/hf/a/b.json", body).client()
	m, err := c.Lookup(context.Background(), "a/b")
	if err != nil {
		t.Fatal(err)
	}
	if got := m.FoundationModelsIn("us-west-2"); len(got) != 1 {
		t.Errorf("an entry with no region list was treated as offered nowhere, dropping a usable option")
	}
	if got := m.FoundationModelsIn("eu-central-1"); len(got) != 0 {
		t.Error("inheriting the parent's regions must not widen them")
	}
}

func TestContextCancellationPropagates(t *testing.T) {
	c := newPages(t).file("/hf/a/b.json", `{"hfId":"a/b"}`).client()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := c.Lookup(ctx, "a/b"); err == nil {
		t.Error("a cancelled context returned a mapping")
	}
}
