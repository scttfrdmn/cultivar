package bedrock

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// DefaultMapEndpoint is the hf-bedrock-map v1 API root.
const DefaultMapEndpoint = "https://scttfrdmn.github.io/hf-bedrock-map/api/v1"

// maxMapBytes caps a response. index.json is ~70 KB for 132 models; 8 MB leaves
// room for the mapping to grow an order of magnitude and still bounds a broken or
// hostile response.
const maxMapBytes = 8 << 20

// Client reads the hf-bedrock-map static API.
type Client struct {
	endpoint string
	http     *http.Client
	now      func() time.Time
}

// Option configures a [Client].
type Option func(*Client)

// WithEndpoint overrides the API root, for fixtures and tests.
func WithEndpoint(endpoint string) Option {
	return func(c *Client) { c.endpoint = strings.TrimRight(endpoint, "/") }
}

// WithHTTPClient overrides the HTTP client.
func WithHTTPClient(h *http.Client) Option {
	return func(c *Client) { c.http = h }
}

// WithClock overrides the clock, so fixtures produce stable observation times.
func WithClock(now func() time.Time) Option {
	return func(c *Client) { c.now = now }
}

// NewClient returns an hf-bedrock-map client.
func NewClient(opts ...Option) *Client {
	c := &Client{
		endpoint: DefaultMapEndpoint,
		http:     &http.Client{Timeout: 30 * time.Second},
		now:      time.Now,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// mappingJSON mirrors a per-model document and an index entry, which are byte-identical
// (verified live for qwen/qwen3-32b on 2026-07-27).
type mappingJSON struct {
	HFID      string   `json:"hfId"`
	OnBedrock bool     `json:"onBedrock"`
	Regions   []string `json:"regions"`
	Bedrock   []struct {
		ModelID    string   `json:"modelId"`
		Catalog    string   `json:"catalog"`
		Confidence string   `json:"confidence"`
		Regions    []string `json:"regions"`
	} `json:"bedrock"`
}

func (j mappingJSON) toMapping(fallbackID string) *Mapping {
	id := j.HFID
	if id == "" {
		id = fallbackID
	}
	m := &Mapping{
		HFID:      id,
		OnBedrock: j.OnBedrock,
		Regions:   j.Regions,
		Entries:   make([]Entry, 0, len(j.Bedrock)),
	}
	for _, b := range j.Bedrock {
		regions := b.Regions
		if len(regions) == 0 {
			// An entry with no region list inherits the parent's. Treating it as
			// "offered nowhere" would drop a usable option.
			regions = j.Regions
		}
		m.Entries = append(m.Entries, Entry{
			ModelID:    b.ModelID,
			Catalog:    Catalog(b.Catalog),
			Confidence: Confidence(b.Confidence),
			Regions:    regions,
		})
	}
	return m
}

// indexJSON mirrors the bulk index.
type indexJSON struct {
	Version     string                 `json:"version"`
	GeneratedAt string                 `json:"generatedAt"`
	Regions     []string               `json:"regions"`
	Count       int                    `json:"count"`
	Models      map[string]mappingJSON `json:"models"`
}

// Lookup resolves one HF repo id.
//
// A 404 is a definitive answer, not an error: it means the repo is not served by
// Bedrock in the mapped regions, which is the common case (94 of 132 mapped repos
// have no token price, and the great majority of HF repos are not mapped at all).
// Returning an error for it would make the ordinary path look like a failure and
// tempt callers into ignoring real failures.
func (c *Client) Lookup(ctx context.Context, hfID string) (*Mapping, error) {
	key := normalizeID(hfID)
	if key == "" {
		return nil, fmt.Errorf("empty model id")
	}
	url := c.endpoint + "/hf/" + key + ".json"

	var doc mappingJSON
	published, found, err := c.getJSON(ctx, url, &doc)
	if err != nil {
		return nil, err
	}
	if !found {
		return &Mapping{
			HFID:       hfID,
			OnBedrock:  false,
			ObservedAt: c.now().UTC(),
			Source:     url,
		}, nil
	}

	m := doc.toMapping(hfID)
	m.PublishedAt = published
	m.ObservedAt = c.now().UTC()
	m.Source = url
	return m, nil
}

// Index fetches the bulk mapping, keyed by lowercased HF id.
//
// Used by `cmd/refresh` and by any comparison over more than a handful of models:
// one 70 KB fetch beats 132 round trips. Unlike a per-model lookup it carries
// generatedAt, so mappings from here have a real age rather than a Last-Modified
// approximation.
func (c *Client) Index(ctx context.Context) (map[string]*Mapping, error) {
	url := c.endpoint + "/index.json"
	var doc indexJSON
	published, found, err := c.getJSON(ctx, url, &doc)
	if err != nil {
		return nil, err
	}
	if !found {
		return nil, fmt.Errorf("GET %s: 404; the bulk index should always exist", url)
	}

	generated := published
	if doc.GeneratedAt != "" {
		t, err := time.Parse(time.RFC3339, doc.GeneratedAt)
		if err != nil {
			return nil, fmt.Errorf("GET %s: parse generatedAt %q: %w", url, doc.GeneratedAt, err)
		}
		generated = t.UTC()
	}
	observed := c.now().UTC()

	out := make(map[string]*Mapping, len(doc.Models))
	for key, entry := range doc.Models {
		m := entry.toMapping(key)
		m.GeneratedAt = generated
		m.PublishedAt = published
		m.ObservedAt = observed
		m.Source = url
		out[normalizeID(key)] = m
	}
	// count is the mapping's own claim about its size. A mismatch means the document
	// was truncated or generated inconsistently, and silently proceeding would make
	// a partial index look complete — every missing model reads as "not on Bedrock".
	if doc.Count != 0 && doc.Count != len(out) {
		return nil, fmt.Errorf("GET %s: index claims %d models but carries %d", url, doc.Count, len(out))
	}
	return out, nil
}

// getJSON fetches and decodes a document. The bool reports whether it exists; a
// 404 is (false, nil).
func (c *Client) getJSON(ctx context.Context, url string, out any) (published time.Time, found bool, err error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return time.Time{}, false, err
	}
	req.Header.Set("Accept", "application/json")

	resp, err := c.http.Do(req)
	if err != nil {
		return time.Time{}, false, fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusNotFound {
		return time.Time{}, false, nil
	}
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return time.Time{}, false, fmt.Errorf("GET %s: %d: %s", url, resp.StatusCode, strings.TrimSpace(string(body)))
	}

	data, err := io.ReadAll(io.LimitReader(resp.Body, maxMapBytes))
	if err != nil {
		return time.Time{}, false, fmt.Errorf("GET %s: read body: %w", url, err)
	}
	if err := json.Unmarshal(data, out); err != nil {
		// GitHub Pages serves an HTML 404 page for some missing paths and JSON for
		// others; say which URL failed so the cause is obvious.
		return time.Time{}, false, fmt.Errorf("GET %s: decode: %w", url, err)
	}

	// Per-model documents carry no generatedAt, so Last-Modified is the only
	// freshness signal on that path.
	if lm := resp.Header.Get("Last-Modified"); lm != "" {
		if t, err := http.ParseTime(lm); err == nil {
			published = t.UTC()
		}
	}
	return published, true, nil
}
