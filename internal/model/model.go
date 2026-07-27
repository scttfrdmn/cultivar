// Package model resolves a Hugging Face repo id to the facts that decide what
// hardware can serve it: how much GPU memory the weights need, how much the KV
// cache adds at a given context length, and whether the repo can be downloaded at
// all.
//
// The sizing here is the reason the whole package exists. An early draft of this
// project paired Qwen/Qwen3-32B (61 GiB in BF16) with a g5.2xlarge (22.9 GiB of
// VRAM). The model cannot load; the recommendation was worse than useless because
// it looked authoritative. Emitting that must be impossible, so every candidate
// instance is checked against a VRAM requirement derived from the repo's own
// metadata, and a requirement that cannot be derived is reported as unknown rather
// than estimated.
package model

import (
	"fmt"
	"math"
	"sort"
	"strings"
	"time"

	"github.com/scttfrdmn/cultivar/internal/report"
)

// GiB is one binary gigabyte. GPU memory is reported in MiB by
// DescribeInstanceTypes, and model weights in bytes by the HF API, so everything
// here works in bytes and converts once at the edge.
const GiB = 1024 * 1024 * 1024

// Gate describes whether a repo can be downloaded without extra credentials.
type Gate string

const (
	// GateNone is a public repo.
	GateNone Gate = "none"
	// GateAuto requires an accepted license, granted automatically on request.
	GateAuto Gate = "auto"
	// GateManual requires a human to approve access. meta-llama/Llama-3.3-70B-Instruct
	// is the canonical case: without an approved token the download 401s *after*
	// the GPU is running, so the user pays for an instance that can never serve.
	GateManual Gate = "manual"
)

// RequiresToken reports whether serving this repo needs an HF token. Checked at
// compare time as a caveat and at deploy time as a hard precondition.
func (g Gate) RequiresToken() bool { return g == GateAuto || g == GateManual }

// Model is the resolved metadata for one Hugging Face repo.
type Model struct {
	// ID is the canonical repo id, e.g. "Qwen/Qwen3-32B".
	ID string

	// Gate is the access requirement.
	Gate Gate

	// Parameters is safetensors.parameters verbatim: a count per dtype. It is a
	// map and not a total on purpose — see [DTypeBytes].
	Parameters map[string]int64

	// Architectures from config.json, e.g. ["Qwen3ForCausalLM"].
	Architectures []string

	// HasSafetensors reports whether the repo publishes safetensors weights. A
	// GGUF-only repo is a different serving path (llama.cpp) with different sizing,
	// and cannot be sized from safetensors.parameters at all.
	HasSafetensors bool

	// QuantMethod from config.json's quantization_config, e.g. "mxfp4". Empty when
	// the checkpoint is unquantized.
	QuantMethod string

	// Config holds the sizing fields from config.json. Zero values mean the field
	// was absent, which makes the KV cache unsizable rather than zero.
	Config Config

	// ObservedAt is when this metadata was read.
	ObservedAt time.Time
}

// Config is the subset of config.json needed for memory sizing.
type Config struct {
	MaxPositionEmbeddings int    // context length ceiling
	HiddenSize            int    // model dim
	NumHiddenLayers       int    // layer count
	NumAttentionHeads     int    // query heads
	NumKeyValueHeads      int    // KV heads; < NumAttentionHeads means GQA
	HeadDim               int    // per-head dim; derived from hidden/heads when absent
	TorchDType            string // checkpoint dtype, e.g. "bfloat16"
}

// EffectiveHeadDim returns head_dim, falling back to hidden_size/num_attention_heads
// when the field is absent. Reports false when neither is derivable.
//
// The fallback is not always right — Qwen3-32B publishes head_dim 128 while
// hidden_size/heads is 80 — which is why the explicit field wins when present.
func (c Config) EffectiveHeadDim() (int, bool) {
	if c.HeadDim > 0 {
		return c.HeadDim, true
	}
	if c.HiddenSize > 0 && c.NumAttentionHeads > 0 {
		return c.HiddenSize / c.NumAttentionHeads, true
	}
	return 0, false
}

// WeightBytes returns the GPU memory the weights occupy, summing each dtype's
// count times its own width.
//
// Returns unavailable — never a guess — when the repo publishes no
// safetensors.parameters, or when any dtype key is unrecognized. A single unknown
// key makes the whole total wrong by an unknown factor, so a partial sum would be
// worse than no answer: it would be plausible.
func (m Model) WeightBytes() report.Amount {
	if len(m.Parameters) == 0 {
		if !m.HasSafetensors {
			return report.Unavailable(report.UnitGiB,
				m.ID+" publishes no safetensors weights; size it from the GGUF variant instead")
		}
		return report.Unavailable(report.UnitGiB,
			m.ID+" reports no safetensors.parameters; cannot size the weights")
	}

	var total float64
	dtypes := make([]string, 0, len(m.Parameters))
	for dtype := range m.Parameters {
		dtypes = append(dtypes, dtype)
	}
	sort.Strings(dtypes) // deterministic source strings

	for _, dtype := range dtypes {
		width, ok := DTypeBytes(dtype)
		if !ok {
			return report.Unavailable(report.UnitGiB,
				fmt.Sprintf("%s reports an unrecognized dtype %q; guessing its width would be wrong by an integer factor", m.ID, dtype))
		}
		total += float64(m.Parameters[dtype]) * width
	}

	parts := make([]string, 0, len(dtypes))
	for _, dtype := range dtypes {
		width, _ := DTypeBytes(dtype)
		parts = append(parts, fmt.Sprintf("%s x %gB", dtype, width))
	}
	return report.Live(total/GiB, report.UnitGiB,
		fmt.Sprintf("HF safetensors.parameters for %s (%s)", m.ID, strings.Join(parts, " + ")),
		m.ObservedAt)
}

// TotalParameters returns the parameter count across all dtypes. Useful for
// display ("32.8B parameters") but never for sizing — that is [Model.WeightBytes].
func (m Model) TotalParameters() int64 {
	var total int64
	for _, n := range m.Parameters {
		total += n
	}
	return total
}

// KVCacheBytesPerToken returns the KV cache cost of a single token across all
// layers, at the given cache dtype width in bytes.
//
// Formula: 2 (K and V) x layers x kv_heads x head_dim x width. Note kv_heads, not
// attention heads: Qwen3-32B has 64 query heads against 8 KV heads, so using the
// wrong one overstates the cache 8x.
func (m Model) KVCacheBytesPerToken(dtypeWidth float64) report.Amount {
	headDim, ok := m.Config.EffectiveHeadDim()
	if !ok || m.Config.NumHiddenLayers == 0 || m.Config.NumKeyValueHeads == 0 {
		return report.Unavailable(report.UnitCount,
			m.ID+" config.json lacks the layer/head fields needed to size the KV cache")
	}
	perToken := 2 * float64(m.Config.NumHiddenLayers) * float64(m.Config.NumKeyValueHeads) * float64(headDim) * dtypeWidth
	return report.Live(perToken, report.UnitCount,
		fmt.Sprintf("KV cache per token for %s = 2 x %d layers x %d kv_heads x %d head_dim x %gB",
			m.ID, m.Config.NumHiddenLayers, m.Config.NumKeyValueHeads, headDim, dtypeWidth),
		m.ObservedAt)
}

// SizingRequest describes the serving configuration being sized.
type SizingRequest struct {
	// ContextTokens is the context length to reserve cache for. Zero means the
	// model's max_position_embeddings, which is often far larger than needed:
	// Qwen3-32B's 40,960-token ceiling costs ~10 GiB of cache, so capping context
	// is a real lever on which instances qualify.
	ContextTokens int

	// Concurrency is the number of simultaneous full-context sequences to hold.
	// Zero is treated as 1. This is the parameter that makes cache dominate: at
	// concurrency 8, Qwen3-32B's cache exceeds its weights.
	Concurrency int

	// KVCacheDTypeBytes is the cache element width. Zero defaults to 2 (fp16/bf16,
	// what vLLM and SGLang use unless told otherwise).
	KVCacheDTypeBytes float64

	// OverheadFraction reserves room for activations, CUDA graphs, fragmentation,
	// and the framework itself, as a fraction of weights plus cache. Zero defaults
	// to [DefaultOverheadFraction].
	OverheadFraction float64
}

// DefaultOverheadFraction is the headroom reserved beyond weights and KV cache.
//
// 15% is a deliberately blunt allowance for activations, CUDA graph capture,
// allocator fragmentation, and the framework's own footprint. It is not measured,
// which is why [Sizing.Total] carries derived rather than live provenance and why
// the fraction is an explicit field on [SizingRequest] rather than a constant
// buried in the arithmetic. Being wrong here is much cheaper than being wrong
// about dtype widths: it shifts a fit decision at the margin instead of by 2x.
const DefaultOverheadFraction = 0.15

// Sizing is the memory a serving configuration needs, broken out so a report can
// show which component dominates.
type Sizing struct {
	Weights  report.Amount // GiB
	KVCache  report.Amount // GiB
	Overhead report.Amount // GiB
	Total    report.Amount // GiB

	ContextTokens int
	Concurrency   int
}

// Size computes the GPU memory required to serve this model under req.
//
// Unavailability propagates: if the weights cannot be sized, the total is
// unavailable, and the caller reports "cannot size this model" instead of
// recommending hardware. That is the whole design — a missing input must not
// become a confident number.
func (m Model) Size(req SizingRequest) Sizing {
	ctx := req.ContextTokens
	if ctx <= 0 {
		ctx = m.Config.MaxPositionEmbeddings
	}
	conc := req.Concurrency
	if conc <= 0 {
		conc = 1
	}
	width := req.KVCacheDTypeBytes
	if width <= 0 {
		width = 2
	}
	overheadFraction := req.OverheadFraction
	if overheadFraction <= 0 {
		overheadFraction = DefaultOverheadFraction
	}

	s := Sizing{ContextTokens: ctx, Concurrency: conc}
	s.Weights = m.WeightBytes()

	if ctx <= 0 {
		s.KVCache = report.Unavailable(report.UnitGiB,
			m.ID+" has no max_position_embeddings and no explicit context length was given")
	} else {
		perToken := m.KVCacheBytesPerToken(width)
		s.KVCache = report.Convert(perToken, float64(ctx)*float64(conc)/GiB, report.UnitGiB,
			fmt.Sprintf("KV cache for %d tokens x %d concurrent", ctx, conc))
	}

	base, err := report.Sum(report.UnitGiB, "weights + KV cache", s.Weights, s.KVCache)
	if err != nil {
		// Both operands are constructed with UnitGiB above, so this is unreachable;
		// surface it rather than dropping it if that ever changes.
		s.KVCache = report.Unavailable(report.UnitGiB, "KV cache: "+err.Error())
		s.Overhead = report.Unavailable(report.UnitGiB, "overhead: "+err.Error())
		s.Total = report.Unavailable(report.UnitGiB, "total: "+err.Error())
		return s
	}

	s.Overhead = report.Scale(base, overheadFraction,
		fmt.Sprintf("activation/framework overhead at %.0f%%", overheadFraction*100))
	total, err := report.Sum(report.UnitGiB, "total GPU memory", base, s.Overhead)
	if err != nil {
		s.Total = report.Unavailable(report.UnitGiB, "total: "+err.Error())
		return s
	}
	s.Total = total
	return s
}

// FitsIn reports whether a GPU pool of the given size can hold this sizing, and
// why not when it cannot.
//
// An unsizable model returns false: refusing to recommend is correct when the
// requirement is unknown, since the failure mode being prevented is a confident
// recommendation of hardware that cannot load the weights.
func (s Sizing) FitsIn(availableGiB report.Amount) (bool, string) {
	need, ok := s.Total.Value()
	if !ok {
		return false, "cannot size this model: " + s.Total.Source()
	}
	have, ok := availableGiB.Value()
	if !ok {
		return false, "unknown GPU memory: " + availableGiB.Source()
	}
	if need > have {
		return false, fmt.Sprintf("needs %.1f GiB, has %.1f GiB", need, have)
	}
	return true, ""
}

// Headroom returns the unused GPU memory after this sizing, as a fraction of the
// pool. Negative when the model does not fit, which is more informative than a
// boolean: -0.3 says "30% over" rather than just "no".
func (s Sizing) Headroom(availableGiB report.Amount) report.Amount {
	need, nok := s.Total.Value()
	have, hok := availableGiB.Value()
	if !nok {
		return report.Unavailable(report.UnitFraction, "cannot size this model: "+s.Total.Source())
	}
	if !hok {
		return report.Unavailable(report.UnitFraction, "unknown GPU memory: "+availableGiB.Source())
	}
	if have == 0 {
		return report.Unavailable(report.UnitFraction, "GPU memory pool is zero: "+availableGiB.Source())
	}
	return report.Derived(math.Round((have-need)/have*1e6)/1e6, report.UnitFraction,
		fmt.Sprintf("headroom after %.1f GiB of %.1f GiB", need, have))
}
