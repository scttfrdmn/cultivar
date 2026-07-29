# cultivar

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Pick a Hugging Face model. Get a working endpoint on AWS. Skip the AWS part.

```bash
cultivar compare Qwen/Qwen3-32B
```
```
Use Bedrock. Self-hosting Qwen/Qwen3-32B would need ~4,200 tok/s sustained
24/7 just to break even against $0.26/1M tokens.

  cultivar deploy Qwen/Qwen3-32B --bedrock
```

Everything between "I want this model" and "here is my endpoint" — instance
families, VRAM math, quota codes, spot placement scores, capacity-block windows,
Price List component filters, Bedrock tier multipliers — is AWS trivia that has
nothing to do with your model. `cultivar` absorbs it and gives you an answer.

## Why the headline command is `compare`, not `deploy`

Because the honest answer is usually **don't self-host.**

Bedrock serves some models serverless at a per-token price. Renting a GPU means
paying by the hour whether or not you're using it. Cross the two and the required
sustained throughput to justify a GPU is enormous — often thousands of tokens per
second, 24 hours a day. Most workloads aren't close.

So `cultivar` tells you the truth first and deploys second. When self-hosting
does win, it launches; when it doesn't, it wires you to Bedrock instead.

## What it compares

Five ways to get inference capacity, on one honest basis:

| Mode | Priced as | Catch |
|---|---|---|
| Bedrock serverless | $/1M tokens | only ~29% of models qualify |
| EC2 on-demand | $/hr | capacity may not exist for GPUs |
| EC2 Spot | $/hr | GPU placement scores are ~1–3/10 |
| Capacity Block for ML | upfront fee | you buy the whole block |
| SageMaker endpoint | $/hr | always costs more than EC2 |

The comparison is **point-in-time**. GPU availability has a shelf life of hours,
so every report is stamped with when it was made, which regions it covers, and
what it assumed. Run it again before you spend money.

## Availability is the hidden dimension

A price for capacity you cannot obtain is fiction, and most cost tables quote
exactly that. `cultivar` treats obtainability as a column, not a footnote: spot
placement score, availability-zone footprint, your account's actual quota
headroom, and whether a capacity block is purchasable right now.

Bedrock's real advantage often isn't price — it's that a token price is
unconditional, while a GPU quote is contingent on capacity, quota, and a
reservation window.

## Every number is live

No hardcoded rates. Each figure in a report carries its provenance
(`live`, `derived`, `external`, `unavailable`), and a price that can't be
resolved renders as *unpriced* rather than as a guess. Stale hardcoded AWS rates
have been found four separate times across this ecosystem; the discipline is
structural here rather than aspirational.

## The report is a contract

Reports are `report.v1`, a frozen JSON Schema you can get from the binary that
produced them:

```bash
cultivar schema | jq '.["$defs"].amount'
```

Compatibility within `report.v1` is additive: fields may be added and
enum-valued strings may gain members — new AWS states are the normal case, not an
error — but nothing is removed, renamed, retyped, or redefined. Consumers must
ignore what they don't recognize, which is why no object in the schema sets
`additionalProperties: false`.

The rule the format exists for: an absent value and zero are not
interchangeable. Money that couldn't be resolved is `null` with provenance
`unavailable`, never `0`. `p5e.48xlarge` genuinely has no on-demand rate, and
reporting that as `$0` — or as the family-based guess that yields $9.60 — is the
specific failure this schema prevents.

Every report also embeds the assumptions that produced it: the traffic mix,
assumed utilization, context length, and the throughput figure with its own
provenance. A recommendation without its inputs is unfalsifiable, and two runs a
day apart can disagree because a price moved, because a region was added, or
because an assumption changed — only the first two are facts about AWS.

## Status

Early. See the [milestones](https://github.com/scttfrdmn/cultivar/milestones) for
what's built and what's next. Work is tracked as issues, not in this file.

## Built on

- [truffle](https://github.com/spore-host/truffle) — instance discovery, pricing, quotas, capacity blocks
- [spawn](https://github.com/spore-host/spawn) — EC2 lifecycle, TTLs, DNS, plugins
- [hf-bedrock-map](https://github.com/scttfrdmn/hf-bedrock-map) — is this model already on Bedrock?

This repo also hosts the `hf-inference-server` spore.host plugin (vLLM, SGLang,
llama.cpp), installable directly:

```bash
spawn plugin install github:scttfrdmn/cultivar/hf-inference-server \
  --instance <id> --config model=Qwen/Qwen3-32B --config engine=vllm
```

## License

Apache-2.0
