# CLAUDE.md

Context for picking this project up as an agent. Read this before making changes.

## What this is

A CLI that takes a Hugging Face model id and answers: what is the cheapest,
actually-obtainable way to serve this on AWS — then does it. Part of the
spore.host suite (truffle, spawn, lagotto, inkcap).

**The product is the verdict, not the table.** Default output is one
recommendation in one sentence. The five-mode comparison, provenance tags, and
break-even arithmetic live behind `--explain`. If you find yourself making the
default output denser, you are working against the point: the user does not want
to learn AWS, they want their model served.

The headline command is `compare`, not `deploy`, because self-hosting usually
loses to Bedrock serverless. `deploy` is what happens when comparison says
otherwise.

## Non-negotiables

**Every number is live and provenanced.** No hardcoded AWS rates, ever. Each
numeric field carries `live | derived | external | unavailable`. An unresolvable
price renders as *unpriced*, never as an estimate. This is enforced in CI, not by
convention — stale hardcoded rates have been found four times in this ecosystem
(spawn#447, the SageMaker premium, the Capacity Block discount, and
spore-host/libs#29, where a GPU price was fabricated 12× low with `err == nil`).

**Every report is reproducible.** Embed the inputs: input:output blend ratio,
assumed utilization, the tok/s figure and its source, `generatedAt`, region set,
tool version. Without those the recommendation is unfalsifiable and the history
log can't be compared across runs whose assumptions silently changed.

**Cost control is existential.** Inherited from spawn. Any test that launches a
billable instance requires an explicit TTL, explicit termination, and an
independent leak check afterward. `compare` and everything under it touches only
free read-only APIs — keep it that way.

**Fix pricing bugs upstream in truffle, don't absorb them here.** truffle is the
suite's pricing authority. When it gets a price wrong or can't answer, file an
issue there (see "Upstream" below) and wrap locally as a stopgap — do not patch
truffle from here, and do not quietly reimplement what it should own.

## Layout

- `cmd/cultivar/` — CLI entry (cobra; follow truffle's `docs/flag-conventions.md`:
  root `-o/--output`, `-y/--yes` on destructive verbs, `-r/--regions`)
- `internal/model/` — HF metadata; dtype-aware VRAM sizing
- `internal/bedrock/` — hf-bedrock-map client + token-price resolution
- `internal/ec2/` — on-demand, spot, capacity blocks, quotas
- `internal/sagemaker/` — `Hosting`-preferring price wrapper
- `internal/compare/` — candidate selection, break-even, verdict (the real product)
- `internal/engine/` — vLLM | SGLang | llama.cpp
- `internal/report/` — `report.v1` schema, provenance, renderers
- `internal/history/` — write-once run records
- `hf-inference-server/plugin.yaml` — the spore.host plugin. **Must stay at repo
  root**: spawn resolves `github:owner/repo/<name>` to
  `raw.githubusercontent.com/owner/repo/<ref>/<name>/plugin.yaml`. Specs are
  capped at 64 KiB.
- `cmd/refresh/` — regenerates `docs/` + history (one cron, not two)
- `docs/` — GitHub Pages root. Needs `.nojekyll`.

Standard files only at the top level. **No ROADMAP.md, no docs/plan.md, no
milestone markdown** — work lives in GitHub Issues and the project board.

## Verified AWS traps this tool exists to absorb

Each of these is measured, not theoretical. Issues labeled `correctness-trap`
each encode one with a regression test; don't collapse them.

1. **truffle's default pricer fabricates GPU prices.** `newDefaultOnDemandPricer`
   falls back to a static table with no g7/g7e/g6e/p5/p5e/p6 entries, which then
   *estimates* from vCPU count and returns `err == nil`. g7e.4xlarge → $0.80 vs
   real $4.00; p6-b200.48xlarge → $9.60 vs real $113.93; p5e.48xlarge → $9.60
   despite having **no on-demand price at all**. Always use
   `SetOnDemandPricer(aws.NewAWSOnDemandPricer(cfg))`. (spore-host/libs#29)
2. **Bedrock token price is a blend of two meters across four tiers.** Standard
   input $0.15/1M and output $0.60/1M for Qwen3-32B, but `priority` is 1.75× and
   `flex`/`batch` are 0.5×. Taking the first Price List row yields *priority*.
   The blend also depends on an input:output ratio — 3:1 gives $0.2625/1M, 1:1
   gives $0.375 (+43%), moving g7e.4xlarge's break-even from 4,233 to 2,963 tok/s.
   Note the direction: a pricier blend makes self-hosting *easier* to justify, so
   output-heavy traffic **lowers** the bar. Input-heavy is the conservative end.
   Make the ratio explicit and print it. (truffle#111)
3. **modelId ↔ Price List name is not a key join.** hf-bedrock-map gives
   `qwen.qwen3-32b-v1:0`; Price List keys on `"Qwen3 32B"`. `usagetype` is
   internally inconsistent (`USE1-Qwen3-32B-input-tokens` vs
   `USE1-qwen.qwen3-32b-mantle-output-tokens-standard`). Paginate: 1,013 rows in
   us-east-1 and Qwen3-32B isn't in the first 100.
4. **Bedrock only rescues ~29% of models.** Of 132 reverse-lookupable HF ids, 38
   have a `foundation-model` path; 94 are marketplace-only, where you rent an
   instance and there is **no token price**. Scan `bedrock[]` for any
   foundation-model entry and dedupe; never read `bedrock[0]`.
5. **`safetensors.parameters` must not be summed.** It's dtype-keyed:
   `openai/gpt-oss-120b` is `{BF16: 2.17e9, U8: 118.2e9}` — naive sum × 2 bytes
   gives ~240 GB vs ~122 GB correct. Also `gated: "manual"` blocks deploy without
   `HF_TOKEN`.
6. **Capacity blocks return durations you didn't ask for.** A 24h request in
   us-west-2 returned both a 24h block ($1146.24, +2 days) and a **19h** block
   ($935.30, +25 min). Read `CapacityBlockDurationHours` off each offering.
   Partials are ~3% *pricier* per hour — their advantage is lower outlay and an
   immediate start. 19h is not a multiple of 12 or 24, so don't assume granularity.
7. **A capacity block only wins above 75% utilization.** p5.48xlarge is $41.53
   CB vs $55.04 on-demand, but you prepay the whole block: break-even is 18.1 of
   24 hours. So `deploy` should favor CBs and `benchmark` should refuse them — a
   1–2h benchmark on a CB costs 5–8× more per used hour, and a CB-only type like
   p5e.48xlarge has a $935 floor.
8. **SageMaker is always more expensive than EC2**, but not by one number: +25%
   g-family, +15% p-family, +20% p6-b300. Derive it live. And never take the
   cheapest matching row — `USE1-TrainingPlanUpfrontFee:ml.p4d.24xlarge` is
   $13.57 against a real `Hosting` rate of $25.25, which fabricates a "SageMaker
   is cheaper" result. The right component for inference is `Hosting`.
   truffle#107 is **fixed in v0.48.0** (verified 2026-07-28, correcting an
   earlier note here that it was still open): `SageMakerPriceFor` takes a
   `SageMakerUsage` and the default preference now leads with `Hosting`, so the
   order-dependent $25.9100-vs-$25.2513 flip on ml.p4d.24xlarge is gone and usage
   is part of the cache key. Call `SageMakerPriceFor(..., aws.UsageInference)`
   rather than `SageMakerPrice` — the bare method is only right by convention,
   and naming the usage is what records that this is an endpoint rate and not a
   HyperPod one. So #27 is a thin call plus a regression test, not a wrapper.
9. **`DryRun` is not an availability probe.** `run-instances --dry-run` returns
   "would have succeeded" for instances you cannot actually launch. It validates
   permissions and parameters, not capacity or quota.
10. **Region matters more than expected.** us-east-2 has the full modern GPU
    lineup at the cheapest rates — 61 GPU types across 15 families. us-west-1 has
    9 types across 3: `g4dn`, `p5`, `p5en` (verified 2026-07-28, correcting an
    earlier note here that it had none). The correction matters in the direction
    that costs money: us-west-1 is thin but carries the *expensive* end, so
    p5.48xlarge will serve a 71 GiB model there at $55/hr while no g6e/g7e exists
    to serve it for $3. A fit check alone recommends it happily. Same instance
    varies ~70% across regions. Always include us-east-2.
11. **Quota is the gate that actually stops people.** P-family on-demand quota is
    often 0 by default and an increase takes days. Check it in `compare`, not as
    a deploy-time surprise. Codes: L-417A185B (On-Demand P), L-DB2E81BA
    (On-Demand G/VT), L-7212CCBC (All P Spot).

## Testing without spending money

Everything `compare` touches is read-only and free: Price List,
DescribeInstanceTypes, DescribeCapacityBlockOfferings, GetSpotPlacementScores,
Service Quotas. Money only enters at `deploy` and `benchmark`.

- Golden JSON fixtures for Price List / EC2 / HF — spawn's `substrate` emulator
  covers **none** of the three APIs this tool most depends on.
- `substrate` for the EC2/quota paths truffle already exercises.
- One opt-in `-tags live` suite hitting the free read-only APIs, to catch AWS
  schema drift. (That's how trap #2 would have surfaced early.)

## Environment

`AWS_PROFILE=aws` → account 942542972736. ECR Public alias
`public.ecr.aws/f8g1e7l5`. SageMaker DLC registry `763104351884`.

## Upstream

File issues on the relevant repo rather than working around them here.

Open: truffle#108 (obtainability signal), #109 (capacity blocks), #110 (dropped
per-region errors), #111 (no Bedrock pricing); spawn#447 (stale slurm prices),
#448 (plugin registry + bare-name 404); spore-host/libs#29 (fabricated GPU
prices).

Fixed upstream: truffle#107 (SageMaker component) in v0.48.0 — use
`SageMakerPriceFor` with `aws.UsageInference`; truffle#114 (static pricer
fallback returning a plausible wrong rate); truffle#106 (nil-matcher panic).

**Check the version before writing a stopgap.** These get fixed faster than the
notes here get updated, and #107 sat in this file as "still open" for a day after
v0.48.0 shipped the exact API that was requested. Two of the three fixes above
landed as the shape proposed in the issue, so filing with a concrete API sketch
is worth the effort. When an issue closes, delete the wrapper rather than keeping
both — and re-run the live suite against the new version, which is what catches a
fix that changed behavior in passing.
