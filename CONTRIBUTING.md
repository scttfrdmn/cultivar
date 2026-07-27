# Contributing

## Ground rules

**Never hardcode an AWS rate.** Every numeric value in a report must come from a
live API and carry its provenance. If a price can't be resolved, the answer is
*unpriced* — not an estimate, not a fallback, not zero. CI enforces this. See
CLAUDE.md for the four separate occasions stale hardcoded rates have burned this
ecosystem.

**Fix pricing bugs upstream.** [truffle](https://github.com/spore-host/truffle) is
the pricing and discovery authority for the spore.host suite. If it returns a wrong
price or can't answer a question, open an issue there and wrap locally as a
stopgap. Don't reimplement what truffle should own, and don't patch truffle from
this repo.

**Keep the default output short.** The product is a one-sentence verdict. Detail
belongs behind `--explain`. A PR that makes the default output denser needs to
justify why.

**Work is tracked in GitHub Issues**, not in markdown files. Don't add
ROADMAP.md, TODO.md, or design docs at the top level — use an issue, and the
milestone it belongs to.

## Cost control

This is existential, inherited from spawn. Any test or manual run that launches a
billable instance must:

1. set an explicit TTL,
2. terminate explicitly when done, and
3. be followed by an independent leak check (list running instances yourself; do
   not trust the teardown path you just exercised).

`compare` and everything beneath it must stay on free read-only APIs. If a change
makes `compare` spend money, that's a design bug.

## Before opening a PR

```bash
make check     # build, vet, gofmt, test
```

- Add a CHANGELOG entry under `[Unreleased]` — part of the change, not a
  follow-up.
- If your change encodes an AWS behavior you verified, add a regression test with
  a golden fixture, and cite the measured evidence in the PR body (actual API
  output, numbers, date). Issues labeled `correctness-trap` each pin one verified
  trap; those tests are the point of the label.
- Tests must not require credentials by default. Live-API tests go behind
  `-tags live`.

## Commits

Conventional commits (`feat:`, `fix:`, `docs:`, `chore:`), matching the rest of
the suite.
