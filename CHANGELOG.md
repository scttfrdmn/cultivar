# Changelog

All notable changes to **cultivar** are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `report.v1.json`: the frozen JSON Schema for the report envelope, embedded in the
  binary and returned by `report.Schema()`. Additive within report.v1 — fields may be
  added and enum-valued strings may gain members; consumers must ignore what they do
  not recognize.
- `cultivar schema` prints it, so the contract is obtainable from the binary that
  produced the report in front of you.
- Initial repository scaffold: module, license, CI, and the standard files.

### Fixed
- An `Amount` whose provenance is `unavailable` no longer decodes when it carries a
  number. The value was invisible to every accessor while still sitting in the JSON for
  a consumer to read as a price.
- `subject.observedAt` is normalized to UTC on the wire. It previously serialized with a
  local offset while `generatedAt` serialized as `Z`, so the two timestamps a reader
  compares to judge staleness came out in different frames.
