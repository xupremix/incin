# HND-004d validation summary

Branch: `develop`

Implementation source commit: `c8cd8fa` (`docs: fix warning-denied rustdoc links`).

## Feature contract

- `tools/feature-matrix.sh` is the single source of truth for the supported backend/facade feature contract.
- The final run covered 32 explicit compile rows and ended with `=== Supported feature contract matrix: PASS ===`.
- CI and `tools/ci-local.sh` invoke this same matrix.
- Exhaustive cargo-hack powersets remain enabled only for the smaller core/macros/diagnostics spaces. Historical backend and facade Cartesian spaces were approximately 8,212 and 36,608 rows respectively; they are not claimed as the supported contract.
- Accelerator and distributed rows in the supported matrix are compile checks; hardware/runtime behavior remains covered by the focused runtime suites.

## Validation

- `feature-matrix.log` and `final-matrix.log`: supported matrix passed.
- `focused-validation.log`: formatting, core/backend/facade/macros tests, conformance, transformer, distributed compile, and generated-doc checks passed.
- `final-gates.log`: workspace tests, distributed doctests, warning-denied rustdoc, clippy, ledger, budgets, docs, and shape checks passed.
- `generated_docs` passed after regenerating `docs/capabilities.md`.
- The warning-denied rustdoc gate exposed stale intra-doc links; those links were corrected in `c8cd8fa`, and the gate passed on rerun.

The optional fresh 384-row core cargo-hack run was intentionally not completed because core implementation was unchanged and the existing exact core evidence was already available; it is not represented as a new pass here.
