# Intentional limitations carried into 0.1.0

These are deliberate scope boundaries, not defects, and each is already stated
in user-facing documentation. They are repeated here so the release decision is
made against them explicitly.

## Backend maturity

CPU is the complete and verified backend: all 158 backend-executable catalog
operations have CPU executors. CUDA, WGPU, and Metal are previews advertising
70, 46, and 25 operations respectively. Those are advertisement counts, not
verified capability: Metal's executors are described as stubs pending
MTL-002/003, and neither CUDA nor Metal has an execution runner in CI.
`docs/capabilities.md` is generated from the registrations and is authoritative
per operation.

No accelerator hardware was present in this environment, so nothing in this
bundle claims accelerator execution.

## Preview surfaces

- Compiled execution is a feature-gated, preview-only CPU reference evaluator
  under `incin::experimental::compiled`. Its plan snapshots are not a
  deployment format or a portable artifact ABI.
- Distributed execution and ONNX import are experimental or partial where their
  dedicated documentation says so.
- Quantized operations are backend-authoring functionality, not a stable
  `Tensor` method surface. Training through them is not claimed.
- `tuning` (the `autotune` feature) is excluded from the API baselines by
  feature selection and carries no compatibility promise.

## Publication state

0.1.0 has not been published to crates.io. README install snippets point at the
git repository by branch and say so. `cargo-semver-checks` records a skip rather
than a pass until the first release tag exists, which is correct for a first
release but means no semver diff has run against a published predecessor.
