# Project status

This is the concise current-state report for the repository. Historical
foundation evidence remains in `audit-evidence/` and the foundation documents;
it is not repeated here as if it were a current API description.

## Current verified surface

| Area | Current status | Evidence or boundary |
|---|---|---|
| Core tensor execution | Stable CPU tensor methods use exact operation descriptors, validated metadata, and canonical dispatch. | Generated capability and operation-semantics documents; focused and workspace tests. |
| CPU backend | Backend-executable catalog operations have canonical CPU executors. | `audit-evidence/FND-005/cpu-migration-status.md`; accelerator hardware is not implied. |
| Shapes and invariant types | Static, mixed, and dynamic shapes use checked construction. State paths reject empty or dotted components. | Core state tests and macro compile-fail test. |
| Autograd and optimizers | Forward, backward, typed gradients, AdamW updates, rollback, and optimizer state restore are supported on CPU. | `crates/incin/tests/optim_tests.rs` and `transformer_block.rs`. |
| Neural-network layers | Linear, normalization, recurrent, convolutional, activation, loss, and container layers are available at their documented feature tiers. | Layer tests and rustdoc examples. |
| Transformer proof | A four-token, single-head self-attention block with residual MLP is executable on CPU. | `crates/incin/tests/transformer_block.rs`; compile benchmark includes it. |
| Data loading | Zero-worker iteration is lazy and fetches only the next batch; worker-backed loading remains available. | `incin-data` tests. |
| State and serialization | Typed state traversal supports exact snapshots and transactional restore. | State tests and Transformer round-trip proof. |
| Exported snapshots | Export validation includes source coverage, dependency checks, public API checks, and a minimal Cargo check. | `tools/export-snapshot.sh`; generated output policy is documented below. |
| Documentation | Rustdoc, Book examples, and generated operation/capability documents are checked against source. | `docs/README.md`, `mdbook build docs/book`, and project validation commands. |
| Test backends | Every test that needs a backend uses a real one. There is no shape-only stand-in, so a passing test implies the operation both exists and computes. | `crates/incin/tests/consumer-fixtures/dummy-backend-absent`; `crates/incin-core/tests/distributions.rs`. |

## Feature boundaries

- CPU is the verified execution backend in this environment, and the complete
   one: all 164 backend-executable catalog operations have CPU executors.
 - CUDA, WGPU, and Metal are previews advertising 159, 64, and 31 operations
   respectively (counted from the generated matrix, not maintained here).
   Each covers arithmetic, reductions, `matmul`, and convolution/pooling;
   WGPU adds the unary activations, `softmax`, and `rms_norm`, and CUDA adds
   the rest of the normalization family through `batch_norm`, and — unlike
   the other two previews — the loss functions, `embedding`, and `dropout`,
   all with training rows. These are advertisement counts rather than verified
  capability: the Metal shader and MPS infrastructure from MTL-001/002/003
  is complete and the gap is operation coverage on top of it, and neither
  CUDA nor Metal has an execution runner in CI. `docs/capabilities.md` is
  generated from the registrations and is authoritative per operation,
  including its `Training` column.
- Building the workspace does not require `protoc`. The ONNX protobuf module is
  checked in and regenerated with `cargo xtask onnx`.
- `incin::test_utils` gates deterministic fault injection only. The shape-only
  `DummyBackend` is removed, including from the feature that used to carry it.
- The declared MSRV is 1.88, held by a CI job pinned to that toolchain.
- CUDA and Metal are feature-compiled where dependencies permit, but no
  hardware execution claim is made without the device.
- WGPU has a supported software-adapter path for its documented subset.
- Distributed execution and ONNX import remain experimental or partial where
  their dedicated documentation says so. Compiled execution is a separately
  gated preview-only CPU reference evaluator under
  `incin::experimental::compiled`; its plan snapshots are not a deployment
  format or portable ABI.
- Quantized operations are backend-authoring functionality, not a stable
  `Tensor` method surface, and training through them is not claimed.

## Validation vocabulary

- **Verified** means the named command ran successfully on the current tree.
- **Implemented** means source exists, but the report does not claim a broader
  runtime matrix than the evidence names.
- **Partial** means only the documented subset is implemented.
- **Hardware-blocked** means compilation may be checked, but execution needs a
  device or platform library unavailable here.

## Generated documentation policy

The source of truth for generated documents is Rust source and its generator
tests. Regenerate with the commands in `docs/README.md`; do not hand-edit
generated tables. `docs/book/book/` is mdBook build output and is intentionally
ignored by Git. CI builds it from `docs/book/src`, so an export contains the
source tree and not a stale generated directory.

## Historical record

For the foundation sequence, migration counts, archived command logs, and
known caveats, see `docs/FROZEN_FOUNDATIONS.md`, `docs/HANDOFF.md`, and the
corresponding `audit-evidence/FND-*` directories. Those records preserve why a
decision was made; this file describes what a current reader may rely on.
