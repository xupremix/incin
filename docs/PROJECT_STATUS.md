# Project status

This is the concise current-state report for the repository. Historical
foundation evidence remains in `audit-evidence/` and the foundation documents;
it is not repeated here as if it were a current API description.

## Implemented surface and evidence

The references below identify implementation and test coverage; they do not
claim that every named test has run on the reader's current checkout.

| Area | Current status | Evidence or boundary |
|---|---|---|
| Core tensor execution | Stable CPU tensor methods use exact operation descriptors, validated metadata, and canonical dispatch. | Generated capability and operation-semantics documents; focused and workspace tests. |
| CPU backend | Backend-executable catalog operations have canonical CPU executors. | `audit-evidence/FND-005/cpu-migration-status.md`; accelerator hardware is not implied. |
| Shapes and invariant types | Static, mixed, and dynamic shapes use checked construction. State paths allow an empty root; `try_child` rejects empty or dotted components. | `docs/INVARIANT_TYPES.md`; core state tests and macro compile-fail test. |
| Autograd and optimizers | Forward, backward, typed gradients, AdamW updates, rollback, and optimizer state restore are supported on CPU. | `crates/incin/tests/optim_tests.rs` and `transformer_block.rs`. |
| Neural-network layers | Linear, normalization, recurrent, convolutional, activation, loss, and container layers are available at their documented feature tiers. | Layer tests and rustdoc examples. |
| Transformer layers | `MultiHeadAttention` (grouped-query, rotary), `FeedForward`, and the encoder/decoder layer pair compose into a decoder-only model that trains on CPU. The earlier hand-composed block remains as the oracle they are checked against. | `crates/incin/tests/gpt_decoder_model.rs`, `transformer_layers.rs`, and `transformer_block.rs`; compile benchmark includes the last. |
| Data loading | Zero-worker iteration is lazy and fetches only the next batch; worker-backed loading remains available. | `incin-data` tests. |
| State and serialization | Typed state traversal supports exact snapshots and transactional restore. | State tests and Transformer round-trip proof. |
| Exported snapshots | Export validation includes source coverage, dependency checks, public API checks, and a minimal Cargo check. | `tools/export-snapshot.sh`; generated output policy is documented below. |
| Documentation | Rustdoc, Book examples, and generated operation/capability documents are checked against source. | `docs/README.md`, `mdbook build docs/book`, and project validation commands. |
| Test backends | Every test that needs a backend uses a real one. There is no shape-only stand-in, so a passing test implies the operation both exists and computes. | `crates/incin/tests/consumer-fixtures/dummy-backend-absent`; `crates/incin-core/tests/distributions.rs`. |

## Feature boundaries

- CPU has executors for every backend-executable catalog operation. The current
  counts are generated in `docs/operation-coverage.md` and
  `audit-evidence/FND-005/cpu-migration-status.md`; completeness does not mean
  every dtype, layout, or training combination is supported.
- CUDA, WGPU, and Metal are previews with different operation subsets.
  `docs/capabilities.md` is generated from the registrations and records the
  exact dtype, layout, rank, and training restrictions. These are capability
  declarations, not evidence of hardware execution. In particular, Metal's
  spatial capability group is empty: convolution and pooling are not supported
  merely because shader and MPS infrastructure exists. The scheduled/manual
  hardware matrix configures Metal execution on macOS runners. CUDA execution
  requires `HARDWARE_CUDA_RUNNER`; scheduled CUDA jobs skip when it is unset.
  Workflow configuration alone does not establish successful hardware execution.
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

## Hardware runs

`.github/workflows/hardware.yml` is the scheduled hardware matrix: it runs
every Monday at 05:00 UTC and on `workflow_dispatch` with a `job` input
(`all`, `cuda`, `wgpu`, `metal`, `dist2-network`, `multinode`). The `select`
job resolves which suites have a registered runner, and each hardware job
that executes uploads its own artifact named `hardware-<job>-<attempt>`
holding a dated result record (`result-<job>.md`: UTC timestamp, run id and
attempt, requested suite, runner OS/arch, runner-variable state, job
conclusion) plus the suite log. A dated execution claim cites the run id,
the job, and that artifact; a job that was skipped uploads nothing, because
it executed nothing.

Runner labels come from the `HARDWARE_CUDA_RUNNER` and
`HARDWARE_WGPU_RUNNER` repository variables. Both were unset as of
2026-09-22 (check with `gh api repos/xupremix/incin/actions/variables`), so
the `cuda`, `wgpu-native`, `dist2-network`, and `multinode` jobs conclude
*skipped* - they never queue on a label no runner carries - and the final
`Hardware Coverage Conclusion` job fails the run with the coverage verdict
*skipped, not success* rather than letting it read as a pass. Registering a
self-hosted runner and setting those two variables is repository-settings
work outside this tree; until that happens there is no dated CUDA,
native-WGPU, or multi-rank execution run to name here, and this file does
not claim one. The suites that do run on GitHub-hosted hardware every
schedule - `wgpu-software` (lavapipe) and `metal` (Apple Silicon) - publish
their artifacts whether they pass or fail.

## Dated runs

### 2026-09-22

A single day of landed work closed several long-standing documentation and
implementation gaps. Commits (oldest → newest):

| Commit | What landed |
|---|---|
| `03ec488c` | #121 — WGPU unbroadcast scalar-seed materialization tests |
| `1d75f82f` | unbroadcast rank-deficit fix across CPU/CUDA/WGPU |
| `93e927b3` | #73 — doctests for `cargo incin doctor` |
| `44040a1e` | #73 — doctests for the axis-reduction family |
| `5fb11e0f` | threat-model + precision-policy research updates |
| `823c0978` | Q4_K GGUF export (ggml two-level quantization) |
| `61b7bcc5` | #84/#86/#87/#88 — eight-op CUDA capability gap |
| `81d90a2e` | exec: seal rule-minted evidence, single-validate lowering |
| `5b6821b9` | kernel-generation strategy survey (#111/#112/#85) |
| `0bd705a4` | #112 step 2 — CMP-005 pointwise fuser |
| `ee0f57c7` | Metal unbroadcast cross-backend rank-deficit semantics |
| `98df2b6c` | #90/#106 — dtype-parametric CUDA matmul, kernelized cast |
| `c9c0d035` | #91 Batch A — 24 runtime-verified WGPU pointwise ops |
| `6db7a3e2` | re-export #112 fuser surface; date research notes |
| `8e80361d` | #111 — delete 13 stranded codegen modules (~131 KB) |
| `4023ceed` | #85 — cuBLASLt GEMM with fused epilogues |
| `2dee3e2c` | #73 — close remaining doctest debt (47 examples) |

Verification boundary for this batch:

- **Runtime-verified:** CPU; WGPU Batch A on the local Vulkan adapter
  (`c9c0d035`, `03ec488c`); the CPU-JIT half of the #112 fuser
  (`0bd705a4`).
- **Compile-gated (no device):** CUDA cuBLASLt value tests
  (`4023ceed`, 6 `#[ignore]`d); CUDA dtype-parametric matmul numerics
  (`98df2b6c`, 6 `#[ignore]`d); Metal unbroadcast (`ee0f57c7`,
  host-side only); #112 NVRTC dispatch tail (`0bd705a4`,
  `#[ignore]`d).
- **Hardware runs:** `HARDWARE_CUDA_RUNNER` and
  `HARDWARE_WGPU_RUNNER` both still unset as of 2026-09-22, so no dated
  CUDA or native-WGPU hardware job executed; see Hardware runs above.

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
