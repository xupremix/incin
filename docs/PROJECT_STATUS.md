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
| Mixed-precision autocast | Dispatch-time allowlisted operand casting under the plan's precision policy; installed by `Trainer::fit` / `fit_scaled` for the duration of a run. Master weights stay f32. | Present in this working tree (`incin_core::exec::autocast`); `crates/incin/tests/precision_fixtures.rs`. Not yet a dated commit. |
| Streaming checkpoint load | `ModelExt::load` restores safetensors through a state stream without materializing the whole file. | Present in this working tree (`load_state_streaming`, `SafetensorsStateStream`); `crates/incin-core/tests/streaming_checkpoint.rs`. Not yet a dated commit. |
| Fuzz targets | ONNX parser, state envelope, and GGUF reader have fail-closed fuzz targets under `fuzz/`. | `fuzz/fuzz_targets/*.rs`; `.github/workflows/fuzz.yml` (#48). |

## Feature boundaries

- The catalog currently has 179 canonical operations, 169 of them
  backend-executable and 10 non-backend execution sites. Those counts are
  generated in `docs/operation-coverage.md` and
  `audit-evidence/FND-005/cpu-migration-status.md`; completeness does not mean
  every dtype, layout, or training combination is supported.
- CPU has executors for every backend-executable catalog operation (169 of
  169 advertised in the `docs/capabilities.md` matrix); completeness does not
  mean every dtype, layout, or training combination is supported.
- CUDA, WGPU, and Metal are previews with different operation subsets. The
  current matrix advertises 167 / 137 / 107 operations respectively (CPU 169).
  `docs/capabilities.md` is generated from the registrations and records the
  exact dtype, layout, rank, and training restrictions. These are capability
  declarations, not evidence of hardware execution. In particular, Metal's
  spatial capability group is still empty: convolution and pooling are not
  supported merely because shader and MPS infrastructure exists. The
  scheduled/manual hardware matrix configures Metal execution on macOS runners
  (`HARDWARE_METAL_RUNNER` is optional; the `macos-latest` fallback is real
  Apple Silicon). CUDA execution requires `HARDWARE_CUDA_RUNNER`; scheduled
  CUDA jobs skip when it is unset. Workflow configuration alone does not
  establish successful hardware execution.
- Building the workspace does not require `protoc`. The ONNX protobuf module is
  checked in and regenerated with `cargo xtask onnx`.
- `incin::test_utils` gates deterministic fault injection only. The shape-only
  `DummyBackend` is removed, including from the feature that used to carry it.
- The declared MSRV is 1.88, held by a CI job pinned to that toolchain.
- CUDA and Metal are feature-compiled where dependencies permit, but no
  hardware execution claim is made without the device. Many CUDA value tests
  are `#[ignore = "requires CUDA hardware"]` and only compile-checked in CI.
  Metal's host-side suites run on any OS; real Metal device execution needs
  macOS Apple Silicon (the scheduled `metal` job).
- WGPU has a supported software-adapter path for its documented subset, and
  every-PR CI job `wgpu` ("WGPU Software Adapter Tests") runs that path on
  ubuntu with lavapipe/mesa. WGPU compute (including `matmul`) is still
  `f32`-only; storage admits `u8`/`u32`/`i64`/`bool` but `native_precision`
  refuses non-`f32` compute.
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
work outside this tree (step-by-step procedure in `CONTRIBUTING.md`); until
that happens there is no dated CUDA,
native-WGPU, or multi-rank execution run to name here, and this file does
not claim one. `HARDWARE_METAL_RUNNER` is deliberately optional: while unset
the `metal` job falls back to `macos-latest`, which is real Apple Silicon.
The suites that do run on GitHub-hosted hardware every
schedule - `wgpu-software` (lavapipe) and `metal` (Apple Silicon) - publish
their artifacts whether they pass or fail.

Separately from the weekly matrix, every-PR CI runs a **software-adapter**
WGPU suite (`.github/workflows/ci.yml`, job `wgpu`: mesa/lavapipe on
`ubuntu-latest`). That is execution evidence for the WGPU subset the job
exercises; it is not an NVIDIA device run, and CUDA value tests remain
`#[ignore]`d pending `HARDWARE_CUDA_RUNNER`.

## Dated runs

### 2026-09-22

A single day of landed work closed several long-standing documentation and
implementation gaps. Commits (oldest → newest):

| Commit | What landed |
|---|---|
| `03ec488c` | #121 — WGPU unbroadcast scalar-seed materialization tests |
| `5bba5e40` | #82 — hardware coverage conclusion, per-job artifacts, dated-run docs |
| `93e927b3` | #73 — doctests for `cargo incin doctor` |
| `5fb11e0f` | threat-model + precision-policy research updates |
| `1d75f82f` | unbroadcast rank-deficit fix across CPU/CUDA/WGPU |
| `90feaf0b` | #83 — independent f64 value oracle for CPU conformance |
| `44040a1e` | #73 — doctests for the axis-reduction family |
| `823c0978` | Q4_K GGUF export (ggml two-level quantization) |
| `61b7bcc5` | #84/#86/#87/#88 — eight-op CUDA capability gap |
| `81d90a2e` | exec: seal rule-minted evidence, single-validate lowering |
| `5b6821b9` | kernel-generation strategy survey (#111/#112/#85) |
| `0bd705a4` | #112 step 2 — CMP-005 pointwise fuser |
| `ee0f57c7` | Metal unbroadcast cross-backend rank-deficit semantics |
| `98df2b6c` | #90/#106 — dtype-parametric CUDA matmul, kernelized cast |
| `c9c0d035` | #91 Batch A — 24 runtime-verified WGPU pointwise ops |
| `6db7a3e2` | re-export #112 fuser surface; date research notes |
| `62be4241` | repository update (no behavioral claim) |
| `8e80361d` | #111 — delete 13 stranded codegen modules (~131 KB) |
| `4023ceed` | #85 — cuBLASLt GEMM with fused epilogues |
| `2dee3e2c` | #73 — close remaining doctest debt (47 examples) |
| `98a138fa` | clear the architecture gate the #112/#73 work tripped |
| `631e5a1d` | #91 Batch B — structural, normalization, loss, and matmul rows on Vulkan |
| `b4351902` | this file: record the 2026-09-22 session and post-#111/#85/#91 reality |
| `cf9a2991` | #123 CUDA batch-norm training; #122 bool-mask retirement; #84 CUDA losses/dropout/normals |
| `9e8c2758` | register cast/cublaslt unsafe; refresh the panic inventory |

Verification boundary for this batch:

- **Runtime-verified:** CPU; WGPU Batches A/B (and later C, attention,
  cross-entropy — see 2026-09-23) on the local Vulkan adapter and, for CI,
  the lavapipe software adapter (`c9c0d035`, `03ec488c`, `631e5a1d`,
  `0623e762`); the CPU-JIT half of the #112 fuser (`0bd705a4`).
- **Compile-gated (no device):** CUDA cuBLASLt value tests
  (`4023ceed`, 6 `#[ignore]`d); CUDA dtype-parametric matmul numerics
  (`98df2b6c`, 6 `#[ignore]`d); CUDA losses/dropout/normals/batch-norm
  (`cf9a2991`, `#[ignore]`d suites under `crates/incin-backends/tests/cuda_*`);
  Metal unbroadcast and Metal host-side suites (`ee0f57c7`, host-side only);
  #112 NVRTC dispatch tail (`0bd705a4`, `#[ignore]`d).
- **Hardware runs:** `HARDWARE_CUDA_RUNNER` and
  `HARDWARE_WGPU_RUNNER` both still unset as of 2026-09-22, so no dated
  CUDA or native-WGPU hardware job executed; see Hardware runs above.

### 2026-09-23

Continued gap-closure on the same branch (oldest → newest):

| Commit | What landed |
|---|---|
| `70ff62be` | complete KV-cache, transpose contract, routing ops, and fuzz suite |
| `a680d3dd` | merge `master` into `feat/custom-autograd-dtype` |
| `36ce4f13` | workspace all-features compile, rustfmt drift, panic inventory |
| `cbe7bac5` | #92 — wire Metal Batch-A pointwise `Execute` and advertise the rows |
| `90a71eb7` | #103/#113 — accept NonZero/GroupedMatMul/routing baselines |
| `0623e762` | #91 — broadcast Dyn matmul batch dims; land WGPU attention e2e |
| `7bab5e09` | #101 — `CrossAttention` with rectangular causal mask and memory prefill |
| `7f38cb38` | #97/#99 — DP synchronizer seam and FSDP/ZeRO-1/ZeRO-2 execution |
| `cb915f71` | #102 — typed MoE/Router with dense masked routing |
| `bf042c80` | #92 — Metal normalization, layout, tril/triu, dropout, linear, SDPA |
| `a741225d` | export the new surface; refresh baselines and inventories |
| `67d4c075` | metal-only build without `cpu`; restore `Trainer` `UnwindSafe` |

Fuzz targets committed with this day: `fuzz/fuzz_targets/{onnx_parser,state_envelope,gguf_reader}.rs`
(#48 threat-model work; CI workflow `.github/workflows/fuzz.yml`).

Verification boundary for this batch:

- **Runtime-verified:** CPU; WGPU attention e2e (non-causal, causal,
  grouped-query, rotary, and a training smoke that `backward`s to finite
  non-zero projection gradients — `crates/incin-backends/tests/wgpu_attention.rs`);
  WGPU cross-entropy with backward (`wgpu_cross_entropy.rs`); WGPU Batch B/C
  rows on the software adapter (`631e5a1d`, later suites).
- **Compile-gated / host-side:** Metal Batch-A pointwise and the #92
  normalization/SDPA rows (`cbe7bac5`, `bf042c80` — `metal_gap_ops` /
  `metal_attention_ops` run host-side math on Linux; real device execution is
  the macOS Apple-Silicon hardware job only).
- **Hardware runs:** `HARDWARE_CUDA_RUNNER` / `HARDWARE_WGPU_RUNNER` remain
  unset; `HARDWARE_METAL_RUNNER` optional with `macos-latest` fallback.

### 0.2.0 theme status (as of this tree)

| Theme | Status | Evidence / boundary |
|---|---|---|
| WGPU training path | Advertised and partially runtime-verified | Matrix advertises 137 ops (155 rows / 123 trainable); losses, norms, dropout, embedding, conv/pool, attention all present. Runtime evidence: Batches A/B/C, attention e2e + backward, cross-entropy backward on Vulkan/lavapipe. Compute still `f32`-only; `cmp_*`/`logical_*` unadvertised. |
| Metal coverage | Expanded, still preview and mostly host-verified | Matrix advertises 107 ops (119 rows / 89 trainable), including losses, norms, dropout, embedding, SDPA, linear, tril/triu, and the thirteen unary activations. Spatial (`conv2d`/pooling) still empty; `instance_norm`, `masked_fill`/`where_cond`, `cmp_*`/`logical_*` unadvertised. Device runs only on macOS Apple Silicon. |
| Autocast / mixed precision (#2) | Present in this working tree | `incin_core::exec::autocast`; `Trainer::fit` / `fit_scaled` install it for the run (`train.rs`). `precision_fixtures` asserts `mixed_bf16_autocasts_allowlisted_dispatch_operands` and the f16 twin; f16-without-scaling still rejected (`TrainError::UnsupportedPrecision`). Not yet a dated commit. |
| Streaming checkpoints (#13) | Present in this working tree | `ModelExt::load` restores via `load_state_streaming` / `SafetensorsStateStream` without materializing the whole file (`serialize.rs`, `state.rs`); test `crates/incin-core/tests/streaming_checkpoint.rs`. Not yet a dated commit. |
| Fusion groups (#112) | Landed | CMP-005 legality-checked pointwise fuser (`0bd705a4`, `compiled/fusion.rs`); book section in `deep_lowering.md`. NVRTC dispatch tail still `#[ignore]`d. |
| Fuzzing (#48) | Landed | `fuzz/fuzz_targets/{onnx_parser,state_envelope,gguf_reader}.rs`, workflow `fuzz.yml`. |
| Broadcast/mask hardening | Landed | #100 `BroadcastShape` bounds + compile-fail fixtures; #122 bool-mask retirement (`cf9a2991`). |
| Distributed FSDP/ZeRO (#99) | Landed (CPU-proven lowering) | `7f38cb38`; host-side two-rank protocol on CPU. No NCCL transport ships; multi-rank still needs `HARDWARE_CUDA_RUNNER` (#82). |

### Uncommitted working-tree note

This checkout has a large uncommitted diff on `feat/custom-autograd-dtype`
(autocast module, streaming load, Metal/WGPU/CUDA capability and test
additions, workflow comment edits, generated-doc refreshes). Counts above
come from the **current tree** (generated docs regenerated from it). Commits
noted as "present in this working tree" are not yet in `git log`; do not
cite them as landed until they are.

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
