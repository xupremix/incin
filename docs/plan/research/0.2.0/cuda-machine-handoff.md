Handoff to a CUDA machine

State of this branch. Six pieces of work, all green on CPU, none of them ever
run against a GPU.

Landed here:

1. `log_softmax` and `logsumexp` as catalog operations (#103).
2. `scatter_add` as a catalog operation, with a new
   `DuplicateIndexRule::Accumulate` (#103). CPU only, deliberately: its row
   claims a fixed summation order, and an atomics-based CUDA kernel cannot
   honour that claim.
3. `.dtype::<K>()` documented in the target-API chapter of the book, which had
   only ever covered `.dtype_dynamic`.
4. `conformance/fixtures.rs` split into `fixtures/{mod,contracts,families}.rs`
   after it crossed the 1200-line architecture gate.
5. A `released-consumer` fixture and a scheduled-only CI job that builds against
   the published crate rather than the working tree.
6. The three research notes beside this file.

Verified on this machine: full workspace suite, clippy, fmt, the conformance
oracle at 161 covered operations, the architecture and large-file gates, five
xtask gates, five Python gates, and both `no_std` builds. Not verified: anything
touching CUDA, WGPU or Metal, because this machine has no `nvidia-smi`.

What to do first on the GPU box, in order.

**Run the nine CUDA suites that have never executed.** They are
`cuda_executor.rs`, `cuda_reduce_ops.rs`, `cuda_optimizer.rs`,
`cuda_embedding.rs`, `cuda_shape_dtypes.rs`, `collective_tuning.rs`,
`nccl_contract.rs`, `nccl_two_rank.rs` and `candle_executor.rs`, all under
`crates/incin-backends/tests/`. CI runs `cargo check` on the `cuda` feature and
nothing else, so every expectation in those files is unchecked. Expect real
findings. This is the cheapest high-yield action available and it needs no new
code.

**Recount the gap before believing any issue.** See
`cuda-verification-state.md`: CUDA is missing five rows against CPU, not the 88
that #105 states. Four of the five are the operations this branch just added.
The CUDA breadth issues (#86, #87, #88, #106) appear to be largely closed by
commits that predate this branch, and should be re-read against
`docs/capabilities.md` rather than trusted.

**Then pick one of two directions**, not both.

Verification: port the conformance oracle to CUDA. That is the other half of
#83, and it converts 165 advertised rows into 165 rows that have answered for
themselves. `advertised_tuples(DeviceKind)` is already backend-parameterised;
what is CPU-specific is the operand builder in `conformance/operands.rs` and the
`run_cpu_self_check` entry point.

Codegen: adopt `src/codegen/` for one operation family. See
`codegen-adoption.md`. Both emitters already end at the same NVRTC dispatcher,
so this is a swap of the source producer rather than new infrastructure, and CUDA
pointwise is the smallest end-to-end proof. If it does not land, deleting
`codegen/` is the honest alternative.

Open questions carried over, none of them blocking:

- Should `DuplicateIndexRule` become `#[non_exhaustive]`? The breaking change is
  already in flight on this branch, so this is the free moment.
- `quantized_matmul` has a three-way contract disagreement: the catalog gives it
  `OutputRule::MatMul` (rhs `[K, N]`), the CPU kernel reads rhs as `[N, K]`, and
  the CUDA declaration says `[K, N]` while its `.cu` takes the left operand as
  `const float*` against a `q8_0` descriptor. It is the one operation the oracle
  cannot cover. Worth its own issue.
- CMP-005 has no task file and no issue. See `compiled-fusion-lowering.md`.
- `PRF-007` step 5 is checked and the tree contradicts it. See
  `codegen-adoption.md`.

Repository conventions that are easy to trip over: the ledger under `docs/plan/`
is closed and new work goes to GitHub issues; five documents are generated and
must be regenerated with `INCIN_DOCS=overwrite` rather than edited; the panic
audit keys on `file:line` so unrelated edits redden it; and `cargo xtask
feature-matrix` runs `cargo-hack` and can fill a small disk.
