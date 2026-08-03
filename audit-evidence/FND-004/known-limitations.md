# FND-004 known limitations

Recorded from the FND-004 checkout. Every item here is a limitation of this
task's result, not a deferred promise that the result already satisfies.

## Formatter

`cargo fmt --all -- --check` exits `1`. This is **pre-existing drift outside the
FND-004 diff** and is not repaired here, because the run's scope forbids broad
repository formatting cleanup.

- Full output: `test-results/fmt-workspace-gate.txt`
- 16 files report drift; none of them is a file FND-004 changed.
- Every Rust file in the FND-004 diff is separately proved formatted-clean:
  `rustfmt --edition 2024 --check` over exactly the changed file list exits `0`
  with empty output.

The criterion "workspace formatter is clean" is therefore **BLOCKED** by
pre-existing drift; the criterion "this task's files are formatted" **PASSES**.

During this task, running `rustfmt` on a crate root was found to recurse through
the whole `mod` tree and reformat unrelated files. Six such files
(`compiled/artifact.rs`, `dist/mod.rs`, `nn/param.rs`, `serialize.rs`,
`tensor/dtype.rs`, `tensor/mod.rs`) were reverted to their committed content so
that no unrelated formatting entered this commit.

## Hardware and platform

- **CUDA**: `cargo check -p incin-backends --no-default-features --features cuda`
  exits `0`. This proves **compilation only**. No CUDA device or driver is
  present in this environment and **no CUDA kernel was executed**. The CUDA
  hardware conformance test remains `#[ignore]`.
- **Metal**: `cargo check -p incin-backends --no-default-features --features metal`
  exits `0`. The host is Linux. This proves **compilation only**; no Metal
  runtime, device, or kernel execution is claimed. The previously archived
  `E0425` failure in `metal/executor.rs` is fixed and no longer reproduces.
- **WGPU**: a software adapter is available here, so WGPU descriptor tests did
  execute. This is environment-specific and is not a portability claim.
- **Candle**: `--features external-candle` compiles; no execution is claimed.

## Scope deliberately left to FND-005

FND-004 freezes semantics and descriptors. It does **not** migrate execution.

- The CPU descriptor executors still cover a subset of the catalog
  (`Add`/`Sub`/`Mul`/`Div`, `BroadcastAs`, `ReshapeExact`, the 14 registered
  reductions, `Conv2dExact`, `MaxPool2d`, `AvgPool2d`, `MatMulExact`) and still
  delegate to the legacy operation-family traits internally.
- Stable `Tensor` methods still call the legacy operation-family traits. Ending
  that dual architecture is FND-005's defining task.
- Broad legacy family capability rows (`Pointwise`, `Reduction`, `Reshape`,
  `MatMul`, `Conv2d`, `Pool2d`, `Storage`, `Fill`, `Random`, `Normalization`,
  `Broadcast`) remain registered for legacy callers. They can no longer make an
  **exact** query supported  -  `an_exact_query_never_resolves_through_a_broad_family_row`
  regression-tests that  -  but they are still present and FND-005 removes them.

## Conformance vector coverage

`SEMANTIC_CONFORMANCE_VECTORS` is a frozen **minimum** set of ten vectors, one
per required semantic class. It is not per-operation coverage. FND-005 is
responsible for the full CPU vector set and the finite-difference gradient
checks; this task only fixes the shared vector *shape* that those suites consume.

## Semver tooling

`cargo semver-checks` is not run here. `cargo public-api -p incin` is run and
archived instead (`test-results/public-api-gate.txt`); the `incin` facade is
**unchanged** by FND-004 at 756 public items.

`incin-core`'s `exec::catalog` module removed the `DTypeRule::BooleanResult`
enum variant, which no operation constructed and which no `DTypeId` could
represent (there is no boolean dtype). `incin-core` is an internal `0.0.0` crate
and is not a promised public extension surface, so this is recorded rather than
gated.
