# FND-005 known limitations

Recorded from the FND-005 checkout. Every item is a limitation of this task's
result, not a deferred promise the result already satisfies.

## Remaining limitations after canonical CPU migration

All 158 backend-executable catalog operations now use exact descriptor
execution from the stable tensor path. The old operation-family traits remain
only as backend-local adapters for fused kernels, host readback, tracing, and
compatibility tests. They are not a second stable tensor execution path.

Thirteen catalog entries use execution sites that the current `Execute` contract
cannot carry. They mutate through an operand, produce storage on another
backend, or act on autograd state. Their blocking reasons are recorded in the
generated CPU migration status and are not counted as missing CPU executors.

## What the gradient checks prove, and what they do not

The four finite-difference checks in `cpu::canonical::tests` use the repository's
existing `gradcheck`, which **ignores any element whose absolute difference is
below `1e-3`**. They therefore catch a gradient that is structurally wrong -
missing, misrouted, or wrongly scaled - and do not resolve differences finer
than that ceiling.

The step size is `1e-2` rather than the more usual `1e-4`, and that choice is
deliberate. Every function checked is a polynomial of degree at most two, so a
central difference has no truncation error; the only error is f32 cancellation,
which *shrinks* as the step grows. At `1e-4` the same gradients came out about
1% off from cancellation alone. Loosening the tolerance to absorb that would
have hidden real errors of the same magnitude, so the step was fixed instead.

Exact agreement between the canonical and legacy gradients is asserted
separately and without tolerance, by `canonical_and_legacy_gradients_are_identical`.
That assertion was mutation-tested: comparing the wrong operand's gradient makes
it fail.

## Result dtype mislabelling, resolved

This section recorded three CPU kernels that returned storage carrying a dtype
the caller did not ask for and was not told about. They were found by measuring
every advertised dtype against the dtype of the storage that came back, rather
than by reading the code. All three are fixed, and the entry is kept because
the shape of the defect is worth not reintroducing.

- `scaled_dot_product_attention` returned `f32` for a `u8`, `u32`, `i64`,
  `bf16`, `f16` or `f64` operand. It has no dtype handling of its own: it is
  composed from `matmul` and `softmax`, and both narrowed. `matmul` wrote every
  result into an f32 buffer even though its read path was already generic and
  it already accumulated a widened operand in f64, and `max_axis_with_indices`
  did the same, which reached `softmax` through `log_softmax`. Both convert
  through the operand's own buffer now.
- `topk` built its value buffer as `f32` whatever the operand held. It converts
  through the operand's buffer too, which is what let its capability row stop
  being narrower than its kernel.
- `argmax`, `argmin`, `argsort` and `topk` took an index dtype as a type
  parameter and ignored it, the first two always building `i64` and the other
  two always `u32`. They build the dtype they were asked for, out of `u8`,
  `u32` and `i64`, and check that the indices fit rather than truncating.

The canonical rows had been narrowed to what each kernel labelled correctly, so
the canonical path refused these requests rather than answering them wrongly.
That was a containment measure and not a fix, and it is undone: the requests are
forwarded now, and `an_index_reduction_produces_the_index_dtype_it_was_asked_for`
asserts the produced dtype rather than the absence of an error.

The index defect was worse than a mislabel. `Tensor::argmax` types its result
`u32` while the kernel filled `i64`, so the frontend rejected the storage its
own backend had just produced and the public method could not succeed at all.

## Rank bounds are measured, then narrowed to the descriptor

Every rank bound in `descriptor_min_rank` and `descriptor_max_rank` was first
measured by executing the operation at ranks zero through four and recording
the lowest and highest that succeeded. Where the descriptor's own validator is
stricter than the kernel, the narrower bound is registered: `instance_norm`
tolerates less than four axes in the kernel but `InstanceNormAttributes`
requires exactly `[batch, channels, height, width]`, and a row wider than its
own validator advertises requests that can never reach the backend.

The reverse case, a kernel stricter than its descriptor, would be a real gap.
None was found among the migrated identities, but the checks are per-operation
and prove nothing about the ones not yet migrated.

## Conformance coverage

The parity tests cover the migrated identities against the legacy method the
catalog names as each one's source. They are **parity** tests, not reference-vector
tests: they prove the canonical path computes what the legacy path computes, not
that the legacy path is numerically correct against an external oracle. The
full CPU reference vector set FND-005 calls for does not exist yet.

`SEMANTIC_CONFORMANCE_VECTORS` remains the frozen ten-vector minimum from
FND-004, one per semantic class. It is not per-operation coverage.

## Formatter

`cargo fmt --all -- --check` exits `1`. This is **pre-existing drift outside the
FND-005 diff**; the run's scope forbids broad repository formatting cleanup.

- **20 files report drift; none is a file FND-005 changed.** The intersection of
  the drift set and the FND-005 changed-file list is empty, and every Rust file
  in the FND-005 diff is separately proved formatted-clean:
  `rustfmt --edition 2024 --check` over exactly that list exits `0`.

### Correction to FND-004's recorded formatter count

FND-004's `summary.md` and final report state **16** drifted files. The actual
drift at FND-004's own commit was **22**. The recorded number came from a gate
run captured before two later steps in that task, so it does not describe the
tree that was committed:

- Six files - `compiled/artifact.rs`, `dist/mod.rs`, `nn/param.rs`,
  `serialize.rs`, `tensor/dtype.rs`, `tensor/mod.rs` - were reverted to their
  committed, drifted content *after* the log was taken, so their drift is
  missing from it. They are unchanged by FND-005 and are drifted now.
- Two files - `metal/executor.rs` and `wgpu/executor.rs` - were formatted after
  the log was taken, so the log records drift they no longer have.

Net: FND-004 understated the drift at its own commit by six files. The
substantive claim it made - that none of the drifted files were files that task
changed - still holds, and holds for FND-005 too. The count did not.

As in FND-004, running `rustfmt` on a crate root recurses the whole `mod` tree
and reformats unrelated files. Every formatting run in this task was over an
explicit file list, and the working tree was compared before and after each one
to confirm no unrelated file changed.

## Hardware and platform

- **CUDA**: `cargo check -p incin-backends --no-default-features --features cuda`
  exits `0`. **Compilation only.** No CUDA device or driver is present and no
  CUDA kernel was executed. The hardware conformance test remains `#[ignore]`.
- **Metal**: the same check exits `0` on a Linux host. **Compilation only**; no
  Metal runtime, device, or kernel execution is claimed.
- **WGPU**: a software adapter is available here, so WGPU descriptor tests did
  execute. Environment-specific, not a portability claim.
- **Candle**: `--features external-candle` compiles; no execution is claimed.

No canonical `Execute<Descriptor<op::X>>` executor was written for CUDA, WGPU or
Metal. Their exact capability rows are unchanged apart from the training view
rows described in `summary.md`, and they continue to execute through the grouped
legacy adapters.

## Public API

`cargo public-api -p incin` reports **1,164 items**. FND-005 does not change the
stable facade, which is another way of saying the step that ends the supertrait
architecture has not started, since doing so necessarily changes it.

`incin-core` gained `exec::dispatch` and `exec::CanonicalError`. `incin-core` is
an internal `0.0.0` crate and is not a promised public extension surface, so this
is recorded rather than gated. `cargo semver-checks` was not run.

## Diagnostic text

Adding `exec::dispatch` changed how rustc renders one type path in a
compile-fail fixture: `Descriptor` became `incin_core::exec::Descriptor`. The
`.stderr` was re-blessed. The error code (`E0451`) and the three private fields
it names are unchanged, so the fixture still proves the same thing.
