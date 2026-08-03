# FND-005 known limitations

Recorded from the FND-005 checkout. Every item is a limitation of this task's
result, not a deferred promise the result already satisfies.

## The task is incomplete, and this is the headline limitation

FND-005 passes only when "stable CPU tensor methods no longer rely on the old
monolithic operation supertrait architecture". They still do:

- `Backend` still requires all nine operation-family supertraits.
- 850 references to those traits remain across 75 files.
- 24 of 174 catalog operations have a canonical CPU executor.
- The stable `Tensor` methods call the family traits, not `dispatch::execute`.

The canonical path is real, exercised and verified, but it is a second path
that runs alongside the legacy one rather than a replacement for it. Anything
that reads this task as ending the dual architecture is reading it wrong.

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

## Conformance coverage

The parity tests cover the 24 migrated identities against the legacy method the
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

`cargo public-api -p incin` reports **756 items, identical to the FND-003 and
FND-004 baseline**. FND-005 does not change the stable facade - which is another
way of saying step 2 of the remaining work has not started, since ending the
supertrait architecture necessarily changes it.

`incin-core` gained `exec::dispatch` and `exec::CanonicalError`. `incin-core` is
an internal `0.0.0` crate and is not a promised public extension surface, so this
is recorded rather than gated. `cargo semver-checks` was not run.

## Diagnostic text

Adding `exec::dispatch` changed how rustc renders one type path in a
compile-fail fixture: `Descriptor` became `incin_core::exec::Descriptor`. The
`.stderr` was re-blessed. The error code (`E0451`) and the three private fields
it names are unchanged, so the fixture still proves the same thing.
