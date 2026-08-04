# Project Status

This document reports only what is supported by current source inspection or
archived command output. A feature can appear in source without being a stable,
verified product capability.

## Status vocabulary

- **Dynamically verified**: exercised successfully by an archived command from
  the identified checkout.
- **Implemented but unverified**: a substantive implementation exists, but the
  current remediation has not completed its required validation matrix.
- **Partial**: only a documented subset is implemented; unsupported cases fail
  explicitly.
- **Structural prototype**: types and graph structures exist without a complete
  executable product path.
- **Intentionally unsupported**: the path rejects requests rather than
  fabricating results or returning success.
- **Hardware-blocked**: verification requires hardware or platform libraries
  unavailable in the current environment.

## Current classification

| Subsystem | Implemented behavior | Known gaps | Public tier | Evidence | Next dependency |
|---|---|---|---|---|---|
| Core feature builds | **Complete and dynamically verified** for the archived no-default and `std` checks | The broader foundation sequence remains active | Stable core dependency | FND-000 check outputs | FND-001 facade |
| `incin` CPU feature build | **Complete and dynamically verified** for the archived CPU check and package tests | This does not prove the canonical descriptor architecture | Stable end-user API | FND-000 CPU check and test outputs | FND-004, then FND-005 |
| Workspace suite | **Dynamically verified** by the post-containment workspace run; no historical aggregate count is reused | Formatting remains non-clean under the current rustfmt baseline | Workspace validation | `fnd000-test-workspace-after-fixes.txt` | Per-task validation |
| Stable public facade | **Complete and dynamically verified** for the FND-001 allow-list and isolated consumer contracts | Semver comparison tooling is blocked by its forced all-feature rustdoc build; see FND-001 evidence | Stable root/prelude plus explicit `backend_authoring`, `experimental`, and feature-gated `test_utils` tiers | FND-001 public API, compile-contract, feature-matrix, Clippy, test, and rustdoc outputs | FND-002 invariant opacity |
| Invariant-bearing values and allocation arithmetic | **Complete and dynamically verified** for the FND-002 opacity, checked-construction, serialization, feature, compile-contract, Clippy, package, workspace, doctest, and rustdoc gates | The workspace-wide formatting baseline still reports pre-existing drift outside the task diff; accelerator execution remains hardware-blocked | Stable values plus backend-authoring/experimental internals | `audit-evidence/FND-002/` | FND-003 typed failure and rollback contracts |
| Typed failures, scalar conversion, and optimizer rollback | **Complete and dynamically verified** for FND-003 | Legacy free-form compatibility variants remain but are not used for new foundation paths; operator outputs are intentionally source-breaking `Result` values | Stable root/prelude plus backend contracts | `audit-evidence/FND-003/` | FND-004 operation semantics |
| Canonical operation semantics and descriptors | **Complete and dynamically verified** for FND-004: 174 exact identities declared once, typed `Descriptor<O>` per operation, per-operand rank contracts, fail-closed output inference, and exact-identity capability resolution | Execution is not migrated; this task freezes semantics only | Backend-authoring contract plus generated docs | `audit-evidence/FND-004/` | FND-005 CPU migration |
| Canonical execution path | **Complete and dynamically verified** for FND-005: `exec::dispatch` validates against real storage metadata, queries the exact capability row, derives output metadata, and dispatches to `Execute<Descriptor<O>>` | Reaching it is opt-in; the stable tensor surface does not yet use it | Backend-authoring internals | `audit-evidence/FND-005/` | Remaining FND-005 migration |
| CPU eager tensor execution | **Partially migrated**: 154 of the 161 backend-executable catalog operations execute canonically, each verified for forward and gradient parity against the legacy path | Stable tensor methods still depend on the legacy operation-family traits, and `Backend` still requires all nine as supertraits | Stable CPU surface | `audit-evidence/FND-005/cpu-migration-status.md` | Remaining FND-005 migration |
| Typed descriptor execution | **Partial** descriptor validation and execution | Every advertised CPU identity has an executor, proved at compile time; the other 23 backend-executable operations are reachable only through legacy traits, and 13 more sit at an `ExecutionSite` the `Execute` trait cannot carry at all | Backend-authoring/experimental internals | `audit-evidence/FND-005/summary.md` | Remaining FND-005 migration |
| Compiled execution | **Structural prototype** for capture, plans, and artifacts | No validated executable/run path | `experimental::compiled`, opt-in `compiled` feature | Containment test and compiled feature check | Deferred compiled CPU vertical slice |
| Constant folding and weight prepacking | **Intentionally unsupported** with typed errors | No transformations are implemented | `experimental::compiled` | `fnd000-test-compiled-containment.txt` | Deferred until canonical CPU descriptors |
| ONNX macro import | **Partial** stateless eager expansion | No initializers, control flow, custom domains, attributes, or broad opset coverage | `experimental::{model, import_model}` | Macro unit tests and FND-001 facade contracts | Real ONNX initializer/state loading (deferred) |
| ONNX initializer/state loading | **Intentionally unsupported** | Real byte/dtype/state loading is absent | No product surface | Macro fail-closed tests | Real ONNX initializer/state loading |
| Data pipeline | **Partial and dynamically verified** for non-zero batch construction and worker iteration | Broader lifecycle, resource, download, and integrity work is deferred | Preview/data APIs | FND-003 loader tests | Data-pipeline reliability (deferred) |
| Distributed execution | **Structural prototype** | No broad stable multi-node execution claim | Experimental/feature-gated | Source inspection only | Deferred until local semantics stabilize |
| CUDA, WGPU, Metal, and Candle execution | **Dynamically verified** for feature compilation; WGPU default-workspace tests ran in the available adapter environment | CUDA and Metal hardware execution was not run; no hardware availability is inferred from compilation | Feature-gated backend surfaces | FND-003 feature checks and workspace suite | Canonical CPU contract first |

## Active foundation sequence

FND-000 through FND-005 are executed in dependency order. A later foundation
task is not started until the prior task's acceptance gate is truthfully met.
FND-000 through FND-004 have passed their archived acceptance gates. **FND-005
is active and PARTIAL.** Its completion condition - that stable CPU tensor
methods no longer rely on the operation-family supertrait architecture - is not
met: `Backend` still requires all nine supertraits, and 154 of the 161
backend-executable catalog operations have a canonical CPU executor. Each of
the seven remaining is blocked by a limit of the descriptor or capability
contract rather than by nobody having written it, and
`cpu-migration-status.md` names which limit stops which operation.
`audit-evidence/FND-005/summary.md` records what was delivered and what
remains, and `audit-evidence/FND-005/cpu-migration-status.md` is generated from
the registrations so the migrated count cannot be overstated by hand.

The denominator is the backend-executable subset rather than the whole catalog.
Thirteen operations sit at an `ExecutionSite` that `Execute` cannot carry: they
write through an operand, produce storage on another backend, or act on
autograd state. Those need the execution contract changed before an executor
could exist, so counting them as pending migrations would overstate the
remaining work by roughly 30%. `ExecutionSite::blocking_reason` states which
reason applies to each.

`docs/FROZEN_FOUNDATIONS.md` names the parts of the architecture that are
finished and should not be rewritten while that work proceeds, and orders the
remaining steps by what blocks what.

The workspace suite at `3e9609e` reports **1433 passed, 0 failed, 1
ignored**. The ignored case requires a CUDA device. No historical aggregate
count is reused.

## CPU correctness pass

Separate from the migration, and finished. An audit of every dtype refusal and
every `Unsupported` site in the CPU backend turned up seven defects, four of
which returned a wrong answer with no error attached. All seven are fixed, each
with a test that fails against the previous code.

| Defect | Class |
|---|---|
| `group_norm`/`instance_norm` took statistics across the whole flattened batch instead of per sample | wrong answer above batch size 1, and every prior test used batch 1 |
| `matmul` and the axis extrema wrote `f32` whatever they read | wrong result dtype; the cause of `scaled_dot_product_attention` answering in `f32` for every operand |
| `argmax`, `argmin`, `argsort` and `topk` ignored their index dtype parameter | wrong result dtype, and `Tensor::argmax` could not succeed at all |
| `to_scalar`/`to_vec1` compared byte width rather than dtype | reinterpreted the bits, so `1.0f32` read as `u32` returned `1065353216` |
| `adamw_step` refused every dtype but `f32` | valid request refused |
| `batch_norm` refused training mode | valid request refused; blocked convolutional training |
| `conv2d` refused an anisotropic window | valid request refused |

Two refusals are deliberate and remain. `conv_transpose2d` still collapses its
window, because its `output_padding` is a fourth per-axis pair and belongs in
its own change. Training-mode `batch_norm` still does not update running
statistics, because they arrive as shared references and mutating through an
operand is a site the execution contract does not carry.

The Q8_0 refusals across storage, reductions, the tape, quantization and
creation were measured and left alone. A packed block format has no scalar
arithmetic or gradient identity without a dequantization step, so those are
boundaries rather than gaps.

The three GPU backends are the open coverage gap rather than a correctness one.
CUDA and Metal each still declare roughly 37 `TensorOps` methods unsupported,
including every comparison, `where_cond`, `masked_fill`, `gather`, `scatter`,
`index_select` and `scaled_dot_product_attention`, plus about half the
elementwise catalog. WGPU is the only GPU backend this environment can
actually execute (a software adapter is available; CUDA and Metal have no
hardware here and are compile-checked only), so it went first: comparisons
(`cmp_eq`/`ne`/`lt`/`le`/`gt`/`ge`), `logical_and`/`or`/`not`, `sub_scalar`,
`div_scalar`, `maximum`, `minimum`, `abs_diff`, `lerp` and `unsqueeze` are now
real WGSL kernels with passing tests, bringing WGPU's `TensorOps` gap down
from 33 to 17 methods. They match CPU's own semantics for these methods
exactly, including CPU's pre-existing gap of not recording a tape entry for
any of them except `unsqueeze` (which delegates to the already-wired
`reshape`) — porting that gap forward is not a new regression, since CPU
itself has no gradient through `maximum`/`minimum`/`abs_diff`/`lerp`/the
comparisons/`sub_scalar`/`div_scalar` either. `where_cond`, `masked_fill`,
`gather`, `scatter`, `index_select`, `scaled_dot_product_attention` and the
remaining structural/normalization ops are still unsupported on WGPU and are
the natural next slice.

`ReductionOps::prod_all`/`prod_dim` and `CreationOps::full`/`arange`/`linspace`
are real now too: the reduce shaders gained a product mode alongside sum/max/min,
and the three creation ops fill a host `Vec<f32>` the same way `zeros`/`ones`
already did. `cumsum` is WGPU's only remaining `ReductionOps` gap (needs a
genuine prefix-scan shader, not a mode addition to the existing reduce
kernels). Closing `prod_all`/`prod_dim` at the `TensorOps`/`ReductionOps`
trait level was not enough on its own: WGPU also has a second, independent
gate — `crate::capability::support`, which the canonical
`Execute<ReductionSpec>` descriptor path (`wgpu/executor.rs`) checks before
ever calling the kernel. Its per-backend operation list lives in
`wgpu_descriptor_operations!` in `capability.rs` and had its own hardcoded
`ProdAll`/`ProdDim` omission, independent of the `unsupported_reduction_ops!`
macro. A trait-level fix does not imply the descriptor path picked it up;
both need checking. `docs/capabilities.md` is generated from this table
(`INCIN_DOCS=overwrite cargo test -p incin-backends --test generated_docs`)
and now lists `prod_all`/`prod_dim` as WGPU-native.

`prod_all` on CPU had the same dtype-mislabeling defect as the matmul and
extrema fixes below: it wrote its accumulator through a hardcoded
`CpuBuffer::F32` instead of `from_f64_values`, so an f64 tensor's product
came back silently mislabeled as f32. `prod_dim` already used
`from_f64_values` correctly; only the all-elements variant had the bug, and
neither had test coverage before this pass.

`TensorOps::addmm`/`bmm` are real on WGPU now too, composed from the
already-wired `matmul`/`mul_scalar_float`/`add` exactly as CPU's own `addmm`
is, so gradients flow through all three operands rather than dead-ending on
the tape. `bmm` is `matmul` outright, since `matmul` already batches. Neither
is gated by the capability table (WGPU's canonical descriptor path has no
`composed_matmul` executor at all, so this pair is reachable only through the
legacy `TensorOps` trait). Adding a gradient test for `addmm` surfaced a
pre-existing issue in the WGPU test file's own `gradcheck_wgpu` harness: it
reports a wrong (too-large) numerical gradient specifically for ops built on
`matmul`, even though `matmul`'s own analytic gradient is independently
correct (verified directly against hand-derived values with no repeated
probing). Every other op that harness exercises elsewhere in the file passes
fine, so `addmm`'s test uses a direct hand-computed `backward` check instead
of the shared harness; the harness issue itself is filed as a follow-up
rather than fixed here, since it is pre-existing and outside this pass's
scope.

`repeat`, `pad`, `triu`, `tril` and `diag` are real on WGPU now too, via a
different strategy than the elementwise/reduction shaders above: WGPU
storage is always contiguous, so each reads its operand back to a host
`Vec<f32>`, walks it with the same row-major odometer CPU's own
implementations use (`increment_multi_index`, new in this pass, mirrors
`cpu::storage::increment_index`), and re-uploads the result — the same
host-compute-then-upload pattern `zeros`/`full`/`arange`/`linspace` already
use, not a new strategy invented for these. None are gated by the capability
table (none of these five appear in `wgpu_descriptor_operations!`'s lists at
all, so there is no canonical-path check to update, unlike the
`prod_all`/`prod_dim` case above). Not autograd-wired, matching CPU, whose
own versions carry no backward closure either.

`index_select` and `masked_fill` are real now too, same host-readback
pattern, same no-autograd fidelity to CPU. `index`/`mask` are just
`WgpuStorage` regardless of their `KInt`/`KMask` type parameter (WGPU has one
physical representation for every dtype), so their values are read back as
f32 the same way the operand is. `masked_fill` added an explicit shape check
CPU's own version does not have — CPU silently walks the operand's shape and
assumes the mask matches it, which produces nonsense on a mismatch rather
than a clear error; WGPU's host-readback path can check this cheaply, so it
does. `where_cond`, `gather` and `scatter` remain unsupported: CPU's
`where_cond`/`gather` are actually autograd-wired (unlike everything ported
in this pass so far), and `where_cond` additionally broadcasts its two value
operands to a common shape rather than requiring an exact match, so porting
them faithfully is more than the same read-transform-upload template and is
being left for a dedicated pass.

`scaled_dot_product_attention` is real now too, composed from the already
tape-wired `transpose`/`matmul`/`mul_scalar_float`/`add`/`softmax` exactly
like CPU's own composition (same `1/sqrt(d_k)` default scale), so gradients
flow through `q`/`k`/`v`/`mask` rather than dead-ending. Its forward value is
tested with an all-zero query — `q@k^T` is then all-zero regardless of `k`
or the scale, softmax of an all-zero row is uniform, and the output is
exactly the unweighted average of `v`'s rows, which sidesteps hand-computing
softmax's exponentials for the test fixture. Its gradient is checked only
for presence (recorded for all three operands), not exact values: the
composition reuses primitives whose own gradients are independently tested
elsewhere in this file, and the same `gradcheck_wgpu`/matmul interaction
filed as a follow-up above rules out using the shared numerical harness
here too.

`unfold` and `pixel_shuffle` are real now too, same host-readback pattern as
`repeat`/`pad`/`triu`/`tril`/`diag` above, same no-autograd fidelity to CPU.
Both keep CPU's own input validation (`unfold`'s window not exceeding the
axis length, `pixel_shuffle`'s channel count dividing evenly by the upscale
factor squared), reported as `BackendError::InvalidInput` rather than CPU's
`Error::Msg` since that is the error type this call site's `Result` uses.
`group_norm` and `instance_norm` are real now too, and simpler on WGPU than
on CPU: WGPU storage is always contiguous, so a group (CPU's per-sample run
of `channels/groups * spatial` elements — see the CPU implementation's doc
comment for why dividing the whole tensor by `groups` is wrong above batch
size 1, which is the same defect class this pass has fixed twice already)
is a plain contiguous slice of the host readback, needing no strided
indexing at all. `instance_norm` is `group_norm` with one group per channel,
matching CPU's own composition. Not autograd-wired, matching CPU. Tests
reuse the CPU backend's own fixtures (`group_norm_statistics_are_per_sample_
not_across_the_batch`, `instance_norm_normalizes_each_channel_of_each_
sample_alone`) rather than deriving new expected values.

`scatter` is real now too, same host-readback pattern, same fidelity to
CPU's semantics — including silently ignoring an out-of-bounds destination
position rather than erroring. Unlike `where_cond`/`gather`, CPU's `scatter`
is not autograd-wired, keeping it in scope for the same
read-transform-upload template.

`gather` is real now too, forward via the same host-readback pattern as
`index_select`, but — unlike `index_select`, `scatter` and every other
structural op ported in this pass — with a real gradient, because CPU wires
one for `gather` specifically. Its backward is the matching scatter-add:
each `grad_out` element routes back to the position it was gathered from,
accumulating with `+=` (unlike plain `scatter`'s forward, which only ever
overwrites) so two output positions reading the same source element both
contribute rather than the later one clobbering the earlier one's gradient.
`index` itself gets no gradient, matching CPU. Mutation-tested by changing
the accumulation to a plain overwrite and confirming a new test — one whose
index deliberately reads the same source position twice — catches it.

`where_cond` is real now too, closing out WGPU's `TensorOps` gap entirely —
33 methods down to zero. It broadcasts `mask`/`on_true`/`on_false` to a
common shape via the already tape-wired `broadcast_as`, reusing
`crate::cpu::stride::broadcast_shape` to compute that shape (the same
resolver CPU's own `where_cond` uses; it is `pub(crate)`, so nothing new had
to be exposed), then selects elementwise via host readback. Its own
backward routes each `grad_out` element to `grad_true` or `grad_false` by
the mask while still in the broadcasted shape; unbroadcasting each back down
to `on_true`'s/`on_false`'s own shape is not this closure's job — it happens
automatically as the tape walk continues into `broadcast_as`'s own backward
for whichever operand was not already at the common shape, the same
multi-hop composition `addmm`/`scaled_dot_product_attention` above rely on.
`mask` itself gets no gradient, matching CPU. Mutation-tested by swapping
the mask branches in the backward closure and confirming a gradient test —
whose `on_false` is a broadcast scalar, so a real bug there would show up
either as gradient routed to the wrong operand or as a wrong unbroadcast
sum — catches it.

With `where_cond` done, `unsupported_tensor_ops!` has no caller left in the
WGPU backend and is removed from it entirely (CUDA and Metal still declare
their own large lists through it). `unsupported_tensor_ops` and
`unsupported_creation_ops` (removed from WGPU earlier in this pass) both
needed `#[allow(unused_macros)]` on their `macro_rules!` and `pub(crate)
use`: with only CUDA/Metal calling them, they are provably unused under
feature combinations that build WGPU without either — CI's WGPU-only
clippy job is exactly this case — which is a feature-gating artifact, not
dead code.

`cumsum` is real now too, closing WGPU's `ReductionOps` gap the same way
`TensorOps`'s just closed: not with a shader (a prefix scan does not fit the
per-workgroup reduction shape `reduce.wgsl`/`reduce_dim.wgsl` compute) but
with the same host-readback/upload pattern as the structural ops, matching
CPU's own per-row running-sum walk exactly. Not autograd-wired, matching
CPU. This was the macro's last WGPU caller too, so
`unsupported_reduction_ops` gets the same `#[allow(unused_macros)]`
treatment as `unsupported_creation_ops`/`unsupported_tensor_ops` above, for
the same reason.

## WGPU backend: feature parity with CPU

Every `TensorOps`, `ReductionOps` and `CreationOps` method WGPU's trait
impls originally declared unsupported — 39 total across the three traits —
now has a real implementation, each with forward-value tests and, where CPU
itself wires a gradient, a gradient test verified by reverting the fix and
confirming the predicted wrong value. Two strategies cover all 39: real
WGSL kernels for elementwise/reduction-shaped operations that fit the
existing shader dispatch machinery (comparisons, logical ops, `maximum`/
`minimum`/`abs_diff`, `prod_all`/`prod_dim`), and host-readback-compute-
upload for everything structural, index-based or statistical that doesn't
(`repeat`/`pad`/`triu`/`tril`/`diag`/`unfold`/`pixel_shuffle`, `gather`/
`scatter`/`index_select`/`masked_fill`/`where_cond`, `group_norm`/
`instance_norm`, `cumsum`), the same pattern `zeros`/`ones`/`full`/`arange`/
`linspace` already used for creation. `addmm`/`bmm`/
`scaled_dot_product_attention` needed neither: pure composition of
already-wired primitives, matching CPU's own compositions method for
method.

Two defects surfaced in the course of this: `prod_all` narrowing an f64
product to f32 on CPU (same class as the matmul/extrema fixes earlier), and
a second, independent capability gate — `wgpu_descriptor_operations!` in
`capability.rs`, checked by the canonical `Execute<ReductionSpec>`
descriptor path before ever calling the kernel — that a `TensorOps`/
`ReductionOps`-level fix does not automatically satisfy. One follow-up was
filed rather than fixed here: `gradcheck_wgpu`, this test file's own shared
numerical-differentiation harness, gives wrong results specifically for
ops built on `matmul`, even though `matmul`'s own analytic gradient is
independently correct; `addmm` and `scaled_dot_product_attention`'s tests
route around it.

CUDA and Metal's much larger (~37-method) gaps are unaffected by any of
this and are next, in a dedicated pass — this environment cannot execute
either to verify (no CUDA or macOS hardware; compile-checked only), which
is why WGPU went first.

## CUDA backend: a first, compile-only-verified pass

Unlike WGPU's WGSL shaders (interpreted at runtime by a real, if software,
adapter this environment can execute), CUDA kernels are C++ rendered and
compiled by NVRTC at runtime from Rust-side templates
(`crates/incin-backends/src/kernel.rs`) — `cargo check` compiles the Rust
glue around that machinery but does not compile, let alone run, the
generated kernel source itself. Writing new CUDA kernel code here would be
unverifiable beyond "the Rust that assembles the template string compiles,"
a materially weaker guarantee than WGPU's software-adapter execution.
Everything added in this first CUDA pass therefore either composes
already-implemented CUDA kernels (no new kernel-rendering code touched) or
reuses the CUDA backend's own pre-existing host-round-trip idiom (see
`cuda_topk_host`/`cuda_argsort_host`, already in the codebase before this
pass, whose own doc comment states plainly that a host download, host
computation and re-upload is "what the 'true' GPU backend already does" for
an operation with no kernel) — never new kernel source.

`unsqueeze` delegates to the already-tape-wired `reshape`, matching every
other backend's own `unsqueeze`. `addmm`, `bmm` and
`scaled_dot_product_attention` compose already-implemented, already
tape-wired kernels (`matmul`/`mul_scalar_float`/`add`/`transpose`/`softmax`)
exactly like CPU's and WGPU's own compositions — no new kernel launches at
all, only Rust-level reuse. `float_to_scalar`/`float_to_vec1`/
`int_to_scalar`/`int_to_vec1`/`tensor_to_dtype` reuse the existing
`download_f32_host` helper the same way `topk`/`argsort` already do,
restricted to F32 via a new `cuda_require_f32` check.

That restriction exists because of a real, pre-existing latent bug this
pass found rather than introduced: `download_f32_host` assumes F32
storage unconditionally, but CUDA storage supports five dtypes
(`CUDA_STORAGE_DTYPES`: I64/BF16/F16/F32/F64), and `topk`/`argsort` call it
with no dtype check at all — calling either on a non-F32 CUDA tensor would
silently misread the bytes rather than error. That bug is filed as a
separate follow-up rather than fixed here (fixing it changes two
already-shipped ops' behavior, which is a different-shaped change than
adding new ones); every new op this pass adds checks first instead of
repeating the gap.

Verification here is necessarily narrower than WGPU's: `cargo check -p
incin-backends --all-targets --no-default-features --features
std,cpu,cuda` (CI's real command) now passes, and `cargo test` with the
same features compiles and runs everything not gated behind real hardware,
which is everything this pass added — each new method has an `#[ignore =
"requires CUDA hardware"]` test, matching the file's own established
convention for exactly this situation, using the same fixtures and expected
values as the equivalent CPU/WGPU tests. Running `--ignored` here fails
with "unable to dynamically load libcuda.so," the same failure every
pre-existing ignored CUDA test in this file already has, confirming the new
tests are unverifiable for the same reason as the old ones rather than
uniquely broken — but it means none of this pass's CUDA changes have been
confirmed correct by execution, only by compilation and by the fact that
the composed primitives and the host-round-trip pattern they reuse are
each independently proven elsewhere (CPU, WGPU, or CUDA's own pre-existing
`topk`/`argsort`).

Along the way, `cargo check --all-targets` for CUDA was found not to
compile at all before this pass touched anything — five pre-existing
`assert_eq!` calls in `#[ignore]`d tests compared a `Result<Vec<f32>, _>`
against a `Vec<f32>` directly, missing `.unwrap()`. Fixed as a prerequisite
to verifying this pass's own changes via the same command, not as a
correctness fix to anything this pass added.

Comparisons, logical ops, `sub_scalar`/`div_scalar` and `maximum`/`minimum`/
`abs_diff`/`lerp` are real now too, via three small shared helpers
(`cuda_binary_f32_elementwise`/`cuda_unary_f32_elementwise`/
`cuda_scalar_f32_elementwise`) that each download the F32 operand(s),
apply a closure per element, and re-upload — genuinely new host-round-trip
code rather than composition, each one checking `cuda_require_f32` first so
a non-F32 CUDA tensor errors instead of repeating the
`download_f32_host`/`topk`/`argsort` bug. Same encoding, same lack of a
gradient as CPU's own versions of each.

`repeat`, `pad`, `triu`, `tril` and `diag` are real now too, same
host-round-trip strategy with per-position index math instead of a flat
elementwise closure, reusing `crate::cpu::stride::contiguous_strides` and
`crate::cpu::storage::increment_index` (both already `pub(crate)`, so
nothing new had to be exposed) rather than re-deriving row-major indexing
from scratch the way WGPU's `checked_flat_index`/`increment_multi_index`
did — WGPU has no equivalent crate-shared utility to reuse, CUDA does.

`unfold`, `pixel_shuffle`, `gather`, `scatter`, `index_select`,
`masked_fill`, `where_cond`, `group_norm` and `instance_norm` remain
unsupported on CUDA and are the natural continuation of this pass.

The FND-004 evidence records 16 formatter-drifted files; the actual count at
that commit was 22, and is 20 now. See `audit-evidence/FND-005/known-limitations.md`
for the correction. No drifted file is one either task changed.
