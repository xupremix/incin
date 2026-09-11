# Decisions taken, and the options not taken

## D1. Base branch
**Taken:** continue on `feat/custom-autograd-dtype`.
**Not taken:** rebuild on master (duplicates ~5000 lines); branch off the feature
branch (keeps history separate, but splits review across two branches).
**Back out:** `git checkout master` and cherry-pick the fix commits.

## D2. How `training = true` gets verified (F6)
**Taken:** extend the existing CPU conformance oracle. Run every training tuple
inside `GradMode::Enabled` and compare `tape::depth()` before and after. New
`Verdict::RecordedNothing`.
**Not taken:**
- A parallel purpose-built test. Rejected: the oracle already enumerates the
  capability rows into their product; a second enumeration would drift from it.
- A `differentiable` column on the capability rule. Rejected for now: whether an
  operation carries gradient is a property of the *operation*, not of a
  backend's row, so a per-row column is the wrong shape. It would also churn
  four capability tables, `docs/capabilities.md`, and the public-API baseline.
**Back out:** revert the `conformance/mod.rs` hunk; nothing else depends on it.

## D3. What `training = true` means
**Taken:** "this operation runs under training", not "this operation carries
gradient". So an operation whose derivative is genuinely zero (`sign`, `floor`,
`ceil`, `round`, `trunc`) may record nothing and still be honest.
**Consequence:** `RecordedNothing` cannot be a finding on its own. It needs a
declared list of operations that are expected not to record, with a reason each.
**Not taken:** redefining the row to mean "carries gradient", which would force
`training = false` on `sign`/`floor`/`ceil`/`round` and change what the
dispatcher admits under training. That is a behaviour change to admission, not
just to a test.

## D4. Operations still recording nothing
Fixed because they have ordinary derivatives: `sin`, `cos`, `log2`, `log10`.
Remaining, and the plan for each:
- `Frac` -- derivative 1 a.e. Record.
- `Fmod`, `Remainder` -- d/dx = 1, d/dy = -trunc(x/y) / -floor(x/y). Record.
- `Trunc` -- derivative 0 a.e. Declared no-gradient.
- `OneHot` -- input is indices. Declared no-gradient.
- `ToDType` -- float to float should pass gradient through with a cast back;
  float to integer should not. Record for the float-to-float case.
  **Alternative if the cast-back proves messy:** declare it no-gradient and
  document that a mid-graph dtype change detaches, which is what it does today
  silently. Worse for researchers who cast to f64 for stability mid-graph.

## D5. Public gradcheck placement
**Taken:** `incin_core::exec::gradcheck`, generic over a new `GradCheckStorage`
trait, with the CPU implementation in a non-test module.
**Not taken:** exposing `cpu::gradcheck` directly (CPU only, and its module is
`#[cfg(test)]`); taking a `backward` closure parameter instead of putting
`backward_from` on the trait (every caller would spell it the same way).
**Back out:** revert the module and its `exec/mod.rs` re-exports; the CPU impl
module goes with it.

## D6. `unbroadcast` visibility
**Taken:** align *downward*. WGPU's `pub` becomes `pub(crate)`, matching CPU,
CUDA and Metal.
**Not taken:** aligning upward the way the recording seam was aligned. The four
are not one contract; they differ in scalar-seed expansion and in which reduce
kernel they use. Exporting them as one API invites a downstream recipe to
depend on CPU's edge cases and meet CUDA's.
**If it is ever needed downstream:** put it on a trait beside `TapeStorage`,
written once, rather than exporting four functions.

## D7. Per-input `Option` in backward returns (Candle / PyTorch `needs_input_grad`)
**Not taken.** `BackwardFn` returns `Vec<S>` from 127 sites across four
backends. `Vec<Option<S>>` is 127 mechanical edits, and the arity check already
turns the main failure (a short return) into a structured error. Changing only
`DifferentiableOp::backward` does not work: `input_ids` is fixed at record
time, so a `None` at position `i` has nothing to remove. Doing it properly
means the core walk accepting optional contributions, which is the same edit.
**Revisit:** with GRD-006, when `BackwardFn` is being changed anyway.

## D8. `Tensor::apply_op` scope
**Taken:** single input, shape preserving, gradient marker and dtype inherited
from the receiver.
**Not taken:** a fully general `apply_op_n` returning arbitrary shapes. The
output geometry of a shape-changing custom operation is something only the
caller knows, so a general version has to ask for it, which is
`execute_shaped_n` with extra steps.
**Superseded in part by D15**, which takes the version that does ask for the
geometry and inherits everything else. The objection above stands: what D15
declines is inferring the shape, not accepting it.
**Back out:** delete the method; nothing else depends on it.

## D9. Higher-order gradients
**Not taken, deliberately.** Documented as a 0.1 ceiling instead. Requires
changing `BackwardFn` from storage to tensors and lifting the
`GradMode::Disabled` wrapper in all four backends. Half-delivered it is worse
than a stated absence.

## D10. Partial-gradient guard in the optimizer
**Not taken.** A frozen parameter and a broken chain are indistinguishable at
the optimizer. The total-skip case is already refused. The real fix is the
conformance check (D2), which stops the chain breaking.

## D11. `Tensor::apply_op` output layout
**Taken:** return the `Dyn` layout.
**Not taken:** `RowMajor`, matching every built-in unary in the same impl block.
Rejected because `RowMajor` is a contiguity claim, and the crate cannot make it
on behalf of a kernel it did not write. `Dyn` claims nothing, which is the true
description, and it composes with the built-in successors regardless -- proven
by chaining a `relu` in `a_custom_operation_is_one_call_from_a_tensor`.
**Revisit:** if an author wants to assert contiguity, the honest shape is an
opt-in on the operation rather than a blanket promise here.

## D12. `GradMode::scope` in the public gradient checker
**Taken:** `restrict`, not `scope`. `scope` installs a thread-local and is gated
on `std`; `gradcheck` is not gated, so calling `scope` broke every no-`std`
configuration.
**Not taken:** gating `gradcheck` on `std`. Rejected: nothing about central
differences needs an allocator-plus-threads environment, and the module would
then be absent exactly where a hand-written kernel is most likely to be wrong.
**Note for later:** the CPU suite cannot catch this class of break, because
`cpu` implies `std`. `cargo xtask feature-matrix` is the only gate that does,
and it is worth running after any change to a module in `incin-core` that is
not itself feature-gated.

## D13. Where this record lives
**Taken:** `docs/plan/research/0.2.0/`, beside `what-to-take-from-sota.md` and
`grd-006-graph-owned-tapes.md`, which are the same genre.
**Not taken:** a session scratchpad. The whole point of recording the options
not taken is that somebody can return to them after this session ends, which a
scratchpad file cannot support.

## D14. Pre-existing drift, now all closed
Four things were found while running the gates and deliberately not fixed in
an autograd commit, because each predated this work and folding a fix in would
have hidden it. Re-checked on 2026-09-11: all four are closed, and three of
them were closed by somebody else while this record sat. The list is kept
rather than deleted because a stale finding is worth more as a dated
correction than as a silent removal.

- **Closed by `2ef12574`.** `tools/build-api-examples.py` emitted one blank
  line the committed `crates/incin/tests/api_examples.rs` did not have, and
  `--check` was lenient about it. That commit made the gate compare rather
  than regenerate. Verified by running the generator three times in a row
  against a clean tree: no diff on any run.
- **Closed, and the original claim does not reproduce.**
  `crates/incin/tests/api_examples.rs` was recorded as warning `unused import:
  incin::prelude::*`. It carries `#![allow(unused_variables, unused_imports,
  clippy::type_complexity)]` and has since `ba6bec4f`, the commit that created
  it, and neither `cargo build --tests` nor `cargo clippy --test api_examples`
  under the CI feature set emits any warning. Recorded as an error in the
  original finding rather than as a fix.
- **Closed.** `crates/incin-backends/src/cuda/backend/tests.rs:903` imported
  `crate::cpu`, so `--all-targets` clippy on a `cuda`-without-`cpu`
  configuration failed. That configuration now passes clean.
- **Closed by `c9b78ec0`.** `test_cuda_jit_kernel_forward_and_backward` was
  `#[cfg(feature = "cuda")]` but not `#[ignore]`, unlike the CUDA tests beside
  it, so it launched a kernel rather than only compiling one and
  `cargo test --workspace --all-features` could not pass on a machine with the
  toolkit and no driver. `hardware.yml` states the rule in its own header, and
  runs the plain suite and then the ignored suite on the CUDA runner, so the
  attribute costs no coverage.

One method note, because it cost time. A workspace-wide `cargo test` run
overlapping a `tools/build-api-examples.py` run reports compile errors in
`crates/incin/tests/api_examples.rs` that are not real: the generator rewrites
that tracked file in stages, compiling and dropping failing examples as it
goes, so a concurrent reader sees an intermediate state. Run the generators
and the suites one at a time.

## D15. `apply_op_n` operand type
**Taken:** `others: &[&B::Storage<K>]`, with the output shape as a parameter.
**Not taken:** `others: &[&Self]`, which is what the architecture review's
section 3.3 proposed. A slice of `&Self` forces every operand to share the
receiver's shape *and* layout, so it cannot express the case that section
itself gives as the motivation, a two-operand kernel over `[m, k]` and
`[k, n]`. It would also leave the review's own claim that this method "removes
the only remaining reason a user touches `TensorHandle`" false, because a
shape-mismatched operand would still need a handle. Dispatch never asked for
the uniformity: `TensorHandle::from_storage` is parameterized by backend,
dtype and placement, not by shape, so the storage reference is the widest
operand type that keeps the call type-safe.
**Not taken either:** an erasure trait (`&[&dyn StorageRef<B, K>]`) to keep
`&[&w]` at the call site. It buys one `.inner()` per argument and costs public
surface plus dynamic dispatch on a method whose whole point is that it is
small.
**Consequence accepted:** the gradient mode comes from the receiver's marker
alone, so a `Grad` receiver records even when an operand's storage came from a
`NoGrad` tensor. That is the rule `apply_op` and `execute_shaped_n` already
follow rather than a new one, so it is documented on the method instead of
being guarded against.
**Second consequence, and the harder one:** the result is labelled with the
receiver's device, and nothing compares the operands against it. Unlike the
gradient marker this was not previously a question, because one input cannot
disagree with itself. **Not taken:** refusing operands whose device differs
from the receiver's. The canonical path's check reads `row.same_device`, which
`table.rs` derives from the semantic profile and sets to false for `Transfer`
and `Creation`, so a per-operation answer is the framework's existing position
rather than an oversight. A custom operation has no row, so a strict check
here would decide for every author that their operation is same-device, and
the one class of custom operation that most obviously is not is the one a
`Transfer` profile would describe. It is also unreachable on CPU, where
`DeviceId::cpu()` is a singleton and `CpuStorage` exposes no constructor
taking metadata, so the guard could not be given a failing test in a CPU-only
CI. Documented under `# Devices` on the method instead.
**Revisit:** if custom operations gain a declared profile, the check becomes
free and should be taken. The multi-device configurations that could exercise
it (`distributed-nccl`, multi-ordinal CUDA) are the ones to test it from.
**Back out:** delete the method and the `Concat2` fixture; nothing else
depends on either.

## D16. Capability-group traits
**Taken:** `ExecuteInto<O, K>` alone, which halves what a dispatched-built-in
recipe pays per operation and is portable.
**Not taken:** the generated per-group traits (`ElementwiseOps<K>` and
friends) that the architecture review's section 3.4 proposes as the version
that "stops scaling with the recipe". They were built, they generate cleanly,
and they must not ship, because the premise they rest on is false.

The review's argument is that "the group is the unit a backend implements and
the completeness proof enforces it". The proof exists, and all four backends
carry it: `assert_every_advertised_row_executes` and its CUDA, WGPU and Metal
twins. What it proves is that a backend implements everything *it* advertises.
It does not prove, and nothing proves, that the four backends advertise the
same members, and they do not. Counting members per group from
`declarations.rs`:

| group | cpu | cuda | wgpu | metal | common |
|---|---|---|---|---|---|
| elementwise | 48 | 48 | 17 | 4 | 4 |
| native_tensor | 33 | 31 | 4 | 0 | 0 |
| reduction | 17 | 15 | 14 | 6 | 6 |
| composed_reduction | 11 | 11 | 0 | 0 | 0 |
| normalization | 6 | 5 | 0 | 0 | 0 |

Only six of the twenty groups have identical membership across all four, and
they are the small ones: `broadcast`, `reshape`, `filling`, `sampling`,
`readback`, `matmul`. So an `ElementwiseOps` generated from the CPU
declaration is satisfiable by CPU and CUDA and by neither WGPU nor Metal, and
a generic kernel bounded by it would compile, pass on CPU, and silently not
exist for half the backends. Generated from the intersection instead it has
four members out of forty-eight, and it *shrinks* whenever a backend is added,
which makes the meaning of a published bound depend on unrelated future work.
Both are worse than naming the two operations a kernel actually uses.

**How the false premise survived:** the review says all three candidates were
"compiled and run", which is consistent with having been compiled against the
CPU backend only. Compiling a group trait proves nothing about whether any
backend satisfies it: a blanket impl with an unsatisfiable bound is a trait
nobody implements, not an error. The assertion that catches it is
`const fn satisfies<B: ElementwiseOps<f32>>()` instantiated at a concrete
backend, which is how the divergence above was found.

**Worth keeping from the attempt,** because it is a fact about the catalog
rather than about the groups: five of the twenty groups could never have a
group trait whatever the membership question, because `ExecuteInto` requires
an operation's output to convert to storage and one member of each returns
something else. `readback` returns host scalars, vectors and bytes, which is
the group that exists because `Execute::Output` is an associated type at all;
`filling` and `sampling` carry the `Variable*` forms, which return a variable;
`reduction` carries `TopK`, which returns a values/indices pair; and
`composed_tensor` carries `Chunk` and `Split`, which return a `Vec`. The other
fifteen are clean.

**Revisit:** if backend parity closes, the membership objection goes with it
and the generator is twenty lines. It is parked in the session scratchpad
rather than committed, because a parked file in the tree is surface without a
consumer. The measurement above is the thing worth keeping, and the command
that reproduces it is a member count per group over `declarations.rs`.

