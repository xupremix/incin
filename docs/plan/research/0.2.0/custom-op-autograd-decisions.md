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

## D14. Pre-existing drift left alone
Three things were found while running the gates and deliberately not touched,
because each predates this work and folding a fix into an autograd commit
hides it:
- `tools/build-api-examples.py` emits one blank line that the committed
  `crates/incin/tests/api_examples.rs` does not have, and `--check` is lenient
  about it. **A contributor who regenerates will get a spurious one-line diff
  and no gate will explain it.** Worth fixing in its own commit.
- `crates/incin/tests/api_examples.rs` warns `unused import: incin::prelude::*`
  (from `9a8426b7`).
- `crates/incin-backends/src/cuda/backend/tests.rs:903` imports `crate::cpu`
  (from `69570485`), so `--all-targets` clippy on a `cuda`-without-`cpu`
  configuration fails. CI runs that suite with both features, and the library
  itself builds in every configuration, so this is a property of the test tree
  rather than a break.
