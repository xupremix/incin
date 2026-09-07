# Autograd: what was wrong, what is fixed, and what to take from the field

Branch `feat/custom-autograd-dtype`, eleven new commits on top of the twenty
that were already there.

Verified at the branch tip: full CPU suite, `cargo fmt`, clippy on core and all
four backends, the panic audit, the public-API baseline, hidden items,
architecture, book site, markdown links, the shape audit, the API-example
build, `cargo xtask docs --check`, `budgets`, `feature-matrix` and
`onnx --check`. CUDA, WGPU and Metal compile here, and the `wgpu`-only and
`cuda`-only configurations were checked without `cpu`; none of the three was
*executed*, so anything claimed about those backends is read from source, not
observed.

One of those gates earned its keep late. `feature-matrix` failed after the
gradient checker landed: `GradMode::scope` installs a thread-local and is gated
on `std`, the checker is not, and so `incin-core` stopped building in every
configuration without `std`. The CPU suite could not have caught it, because
`cpu` implies `std`. The fix is `restrict`, the ungated tighten-only direction,
which is the more accurate call there anyway.

## 1. The defects the work found

The eight findings from the review, and where each stands.

| | Finding | Status |
|---|---|---|
| F1 | Extension seam closed: no downstream custom op could join the graph | Was already fixed on the branch; now **proven** by a mixed built-in/custom test, and the last inconsistency (Metal) closed |
| F2 | CUDA seeded and NaN-checked every dtype as f32 | Already fixed on the branch; verified |
| F3 | `.unwrap()` on the CUDA backward path | **Fixed** |
| F4 | Recipe arity enforced by prose | Already fixed on the branch |
| F5 | No higher-order gradients, undocumented; two false doc claims | **Documented**, not implemented, deliberately |
| F6 | `training = true` was an unverified claim | **Fixed**, and it found eight real holes |
| F7 | Thread-local tape, partial-gradient case | Empty-tape case already errors on the branch; partial case deliberately left alone, see §4 |
| F8 | O(n^2) reachability; contradictory doc | **Fixed** |

### The one that mattered most

F6 turned out to be the lever. The dispatcher enforced a capability row in one
direction only: a training query is refused against a row that does not claim
training, and nothing checked the converse. So a row could advertise training
while its kernel recorded no tape node.

That does not produce a wrong number. It produces a graph with a hole in it:
`backward` reaches everything below the hole and nothing above, the optimizer
skips the parameters it found no gradient for, and the run finishes having
trained part of the model. A wrong forward kernel is visible; this is not.

The CPU conformance oracle now runs all 2286 training tuples with recording
enabled and fails the row if the tape did not grow. On first run it named ten
operations. Eight were real:

- **`sin`, `cos`, `log2`, `log10`** carried no gradient at all. Sinusoidal
  position encodings and rotary embeddings are exactly the shapes that hit
  this.
- **`to_dtype`** detached the graph. Casting to `f64` for a delicate step and
  back is ordinary practice, and everything upstream of the cast received
  nothing.
- **`frac`, `fmod`, `remainder`** carried no gradient.

The other two, `one_hot` and (for integer targets) `to_dtype`, are correct as
they were. Each new rule is checked against central differences rather than a
spot value, and the modulus rule is worth a line: every modulus here is
`r = a - b*q` for a locally constant integer `q`, so `dr/da = 1` and
`dr/db = -q`, and `q` is recovered as `(a - r) / b` from the values rather
than recomputed with a rounding rule. That is exact for both `fmod`'s
truncation and `remainder`'s least-non-negative convention, and it means
neither recipe can drift from its kernel by picking the wrong rounding.

`sign`, `floor`, `ceil`, `round`, `trunc` and `one_hot` are declared, by hand
with a reason each, in the oracle's `carries_no_gradient`. A new operation is
a finding until somebody writes down which group it is in, because there is no
way to tell "has no derivative" from "forgot to write one" by reading a
kernel.

## 2. What the field does, and what was taken from it

The question was what to learn from Burn, Candle and PyTorch. Four ideas, and
what happened to each.

### Candle: the entry point is on the tensor

`Tensor::apply_op1` takes a `CustomOp1` and returns a tensor. The author never
touches storage plumbing.

**Taken.** `Tensor::apply_op` is new. Before it, calling a custom operation
meant building a `TensorHandle`, building an execution context, dispatching,
then lifting the storage back with `try_from_storage` while restating the
shape, dtype, device and gradient marker by hand. All four were already known
at the call site. The whole flow is now:

```rust
let squared = x.apply_op::<Square>(NoAttributes)?;
let loss = squared.sum_all()?;
let grads = loss.backward()?;
```

Deliberately single-input and shape-preserving, which is what a fused
activation, a custom loss term or a quantization stub is. It returns the `Dyn`
layout rather than the `RowMajor` every built-in unary returns, and that is a
decision rather than an omission: `RowMajor` is a claim the built-ins can make
because their kernels are known to write contiguous output, and a custom kernel
is by definition not known to. `Dyn` claims nothing and still composes with
every built-in successor, which the mixed-graph test now proves rather than
assumes by chaining a `relu` onto the result. Anything that
changes shape or has several inputs or outputs keeps `execute_shaped_n`, where
the output geometry is something only the caller knows. Two paths, with the
boundary stated, rather than one path that has to ask the caller for
information it already has.

### PyTorch and JAX: the framework owns the saved values

`ctx.save_for_backward` and JAX's returned residuals both put saved values in
a declared slot rather than a closure capture, so the framework can free them.

**Already right here.** `DifferentiableOp` has an associated `Saved` type. The
recipe receives it by reference. This is the piece of the design that most
needs to stay as it is: when `GRD-006` moves saved-tensor lifetime to the
graph, this is the shape that makes it possible.

### Burn: autograd as a decorator backend

`Autodiff<B>` wraps a backend rather than each backend owning a thread-local
tape.

**Not taken, and noted as the destination.** It is what `GRD-006` is reaching
for, and it is the structural answer to the thread-local hazard in §4. It is
also a rewrite, not a change. Taking it now would mean redoing the four
backends while eight gradient holes were still open, which is the wrong order.

### PyTorch: `needs_input_grad`, and Candle's `Option` per input

Both let a rule say "this input takes no gradient" in the type rather than by
returning a zero tensor or by silently returning a short vector.

**Not taken, for a reason worth recording.** `BackwardFn` returns `Vec<S>` and
is called from 127 sites across four backends. Changing it to `Vec<Option<S>>`
is 127 mechanical edits for a benefit the arity check already mostly delivers:
a short return is a structured error, not a silent truncation. The cheaper
version, changing only `DifferentiableOp::backward`, does not work, because
`input_ids` is fixed when the node is recorded and a `None` at position `i`
would have nothing to remove. Doing this properly means the core walk accepting
optional contributions, which is the same 127-site edit. It belongs with the
`GRD-006` rewrite, not before it.

### The one axis where incin is the outlier

Honesty about what the comparison did *not* flatter. `DifferentiableOp` has an
associated `Dtype`, so a recipe is written for one element type: an author who
wants `Square` in `f32` and `f64` writes `Square<f32>` and `Square<f64>` and two
impls. None of Candle, PyTorch or Burn asks that. Candle's `CustomOp1` matches
on the storage's dtype inside one `fwd`; PyTorch's `Function` never sees a dtype
in its signature at all.

This is the remaining place where the design has more going on than it needs to.
It is not fixed here, and it should not be fixed in passing: the associated type
is what lets `supports` be a compile-time claim and what the blanket `Execute`
impl keys on, so removing it means deciding whether dtype dispatch inside a
recipe is checked or runtime-matched. That is a design question, not a
refactor. The place it would be answered is the same place `Saved` lifetime is
answered, which is `GRD-006`.

## 3. What the seam looks like now, and why it is smaller than it was

The concern was that the design was getting confusing. It was, in three
specific ways, and two of them are gone.

**One entry point instead of four steps.** `apply_op`, above.

**One recording seam instead of four accidents.** `tape_record` and
`tape_record_with` are the two names a downstream `Execute` needs, and Metal
had only the first while its `tape` module was the one of the four still
`pub`. The most important boundary in the crate, whether a third party can add
a differentiable operation, differed by backend for no stated reason. All four
now present the same two names, and the modules are uniformly `pub(crate)`.

The same reasoning ran the other way for `unbroadcast`. WGPU's was `pub`, CPU's
and CUDA's `pub(crate)`, Metal's a bare private function. The temptation is to
align them upward the way the recording seam was aligned. That would be wrong:
those four are not one contract, they differ in how a reduced-all-the-way
scalar seed is expanded back and in which reduce kernel they reach for, and
exporting them as one API is how a downstream recipe comes to depend on the
CPU backend's edge cases and meets CUDA's. WGPU's is now `pub(crate)` with the
rest. If un-broadcasting is ever offered downstream it belongs on a trait,
written once.

**One gradient check instead of a paragraph of advice.** The chapter used to
tell authors there was no public helper and to write a central-difference
sweep themselves, including choosing the step size, which is the part that is
easy to get wrong: at `f32` precision the total error is minimised near `1e-2`,
and the `1e-4` that looks conservative sits at its own noise floor where a real
defect and a rounding artifact are indistinguishable.

`incin_core::exec::gradcheck` is now public and backend generic.
`GradCheckStorage` is the whole of what it asks: read one element, perturb one
element, run the backward pass. `GradCheckReport` names the input, the element,
both values and the relative error, and says in its own `Display` that a
constant factor across every element is a missing term while a single element
is usually a boundary. A deliberately halved recipe is in the tests, so the
check has failed a bad gradient as well as passed good ones.

One correction inside that work: the implementation was first written inside
`cpu::gradcheck`, which is `#[cfg(test)]`. The public checker compiled,
documented itself, and had no implementation in any shipped build. It is now
in a module that is not test-gated, and the downstream fixture uses it, which
is the proof it works from outside.

## 4. What was deliberately not done

**A partial-gradient guard.** The optimizer already refuses a step where no
parameter in a group received a gradient, and its error names the thread-local
tape as the likely cause. Extending that to the partial case, where some
parameters get gradients and some do not, would be wrong: a frozen embedding
and an unused head are legitimate and indistinguishable from a broken chain at
that layer. The right fix is upstream, and it is F6, which makes the chain not
break in the first place.

**Higher-order gradients.** `BackwardFn` is over storage, not tensors, so there
is nothing in a recipe's output to differentiate, and all four backends run the
walk under `GradMode::Disabled`. That rules out gradient penalties,
meta-learning and Hessian-vector products. It is now written down as a 0.1
decision rather than discovered, with the note that `DifferentiableOp`'s shape
does not foreclose it. Implementing it means changing `BackwardFn` and every
push site; half-delivered it is worse than absent.

Two doc claims were false and are corrected: both `exec/tape.rs` and the
custom-operations chapter said a recipe may itself record, "as every
convolution backward on the CPU backend does". Those recipes call raw kernel
helpers, and anything they did record would be refused by the disabled scope.
The by-value signature is still right, for re-entrancy on the `RefCell`, which
is what the note now says.

**Rewriting `polar_cartesian.rs`.** It teaches the own-your-own-node-list path,
which a foreign backend still needs, and its header says so. The seam is proven
instead by a new test: built-in `Mul`, custom `Square`, built-in `SumAll`, one
`backward`, checked against `4x^3`. A downstream crate could not have written
that before `tape_record` was public.

## 5. If something here turns out wrong

Every decision above, including the ones not taken and how to back them out, is
in `custom-op-autograd-decisions.md` beside this file. The commits are separable: the CUDA panic fix,
the reachability change, the four trigonometric gradients, the oracle
enforcement, the public gradcheck, the docs, and `apply_op` are each their own
commit and each reverts alone.

The riskiest single change is the oracle enforcement, because it turns a
previously silent condition into a test failure and a future operation will
trip it. That is the intent, and `carries_no_gradient` is where the answer
goes, with a reason.
