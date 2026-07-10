# Native Backend — Design Outline

Milestone: a native, pure-Rust, ownership-based tape-autograd CPU backend for
`kindle`, intended to become the **default/primary** backend (Candle/ndarray
remain as peer options). CUDA/wgpu/Metal targets, kernel fusion, and
distributed training are explicitly **out of scope** for this milestone —
see "Deferred" at the bottom.

This document is meant to be handed to a code-generation pass as a build
spec: it enumerates the exact trait surface to implement, the design
decisions to lock in before writing code, a phase breakdown, and the
definition of done.

---

## 1. Why ownership-based tape autograd (the dfdx approach)

PyTorch/Candle-style autograd shares a mutable tape via `Rc<RefCell<_>>` (or
Python's GC) because tensors can be reused arbitrarily. `dfdx` instead
exploits Rust's move semantics: an op **consumes** its input tensor(s) by
value and returns `(output, backward_closure)`, where the closure owns
whatever it captured from the inputs before they were moved. No shared
mutable state, no runtime borrow checks, and the compiler enforces you can't
use a tensor after it's been consumed by an op unless you explicitly `.clone()`
it — which mirrors PyTorch's `retain_graph` semantics but at compile time.

This fits `kindle` unusually well: the `Tensor<S, B, K, D, G>` type **already**
tracks grad-requirement at the type level via `G: RequiresGrad`. The
extension is that for `G = Grad`, ops return a value whose backward pass is
*just as shape-checked* as the forward pass (same `S`, same `K`), instead of
falling back to a dynamic backward graph like Candle does internally.

## 2. Core data structures

```rust
// crates/kindle-backends/src/native/storage.rs

/// Owned, contiguous-or-strided CPU buffer for one dtype.
pub struct NativeStorage {
    data: NativeBuffer,       // dtype-tagged buffer, see below
    shape: Vec<usize>,
    strides: Vec<usize>,      // element strides, supports non-contig views
    offset: usize,
}

/// Runtime-tagged buffer, mirrors Candle's approach so `Storage<K>` can be
/// a single concrete type regardless of the phantom `K: DType` (matches
/// the existing pattern: Storage<K> doesn't need to encode K in the Rust
/// type, K is a call-site marker only — see DummyBackend/CandleBackend).
pub enum NativeBuffer {
    F32(Rc<Vec<f32>>),
    F64(Rc<Vec<f64>>),
    U8(Rc<Vec<u8>>),
    U32(Rc<Vec<u32>>),
    I64(Rc<Vec<i64>>),
    F16(Rc<Vec<half::f16>>),
    BF16(Rc<Vec<half::bf16>>),
}
```

**Decision: `Rc<Vec<T>>` inside the buffer, not raw ownership.** Even though
the *op* level is ownership-based (consumes-and-returns), the underlying
buffer needs `Rc` so that non-mutating views (`reshape`, `transpose`,
`narrow`, `slice`, `broadcast_as`) can share the backing allocation via
strides instead of copying — same tradeoff Candle/numpy make. This does NOT
reintroduce PyTorch-style shared *mutable* tensors: `NativeStorage` itself is
still moved by value through the op graph; only the immutable backing array
is refcounted. Mutation (in-place ops, optimizer steps) always goes through
`RawVar`, never through a shared `NativeStorage`.

```rust
// The tape/gradient side.

/// One entry: given the id of an output tensor, how to turn its
/// (already-computed) incoming gradient into gradients for its inputs.
struct TapeEntry {
    output_id: TensorId,
    input_ids: Vec<TensorId>,
    backward: Box<dyn FnOnce(&NativeStorage) -> Vec<NativeStorage>>,
}

pub struct NativeGrads {
    tape: Vec<TapeEntry>,
}
```

`TensorId` is a monotonically increasing counter (or pointer identity of the
backing `Rc`), used the same way Candle's `GradStore` keys gradients by
tensor identity. `backward<K>()` walks `tape` in reverse order accumulating
gradients into a `HashMap<TensorId, NativeStorage>`, which is what
`Backend::Grads` resolves to and what `get_grad` reads from.

```rust
pub struct NativeVar(Rc<RefCell<NativeStorage>>);
```

`RawVar` is the **one** place with interior mutability (`Rc<RefCell<_>>`),
because `Param<S, B>` is referenced by both the model (read) and the
optimizer (write on `.step()`) — same shape as `candle::Var`. This is
intentional and scoped: normal tensor flow through ops stays ownership-based;
only the long-lived trainable-parameter slot needs shared mutability.

## 3. Trait surface to implement

`NativeBackend<D: Device>` implements, exactly matching
`crates/kindle-core/src/tensor/backend.rs` (current shape — dtype is a
per-call generic `K: DType`, not a fixed associated type):

- **`Backend`** — `type Storage<K: DType> = NativeStorage; type RawVar =
  NativeVar; type Grads = NativeGrads; type FloatElem = f32; type IntElem =
  i64;` + `shape`, `format_tensor_display/debug`, `backward`, `get_grad`,
  `to_bytes`/`from_bytes`, `var_as_tensor`/`var_from_tensor`/`var_to_device`,
  `assign_var`.
- **`CreationOps<Self>`** — `zeros/ones/rand/randn`, `var_zeros/var_ones/
  var_rand/var_randn`, `tensor_to_device` (no-op on CPU-only, but keep the
  hook for later GPU targets).
- **`NumericOps<Self>`** — `add/sub/mul/div` (broadcasting elementwise, with
  backward rules).
- **`TensorOps<Self>`** — `matmul` (+ backward via `grad_out @ rhs.T` /
  `lhs.T @ grad_out`, with the same batch-broadcast handling the Candle impl
  has), `reshape`, `transpose`, `broadcast_as`, `broadcast_left`, `narrow`,
  `squeeze`, `stack`, `concat`, `slice`, `flatten`, `float_to_scalar/vec1`,
  `int_to_scalar/vec1`, `tensor_to_dtype`.
- **`FloatOps<Self>`** — `relu/gelu/abs/exp/neg/sqrt/log/tanh/sigmoid/swish/
  softmax/add_scalar_float/mul_scalar_float`, each with its backward rule.
- **`ReductionOps<Self>`** — `sum_all/mean_all/max_all/min_all`,
  `sum_dim/sum_keepdim/mean_dim/mean_keepdim/max_dim/max_keepdim/min_dim/
  min_keepdim`, `argmax/argmin` (dim: `Option<usize>`, no backward needed —
  index ops are non-differentiable).
- **`ModuleOps<Self>`** — `layer_norm`, `batch_norm` (both take
  `Option<&Storage<K>>` for weight/bias/running stats — see the `Option`
  handling already established in `CandleBackend`'s implementation for the
  pattern to mirror), `embedding`, `conv1d/conv2d/conv_transpose2d`,
  `max_pool2d/avg_pool2d/adaptive_avg_pool2d`.
- **`LossOps<Self>`** — `mse_loss/l1_loss/bce_with_logits_loss/
  cross_entropy_loss`.

Every method above needs a **forward** implementation from day one, but only
**needs a backward/tape hook when it can be called on a `G = Grad` tensor**
in the existing `kindle-core` op layer — check `crates/kindle-core/src/
tensor/ops/*.rs` and `crates/kindle-core/src/nn/*.rs` for which ops actually
get called with grad-tracking on, to avoid over-building backward rules for
ops that are only ever used in inference paths.

## 4. Phase breakdown (all within this one CPU milestone)

1. **Storage & strides** — `NativeStorage`/`NativeBuffer`, contiguous +
   strided views, broadcasting shape-resolution helpers, `to_bytes`/
   `from_bytes`. No autograd yet. Unit-test shape/stride math directly.
2. **Elementwise forward ops** — `NumericOps`, `FloatOps` forward-only.
   Parity-test outputs against `CandleBackend` (same inputs, same seed,
   assert within tolerance).
3. **Autograd core** — `TapeEntry`, `NativeGrads`, `backward<K>()`,
   `get_grad<K>()`. Wire backward rules for the Phase 2 ops. Validate against
   Candle's gradients on the same small graphs (`autograd_tests.rs` pattern).
4. **Reductions** — forward + backward for `sum/mean/max/min` (all/dim/
   keepdim variants), `argmax/argmin`.
5. **Matmul** — forward + backward, including the >3D batched-broadcast case
   (mirror the batching logic already in `CandleBackend::matmul`). This is
   the highest-risk-of-bugs phase — budget real test time here.
6. **Shape ops** — `reshape/transpose/broadcast_as/broadcast_left/narrow/
   squeeze/stack/concat/slice/flatten`, each with its (often trivial —
   scatter-back) backward rule.
7. **NN module ops** — `layer_norm`, `batch_norm`, `embedding`, `conv1d`,
   `conv2d`, `conv_transpose2d`, `max_pool2d`, `avg_pool2d`,
   `adaptive_avg_pool2d`. Forward first, backward second per op.
8. **Loss ops** — `mse_loss/l1_loss/bce_with_logits_loss/
   cross_entropy_loss`.
9. **Optimizer + `Param` integration check** — confirm `crates/kindle-core/
   src/optim/mod.rs`'s `SGD`/`Gradients` work unmodified against
   `NativeBackend` (they're already generic over `Backend`; this phase is
   about *proving* it, not writing new optimizer code).
10. **Definition of done: new dedicated example/benchmark** — a from-scratch
    example (not swapping into an existing one) that trains a small model
    end-to-end on `NativeBackend` and reports numbers (loss curve, timing)
    directly comparable to the same model on `CandleBackend`.

Optional stretch **within** this milestone (do not let it block Phase 10):
Rayon-parallelized elementwise ops as a cheap, non-fusion perf win.

## 5. Design decisions to lock in before coding

| Decision | Recommendation | Why |
|---|---|---|
| Buffer representation | Runtime-tagged `enum NativeBuffer` (§2), not monomorphized-per-`K::Elem` storage | `Storage<K>` must be one concrete Rust type across all `K: DType` (including `Dyn`), same constraint every other backend in this repo already satisfies |
| Strided views vs copy-only | Real strides from Phase 1 | Avoids a costly rewrite later; `transpose`/`narrow`/`slice`/`broadcast_as` become O(1) metadata ops instead of copies |
| Gradient store keying | Tensor identity (`Rc` pointer or an id counter), `HashMap`-backed | Matches Candle's `GradStore` semantics that `get_grad` already assumes |
| Numerical parity bar | Relative error tolerance (e.g. `1e-5` for f32), not bit-exact | Reduction/matmul summation order will differ from Candle's BLAS calls |
| In-place mutation | Only through `RawVar`/`NativeVar`; never on a live `NativeStorage` mid-graph | Keeps the ownership-based tape model sound — see §2 |
| Threading | Rayon on elementwise ops only, opt-in, not required for Phase 10 done-ness | Real fusion/perf work is explicitly deferred to a later milestone |

## 6. Testing & validation strategy

- Reuse the existing test shape: `crates/kindle-core/tests/nn_tests.rs`,
  `layers.rs`, `autograd_tests.rs`, `broadcast.rs` already exercise the
  `Backend` trait generically — running them against a `NativeBackend` type
  alias (in addition to the existing Candle-backed ones) gives free
  regression coverage as each phase lands.
- Add `crates/kindle-backends/tests/native_parity.rs`: op-by-op forward
  numeric comparison against `CandleBackend` for every op in §3, plus
  gradient comparison for every op with a backward rule.
- Phase 10's new example is the actual "done" signal — parity tests alone
  don't prove the autograd tape is sound end-to-end across a real training
  loop (optimizer step, multiple backward passes, parameter reuse).

## 7. Deferred to future milestones (explicitly not this one)

- **CUDA / wgpu / Metal backends** for `NativeBackend` — build these as
  additional `Storage<K>` implementations or sibling backend types once the
  CPU implementation's op/autograd design has proven itself.
- **Kernel/op fusion** — an optimization pass on top of a stable tape
  representation; premature before the tape shape itself is validated.
- **Distributed training** (data/model parallelism, checkpoint-and-sync
  across devices/machines) — needs a `Sharding`/`Placement` type-level
  concept layered on top of `S`/`D`, which is its own significant design
  effort (see the earlier discussion in this thread: separate *logical*
  shape from *physical* placement, mirroring JAX/PyTorch FSDP, but done
  through Rust's type system rather than at runtime).

---

*Generated from a GSD deep-questioning session, 2026-07-09. See `TODO.md`
for known gaps in the current (Candle/ndarray/burn-based) backend layer that
are separate from this native-backend initiative.*
