# Kindle

## What This Is

Kindle is a Rust deep-learning/tensor library that checks as many invariants
as possible at compile time — shapes, dtypes, devices, and grad-tracking are
all encoded in the `Tensor<S, B, K, D, G>` type, so mismatches like a bad
`matmul` shape or a device mismatch are caught by `rustc`, not at runtime.
Ergonomic macros (`s!`, `idx!`) let users write shapes and slices without
spelling out the full type machinery, so the experience feels close to
PyTorch/NumPy despite the extra compile-time guarantees. Computation is
delegated to a pluggable `Backend` trait — the same `Tensor` code runs on
Candle, ndarray, or (in progress) a native backend, per the user's needs.

## Core Value

Catch shape/dtype/device mistakes at compile time instead of at runtime —
if it compiles, the tensor math is structurally sound.

## Requirements

### Validated

- ✓ Compile-time shape checking via `Tensor<S: Shape, ...>` with `s!`
  macro ergonomics, supporting mixed static/dynamic dims (`s![10, 20]` vs
  `s![dyn, 20]`) — existing
- ✓ Pluggable `Backend` trait abstracting compute so the same model code
  runs on multiple backends (Candle default, ndarray, burn stub) — existing,
  just finished a major refactor moving dtype from a fixed per-backend type
  to a per-call generic (`Storage<K: DType>`) plus an explicit `Device`
  generic on `Tensor`
- ✓ NN layer library (Linear, Conv1d/2d, BatchNorm2d, LayerNorm, Embedding,
  LSTM/RNN, pooling, activations, losses) generic over `Backend` — existing
- ✓ Autograd via the underlying backend (currently delegates to Candle's
  tape) — existing
- ✓ ONNX import (`import_model!` macro, codegen from `.onnx` graphs) and
  export — existing
- ✓ Optimizers (SGD), serialization (safetensors-style), data loading with
  Rayon-backed parallel loaders — existing

### Active

- [ ] Native, pure-Rust, ownership-based tape-autograd backend for CPU —
  becomes the default/primary backend once complete (Candle/ndarray remain
  as peer options)
- [ ] Full trait-surface parity with the existing `Backend`/`CreationOps`/
  `NumericOps`/`TensorOps`/`FloatOps`/`ReductionOps`/`ModuleOps`/`LossOps`
  traits (see `NATIVE_BACKEND_PLAN.md` for the exact method list)
- [ ] Autograd tape design that mirrors dfdx: ops consume inputs by value
  and return `(output, backward_closure)` — no `Rc<RefCell<Tape>>` sharing
  at the tensor level; the one deliberate interior-mutability boundary is
  the trainable-parameter (`RawVar`) slot
- [ ] New dedicated example/benchmark training a model end-to-end on the
  native backend, numerically comparable to the same model on Candle — this
  is the definition of done for the milestone

### Out of Scope

- CUDA / wgpu / Metal backends for the native implementation — deferred
  until the CPU implementation's op/autograd design is proven; bundling all
  four compute targets into one milestone was explicitly rejected as too
  much surface area at once
- Kernel/operator fusion — an optimization pass that only makes sense once
  the tape representation is stable; premature now
- Distributed training (data/model parallelism, cross-device/cross-machine
  sync, checkpointing) — needs a `Sharding`/`Placement` type-level concept
  layered on top of `S`/`D` that separates logical shape from physical
  placement (JAX/PyTorch-FSDP-style, but done through Rust's type system).
  Real design risk; deliberately sequenced as its own future milestone after
  the native backend proves itself, rather than co-designed now
- Replacing Candle/ndarray/burn outright — the native backend becomes the
  *default*, but existing backends stay as peer options users can opt into

## Context

Brownfield project. A large refactor of `kindle-core`'s `Backend` trait
(dtype-per-call generics, explicit `Device` generic on `Tensor`) had just
been carried through to completion across the whole workspace — see
`TODO.md` for the fix log and remaining gaps in the *existing* backend layer
(NdarrayBackend mostly stubbed, BurnBackend orphaned/defunct API, a couple
of trybuild fixture gaps). Full codebase map is in `.planning/codebase/`.

This next milestone (native backend) is deliberately scoped narrower than
the user's full long-term vision, which also includes multi-GPU-target
support (CUDA/wgpu/Metal + fusion) and distributed training. See
`NATIVE_BACKEND_PLAN.md` at the repo root for the detailed technical design
outline (data structures, trait surface, phase breakdown, design decisions)
produced during this project's deep-questioning session.

## Constraints

- **Tech stack**: Rust, edition 2024. Must integrate with the existing
  `Backend`/`CreationOps`/`NumericOps`/`TensorOps`/`FloatOps`/
  `ReductionOps`/`ModuleOps`/`LossOps` trait surface in
  `crates/kindle-core/src/tensor/backend.rs` — no changes to that trait
  shape are in scope for this milestone, only a new implementor of it
- **Compatibility**: Must not break the existing Candle/ndarray backends or
  their test suites
- **Numerical correctness**: Backward-pass gradients must match Candle's
  within a relative-error tolerance (bit-exact not required — summation
  order will legitimately differ)

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Ownership-based tape autograd (dfdx-style), not shared-mutable tape | Fits Rust's move semantics; keeps backward passes just as shape-checked as forward, unlike wrapping an opaque backend autograd | — Pending |
| CPU-only for this milestone; GPU targets deferred | 4 compute targets + fusion in one milestone was too much surface area; CPU-first gets the autograd design validated before adding hardware complexity | — Pending |
| Distributed training deferred to a future milestone | Needs a logical-shape-vs-physical-placement type design that's its own significant effort; don't co-design with the base autograd | — Pending |
| Native backend becomes default/primary; Candle/ndarray/burn stay as peer options | User wants native to be the recommended path without breaking existing users of other backends | — Pending |
| Definition of done = new dedicated example/benchmark, not just op-parity tests | Op parity alone doesn't prove the tape is sound across a real multi-step training loop (optimizer step, parameter reuse) | — Pending |

## Evolution

This document evolves at phase transitions and milestone boundaries.

**After each phase transition** (via `/gsd-transition`):
1. Requirements invalidated? → Move to Out of Scope with reason
2. Requirements validated? → Move to Validated with phase reference
3. New requirements emerged? → Add to Active
4. Decisions to log? → Add to Key Decisions
5. "What This Is" still accurate? → Update if drifted

**After each milestone** (via `/gsd-complete-milestone`):
1. Full review of all sections
2. Core Value check — still the right priority?
3. Audit Out of Scope — reasons still valid?
4. Update Context with current state

---
*Last updated: 2026-07-09 after initialization*
