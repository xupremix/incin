# Architecture Deep Dive

This document details the core runtime and type-level architecture of Incin.

---

## 1. Type-Level Shape System

Incin represents multidimensional shapes as homogeneous type-level linked lists terminated by `Nil`:

```rust
pub struct Nil;
pub struct DimCons<Head, Tail>(pub PhantomData<(Head, Tail)>);
```

For example, a 3D tensor of shape `[Batch, Channels, Hidden]` with dimensions `[16, 3, 256]` is represented in the type system as:

```rust
DimCons<U16, DimCons<U3, DimCons<U256, Nil>>>
```

The macro `s![16, 3, 256]` generates this type at compile time.

### Rank-Independent Trait Implementations
Because shapes are inductively constructed linked lists (`DimCons<Head, Tail>`), all shape manipulation algorithms (transposition, slicing, concatenation, broadcasting) are implemented via inductive trait resolution without any hard-coded rank limits (such as a 6D or 8D maximum).

---

## 2. Tensor Representation & Backend Decoupling

A tensor in Incin is defined by three orthogonal type parameters:

```rust
pub struct Tensor<S, B, D = f32, G = NoGrad>
where
    S: Shape,
    B: Backend,
    D: DType,
    G: GradState,
{
    storage: B::Storage<D>,
    meta: TensorMeta,
    _marker: PhantomData<(S, G)>,
}
```

1. **`S` (Shape)**: Static `DimCons<...>` or dynamic runtime shape tracker.
2. **`B` (Backend)**: Concrete device & compute provider (`Cpu`, `Cuda`, `Wgpu`, `Metal`).
3. **`D` (DType)**: Element type (`f32`, `f16`, `bf16`, `f64`, `i32`, `i64`, `u8`, `q4_0`, `q8_0`).
4. **`G` (GradState)**: Tracks whether this tensor node participates in autograd graph construction (`TrackGrad` vs `NoGrad`).

---

## 3. Storage Model & Zero-Copy Views

All backends share a unified `TensorMeta` layout contract:
* **`shape`**: `[usize]` dimension extents.
* **`strides`**: `[usize]` physical element steps.
* **`offset`**: `usize` byte/element offset into the backing buffer.

### Zero-Copy Operations
* **`narrow(axis, start, length)`**: Adjusts `offset` and updates `shape[axis]`. Strides and underlying data buffers remain completely unchanged.
* **`transpose(dim0, dim1)`**: Swaps `shape[dim0]` with `shape[dim1]` and `strides[dim0]` with `strides[dim1]`. Zero allocations, zero memcpy.
* **`broadcast_to(target_shape)`**: Inserts stride `0` on expanded axes.

---

## 4. Autograd Tape Architecture

Incin uses a non-intrusive reverse-mode automatic differentiation engine:

1. **Forward Pass**: Operations performed on tensors with `TrackGrad` record their backward closures and input/output node IDs onto a thread-local execution tape (`Tape`).
2. **Backward Pass**: Calling `loss.backward()` drains the tape in reverse topological order, accumulating gradients into each leaf parameter's gradient accumulator.
3. **Memory Reclamation**: Once `backward()` completes, all intermediate activations held by tape closures are freed immediately, ensuring peak memory is bounded.
