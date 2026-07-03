# Kindle Design Document

## Objective
To build an experimental statically-typed deep learning framework wrapper around dynamic backends (like Candle/Burn), proving that robust mathematical bounds and shapes can be verified at compile-time in Rust, with zero overhead at runtime.

## Architecture

### 1. The Core `Tensor` Wrapper
```rust
pub struct Tensor<S: Shape, B: Backend, T: DType, D: Device, G: RequiresGrad> {
    pub(crate) inner: B::RawTensor,
    pub(crate) _shape: S::Field,
    // ...
}
```
All tensor parameters (`S`, `T`, `D`, `G`) use zero-sized `PhantomData` fields to guarantee exact mathematical constraints when compiling, without inflating the size of the underlying backend tensor buffer.

### 2. Type-Level Shapes & Traits
Shapes are expressed strictly using const generics:
```rust
pub trait Shape {
    type Field;
    // Mathematical invariants mapping `s![B, C, H, W]` to behavior
}
```
If a user writes `.add()` between two tensors, the Rust compiler will verify `S1 == S2`.
If they don't match, `rustc` rejects it immediately, meaning runtime panics are drastically reduced.

### 3. Procedural Macro Ergonomics (`kindle-macros`)
To compensate for Rust's verbose type signatures, we introduced `proc_macro` attributes.
- **`s![B, C, H, W]`**: Expands a list of dimensions to a type-level nested generic list.
- **`idx![.., 1..2, 0]`**: Translates NumPy/PyTorch slicing logic into successive `narrow` and dimension-dropping operations.
- **`#[kindle::module]` & `#[kindle::forward]`**: In future iterations, these will AST-trace mathematical operations within neural networks to circumvent the type-solver when networks become thousands of layers deep, significantly speeding up compilation times.

### 4. Backends
The `Backend` trait delegates computation. Currently, `CandleBackend` implements dynamic execution logic, providing a `candle::Tensor` to `Tensor::inner`.

### 5. Multi-Threaded DataLoaders
By utilizing Rayon, `DataLoaderExt` exposes `.into_par_loader()`, effortlessly turning any standard Rust Iterator into a CPU-bound asynchronous worker pool.
