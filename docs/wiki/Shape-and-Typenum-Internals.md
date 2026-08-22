# Shape & Typenum Internals

Incin achieves compile-time shape verification in standard, stable Rust without requiring non-type const generics or unstable compiler plugins. It does this by evaluating mathematical relationships at the type level using the `typenum` crate.

---

## 1. Type-Level Unsigned Integers

Numbers in `typenum` are represented as binary trees where:
* `UTerm` represents `0`.
* `UInt<N, B0>` represents `2 * N`.
* `UInt<N, B1>` represents `2 * N + 1`.

### Examples
* `U0` = `UTerm`
* `U1` = `UInt<UTerm, B1>`
* `U2` = `UInt<UInt<UTerm, B1>, B0>`
* `U3` = `UInt<UInt<UTerm, B1>, B1>`
* `U4` = `UInt<UInt<UInt<UTerm, B1>, B0>, B0>`

When rustc emits an error involving raw typenum types, `cargo incin check` and `incin-lsp` automatically parse these binary trees and render them as normal decimal numbers (e.g. `[16, 3, 256]`).

---

## 2. Compile-Time Shape Proof Traits

Every tensor operation defines a trait that asserts valid dimensionality:

### Matrix Multiplication (`MatMulShape`)
```rust
pub trait MatMulShape<Rhs> {
    type OutputShape: Shape;
}

// [M, K] x [K, N] -> [M, N]
impl<M, K, N> MatMulShape<DimCons<K, DimCons<N, Nil>>>
    for DimCons<M, DimCons<K, Nil>>
where
    M: Unsigned,
    K: Unsigned,
    N: Unsigned,
{
    type OutputShape = DimCons<M, DimCons<N, Nil>>;
}
```

If a user writes:
```rust
let a: Tensor<s![4, 8], Cpu> = Cpu.zeros(())?;
let b: Tensor<s![3, 8], Cpu> = Cpu.zeros(())?;
let c = a.matmul(&b)?;
```
`DimCons<U8, Nil>` does not match `DimCons<U3, Nil>`, so rustc fails compilation with `trait bound MatMulShape is not satisfied`.

---

## 3. Dynamic Shapes & Dual-Mode API

When shapes cannot be known at compile time (e.g. variable sequence lengths or dynamic batch sizes), Incin provides the runtime `shape!` macro:

```rust
// Static Shape: compile-time checked
let x: Tensor<s![16, 128], Cpu> = Cpu.randn(())?;

// Runtime Shape: validated upon construction and execution
let batch_size = read_runtime_batch_size();
let y: Tensor<Dyn, Cpu> = Cpu.randn(shape![batch_size, 128])?;
```

Both forms share the exact same kernel implementations, memory allocators, and autograd engine.
