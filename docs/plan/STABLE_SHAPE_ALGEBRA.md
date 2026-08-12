# Stable-Rust Rank-Independent Shape Algebra

## 1. Overview

Incin's canonical shape, dimension, and axis architecture is built on Stable Rust without requiring nightly-only features (`generic_const_exprs`, `specialization`, `negative_impls`, `adt_const_params`, or variadic generics).

This document specifies the design, semantic proof boundaries, structural representation, and compatibility contracts of Incin's rank-independent shape system.

---

## 2. Canonical Dimension Model

### 2.1 Static Extent Tri-State Enum

Static dimension information is represented with a 3-state semantic enum (`StaticExtent`):

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StaticExtent {
    /// Extent is known only at runtime (e.g. dynamic batch/seq length).
    RuntimeUnknown,
    /// Extent is statically known at compile time to be `value`.
    Value(usize),
    /// Statically known to be invalid/overflow/underflow/div-by-zero.
    Invalid,
}
```

This prevents ambiguous `Option<usize>` where `None` could conflate runtime unknown extents with static arithmetic invalidity.

### 2.2 Canonical Static Dimension

The canonical static dimension type is:

```rust
pub struct ConstDim<const N: usize>;
```

Dynamic dimensions (runtime extents) are represented by `usize`.

### 2.3 Derived Static Dimensions & Symbolic Arithmetic

Symbolic compile-time dimensions implement symbolic evaluation:

- `MulDim<A, B>`: Static product of two dimensions.
- `AddDim<A, B>`: Static sum of two dimensions.
- `CheckedSubDim<A, B>`: Static difference of two dimensions (yields `Invalid` on underflow).
- `ExactDivDim<A, B>`: Static exact division (yields `Invalid` on non-zero remainder or div-by-zero).
- `ProductDims<S>`: Product of all static dimensions in recursive shape `S`.

#### Semantic Equality vs. Rust Type Identity
`MulDim<ConstDim<32>, ConstDim<4>>` and `ConstDim<128>` are distinct Rust types, but evaluate to identical `StaticExtent::Value(128)`. The framework evaluates static semantics via `static_extent()`, enforcing semantic compatibility rather than Rust `TypeId` equality.

---

## 3. Named Dimensions

Semantic axis identity (tag) and dimension extent knowledge are orthogonal:

```rust
pub struct NamedDim<Tag, Extent> {
    pub extent: Extent,
    _tag: PhantomData<Tag>,
}
```

This allows representing:
- `NamedDim<Channels, usize>` (named tag + runtime extent)
- `NamedDim<Channels, ConstDim<64>>` (named tag + static extent)
- `NamedDim<Channels, ConstDim<1>>` (named tag + static unit extent, e.g. after `keepdim` reduction)
- `NamedDim<Channels, MulDim<A, B>>` (named tag + derived static extent)

When performing `keepdim` reductions, the `Tag` is preserved while the extent becomes `ConstDim<1>`.

---

## 4. Canonical Fixed-Rank Recursive Shape

Canonical shapes are represented as a recursive cons-list:

```rust
pub struct DimCons<H, T> {
    pub head: H,
    pub tail: T,
}

pub struct Nil;
```

`shape![32, batch, const WIDTH]` lowers into:

```rust
DimCons<
    ConstDim<32>,
    DimCons<
        usize,
        DimCons<
            ConstDim<WIDTH>,
            Nil
        >
    >
>
```

### Runtime Storage Efficiency

Recursive type-level shape structures do **not** become runtime linked lists.
At runtime, all shape extents and strides are flattened into contiguous, inline `ShapeBuf` storage (`INLINE_RANK = 8`). Small ranks require zero heap allocations; larger ranks spill to a single contiguous heap allocation without altering compile-time or runtime semantics.

---

## 5. Generic `[usize; N]` Shape Arguments

Generic arrays `[usize; N]` are constructor adapters for runtime dimensions.
They resolve to `Dyn` and do not form a second known-rank shape engine. Exact
known-rank runtime shapes use `Ranked<R>`.

Array arguments require no macro generation or explicit rank cap. The array
length only describes the input value supplied to the constructor.

---

## 6. Known-Rank Runtime Shapes (`Ranked<R>`)

When the rank `R` is known at compile-time but individual axis extents are runtime-dynamic, `Ranked<R>` preserves rank knowledge through a typenum rank:

```rust
pub struct Ranked<R: Unsigned> { /* runtime values are held by ShapeBuf */ }
```

Operations like reduction without `keepdim` on a `Ranked<typenum::U4>` output
`Ranked<typenum::U3>` rather than degrading to a completely untyped `Dyn`.

---

## 7. Canonical Axis Model

The `axis!(...)` macro provides unified axis selection:

- `axis!(2)` -> Static/runtime index 2
- `axis!(-1)` -> Relative from end
- `axis!(0, 2, -1)` -> Ordered selector sequence
- `axis!(Nchw::CHANNELS)` -> Named schema witness selection

### Structural Axis Cursors

Static axes are represented as type-level structural cursors:

```rust
pub struct Here;
pub struct Next<I>;
pub struct FromEnd<I>;
```

- Index 0 -> `Here`
- Index 1 -> `Next<Here>`
- Index 2 -> `Next<Next<Here>>`
- Index -1 -> `FromEnd<Here>`

---

## 8. Descriptor Axis Representation (`AxisSet`)

`AxisSet` replaces fixed 32-bit `AxisMask(u32)` to support arbitrary framework-supported axis indices (>31):

```rust
pub enum AxisSet {
    Inline(u64),
    Spilled(Vec<usize>),
}
```

---

## 9. Backend Rank Capability (`RankSupport`)

Framework shape representability is decoupled from backend kernel capability:

```rust
pub enum RankSupport {
    Any,
    UpTo(usize),
    Range { min: usize, max: usize },
}
```

A backend may report `UpTo(8)` while the framework shape algebra supports rank 64 or 200.

---

## 10. API Notes

- Exact structural shapes use `DimCons<Head, Tail>` and `Nil`; tuple shapes are not a canonical engine.
- Typenum dimensions (`typenum::U4`) interoperate with `ConstDim<4>`.
- `ConstDim<N>` remains only the adapter for unevaluable const paths; raw literals use recursive typenum integers.
