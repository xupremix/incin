# Asking for a layout at construction

How a caller should say which memory order they want when a tensor is
*created*, rather than recovering a proof afterwards with `into_row_major`.

Code here is illustrative and has not been compiled. Names match the tree as of
`develop` after the density work.

## The gap

Two places in the tree pin the answer to row-major and cannot express anything
else.

`crates/incin-backends/src/target/place.rs`:

    pub type TargetTensor<T, S, K> = Tensor<S, TargetBackend<T>, K, NoGrad>;

Six parameters, five named: the layout slot takes its default, so nothing the
target API returns can carry a proof. And
`crates/incin-backends/src/target/ext.rs`:

    fn allocate_row_major<S: Shape + DynShape, K: ...>(
        &self, values: &[K], dims: Vec<usize>, field: ShapeBuf,
    ) -> Result<TargetTensor<Self, S, K>>

which hardcodes the answer in its name. Every data constructor funnels through
it.

The core constructors are better off -- they are generic over the layout,
bounded on `FreshDense<S>`, so `let t: Dense<s![3, 4], B> = Tensor::zeros(())?`
already works. But `FreshDense` has exactly two implementors:

    impl<S: Shape> FreshDense<S> for Dyn {}
    impl<S: Shape> FreshDense<S> for RowMajor<S> {}

and the seal exists because the premise is *a fresh allocation genuinely is
both*. That premise is the whole design, and it stops holding the moment
someone wants a layout a fresh allocation is not.

## What other frameworks do

**PyTorch** carries two orthogonal parameters, and the naming collides with
ours in a way worth stating in the book:

| Parameter | Values | What it means |
| --- | --- | --- |
| `layout=` | `torch.strided`, `torch.sparse_coo`, `torch.sparse_csr` | storage *scheme* |
| `memory_format=` | `contiguous_format`, `channels_last`, `channels_last_3d`, `preserve_format` | stride *order* |

**Incin's `Layout` is PyTorch's `memory_format`, not PyTorch's `layout`.**

`channels_last` keeps the shape `NCHW` and permutes only the strides, so
`t.shape` is unchanged and `t.stride()` is what moved. Both parameters appear on
factory functions and on `.to()`; `memory_format` also appears on
`.contiguous(memory_format=...)`.

Two properties are the interesting part for us. The default on `*_like`
factories is `preserve_format` -- **carry the operand's** -- which is exactly
the contract we removed, and PyTorch's propagation through ops is a set of
precedence heuristics rather than a rule, with several ops silently falling back
to contiguous. And because the tag is a runtime fact, `.view()` still raises at
runtime on a non-contiguous tensor, which is the failure `reshape_view` exists
to move to compile time.

**JAX/XLA** treats layout as a compiler concern: XLA chooses `minor_to_major`
during compilation, and the user-facing escape hatch attaches to `jit`
boundaries rather than to array constructors. The philosophy is that users
should not specify what the compiler should pick.

**TensorFlow** puts it on the operation instead of the value: `data_format=
"NHWC" | "NCHW"` is an argument to `conv2d`, not a property of the tensor.

**CuTe / CUTLASS** is where `LayoutOf` came from: a `(Shape, Stride)` pair
required to be congruent, composable through a layout algebra, static where the
extents are.

Rust neighbours: `ndarray` marks Fortran order with a runtime builder flag
(`Array::from_shape_vec(shape.f(), data)`); `nalgebra` fixes column-major in the
type with no choice; `candle` carries a runtime `Layout { shape, stride,
start_offset }` and checks `is_contiguous()`; `burn` has no concept and
materialises.

## The proposal

Generalise the sealed trait so that it *carries the strides to allocate with*:

    /// A layout a fresh allocation can be made to satisfy.
    pub trait FreshLayout<S: Shape>: LayoutOf<S> {
        /// The strides an allocation must use for `Self` to describe it.
        fn strides(dims: &[usize]) -> StrideBuf;
    }

    impl<S: Shape> FreshLayout<S> for RowMajor<S> {
        fn strides(dims: &[usize]) -> StrideBuf { suffix_products(dims) }
    }

    impl<S: Shape> FreshLayout<S> for ChannelsLast<S> {
        fn strides(dims: &[usize]) -> StrideBuf { nhwc_strides(dims) }
    }

    impl<S: Shape> FreshLayout<S> for Dyn {
        // Claims nothing, so any allocation satisfies it. Row-major is the
        // canonical choice rather than a claim.
        fn strides(dims: &[usize]) -> StrideBuf { suffix_products(dims) }
    }

The constructor calls `L::strides(dims)`, hands them to the backend, and returns
a tensor claiming `L`. **The claim is then honest by construction**: the value
that produced the strides is the one named in the type, so there is no gap
between what was asked for and what was allocated.

That also dissolves the reason `FreshDense` is sealed. A downstream author
writing `ChannelsLast::strides` incorrectly is lying about their own type, which
is the same trust `Layout::STATIC_STRIDES` already extends to them. The seal
protected against a *different* thing -- an unbounded constructor stamping a
layout onto an allocation that had nothing to do with it -- and that is no
longer possible when the layout chose the allocation.

### Spelling: type-directed, not an argument

Keep the existing pattern. `Dense<S, B>` in the return position already works,
and a `memory_format`-style argument would introduce a value that has to agree
with a type, which is a second thing to check and a second thing to get wrong.

    // Type-directed, consistent with what exists.
    let nhwc: ChannelsLast4<s![N, C, H, W], B> = Tensor::zeros((n, c, h, w))?;

    // Rejected: the value and the type can disagree.
    let nhwc = Tensor::zeros_in(MemoryFormat::ChannelsLast, (n, c, h, w))?;

### Backends must be able to refuse

This is the part that decides whether the design is sound. CPU and CUDA allocate
contiguous today. A `ChannelsLast` request must either be honoured or rejected
through the existing `*_layouts` capability rows -- never silently allocated
row-major and returned with a channels-last type on it.

**A creation API that cannot refuse is a minting press with extra steps**, which
is the exact hazard the seal was introduced to close. The refusal belongs in the
same capability table that already gates strided operands, so a backend that
gains the ability advertises it, and one that has not cannot be asked.

## Worked cases

Assume `ChannelsLast<S>` exists, describing an `NCHW` shape whose strides are
ordered `NHWC`.

### 1. Ask for the default, get what exists today

    let t: Tensor<s![3, 4], B> = Tensor::zeros(())?;

`L` takes its default `Dyn`. Nothing claimed, nothing checked, identical to the
code that compiles now. This is the case that must not regress.

### 2. Ask for row-major and get a proof from the allocation

    let t: Dense<s![3, 4], B> = Tensor::zeros(())?;
    let flat = t.reshape_view::<s![12]>()?;   // compiles: `RowMajor: Contiguous`

Already works. `RowMajor::strides` returns suffix products, which is what the
allocator was doing anyway, so this path is unchanged in behaviour and only
changes in how it is justified.

### 3. Ask for channels-last

    let x: ChannelsLast4<s![N, C, H, W], B> = Tensor::zeros((16, 3, 224, 224))?;

The allocator receives `[C*H*W, 1, C*W, C]` rather than suffix products. The
buffer really is channels-last, and the type says so because the type chose it.

### 4. A layout the backend cannot allocate

    let x: ChannelsLast4<s![N, C, H, W], WgpuBackend> = Tensor::zeros(...)?;
    // Err(BackendError::UnsupportedLayout { backend: "wgpu", layout: ChannelsLast })

Refused at construction, through the capability row. The alternative -- handing
back a row-major buffer with a channels-last type -- is the failure mode the
whole parameter exists to prevent, and it would be undetectable downstream.

### 5. `reshape_view` on channels-last must not compile

    let x: ChannelsLast4<s![1, 3, 4, 4], B> = Tensor::zeros(...)?;
    let flat = x.reshape_view::<s![48]>()?;
    // error: `ChannelsLast<_>` does not implement `Contiguous`

This is the case that makes the whole exercise worth doing: today
`ChannelsLast` does not exist, so `Contiguous` has never had to *exclude*
anything. The bound has been vacuously satisfied by both implementors. A second
layout is what turns `reshape_view`'s bound from decoration into a check, and
until one exists that bound has never been tested.

### 6. Recovering a proof by checking

    let dense = x.into_row_major()?;   // Ok only if the strides happen to match
    let flat = dense.reshape_view::<s![48]>()?;

For a genuinely channels-last tensor this returns `Err`, correctly. There is
deliberately no `assume_row_major`.

### 7. Materialising into a different layout

    let dense: Dense<s![1, 3, 4, 4], B> = x.to_layout()?;   // copies

The counterpart to PyTorch's `.contiguous(memory_format=...)`. Distinct from
(6): `into_row_major` *checks and refuses*, `to_layout` *copies and succeeds*.
Both are needed, and conflating them is how a framework ends up with a method
that sometimes costs nothing and sometimes costs a full copy.

### 8. Host data whose order differs from the device layout

    let values = vec![/* row-major NCHW from a file */];
    let x: ChannelsLast4<s![1, 3, 4, 4], B> = target.tensor_from(&values)?;

The constructor is handed row-major host data and a channels-last target. It
must permute during upload rather than memcpy. This is the case most likely to
be got wrong, because the length check passes either way and only the *order* is
wrong -- so the conformance test has to assert values, not just strides. That is
the same trap the strided-GEMM test was written to avoid.

### 9. A pointwise operation on a channels-last operand

    let y = x.relu()?;   // Dense<s![1, 3, 4, 4], B>, not ChannelsLast

Today's rule already answers this: the result states `RowMajor` because the
kernel writes a fresh packed buffer. **This is the case that proves the "state,
don't carry" decision was right rather than merely defensible** -- carrying
would have returned a channels-last claim over a row-major buffer, which is a
lie no test would catch until someone called `reshape_view` on it.

Whether pointwise *should* preserve channels-last is a separate performance
question. If a backend gains a kernel that writes channels-last output for a
channels-last operand, the signature can change -- but only alongside the
conformance test that shows the buffer really is.

### 10. `Dropout`, the one carrying operation

    let d: Dropout = Dropout::new(0.5);
    let y = d.forward(x)?;   // ChannelsLast in, ChannelsLast out

Its eval branch returns the operand untouched, so the carry is right. Its
training branch allocates -- and under `FreshLayout` it would allocate *with
`L::strides`*, so both branches genuinely produce `L` and the bound tightens
from "a dense buffer satisfies this" to "an allocation can be made to satisfy
this". Strictly better: today `Dropout` accepts only layouts a *dense* buffer
satisfies, which would exclude `ChannelsLast` entirely.

### 11. Generic code over the layout

    fn normalise<S: Shape, B: Backend, L: Layout>(
        t: &Tensor<S, B, f32, NoGrad, Local, L>,
    ) -> Result<Dense<S, B>> { ... }

Unchanged. `L: Layout` accepts any operand; the return type states what the
function allocates. The bound only tightens where a function *creates* something
in the caller's layout.

### 12. A downstream layout

    struct Blocked<S, const N: usize>(PhantomData<S>);
    impl<S: Shape, const N: usize> Layout for Blocked<S, N> { ... }
    impl<S: Shape, const N: usize> FreshLayout<S> for Blocked<S, N> {
        fn strides(dims: &[usize]) -> StrideBuf { ... }
    }

Works without changes to this crate, because `FreshLayout` is unsealed. What it
does *not* get is `Contiguous`, so `reshape_view` stays closed to it -- which is
the correct default for a layout this crate knows nothing about.

## Sequencing

There is no second layout yet. Designing the creation API against `RowMajor`
alone risks fitting it to the one case that already worked, and cases 5, 9 and
10 above are all untestable until a second layout exists.

So: **introduce `ChannelsLast` and `FreshLayout` in the same change**, with a
conformance test that allocates and asserts both the strides and the values.
Otherwise the API's only user is the case that needed no API.

## Open questions

- Does `Contiguous` stay a marker, or become `Contiguous<Order>`? A blocked
  layout is contiguous in a sense `reshape_view` still cannot use.
- Should `transpose_view` restate the layout the way `into_shape` does? A
  transposed `RowMajor<[H, W]>` is a real layout of `[W, H]`, just not
  `RowMajor` -- naming it needs a permutation layout, which is more machinery
  than the current `Dyn` answer.
- Does the capability table gate creation by `LayoutClass`, or does creation
  need its own row? `Strided` is too coarse: a backend may allocate
  channels-last and still refuse an arbitrary stride.
