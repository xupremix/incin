typed layout: one structure parameter, many traits

Design sketch for putting stride, contiguity and alignment in the type. Code
here is illustrative and has not been compiled; the names match the tree as of
`develop`.

## Why one parameter and not four

The facts worth typing are strides, contiguity, alignment, and arguably
quantization block structure. Given as separate parameters that is
`Tensor<S, B, K, G, P, L, A, Q>`, which nobody would want to write or read.

Bundled, it is one: `Tensor<S, B, K, G, P, L>`.

The codebase already established this pattern. `Shape` is a single parameter
with 74 traits over it -- `BroadcastShape`, `SpatialConv2d`, `ReduceAt`,
`SwapAxes` -- so a caller writes `S: Shape` once and bounds the *facts* it needs
where it needs them. Layout should work the same way: one parameter, and
`L: Contiguous` or `L: AlignedTo<U16>` at the call sites that care.

This is also what CuTe does. Its `Layout` is the bundle of `Shape` and `Stride`
with a congruence requirement between them, and the congruence is only
expressible because they are bundled. Two independent parameters would be two
things to keep manually consistent.

## The trait

Mirrors `Shape`'s conventions exactly, including the one that matters most:
silence is never credited.

```rust
/// What the type settles about where a tensor's elements live.
pub trait Layout: 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    /// Per-axis strides in elements, outermost first, `None` where the type
    /// settles nothing. Empty when the rank is unknown.
    ///
    /// Same shape as `Shape::STATIC_EXTENTS`, and for the same reason: a
    /// tensor with a dynamic batch stride and static inner strides is still
    /// worth specialising on.
    const STATIC_STRIDES: &'static [Option<usize>] = &[];

    /// Element offset into the underlying buffer, when settled.
    const STATIC_OFFSET: Option<usize> = None;

    /// Alignment of the base pointer in elements, when settled.
    const STATIC_ALIGNMENT: Option<usize> = None;

    /// How much of the layout came from the type rather than a runtime scan.
    /// Defaults to `Dynamic`, so a `Layout` implemented outside this crate is
    /// credited with nothing it has not shown.
    const PROOF: ProofLevel = ProofLevel::Dynamic;
}
```

The construction-buffer trick from `Shape::STATIC_EXTENTS` applies unchanged: a
fixed `[Option<usize>; MAX_STATIC_RANK]` scratch const, sliced to the rank, with
const promotion putting the result in `.rodata`. Ranks past the buffer report
`&[]` rather than a prefix -- a truncated stride list is a miscompile, an empty
one is a missed optimisation.

## The identity element

Every existing call site must keep compiling, so the parameter needs a default
that claims nothing:

```rust
/// Nothing proven about layout. The default, and what any runtime-shaped or
/// dynamically dispatched tensor carries.
pub struct Unknown;

impl Layout for Unknown {}   // every associated const takes its default

pub struct Tensor<
    S: Shape,
    B: Backend,
    K: DType = f32,
    G: RequiresGrad = NoGrad,
    P: Placement = Local,
    L: Layout = Unknown,      // <- added, defaulted
> { /* .. */ }
```

`Unknown` is to `Layout` what `Dyn` is to `Shape` and `ProofLevel::Dynamic` is
to a proof. Nothing that exists today changes meaning.

## Congruence

The one rule borrowed wholesale from CuTe. A layout is only meaningful against a
shape of the same rank, and letting the two drift is the failure mode a bundle
is supposed to prevent.

```rust
/// `L` describes a tensor of shape `S`.
///
/// Rank-congruent: one stride per extent. Checked once, here, rather than at
/// every operation that touches both.
pub trait LayoutOf<S: Shape>: Layout {}

/// A row-major layout derived from the shape itself.
pub struct RowMajor<S>(PhantomData<fn() -> S>);

impl<S: Shape> Layout for RowMajor<S> {
    const STATIC_STRIDES: &'static [Option<usize>] = /* suffix products of S */;
    const STATIC_OFFSET: Option<usize> = Some(0);
    const PROOF: ProofLevel = S::PROOF;
}
impl<S: Shape> LayoutOf<S> for RowMajor<S> {}
impl<S: Shape> Contiguous for RowMajor<S> {}
```

## Facts as traits

```rust
/// The layout visits memory in one unbroken ascending run.
pub trait Contiguous: Layout {}

/// The base pointer is a multiple of `N` elements.
pub trait AlignedTo<N: Unsigned>: Layout {}
```

Nothing implements `Contiguous` for `Unknown`, so an unproven tensor simply
cannot satisfy a bound that needs it -- and falls back to the runtime path.

## Example 1: reshape stops being a runtime error

The payoff that is a correctness property rather than a speed one. Reshaping a
non-contiguous tensor is a runtime failure in every framework; here it stops
being expressible.

```rust
impl<S: Shape, B: Backend, K: DType, G: RequiresGrad, P: Placement, L: Layout>
    Tensor<S, B, K, G, P, L>
{
    /// Only callable when the layout is proven contiguous.
    pub fn reshape<S2: Shape>(self) -> Result<Tensor<S2, B, K, G, P, RowMajor<S2>>>
    where
        L: Contiguous,
        S: ElementCount<Count = <S2 as ElementCount>::Count>,
    { /* .. */ }
}
```

```rust
let t: Tensor<s![3, 4], B, f32, NoGrad, Local, RowMajor<s![3, 4]>> = /* .. */;
let flat = t.reshape::<s![12]>()?;         // fine: RowMajor is Contiguous

let v = t.transpose_view::<_0, _1>();      // Tensor<s![4,3], .., Permuted<..>>
let bad = v.reshape::<s![12]>()?;          // compile error: Permuted: !Contiguous
```

Today the second case is either a runtime error or, because `transpose`
materialises, silently a copy. Neither is visible at the call site.

## Example 2: views stop copying

This is the cost the measurement found. Every `transpose`, `narrow` and
`broadcast` on CUDA materialises into a fresh contiguous buffer --
`launch_transpose` runs a permutation kernel, and all nineteen
`try_from_parts` call sites pass `contiguous_strides`. An attention block
transposes constantly and pays a full copy each time.

```rust
/// A view: permutes the stride list in the type, touches no memory.
pub fn transpose_view<Lx: StaticCursor, Rx: StaticCursor>(
    self,
) -> Tensor<S::Output, B, K, G, P, L::Swapped>
where
    S: SwapAxes<Lx, Rx>,
    L: SwapStrides<Lx, Rx> + LayoutOf<S>,
{ /* rebinds metadata only */ }
```

The consumer is then compiled against the stride pattern instead of discovering
it at runtime -- which is the whole reason a view can be a view.

Note what this makes live: the strided extent folding on `develop` is correct
and tested but currently unreachable, precisely because views materialise.
It is the prerequisite for this, not an optimisation on top of it.

Caveat that belongs in the scoping, not in the enthusiasm: materialising is not
obviously wrong. A contiguous copy can beat repeated strided reads for a
consumer that touches the data more than once, and it keeps everything
downstream on the fast path. The choice should be measured per consumer, and
typed layout is what makes *having* the choice possible.

## Example 3: alignment becomes a proof

`select_unary_strategy` currently decides packing with a runtime test:

```rust
let aligned = dense && offset.is_multiple_of(width.into());
```

With the fact in the type it is a bound, and the fallback stops being reachable
for callers who can prove it:

```rust
impl<S, B, K, G, P, L> Tensor<S, B, K, G, P, L>
where
    L: Contiguous + AlignedTo<U4>,
{
    /// Always takes the 4-wide packed path; no alignment branch is emitted.
    pub fn relu_packed(&self) -> Result<Self> { /* .. */ }
}
```

## Example 4: reaching the backend

No new channel is needed. `ShapeEvidence` already crosses the boundary on every
`Validated` descriptor and is already read by `KernelSpecialization`. Layout
facts ride the same rail:

```rust
// exec/proof.rs
pub struct ShapeEvidence {
    proof: ProofLevel,
    static_rank: Option<usize>,
    static_numel: Option<usize>,
    static_extents: &'static [Option<usize>],
    static_strides: &'static [Option<usize>],   // added
    static_alignment: Option<usize>,            // added
}

impl ShapeEvidence {
    pub(crate) const fn of_shaped<S: Shape, L: LayoutOf<S>>() -> Self {
        Self {
            proof: S::PROOF.meet(L::PROOF),
            static_extents: S::STATIC_EXTENTS,
            static_strides: L::STATIC_STRIDES,
            static_alignment: L::STATIC_ALIGNMENT,
            /* .. */
        }
    }
}
```

`ProofLevel::meet` already exists and does the right thing: a static shape with
an unknown layout yields the weaker of the two, so nothing is credited that was
not shown. The backend side is a field on `KernelSpecialization` and a guard in
`unrollable_extents`'s shape, both of which already exist in the form this
needs.

## Where quantization belongs, and it is not here

Alignment and strides describe *where* an element sits. Block structure --
`Q8_0`'s 32-element blocks, its scale placement, symmetric versus asymmetric --
describes *what an element is*. That is `K`'s job.

The evidence is the bug already in the tree: `quantized_matmul` has the catalog
giving `OutputRule::MatMul` with rhs `[K, N]`, the CPU kernel reading rhs as
`[N, K]`, and the CUDA declaration saying `[K, N]` while its `.cu` takes the
left operand as `const float*` against a `q8_0` descriptor. That is three
components disagreeing about what a quantized operand *is*, not about where it
lives. Putting block structure in `L` would file it under the wrong heading and
leave the dtype contract just as unchecked.

## Migration

1. Add `Layout`, `Unknown`, `RowMajor<S>`, `LayoutOf<S>`, `Contiguous`. Nothing
   uses them; nothing breaks.
2. Add `L: Layout = Unknown` to `Tensor`. Every existing signature still
   compiles, because the default claims nothing.
3. Have creation operations return `RowMajor<S>` instead of `Unknown`. Now
   `Contiguous` bounds start being satisfiable.
4. Add `reshape`'s `L: Contiguous` bound. This is the first breaking change and
   the first real payoff.
5. Add `transpose_view` beside the materialising `transpose`, so the two can be
   compared rather than swapped blind.
6. Thread the layout fields through `ShapeEvidence`.

Steps 1-3 are additive. The order is chosen so the first thing that can break a
downstream build is also the first thing that fixes a class of runtime error.

## What would make this not worth doing

- If measurement shows materialising views cost little on real workloads, the
  case narrows to reshape safety alone -- still real, but a much smaller prize
  for a sixth type parameter.
- If the inference burden turns out to be severe. `Shape` is already inferred
  through 74 traits; a second bundle interacting with it could make error
  messages considerably worse, and that cost lands on users rather than on the
  library.

Both are answerable before step 4, which is the last additive step.

## Prior art

CuTe (CUTLASS 3.x): `Layout = (Shape, Stride)`, congruent tuples, per-mode
static (`Int<N>`) or dynamic integers, with an algebra -- composition,
complement, division, product -- written to preserve staticness through
transformations. The bundle and the congruence are what this design takes. The
algebra is not proposed here: incin's 74 shape traits are a menu rather than a
calculus, and converting them is a separate and much larger question.

The one structural thing incin cannot copy without deeper change: CuTe shapes
nest (`((_2,_3),_4)`), and the nesting *is* the tiling structure.
`DimCons<H: Dim, T: Shape>` takes a `Dim` as its head, so shapes here are flat.
Flat shapes can carry strides; they cannot carry tiling.

---

## Implementation notes: what the migration plan got wrong

Steps 1 and 2 landed as designed. Step 3 is where the plan breaks, and the
reason is worth recording before anyone picks this up.

**Adding the parameter is free; adopting it is not.** `Tensor` gained
`L: Layout = Unknown` and the entire workspace compiled unchanged -- the only
churn was trybuild snapshots re-rendering the type with a sixth parameter, with
every diagnostic's substance identical. That is the default doing its job.

But an `impl<S, B, K, G, P> Tensor<S, B, K, G, P>` binds `L` to its default, so
**a tensor carrying a proof loses every method defined that way**. The parameter
is additive precisely because nothing produced a non-`Unknown` layout; the
moment something does, that tensor has almost no API.

**And converting those impls is not a rename.** Rewriting eighteen impl headers
to be generic over `L` produced 95 errors, and they were not mechanical. Two
kinds:

- Methods return `Tensor<..>` without naming `L`, so the parameter is
  uninferrable at the construction site.
- Methods call each other across impls, and a caller generic over `L` cannot
  reach a callee whose impl pinned it.

Both reduce to one question the design note never asked: **what layout does an
operation produce?** `a.mul(b)` allocates a fresh dense buffer today, so
`RowMajor<S>` is the truthful answer -- but only because every backend
materialises contiguously, which is a property of the current implementations
rather than of the operation contract. Answering it per operation family is the
actual step 3, and it is a design exercise, not a refactor.

The intermediate state that works, and is what is on `develop`:

- `into_row_major` is the sound way in: a checked promotion from runtime
  strides, defined only on `Unknown`. Deliberately no `assume_row_major`.
- `reshape_view` is the one consumer of `L: Contiguous`, and demonstrates the
  payoff -- reinterpreting a non-contiguous buffer stops compiling.
- Three of the eighteen impls are generic over `L`; the rest still pin it.

Revised ordering for whoever continues:

1. Decide the output layout for each operation family, starting with pointwise
   (where the answer is `RowMajor` and defensible) and stopping at anything that
   materialises for reasons the contract does not state.
2. Convert impls family by family, using that answer, rather than all at once.
3. Only then make creation return `RowMajor<S>`, since until step 1 covers an
   operation the tensor it returns will be stranded without an API.

The measurement that should gate all of it is still unmade: whether
materialising views actually costs anything on real workloads. If it does not,
this stops at reshape safety, which is real but a much smaller prize for a sixth
type parameter and this much churn.
