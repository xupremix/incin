//! What the type settles about *where* a tensor's elements live.
//!
//! [`Shape`] answers how many elements there are and how they are indexed
//! logically. It says nothing about the mapping from a logical coordinate to a
//! buffer offset, which is the stride pattern, and that mapping is what decides
//! whether a kernel can address linearly, whether a packed load is legal, and
//! whether a reshape is even meaningful.
//!
//! Today that mapping is entirely a runtime fact: `TensorMeta` carries strides,
//! and the backend recovers a four-valued `LayoutClass` by scanning them. This
//! module is the type-level half.
//!
//! # One parameter, many traits
//!
//! Strides, offset, alignment and contiguity are four facts, but they are not
//! four type parameters. `Shape` already established the pattern this follows:
//! a single parameter carrying the bundle, with the individual facts exposed as
//! traits and associated constants that call sites bound on where they care.
//! `L: Contiguous` reads well; a `Tensor<S, B, K, G, P, L, A, Q>` does not.
//!
//! Bundling is also what makes congruence expressible. A stride list is only
//! meaningful against a shape of the same rank, and [`LayoutOf`] states that
//! relationship once rather than leaving two independent parameters to be kept
//! consistent by hand.
//!
//! # Where the runtime value lives
//!
//! A layout parameter is a marker, not a container. When it settles nothing,
//! the answer still has to come from somewhere, and that somewhere already
//! exists: `TensorMeta` carries the strides, the offset and the shape of every
//! tensor, and a backend reads them as it always has.
//!
//! This mirrors how shapes work. `Tensor` holds a `ShapeValue<S>`, which pairs
//! the shape *type* with the runtime dimensions; a `Dyn` shape gets its answer
//! from the value, and a static one has the same answer in both places.
//! `Layout` is the type half of the same pairing, with `TensorMeta` as the
//! value half -- so no new field is needed, and
//! [`Tensor::into_row_major`](crate::prelude::Tensor::into_row_major) is
//! exactly the operation that reads the value and, if it agrees, promotes it
//! into the type.
//!
//! The consequence worth noticing is that a *fully static* layout makes the
//! runtime copy redundant. Nothing exploits that yet at the tensor level, but
//! the CUDA pointwise path already does the kernel-side version: when the
//! extents are proven, the `shape` array and its per-launch upload are not
//! emitted at all, because the values are baked into the source instead. The
//! same reasoning applies to strides once enough of the API carries a layout.
//!
//! # Silence is never credited
//!
//! Every associated constant defaults to "nothing known", exactly as
//! `Shape::PROOF` and `Shape::STATIC_EXTENTS` do. A layout implemented outside
//! this crate, or one carried by a tensor that came through a dynamically
//! dispatched path, reports no strides rather than a guess. [`Dyn`] is the
//! identity element and the default, so adding this parameter changes the
//! meaning of nothing that already exists.

use crate::dist::Local;
use crate::shapes::{Dyn, MAX_STATIC_RANK, ProofLevel, Shape};
use core::fmt::Debug;
use core::marker::PhantomData;

/// The stride pattern and alignment a type settles for a tensor's buffer.
///
/// See the [module documentation](self) for why this is one parameter rather
/// than several.
pub trait Layout: 'static + Clone + Debug + Send + Sync + Eq + PartialEq {
    /// Per-axis strides in elements, outermost first.
    ///
    /// `None` in a position means that axis's stride is a runtime fact. Empty
    /// means the rank itself was not settled, or the rank exceeds
    /// [`MAX_STATIC_RANK`].
    ///
    /// Per-axis rather than all-or-nothing for the same reason
    /// `Shape::STATIC_EXTENTS` is: a tensor whose batch stride is dynamic and
    /// whose inner strides are constants is still worth specialising on, and
    /// that is the common transformer case.
    const STATIC_STRIDES: &'static [Option<usize>] = &[];

    /// Element offset of the tensor's first element into its buffer, when the
    /// type settles it.
    const STATIC_OFFSET: Option<usize> = None;

    /// How much of this layout came from the type rather than a runtime scan.
    ///
    /// Defaults to [`ProofLevel::Dynamic`], so an implementation that says
    /// nothing is credited with nothing.
    const PROOF: ProofLevel = ProofLevel::Dynamic;

    /// Construction buffer behind [`STATIC_STRIDES`](Layout::STATIC_STRIDES).
    ///
    /// The same device `Shape::EXTENT_BUF` uses, for the same reason: building
    /// a stride list whose length depends on an associated constant would need
    /// `generic_const_exprs`. The buffer is the vehicle; the slice is the
    /// payload, so what a `Layout` actually carries is two words of `'static`
    /// data rather than a rank-sized array.
    #[doc(hidden)]
    const STRIDE_BUF: [Option<usize>; MAX_STATIC_RANK] = [None; MAX_STATIC_RANK];
}

/// `Self` describes the memory of a tensor whose shape is `S`.
///
/// Rank congruence: one stride per extent. Stated once here rather than
/// re-derived at every operation that touches both halves, which is the whole
/// argument for bundling them.
///
/// Borrowed from CuTe, where `Layout` is a `(Shape, Stride)` pair required to
/// be congruent -- same tuple profile, one stride integer per shape integer.
pub trait LayoutOf<S: Shape>: Layout {}

/// `Self`, re-described against a shape type covering the same runtime dims.
///
/// The general rule is that a layout cannot be carried across a shape change,
/// because it describes one geometry. [`into_shape`] is the exception, and it is
/// worth being precise about why: it changes no dimension. It re-describes the
/// *same* extents under a different shape type, over the same buffer, with the
/// same strides -- `S2::try_from_dims` is what makes it fallible rather than
/// free, and what rules out the case where the two shapes disagree.
///
/// So `RowMajor<S1>` and `RowMajor<S2>` denote identical strides whenever that
/// conversion succeeds, and dropping to [`Dyn`] there discarded a fact that was
/// still true. This trait is the type-level half of that argument: it maps a
/// layout to the way the same layout is spelled against another shape.
///
/// # Why it is not sealed
///
/// Unlike [`FreshDense`], nothing here can be minted. A downstream layout
/// author naming their own `Restated` is describing their own type, which is
/// the same trust already extended to them by [`Layout::STATIC_STRIDES`]. The
/// obligation is stated rather than enforced: `Restated` must denote the same
/// strides `Self` does, for a shape with the same extents.
///
/// [`into_shape`]: crate::prelude::Tensor::into_shape
pub trait RestateFor<S2: Shape>: Layout {
    /// The same layout, spelled against `S2`.
    type Restated: LayoutOf<S2>;
}

/// Claiming nothing about one shape is claiming nothing about any other.
impl<S2: Shape> RestateFor<S2> for Dyn {
    type Restated = Dyn;
}

/// Row-major strides are a function of the extents, and `into_shape` does not
/// change the extents.
impl<S1: Shape, S2: Shape> RestateFor<S2> for RowMajor<S1> {
    type Restated = RowMajor<S2>;
}

/// `Self` is a truthful description of a freshly allocated dense buffer of `S`.
///
/// Constructors like [`Tensor::zeros`](crate::prelude::Tensor::zeros) allocate a
/// packed row-major buffer, so they are entitled to hand back a layout proof
/// rather than [`Dyn`]. This trait is what entitles them, and it is the
/// reason they can do so without also becoming a way to *forge* one.
///
/// # Why a bound and not just a return type
///
/// Naming `RowMajor<S>` in the return type would work, and would break every
/// existing call site that expects the default. Bounding the constructor on this
/// instead lets the caller choose: ask for `Tensor<S, B>` and get `Dyn` as
/// before, ask for [`Dense<S, B>`](Dense) and get a real proof, from the same
/// function and the same allocation.
///
/// # Why it is sealed
///
/// A constructor generic over `L` is a minting press: whatever layout the caller
/// names, it produces a tensor claiming it. That is harmless while the only
/// layouts are `Dyn` and `RowMajor`, because a fresh allocation genuinely is
/// both. It stops being harmless the moment a second real layout exists -- a
/// `ChannelsLast<S>` would be mintable from `zeros` despite a fresh allocation
/// not being channels-last at all, and the proof would be a lie with no unsafe
/// block and no runtime check anywhere near it.
///
/// Sealing means only this module decides what a fresh allocation may claim, so
/// adding `ChannelsLast` cannot silently make it claimable. The compiler asks
/// the question at the point where the answer is known.
pub trait FreshDense<S: Shape>: LayoutOf<S> + sealed::SealedFresh {}

/// A fresh allocation is free to claim nothing.
impl<S: Shape> FreshDense<S> for Dyn {}

/// A fresh allocation is packed row-major, which is exactly this claim.
impl<S: Shape> FreshDense<S> for RowMajor<S> {}

mod sealed {
    /// Prevents [`FreshDense`](super::FreshDense) being implemented downstream.
    ///
    /// Without this, a downstream layout could assert that a fresh allocation
    /// satisfies it and mint the proof from any constructor.
    pub trait SealedFresh {}
    impl SealedFresh for super::Dyn {}
    impl<S> SealedFresh for super::RowMajor<S> {}
}

/// The layout visits memory in one unbroken ascending run with no gaps.
///
/// This is a structural claim, not a numeric one: it holds for a row-major
/// layout whose extents are entirely dynamic, because contiguity is about the
/// *pattern* rather than about the values. That is why it is a marker trait and
/// not a predicate over [`Layout::STATIC_STRIDES`].
///
/// Deliberately not implemented for [`Dyn`]. A tensor that has proven
/// nothing cannot satisfy a bound that needs this, and falls back to the
/// runtime path.
pub trait Contiguous: Layout {}

// There is deliberately no `AlignedTo<N>` trait, and no `STATIC_ALIGNMENT`
// constant, though the design note listed both.
//
// Alignment is a property of the *allocation*, not of the shape: `TensorMeta`
// receives an `allocation_alignment` at construction, and nothing about `S`
// implies it. So no layout derived from a shape -- `RowMajor<S>` included --
// can implement such a trait. Making it real would need a separate wrapper
// layout produced by a checked promotion, in the style of `into_row_major`, and
// a backend that consumes the bound.
//
// Neither exists, and the win is small: the only place the fact would be used
// is `select_unary_strategy`, which decides packing with a single
// `offset.is_multiple_of(width)` test. Shipping the vocabulary without either
// half would be exactly the unexecuted public surface tracked by #111, so it
// waits for a consumer that justifies it.

/// Nothing proven about layout: [`Dyn`] is the identity element.
///
/// [`Dyn`] is the runtime-selected marker the shape, dtype, device and
/// placement slots already use, and it means the same thing in the layout
/// slot -- the compiler settled nothing, so the answer is read from
/// `TensorMeta` at runtime. It is the default for [`Layout`] parameters and
/// what every tensor carries until a more specific layout is threaded through.
/// Every constant stays at its default, and it deliberately does not implement
/// [`Contiguous`].
///
/// # Why the marker is shared rather than its own type
///
/// An earlier version spelled this `Unknown`, a distinct unit struct. The
/// argument for separating them was that `Dyn` is not a pure marker -- it is a
/// [`Shape`] with `Arg = Vec<usize>` and its own `resolve`, because a dynamic
/// shape is *constructed* from runtime dimensions, whereas an unproven layout
/// is constructed from nothing.
///
/// That is true and it costs nothing: a trait impl does not oblige the layout
/// slot to use `Shape::resolve`, and the layout slot never constructs a value
/// at all. What the separation did cost was a second name for one idea. A
/// reader who has learned that `Dyn` means "decided at runtime" in five slots
/// had to learn that the sixth slot spells the same concept differently, and
/// every `where` clause that wanted "unproven anything" named two types.
///
/// The one real objection was legibility of the rendered type, where the marker
/// now appears twice:
///
/// ```text
/// Tensor<Dyn, CpuBackendImpl, f32, NoGrad, Local, Dyn>
/// ```
///
/// A `Dyn` layout carries no information and should be elided from a hover; a
/// `Dyn` *shape* carries a great deal and must never be. The humanizer
/// distinguishes them by **position** rather than by name -- the layout is the
/// sixth of six arguments -- which is what the old spelling was standing in
/// for. Position is the more precise test anyway: it cannot be fooled by an
/// unrelated type that happens to share a name.
impl Layout for Dyn {}

/// `Dyn` describes any shape, because it claims nothing about it.
impl<S: Shape> LayoutOf<S> for Dyn {}

/// The dense row-major layout implied by a shape.
///
/// Strides are the suffix products of the extents, the offset is zero, and the
/// pattern is contiguous by construction. This is what every freshly created
/// tensor has, and what every materialising operation produces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct RowMajor<S>(PhantomData<fn() -> S>);

impl<S: Shape> Layout for RowMajor<S> {
    const STRIDE_BUF: [Option<usize>; MAX_STATIC_RANK] = row_major_strides(S::STATIC_EXTENTS);

    const STATIC_STRIDES: &'static [Option<usize>] = match S::RANK {
        Some(rank) if rank <= MAX_STATIC_RANK => Self::STRIDE_BUF.split_at(rank).0,
        // Unknown rank, or deeper than the buffer: report nothing rather than a
        // prefix. A truncated stride list would be a wrong geometry; an empty
        // one is only a missed specialisation.
        _ => &[],
    };

    const STATIC_OFFSET: Option<usize> = Some(0);

    // A row-major layout is exactly as well known as the shape it came from:
    // every stride is a product of extents, so an unproven extent is the only
    // thing that can make a stride unproven.
    const PROOF: ProofLevel = S::PROOF;
}

impl<S: Shape> LayoutOf<S> for RowMajor<S> {}

/// Contiguous by construction, whatever the extents turn out to be.
impl<S: Shape> Contiguous for RowMajor<S> {}

/// Suffix products of `extents`, innermost stride first at the deepest index.
///
/// `strides[i]` is the product of `extents[i + 1..]`, with the innermost stride
/// being one. An unknown extent makes every stride outside it unknown, because
/// each of those is a product that includes it -- but strides *inside* it stay
/// known, which is the asymmetry that makes per-axis reporting worth having.
#[doc(hidden)]
#[must_use]
pub const fn row_major_strides(extents: &[Option<usize>]) -> [Option<usize>; MAX_STATIC_RANK] {
    let mut out = [None; MAX_STATIC_RANK];
    let rank = extents.len();
    if rank == 0 || rank > MAX_STATIC_RANK {
        return out;
    }
    let mut accumulated: Option<usize> = Some(1);
    let mut axis = rank;
    while axis > 0 {
        axis -= 1;
        out[axis] = accumulated;
        accumulated = match (accumulated, extents[axis]) {
            (Some(product), Some(extent)) => product.checked_mul(extent),
            _ => None,
        };
    }
    out
}

/// A tensor whose buffer is proven dense and row-major.
///
/// The layout parameter is precise but verbose to write out, and the dense case
/// is overwhelmingly the common one. This alias is the ergonomic spelling:
///
/// ```text
/// // instead of
/// fn f(t: Tensor<s![3, 4], B, f32, NoGrad, Local, RowMajor<s![3, 4]>>) {}
/// // write
/// fn f(t: Dense<s![3, 4], B>) {}
/// ```
///
/// Note the shape appears once rather than twice: `RowMajor` is always
/// congruent with the shape it describes, so repeating it is noise the alias
/// removes. That congruence is exactly what [`LayoutOf`] states.
pub type Dense<S, B, K = f32, G = crate::tensor::grad::NoGrad, P = Local> =
    crate::tensor::base::Tensor<S, B, K, G, P, RowMajor<S>>;
