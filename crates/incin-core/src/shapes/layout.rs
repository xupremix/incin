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
//! # Silence is never credited
//!
//! Every associated constant defaults to "nothing known", exactly as
//! `Shape::PROOF` and `Shape::STATIC_EXTENTS` do. A layout implemented outside
//! this crate, or one carried by a tensor that came through a dynamically
//! dispatched path, reports no strides rather than a guess. [`Unknown`] is the
//! identity element and the default, so adding this parameter changes the
//! meaning of nothing that already exists.

use crate::shapes::{MAX_STATIC_RANK, ProofLevel, Shape};
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

    /// Alignment of the base address in *elements*, when the type settles it.
    ///
    /// In elements rather than bytes so a caller can compare it against a
    /// vector width without knowing the dtype's size.
    const STATIC_ALIGNMENT: Option<usize> = None;

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

/// The layout visits memory in one unbroken ascending run with no gaps.
///
/// This is a structural claim, not a numeric one: it holds for a row-major
/// layout whose extents are entirely dynamic, because contiguity is about the
/// *pattern* rather than about the values. That is why it is a marker trait and
/// not a predicate over [`Layout::STATIC_STRIDES`].
///
/// Deliberately not implemented for [`Unknown`]. A tensor that has proven
/// nothing cannot satisfy a bound that needs this, and falls back to the
/// runtime path.
pub trait Contiguous: Layout {}

/// The base address is a multiple of `N` elements.
///
/// Lets a vectorised load be a bound rather than a runtime test. The backend
/// currently decides this with `offset.is_multiple_of(width)` at launch time.
pub trait AlignedTo<N: typenum::Unsigned>: Layout {}

/// Nothing proven about layout.
///
/// The default for [`Layout`] parameters and what every tensor carries until a
/// more specific layout is threaded through. Implements [`Layout`] with every
/// constant at its default, and deliberately implements neither [`Contiguous`]
/// nor [`AlignedTo`].
///
/// This is to `Layout` what `Dyn` is to `Shape`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Unknown;

impl Layout for Unknown {}

/// `Unknown` describes any shape, because it claims nothing about it.
impl<S: Shape> LayoutOf<S> for Unknown {}

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
pub type Dense<S, B, K = f32, G = crate::tensor::grad::NoGrad, P = crate::dist::Local> =
    crate::tensor::base::Tensor<S, B, K, G, P, RowMajor<S>>;
