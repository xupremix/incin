//! The seal that makes a descriptor worth trusting.
//!
//! [`spec`](crate::exec::spec) resolves an operation into a descriptor. That
//! descriptor is internally consistent — its constructors derive every field
//! rather than accepting it — but internal consistency is not the property a
//! backend needs. A backend needs to know that the operation was checked
//! against the *frontend's* shape proof, and a bare `BroadcastSpec` cannot say
//! so, because anyone can build one.
//!
//! [`Validated<O>`] is that statement. It has private fields and a
//! `pub(crate)` constructor, so outside `incin-core` there is no way to
//! produce one except by asking a lowering rule (`EXE-003`) to produce it.
//! When `EXE-006` gives `Execute<O>` a `&Validated<O>` rather than a bare
//! descriptor, "this was validated" stops being a convention that kernels
//! re-check and becomes a type the compiler enforces.
//!
//! # Provenance, not just a boolean
//!
//! "Validated" alone would be too coarse. A shape proved entirely at compile
//! time and one checked at runtime a microsecond ago are both valid, but they
//! justify different amounts of work: the first can specialize a kernel on
//! constants, the second cannot. [`ProofLevel`] records which happened, and
//! travels with the descriptor so a backend can act on it.
//!
//! The level is derived from the shape types by `Shape::PROOF`, never passed
//! in by a caller. This matters: a caller who could assert `Static` for a
//! runtime shape would be handing kernels a constant that does not exist.

use crate::shapes::shape::Shape;
use core::fmt;

#[cfg(feature = "paranoid-validation")]
use crate::shapes::error::ShapeError;

/// How much of an operation's legality the compiler settled.
///
/// The variants are ordered from strongest to weakest, and [`meet`] combines
/// two operands by taking the weaker — an operation is only as proved as its
/// least-proved input.
///
/// [`meet`]: ProofLevel::meet
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ProofLevel {
    /// Rank and every semantic dimension constraint came from type-level data.
    ///
    /// Nothing about the geometry was read from a runtime value, so a backend
    /// may specialize on it: bake extents into a kernel, unroll a loop, pick a
    /// tile size at compile time.
    Static,

    /// Structure is typed, but named or dynamic dimensions were checked once.
    ///
    /// The rank and the *relationships* between axes are compile-time facts;
    /// the sizes are not. A backend can trust the geometry without re-deriving
    /// it, but must read the extents from the descriptor rather than assuming
    /// them.
    Mixed,

    /// Rank and all semantic dimensions were checked at runtime.
    ///
    /// The check still happened — a `Dynamic` descriptor is validated, not
    /// unvalidated — but nothing about it was known before the data existed.
    Dynamic,
}

impl ProofLevel {
    /// The proof carried by shape `S`.
    ///
    /// Reads `Shape::PROOF`, which is folded from each axis's
    /// `Dim::STATIC_SIZE`.
    ///
    /// `PROPOSALS.md` §1.2.1 sketches this as `ProofLevel::of::<L, R>()`, for
    /// two operands. It is one operand here, combined with [`meet`], because
    /// convolution lowers three (input, weight, bias) and a binary-only form
    /// has nowhere to put the third. The two-operand call in the RFC is
    /// `ProofLevel::of::<L>().meet(ProofLevel::of::<R>())`.
    ///
    /// [`meet`]: ProofLevel::meet
    #[must_use]
    pub const fn of<S: Shape>() -> Self {
        S::PROOF
    }

    /// The weaker of two proofs.
    ///
    /// An operation combining a `Static` operand with a `Mixed` one is
    /// `Mixed`: the runtime extent is present in the result regardless of how
    /// well the other side is known. This makes [`ProofLevel`] a meet
    /// semilattice with `Static` as its top, so folding over any number of
    /// operands is order-independent.
    #[must_use]
    pub const fn meet(self, other: Self) -> Self {
        // `Ord` derives from declaration order, strongest first, so the weaker
        // proof is the numerically greater one.
        if (self as u8) >= (other as u8) {
            self
        } else {
            other
        }
    }

    /// The level for a shape of known rank, given whether every axis is
    /// statically sized.
    ///
    /// Exists because one `Shape` impl covers every tuple of a given arity —
    /// `(U2, U3)` and `(U2, usize)` share it — so the tuple's `PROOF` has to
    /// be a fold over its axes rather than a literal. Called from the shape
    /// macros; not useful directly.
    #[doc(hidden)]
    #[must_use]
    pub const fn of_ranked(all_axes_static: bool) -> Self {
        if all_axes_static {
            Self::Static
        } else {
            Self::Mixed
        }
    }

    /// Whether the geometry is entirely a compile-time constant.
    ///
    /// The question a specializing backend asks before baking extents into a
    /// kernel.
    #[must_use]
    pub const fn is_static(self) -> bool {
        matches!(self, Self::Static)
    }

    /// Whether the rank was known before the data existed.
    ///
    /// True for both [`Static`](Self::Static) and [`Mixed`](Self::Mixed). A
    /// backend that can pick an iteration strategy from rank alone — most
    /// pointwise kernels — needs this rather than [`is_static`].
    ///
    /// [`is_static`]: Self::is_static
    #[must_use]
    pub const fn has_static_rank(self) -> bool {
        !matches!(self, Self::Dynamic)
    }
}

impl fmt::Display for ProofLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Static => "static",
            Self::Mixed => "mixed",
            Self::Dynamic => "dynamic",
        })
    }
}

/// A [`ProofLevel`] that came from a shape *type* rather than from a caller's
/// claim about one.
///
/// [`dispatch::execute`](crate::exec::dispatch::execute) is generic over the
/// operation and the backend but not over the operand shapes, so it has no `S`
/// to read [`Shape::PROOF`] from and passes [`Dynamic`](ProofLevel::Dynamic)
/// for everything. The typed tensor frontend *does* hold `S`, and this is the
/// value that carries what it knows across that boundary.
///
/// The reason it is a distinct type rather than a bare `ProofLevel` parameter
/// is provenance. A plain enum argument would let any caller write
/// `ProofLevel::Static` beside whatever metadata it liked, which is precisely
/// the forgery [`Validated`] exists to prevent one layer down. The only
/// constructors here are [`of`](Self::of), which reads the level off a shape
/// type and cannot be told a different answer, and
/// [`dynamic`](Self::dynamic), which claims nothing.
///
/// A `Shape` implemented outside this crate can still overstate its own
/// `PROOF`, exactly as it can today — the trait's default is
/// [`Dynamic`](ProofLevel::Dynamic) so silence is never credited, and a wrong
/// override is a wrong specialization rather than unsoundness. What this type
/// removes is the ability to assert a level with *no* type behind it at all.
///
/// [`Shape::PROOF`]: crate::prelude::Shape::PROOF
/// A descriptor together with the proof that produced it.
///
/// The only way to obtain one outside `incin-core` is from a lowering rule,
/// because its constructor is `pub(crate)` and the fields are private.
/// A backend receiving `&Validated<MatMulSpec>` therefore knows the geometry
/// was checked against the frontend's shape proof, and does not need to check
/// it again.
///
/// The wrapper is deliberately thin: it adds provenance and takes away the
/// ability to forge, and it does nothing else. Everything about the operation
/// is still readable through [`descriptor`](Self::descriptor).
///
/// # Why not just a `validated: bool` field on the descriptor
///
/// Because a field can be set. The guarantee has to live somewhere the caller
/// cannot reach, and in Rust that is the privacy boundary, not a value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Validated<O> {
    descriptor: O,
    proof: ProofLevel,
}

impl<O> Validated<O> {
    /// Seal `descriptor` with the proof that justified it.
    ///
    /// Crate-private on purpose, and the reason this type exists. `EXE-003`'s
    /// lowering rules are its only callers; they hold the shape types the
    /// proof is derived from, which no external caller does.
    pub(crate) const fn new(descriptor: O, proof: ProofLevel) -> Self {
        Self { descriptor, proof }
    }

    /// The resolved operation.
    #[must_use]
    pub const fn descriptor(&self) -> &O {
        &self.descriptor
    }

    /// How much of the operation the compiler settled.
    #[must_use]
    pub const fn proof_level(&self) -> ProofLevel {
        self.proof
    }

    /// Take the descriptor back out, discarding the proof.
    ///
    /// For a backend that wants to store the geometry after dispatch. The
    /// result is an ordinary descriptor with no standing to be trusted, which
    /// is why this consumes `self` rather than copying: a `Validated` and a
    /// bare descriptor should not be alive at once and confusable.
    #[must_use]
    pub fn into_descriptor(self) -> O {
        self.descriptor
    }
}

impl<O: fmt::Debug> fmt::Display for Validated<O> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?} [{} proof]", self.descriptor, self.proof)
    }
}

/// Recompute what the constructors already proved.
///
/// `PROPOSALS.md` §1.2.2 describes `paranoid-validation` as a debug aid that
/// "recomputes logical facts inside an executor", explicitly not the normal
/// contract. That framing sets the shape of this API: it is a check a test or
/// a debug build runs, never something a kernel's correctness depends on.
///
/// The fact worth rechecking is the one `EXE-001` made a constructor
/// obligation — that the output can actually be indexed. If this ever returns
/// `Err`, a descriptor reached a backend without going through a checked
/// constructor, which is a bug in the lowering layer rather than in the
/// caller's shapes.
#[cfg(feature = "paranoid-validation")]
impl<O: super::spec::ExecutionDescriptor> Validated<O> {
    /// Re-derive the descriptor's own invariants and report disagreement.
    ///
    /// Available only under the `paranoid-validation` feature, so a release
    /// build cannot pay for it by accident.
    pub fn audit(&self) -> Result<(), ShapeError> {
        self.descriptor.output_shape().map_or(Ok(()), |shape| {
            shape
                .checked_numel(crate::shapes::error::OperationKind::Storage)
                .map(|_| ())
        })
    }
}

/// Assert a descriptor's invariants when `paranoid-validation` is on.
///
/// Expands to nothing otherwise, so an executor can call it on a hot path and
/// pay only in the builds that asked to pay. Takes a [`Validated`] because a
/// bare descriptor has nothing to audit *against*.
#[macro_export]
macro_rules! paranoid_audit {
    ($validated:expr) => {{
        #[cfg(feature = "paranoid-validation")]
        {
            $crate::exec::proof::__audit_or_panic(&$validated);
        }
        #[cfg(not(feature = "paranoid-validation"))]
        {
            let _ = &$validated;
        }
    }};
}

/// Panicking form of [`Validated::audit`], called by [`paranoid_audit!`].
///
/// Panics rather than returning, because a failure here means an unvalidated
/// descriptor is already in flight and there is no correct way to continue.
#[cfg(feature = "paranoid-validation")]
#[doc(hidden)]
pub fn __audit_or_panic<O: super::spec::ExecutionDescriptor>(validated: &Validated<O>) {
    if let Err(error) = validated.audit() {
        panic!("paranoid-validation: descriptor failed its own invariants: {error:?}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::catalog::NoAttributes;
    use crate::exec::{Descriptor, LogicalTensorMeta, op};
    use crate::prelude::{Dyn, ShapeBuf};
    use typenum::{U2, U3};
    type Static23 = crate::shapes::DimCons<U2, crate::shapes::DimCons<U3, crate::shapes::Nil>>;
    type Mixed23 = crate::shapes::DimCons<U2, crate::shapes::DimCons<usize, crate::shapes::Nil>>;
    type Mixed32 = crate::shapes::DimCons<usize, crate::shapes::DimCons<U3, crate::shapes::Nil>>;
    type MixedDyn =
        crate::shapes::DimCons<usize, crate::shapes::DimCons<usize, crate::shapes::Nil>>;

    // These live inside the module rather than in `tests/` because
    // `Validated::new` is `pub(crate)`. That is the point of the type, so the
    // seal is proved from outside by the `compile_fail` cases instead.

    #[test]
    fn a_fully_typed_shape_proves_itself_statically() {
        assert_eq!(ProofLevel::of::<Static23>(), ProofLevel::Static);
        assert_eq!(ProofLevel::of::<()>(), ProofLevel::Static);
    }

    #[test]
    fn one_runtime_axis_weakens_the_whole_shape() {
        assert_eq!(ProofLevel::of::<Mixed23>(), ProofLevel::Mixed);
        assert_eq!(ProofLevel::of::<Mixed32>(), ProofLevel::Mixed);
        assert_eq!(ProofLevel::of::<MixedDyn>(), ProofLevel::Mixed);
    }

    #[test]
    fn an_array_shape_knows_its_rank_but_not_its_extents() {
        assert_eq!(ProofLevel::of::<[usize; 3]>(), ProofLevel::Mixed);
        assert!(ProofLevel::of::<[usize; 3]>().has_static_rank());
        assert!(!ProofLevel::of::<[usize; 3]>().is_static());
    }

    #[test]
    fn an_unranked_shape_proves_nothing_before_it_exists() {
        assert_eq!(ProofLevel::of::<Dyn>(), ProofLevel::Dynamic);
        assert!(!ProofLevel::of::<Dyn>().has_static_rank());
    }

    #[test]
    fn a_named_axis_is_typed_but_not_statically_sized() {
        crate::dim!(Batch);
        // Naming an axis is what lets the compiler reject mixing it with a
        // different name. It says nothing about how large it is.
        assert_eq!(
            ProofLevel::of::<
                crate::shapes::DimCons<
                    crate::shapes::dim::NamedDim<Batch, usize>,
                    crate::shapes::DimCons<U3, crate::shapes::Nil>,
                >,
            >(),
            ProofLevel::Mixed
        );
    }

    #[test]
    fn meet_takes_the_weaker_operand() {
        assert_eq!(
            ProofLevel::Static.meet(ProofLevel::Mixed),
            ProofLevel::Mixed
        );
        assert_eq!(
            ProofLevel::Mixed.meet(ProofLevel::Dynamic),
            ProofLevel::Dynamic
        );
        assert_eq!(
            ProofLevel::Static.meet(ProofLevel::Static),
            ProofLevel::Static
        );
    }

    #[test]
    fn meet_is_commutative_idempotent_and_topped_by_static() {
        let levels = [ProofLevel::Static, ProofLevel::Mixed, ProofLevel::Dynamic];
        for a in levels {
            assert_eq!(a.meet(a), a, "idempotent");
            assert_eq!(ProofLevel::Static.meet(a), a, "Static is the identity");
            for b in levels {
                assert_eq!(a.meet(b), b.meet(a), "commutative");
                for c in levels {
                    assert_eq!(
                        a.meet(b).meet(c),
                        a.meet(b.meet(c)),
                        "associative, so folding N operands is order-independent"
                    );
                }
            }
        }
    }

    #[test]
    fn a_sealed_descriptor_still_reads_back_exactly() {
        let spec = Descriptor::<op::Add>::infer_runtime(
            NoAttributes,
            vec![
                LogicalTensorMeta {
                    shape: Some(ShapeBuf::from_slice(&[2, 3])),
                    dtype: None,
                    device: None,
                },
                LogicalTensorMeta {
                    shape: Some(ShapeBuf::from_slice(&[1, 3])),
                    dtype: None,
                    device: None,
                },
            ],
        )
        .unwrap()
        .into_descriptor();
        let validated = Validated::new(spec.clone(), ProofLevel::Mixed);

        assert_eq!(validated.descriptor(), &spec);
        assert_eq!(validated.proof_level(), ProofLevel::Mixed);
        assert_eq!(validated.into_descriptor(), spec);
    }

    #[test]
    fn the_proof_is_carried_not_recomputed() {
        // A `Static` stamp on a runtime-built descriptor is a lie the type
        // system cannot catch, which is exactly why `new` is crate-private:
        // only a lowering rule holding the shape types may call it.
        let spec = Descriptor::<op::Add>::infer_runtime(
            NoAttributes,
            vec![
                LogicalTensorMeta {
                    shape: Some(ShapeBuf::from_slice(&[2, 3])),
                    dtype: None,
                    device: None,
                },
                LogicalTensorMeta {
                    shape: Some(ShapeBuf::from_slice(&[2, 3])),
                    dtype: None,
                    device: None,
                },
            ],
        )
        .unwrap()
        .into_descriptor();
        let validated = Validated::new(spec, ProofLevel::of::<Static23>());
        assert!(validated.proof_level().is_static());
    }

    #[cfg(feature = "paranoid-validation")]
    #[test]
    fn audit_passes_for_a_descriptor_from_a_checked_constructor() {
        let spec = Descriptor::<op::Add>::infer_runtime(
            NoAttributes,
            vec![
                LogicalTensorMeta {
                    shape: Some(ShapeBuf::from_slice(&[2, 3])),
                    dtype: None,
                    device: None,
                },
                LogicalTensorMeta {
                    shape: Some(ShapeBuf::from_slice(&[1, 3])),
                    dtype: None,
                    device: None,
                },
            ],
        )
        .unwrap()
        .into_descriptor();
        let validated = Validated::new(spec, ProofLevel::Mixed);
        assert!(validated.audit().is_ok());
    }
}
