/// Semantic tri-state extent classification for a dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StaticExtent {
    /// Extent is known only at runtime (e.g. dynamic batch or sequence length).
    RuntimeUnknown,
    /// Extent is statically known at compile-time to be `value`.
    Value(usize),
    /// Statically known to be invalid (e.g. overflow, underflow, or div-by-zero).
    Invalid,
}

impl StaticExtent {
    /// Returns the value only when the extent is a valid static proof.
    #[inline(always)]
    pub const fn value(self) -> Option<usize> {
        match self {
            StaticExtent::Value(value) => Some(value),
            StaticExtent::RuntimeUnknown | StaticExtent::Invalid => None,
        }
    }

    /// Validates whether a runtime `size` is consistent with this static extent.
    #[inline(always)]
    pub const fn validate_size(self, size: usize) -> bool {
        match self {
            StaticExtent::Value(val) => size == val,
            StaticExtent::RuntimeUnknown => true,
            StaticExtent::Invalid => false,
        }
    }
}

pub trait Dim: 'static + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq {
    /// The same semantic axis after a keep-dimension reduction.
    type KeepDim: Dim;
    /// Whether this axis's *size* is fixed by the type rather than supplied at
    /// runtime.
    const STATIC_SIZE: bool = matches!(Self::STATIC, StaticExtent::Value(_));

    /// Forces invalid type-level arithmetic to fail when the dimension is
    /// consumed by a shape operation.  Keeping this on the dimension trait
    /// lets symbolic expressions remain representable for diagnostics while
    /// preventing them from being silently downgraded to runtime checks.
    const STATIC_VALID: () = {
        if matches!(Self::STATIC, StaticExtent::Invalid) {
            panic!("invalid static dimension expression");
        }
    };

    /// This axis's size, when the type fixes it.
    const STATIC: StaticExtent = StaticExtent::RuntimeUnknown;

    /// Semantic name carried by this axis, when it is named.
    const NAME: Option<&'static str> = None;

    /// Tests semantic tag identity for the runtime named-lookup fallback.
    /// Untagged dimensions never match a named selector.
    #[inline]
    fn matches_tag<Tag: AxisTag>() -> bool {
        false
    }

    /// Returns the precise semantic static extent classification of this dimension.
    #[inline]
    fn static_extent(&self) -> StaticExtent {
        Self::STATIC
    }

    /// Returns a concrete value for a dimension whose type proves its extent.
    /// Runtime and invalid symbolic extents are rejected instead of being
    /// represented by a fabricated dimension value.
    #[inline]
    fn static_size() -> Result<usize, crate::shapes::error::ShapeError> {
        Self::STATIC
            .value()
            .ok_or(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: 1,
            })
    }

    /// The user-facing constructor argument (e.g. `()` for compile-time-
    /// fixed dimensions, `usize` for runtime-sized ones).
    type Arg: Clone + Default + core::fmt::Debug;
    /// Resolves an argument at the Shape/ShapeBuf boundary without retaining a
    /// dimension value.
    #[inline]
    fn resolve_arg(
        _arg: Self::Arg,
    ) -> core::result::Result<usize, crate::shapes::error::ShapeError> {
        match Self::STATIC {
            StaticExtent::Invalid => Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: 1,
            }),
            StaticExtent::Value(value) => {
                if Self::validate_size(value) {
                    Ok(value)
                } else {
                    Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                        operation: crate::shapes::error::OperationKind::Storage,
                        rank: 1,
                    })
                }
            }
            StaticExtent::RuntimeUnknown => {
                Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                    operation: crate::shapes::error::OperationKind::Storage,
                    rank: 1,
                })
            }
        }
    }

    /// Validates a runtime axis value without constructing a second runtime
    /// dimension representation.
    #[inline]
    fn validate_size(size: usize) -> bool {
        Self::STATIC.validate_size(size)
    }
}

/// Projects only genuinely concrete static extents into the arithmetic
/// namespace.  In particular, a semantic named axis is static only when its
/// extent is static too; `NamedDim<Tag, usize>` intentionally does not
/// implement this trait.
pub trait ConcreteStaticExtent: Dim {
    /// The underlying typenum natural used by static arithmetic.
    type Nat: typenum::Unsigned;
}

#[doc(hidden)]
pub trait BroadcastChoice<WhenOne, WhenZero> {
    type Output;
}

impl<WhenOne, WhenZero> BroadcastChoice<WhenOne, WhenZero> for typenum::B1 {
    type Output = WhenOne;
}

impl<WhenOne, WhenZero> BroadcastChoice<WhenOne, WhenZero> for typenum::B0 {
    type Output = WhenZero;
}

#[doc(hidden)]
pub type BroadcastSameNat<L, R> =
    <<L as typenum::IsEqual<R>>::Output as BroadcastChoice<L, ()>>::Output;
#[doc(hidden)]
pub type BroadcastRightNat<L, R> =
    <<R as typenum::IsEqual<typenum::U1>>::Output as BroadcastChoice<L, BroadcastSameNat<L, R>>>::Output;
#[doc(hidden)]
pub type BroadcastStaticNat<L, R> =
    <<L as typenum::IsEqual<typenum::U1>>::Output as BroadcastChoice<R, BroadcastRightNat<L, R>>>::Output;

impl<L, R> ConcreteStaticExtent for BroadcastExtent<L, R>
where
    L: Dim + ConcreteStaticExtent,
    R: Dim + ConcreteStaticExtent,
    L::Nat: typenum::Unsigned + typenum::IsEqual<typenum::U1> + typenum::IsEqual<R::Nat>,
    R::Nat: typenum::Unsigned + typenum::IsEqual<typenum::U1>,
    <L::Nat as typenum::IsEqual<typenum::U1>>::Output:
        BroadcastChoice<R::Nat, BroadcastRightNat<L::Nat, R::Nat>>,
    <R::Nat as typenum::IsEqual<typenum::U1>>::Output:
        BroadcastChoice<L::Nat, BroadcastSameNat<L::Nat, R::Nat>>,
    <L::Nat as typenum::IsEqual<R::Nat>>::Output: BroadcastChoice<L::Nat, ()>,
    BroadcastStaticNat<L::Nat, R::Nat>: typenum::Unsigned,
{
    type Nat = BroadcastStaticNat<L::Nat, R::Nat>;
}

impl<T> ConcreteStaticExtent for T
where
    T: Dim + typenum::Unsigned,
{
    type Nat = T;
}

/// Semantic identity carried by a named axis.
///
/// Implementations are zero-sized tags. Runtime extent values and static
/// extent knowledge belong to the `NamedDim<Tag, Extent>` extent parameter and
/// to the validated `ShapeBuf`, never to the tag itself.
pub trait AxisTag:
    'static + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq
{
    const NAME: &'static str;

    /// Creates the canonical semantic selector for this axis tag.
    ///
    /// The selector contains only the tag. It resolves its current position
    /// against the shape at the operation boundary, so transposing a named
    /// axis cannot leave a stale position behind.
    #[inline]
    fn selector() -> crate::shapes::idx::NamedAxisSelector<Self>
    where
        Self: Sized,
    {
        crate::shapes::idx::NamedAxisSelector::default()
    }
}

/// Namespace marker used by a group of tags declared in one `dim!` call.
/// The marker is type-level only and never stores an extent or a position.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct AxisSchema<Root>(core::marker::PhantomData<Root>);

/// Stable semantic identity for a named axis.
///
/// `Id` is assigned within `Schema` by the declaration macro. It is an
/// identity key, not a current structural position. Exact named lookup is
/// deliberately not inferred from this trait alone; callers that cannot prove
/// a unique structural match must resolve the name at runtime.
pub trait AxisIdentity: AxisTag {
    type Schema: 'static;
    type Id: typenum::Unsigned;
}

/// Semantic proof boundary for comparing two dimensions statically or dynamically.
pub trait DimCompatible<Rhs: Dim>: Dim {
    const STATIC_ASSERT: () = match (Self::STATIC, Rhs::STATIC) {
        (StaticExtent::Invalid, _) | (_, StaticExtent::Invalid) => {
            panic!("Invalid static dimension expression");
        }
        (StaticExtent::Value(lhs), StaticExtent::Value(rhs)) => {
            assert!(lhs == rhs, "Statically incompatible dimensions");
        }
        _ => (),
    };

    /// Runtime compatibility check between two dimension instances.
    fn check_compatible(&self, _rhs: &Rhs) -> Result<(), crate::shapes::error::ShapeError> {
        Self::STATIC_ASSERT;
        // Runtime extents are compared by ShapeBuf, where both values are
        // available. A dimension specification cannot compare values it does
        // not own.
        Ok(())
    }
}

impl<L: Dim, R: Dim> DimCompatible<R> for L {}

const fn broadcast_static(lhs: StaticExtent, rhs: StaticExtent) -> StaticExtent {
    match (lhs, rhs) {
        (StaticExtent::Invalid, _) | (_, StaticExtent::Invalid) => StaticExtent::Invalid,
        (StaticExtent::Value(lhs), StaticExtent::Value(rhs)) => {
            if lhs == rhs {
                StaticExtent::Value(lhs)
            } else if lhs == 1 {
                StaticExtent::Value(rhs)
            } else if rhs == 1 {
                StaticExtent::Value(lhs)
            } else {
                StaticExtent::Invalid
            }
        }
        (StaticExtent::Value(1), other) | (other, StaticExtent::Value(1)) => other,
        (StaticExtent::Value(value), StaticExtent::RuntimeUnknown)
        | (StaticExtent::RuntimeUnknown, StaticExtent::Value(value)) => {
            if value == 1 {
                StaticExtent::RuntimeUnknown
            } else {
                StaticExtent::Value(value)
            }
        }
        _ => StaticExtent::RuntimeUnknown,
    }
}

/// Symbolic output extent for a broadcast pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct BroadcastExtent<L, R>(pub core::marker::PhantomData<(L, R)>);

impl<L: Dim, R: Dim> Dim for BroadcastExtent<L, R> {
    type KeepDim = typenum::U1;
    const STATIC: StaticExtent = broadcast_static(L::STATIC, R::STATIC);
    type Arg = usize;
    fn resolve_arg(
        arg: Self::Arg,
    ) -> core::result::Result<usize, crate::shapes::error::ShapeError> {
        match Self::STATIC {
            StaticExtent::Invalid => Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: 1,
            }),
            StaticExtent::Value(value) if value == arg => Ok(value),
            StaticExtent::Value(_) => Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: 1,
            }),
            StaticExtent::RuntimeUnknown => Ok(arg),
        }
    }
}

impl Dim for usize {
    type KeepDim = typenum::U1;
    /// A runtime dimension's argument is just its size.
    type Arg = Self;

    fn resolve_arg(
        arg: Self::Arg,
    ) -> core::result::Result<usize, crate::shapes::error::ShapeError> {
        Ok(arg)
    }
}

/// Generates a Named Tensor dimension (symbolic dimension).
/// This creates a strong type that wraps `usize` for runtime shape tracking,
/// ensuring that symbolic dimensions match at compile time.
///
/// ```rust
/// incin_core::dim!(Batch, Seq);
/// ```
#[macro_export]
macro_rules! dim {
    ($( $(#[$meta:meta])* $name:ident ),+ $(,)?) => {
        $crate::__incin_dim_declare!(@first $( $(#[$meta])* $name ),+ );
    };
}

#[doc(hidden)]
#[macro_export]
macro_rules! __incin_dim_declare {
    (@first $(#[$first_meta:meta])* $first:ident $(, $(#[$rest_meta:meta])* $rest:ident)* ) => {
        $(#[$first_meta])*
        #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $first;

        impl $crate::shapes::AxisTag for $first {
            const NAME: &'static str = stringify!($first);
        }
        impl $crate::shapes::AxisIdentity for $first {
            type Schema = $crate::shapes::AxisSchema<$first>;
            type Id = $crate::typenum::U0;
        }

        $crate::__incin_dim_declare!(@rest $crate::shapes::AxisSchema<$first>; $crate::typenum::U1; $( $(#[$rest_meta])* $rest ),* );
    };
    (@rest $schema:ty; $id:ty; ) => {};
    (@rest $schema:ty; $id:ty; $(#[$meta:meta])* $name:ident $(, $(#[$rest_meta:meta])* $rest:ident)* ) => {
        $(#[$meta])*
        #[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name;

        impl $crate::shapes::AxisTag for $name {
            const NAME: &'static str = stringify!($name);
        }
        impl $crate::shapes::AxisIdentity for $name {
            type Schema = $schema;
            type Id = $id;
        }

        $crate::__incin_dim_declare!(@rest $schema; $crate::typenum::Sum<$id, $crate::typenum::U1>; $( $(#[$rest_meta])* $rest ),* );
    };
}

/// A const-generic adapter for fixed extents that Stable Rust cannot expose as
/// a proc-macro literal (for example `shape![const Model::WIDTH]`).
///
/// Raw literals use the macro's recursive typenum representation and therefore
/// retain typenum arithmetic. `ConstDim` is intentionally only the semantic
/// adapter for an unevaluated const path; it is not a finite literal catalogue
/// and does not claim the same normalized arithmetic output types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct ConstDim<const N: usize>;

impl<const N: usize> Dim for ConstDim<N> {
    type KeepDim = typenum::U1;
    const STATIC: StaticExtent = StaticExtent::Value(N);
    type Arg = ();
}

macro_rules! static_op {
    ($( $id1:ident $id2:ident )+) => {
        $(
            const fn $id1(lhs: StaticExtent, rhs: StaticExtent) -> StaticExtent {
                match (lhs, rhs) {
                    (StaticExtent::Invalid, _) | (_, StaticExtent::Invalid) => StaticExtent::Invalid,
                    (StaticExtent::Value(lhs), StaticExtent::Value(rhs)) => match lhs.$id2(rhs) {
                        Some(v) => StaticExtent::Value(v),
                        None => StaticExtent::Invalid,
                    },
                    _ => StaticExtent::RuntimeUnknown,
                }
            }
        )+
    };
}

static_op! {
    static_mul checked_mul
    static_add checked_add
    static_sub checked_sub
}

const fn static_exact_div(lhs: StaticExtent, rhs: StaticExtent) -> StaticExtent {
    match (lhs, rhs) {
        (StaticExtent::Invalid, _)
        | (_, StaticExtent::Invalid)
        | (StaticExtent::Value(_), StaticExtent::Value(0)) => StaticExtent::Invalid,
        (StaticExtent::Value(lhs), StaticExtent::Value(rhs)) => {
            if lhs % rhs == 0 {
                StaticExtent::Value(lhs / rhs)
            } else {
                StaticExtent::Invalid
            }
        }
        _ => StaticExtent::RuntimeUnknown,
    }
}

macro_rules! static_op_dim {
    ( $name:ident $op:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
        pub struct $name<A, B>(pub core::marker::PhantomData<(A, B)>);

        impl<A: Dim, B: Dim> Dim for $name<A, B> {
            type KeepDim = typenum::U1;
            const STATIC: StaticExtent = $op(A::STATIC, B::STATIC);

            type Arg = usize;

            fn resolve_arg(
                arg: Self::Arg,
            ) -> core::result::Result<usize, crate::shapes::error::ShapeError> {
                match Self::STATIC {
                    StaticExtent::Invalid => {
                        Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                            operation: crate::shapes::error::OperationKind::Storage,
                            rank: 1,
                        })
                    }
                    StaticExtent::Value(value) if value == arg => Ok(value),
                    StaticExtent::Value(_) => {
                        Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                            operation: crate::shapes::error::OperationKind::Storage,
                            rank: 1,
                        })
                    }
                    StaticExtent::RuntimeUnknown => Ok(arg),
                }
            }
        }
    };
}

static_op_dim!( AddDim static_add );

static_op_dim!( CheckedSubDim static_sub );

static_op_dim!( ExactDivDim static_exact_div);

/// A dimension that pairs a semantic tag with a dimension extent (e.g. `NamedDim<Channels, ConstDim<64>>`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct NamedDim<Tag, Extent> {
    _tag: core::marker::PhantomData<(Tag, Extent)>,
}

impl<Tag, Extent> ConcreteStaticExtent for NamedDim<Tag, Extent>
where
    Tag: AxisTag,
    Extent: ConcreteStaticExtent,
{
    type Nat = Extent::Nat;
}

impl<Tag: AxisTag, Extent: Dim> NamedDim<Tag, Extent> {
    /// Constructs the zero-sized semantic axis marker.
    ///
    /// Runtime extents are carried by `ShapeBuf`, not by this type. Static
    /// extents remain part of `Extent` and therefore remain available to the
    /// type system without storing a duplicate value.
    pub const fn new() -> Self {
        Self {
            _tag: core::marker::PhantomData,
        }
    }
}

impl<Tag: AxisTag, Extent: Dim> Dim for NamedDim<Tag, Extent> {
    type KeepDim = NamedDim<Tag, typenum::U1>;
    const STATIC: StaticExtent = Extent::STATIC;
    const NAME: Option<&'static str> = Some(Tag::NAME);
    type Arg = Extent::Arg;

    #[inline]
    fn matches_tag<Other: AxisTag>() -> bool {
        core::any::TypeId::of::<Tag>() == core::any::TypeId::of::<Other>()
    }

    fn resolve_arg(
        arg: Self::Arg,
    ) -> core::result::Result<usize, crate::shapes::error::ShapeError> {
        Extent::resolve_arg(arg)
    }
}

/// A checked product of two dimension specifications `A` and `B`.
///
/// Used internally to track the resulting size when two dimensions are flattened or multiplied.
/// Runtime values are supplied through the surrounding `ShapeBuf`; this type
/// carries only the symbolic product contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct MulDim<A, B>(pub core::marker::PhantomData<(A, B)>);

impl<A: Dim, B: Dim> Dim for MulDim<A, B> {
    type KeepDim = typenum::U1;
    /// The product of the factors' extents, checked. An overflowing product
    /// answers `None` rather than a wrapped constant: a wrong extent baked into
    /// a kernel is worse than no extent at all.
    const STATIC: StaticExtent = match (A::STATIC, B::STATIC) {
        (StaticExtent::Value(a), StaticExtent::Value(b)) => match a.checked_mul(b) {
            Some(value) => StaticExtent::Value(value),
            None => StaticExtent::Invalid,
        },
        (StaticExtent::Invalid, _) | (_, StaticExtent::Invalid) => StaticExtent::Invalid,
        _ => StaticExtent::RuntimeUnknown,
    };

    type Arg = usize;

    fn resolve_arg(
        arg: Self::Arg,
    ) -> core::result::Result<usize, crate::shapes::error::ShapeError> {
        match Self::STATIC {
            StaticExtent::Invalid => Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: 1,
            }),
            StaticExtent::Value(value) if value == arg => Ok(value),
            StaticExtent::Value(_) => Err(crate::shapes::error::ShapeError::TargetShapeRejected {
                operation: crate::shapes::error::OperationKind::Storage,
                rank: 1,
            }),
            StaticExtent::RuntimeUnknown => Ok(arg),
        }
    }
}

use typenum::{Bit, UInt, UTerm, Unsigned};

use crate::exec::ProofLevel::Static;

impl Dim for UTerm {
    type KeepDim = typenum::U1;
    /// Zero, which is a real extent and not a missing one.
    const STATIC: StaticExtent = StaticExtent::Value(0);

    /// No argument needed — `UTerm` (typenum's zero) is always size 0.
    type Arg = ();
}

impl<U, B> Dim for UInt<U, B>
where
    U: Unsigned + Dim,
    B: Bit + Default + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq + 'static,
    UInt<U, B>: Unsigned
        + Default
        + Copy
        + Clone
        + core::fmt::Debug
        + Send
        + Sync
        + Eq
        + PartialEq
        + 'static,
{
    type KeepDim = typenum::U1;
    /// The `typenum` value, which is where every static extent comes from.
    const STATIC: StaticExtent = StaticExtent::Value(<Self as Unsigned>::USIZE);

    /// No argument needed — the size is fixed by the `typenum` type itself.
    type Arg = ();
}
