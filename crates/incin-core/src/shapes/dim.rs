/// The core dimension trait, implemented by all types that can represent a single tensor axis.
///
/// A `Dim` is a type-level description of a single tensor dimension. It can be a compile-time
/// constant dimension (e.g. `typenum::U128` for a static 128-element axis), or a runtime value
/// (`usize` for a fully dynamic axis).
///
/// In practice, you rarely need to implement or use `Dim` directly. The `s![]` macro generates
/// the correct implementations automatically. Custom symbolic dimensions can be created via `sym_dim!`.
pub trait Dim: 'static + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq {
    /// Whether this axis's *size* is fixed by the type rather than supplied at
    /// runtime.
    ///
    /// This is the per-axis input to
    /// [`ProofLevel`](crate::exec::ProofLevel): a shape whose every axis is
    /// statically sized carries a `Static` proof, and one axis that is not
    /// weakens the whole shape to `Mixed`. Note that a *named* dimension
    /// (`dim!(Batch)`) is `false` here — naming an axis makes it distinct in
    /// the type system, which is why it can be checked at compile time, but its
    /// size is still a runtime value.
    ///
    /// It defaults to `false` so that a `Dim` implemented outside this crate
    /// claims no static proof it has not demonstrated. Claiming `true`
    /// incorrectly would let a descriptor be built on a size the compiler never
    /// checked.
    const STATIC_SIZE: bool = false;

    /// The user-facing constructor argument (e.g. `()` for compile-time-
    /// fixed dimensions, `usize` for runtime-sized ones).
    type Arg: Clone + Default + core::fmt::Debug;
    /// Returns this dimension's size.
    fn size(&self) -> usize;
    /// Attempts to construct this dimension from a runtime `size`,
    /// returning `None` if `size` doesn't match a compile-time-fixed value.
    fn from_size(size: usize) -> Option<Self>;
    /// Constructs this dimension from its constructor argument.
    fn from_arg(arg: Self::Arg) -> Self;
    /// Returns the constructor argument that would reproduce this dimension.
    fn arg(&self) -> Self::Arg;
}

/// A dimension whose *type* is not the literal `U1`.
///
/// Broadcasting stretches an axis of extent 1 to meet its partner, so the rule
/// relating a pair of axes has three cases: the two types agree, the left is
/// `U1`, or the right is `U1`. Those cases only stay disjoint — and the impls
/// expressing them only stay coherent — if "not `U1`" is sayable, and Rust has
/// no negative bound to say it with. This marker says it structurally instead.
///
/// A canonical `typenum` value is either `UTerm`, which is zero, or a
/// `UInt<U, B>`, and `U1` is the single shape `UInt<UTerm, B1>`. So everything
/// that is not `U1` is either `UTerm` or has a nested `UInt` in its high bits,
/// which is exactly what the two impls below name. A `dim!` name carries a
/// runtime size and is never the type `U1`, so the macro implements this for
/// each name it defines.
///
/// A `usize` axis is deliberately absent: it is not the type `U1`, but neither
/// is it statically sized, and the mixed broadcast families relate it by their
/// own rules.
pub trait NotOne: Dim {}

/// `UTerm` is typenum's zero.
impl NotOne for UTerm {}

/// A nested `UInt` in the high bits means a value of 2 or more.
impl<U, B, C> NotOne for UInt<UInt<U, B>, C> where UInt<UInt<U, B>, C>: Dim {}

impl Dim for usize {
    /// The canonical runtime axis: its size arrives with the data.
    const STATIC_SIZE: bool = false;

    /// A runtime dimension's argument is just its size.
    type Arg = Self;

    #[inline(always)]
    /// Itself.
    fn size(&self) -> usize {
        *self
    }
    #[inline(always)]
    /// Always succeeds — any `usize` is a valid runtime dimension.
    fn from_size(size: usize) -> Option<Self> {
        Some(size)
    }

    #[inline(always)]
    /// Identity.
    fn from_arg(arg: Self::Arg) -> Self {
        arg
    }

    #[inline(always)]
    /// Identity.
    fn arg(&self) -> Self::Arg {
        *self
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
        $(
            $(#[$meta])*
            #[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
            pub struct $name(pub usize);

            impl $crate::prelude::Dim for $name {
                /// The wrapped runtime size.
                type Arg = usize;

                #[inline(always)]
                /// The wrapped size.
                fn size(&self) -> usize {
                    self.0
                }

                #[inline(always)]
                /// Always succeeds — wraps any `usize`.
                fn from_size(size: usize) -> Option<Self> {
                    Some(Self(size))
                }

                #[inline(always)]
                /// Wraps `arg`.
                fn from_arg(arg: Self::Arg) -> Self {
                    Self(arg)
                }

                #[inline(always)]
                /// Unwraps the size.
                fn arg(&self) -> Self::Arg {
                    self.0
                }
            }

            impl $crate::prelude::StaticOrNamedDim for $name {}

            // A name is a distinct type, never the type `U1`, so it may sit on
            // the non-stretched side of a broadcast. Its runtime size may still
            // be 1; that is a value, and nothing here claims otherwise.
            impl $crate::prelude::NotOne for $name {}
        )+
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! sym_dim {
    ($($tokens:tt)*) => {
        $crate::dim!($($tokens)*);
    };
}

#[macro_export]
#[doc(hidden)]
macro_rules! symbolic_dim {
    ($($tokens:tt)*) => {
        $crate::dim!($($tokens)*);
    };
}

/// A mathematical product of two Dimensions `A` and `B`.
///
/// Used internally to track the resulting size when two dimensions are flattened or multiplied.
/// For example, after `t.reshape::<s![-1, A_times_B]>()`, the last dimension's type would be `ProdDim<A, B>`.
/// It preserves static dimensionality information across such operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProdDim<A, B>(pub usize, core::marker::PhantomData<(A, B)>);

impl<A: Dim, B: Dim> Dim for ProdDim<A, B> {
    /// A product is statically sized exactly when both factors are. This is
    /// what keeps a `reshape` that folds two static axes together from
    /// degrading the result's proof to `Mixed`.
    const STATIC_SIZE: bool = A::STATIC_SIZE && B::STATIC_SIZE;

    /// The pair of constituent dimensions' own arguments.
    type Arg = (A::Arg, B::Arg);

    #[inline(always)]
    /// The precomputed product `A::size() * B::size()`.
    fn size(&self) -> usize {
        self.0
    }

    #[inline(always)]
    /// Always succeeds — any `usize` product is accepted as-is (the
    /// individual `A`/`B` factors are not recoverable from the product alone).
    fn from_size(size: usize) -> Option<Self> {
        Some(Self(size, core::marker::PhantomData))
    }

    #[inline(always)]
    /// Constructs `A`/`B` from their arguments and stores their product.
    fn from_arg(arg: Self::Arg) -> Self {
        let a = A::from_arg(arg.0);
        let b = B::from_arg(arg.1);
        Self(a.size() * b.size(), core::marker::PhantomData)
    }

    #[inline(always)]
    /// Returns `A`/`B`'s default arguments (the actual product size is
    /// tracked separately in `self.0`, not reconstructible from `Arg` alone).
    fn arg(&self) -> Self::Arg {
        (
            <A::Arg as core::default::Default>::default(),
            <B::Arg as core::default::Default>::default(),
        )
    }
}

impl<A: Dim + Default, B: Dim + Default> Default for ProdDim<A, B> {
    fn default() -> Self {
        Self(
            A::default().size() * B::default().size(),
            core::marker::PhantomData,
        )
    }
}

use typenum::{Bit, UInt, UTerm, Unsigned};

impl Dim for UTerm {
    /// Fixed by the type: `UTerm` is typenum's zero and denotes size 0.
    const STATIC_SIZE: bool = true;

    /// No argument needed — `UTerm` (typenum's zero) is always size 0.
    type Arg = ();

    #[inline(always)]
    /// Always 0.
    fn size(&self) -> usize {
        0
    }

    #[inline(always)]
    /// Succeeds only for `size == 0`.
    fn from_size(size: usize) -> Option<Self> {
        if size == 0 { Some(UTerm) } else { None }
    }

    #[inline(always)]
    /// No-op: `UTerm` has only one value.
    fn from_arg(_: Self::Arg) -> Self {
        UTerm
    }

    #[inline(always)]
    /// No-op.
    fn arg(&self) -> Self::Arg {}
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
    /// Fixed by the type: the size is the `typenum` value itself.
    const STATIC_SIZE: bool = true;

    /// No argument needed — the size is fixed by the `typenum` type itself.
    type Arg = ();

    #[inline(always)]
    /// The compile-time-known `typenum` value.
    fn size(&self) -> usize {
        Self::USIZE
    }

    #[inline(always)]
    /// Succeeds only when `size` matches this exact compile-time value.
    fn from_size(size: usize) -> Option<Self> {
        if size == Self::USIZE {
            Some(Default::default())
        } else {
            None
        }
    }

    #[inline(always)]
    /// No-op: the value is fixed by the type, not the argument.
    fn from_arg(_: Self::Arg) -> Self {
        Default::default()
    }

    #[inline(always)]
    /// No-op.
    fn arg(&self) -> Self::Arg {}
}
