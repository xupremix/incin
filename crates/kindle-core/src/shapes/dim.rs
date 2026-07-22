/// The core dimension trait, implemented by all types that can represent a single tensor axis.
///
/// A `Dim` is a type-level description of a single tensor dimension. It can be a compile-time
/// constant dimension (e.g. `typenum::U128` for a static 128-element axis), or a runtime value
/// (`usize` for a fully dynamic axis).
///
/// In practice, you rarely need to implement or use `Dim` directly. The `s![]` macro generates
/// the correct implementations automatically. Custom symbolic dimensions can be created via `symbolic_dim!`.
pub trait Dim: 'static + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq {
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

impl Dim for usize {
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
/// kindle_core::symbolic_dim!(Batch, Seq);
/// ```
#[macro_export]
macro_rules! symbolic_dim {
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
        )+
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
