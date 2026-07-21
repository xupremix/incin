/// The core dimension trait, implemented by all types that can represent a single tensor axis.
///
/// A `Dim` is a type-level description of a single tensor dimension. It can be a compile-time
/// constant dimension (e.g. `typenum::U128` for a static 128-element axis), or a runtime value
/// (`usize` for a fully dynamic axis).
///
/// In practice, you rarely need to implement or use `Dim` directly. The `s![]` macro generates
/// the correct implementations automatically. Custom symbolic dimensions can be created via `symbolic_dim!`.
pub trait Dim: 'static + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq {
    /// Auto-generated documentation for Arg.
    type Arg: Clone + Default + core::fmt::Debug;
    /// Auto-generated documentation for size.
    fn size(&self) -> usize;
    /// Auto-generated documentation for from_size.
    fn from_size(size: usize) -> Option<Self>;
    /// Auto-generated documentation for from_arg.
    fn from_arg(arg: Self::Arg) -> Self;
    /// Auto-generated documentation for arg.
    fn arg(&self) -> Self::Arg;
}

impl Dim for usize {
    /// Auto-generated documentation for Arg.
    type Arg = Self;

    #[inline(always)]
    /// Auto-generated documentation for size.
    fn size(&self) -> usize {
        *self
    }
    #[inline(always)]
    /// Auto-generated documentation for from_size.
    fn from_size(size: usize) -> Option<Self> {
        Some(size)
    }

    #[inline(always)]
    /// Auto-generated documentation for from_arg.
    fn from_arg(arg: Self::Arg) -> Self {
        arg
    }

    #[inline(always)]
    /// Auto-generated documentation for arg.
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
    ($($name:ident),+ $(,)?) => {
        $(
            #[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
            pub struct $name(pub usize);

            impl $crate::prelude::Dim for $name {
                /// Auto-generated documentation for Arg.
                type Arg = usize;

                #[inline(always)]
                /// Auto-generated documentation for size.
                fn size(&self) -> usize {
                    self.0
                }

                #[inline(always)]
                /// Auto-generated documentation for from_size.
                fn from_size(size: usize) -> Option<Self> {
                    Some(Self(size))
                }

                #[inline(always)]
                /// Auto-generated documentation for from_arg.
                fn from_arg(arg: Self::Arg) -> Self {
                    Self(arg)
                }

                #[inline(always)]
                /// Auto-generated documentation for arg.
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
    /// Auto-generated documentation for Arg.
    type Arg = (A::Arg, B::Arg);

    #[inline(always)]
    /// Auto-generated documentation for size.
    fn size(&self) -> usize {
        self.0
    }

    #[inline(always)]
    /// Auto-generated documentation for from_size.
    fn from_size(size: usize) -> Option<Self> {
        Some(Self(size, core::marker::PhantomData))
    }

    #[inline(always)]
    /// Auto-generated documentation for from_arg.
    fn from_arg(arg: Self::Arg) -> Self {
        let a = A::from_arg(arg.0);
        let b = B::from_arg(arg.1);
        Self(a.size() * b.size(), core::marker::PhantomData)
    }

    #[inline(always)]
    /// Auto-generated documentation for arg.
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
    /// Auto-generated documentation for Arg.
    type Arg = ();

    #[inline(always)]
    /// Auto-generated documentation for size.
    fn size(&self) -> usize {
        0
    }

    #[inline(always)]
    /// Auto-generated documentation for from_size.
    fn from_size(size: usize) -> Option<Self> {
        if size == 0 { Some(UTerm) } else { None }
    }

    #[inline(always)]
    /// Auto-generated documentation for from_arg.
    fn from_arg(_: Self::Arg) -> Self {
        UTerm
    }

    #[inline(always)]
    /// Auto-generated documentation for arg.
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
    /// Auto-generated documentation for Arg.
    type Arg = ();

    #[inline(always)]
    /// Auto-generated documentation for size.
    fn size(&self) -> usize {
        Self::USIZE
    }

    #[inline(always)]
    /// Auto-generated documentation for from_size.
    fn from_size(size: usize) -> Option<Self> {
        if size == Self::USIZE {
            Some(Default::default())
        } else {
            None
        }
    }

    #[inline(always)]
    /// Auto-generated documentation for from_arg.
    fn from_arg(_: Self::Arg) -> Self {
        Default::default()
    }

    #[inline(always)]
    /// Auto-generated documentation for arg.
    fn arg(&self) -> Self::Arg {}
}
