pub trait Dim: 'static + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq {
    type Arg;
    fn size(&self) -> usize;
    fn from_size(size: usize) -> Option<Self>;
    fn from_arg(arg: Self::Arg) -> Self;
}

impl Dim for usize {
    type Arg = Self;

    #[inline(always)]
    fn size(&self) -> usize {
        *self
    }
    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        Some(size)
    }

    #[inline(always)]
    fn from_arg(arg: Self::Arg) -> Self {
        arg
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
                type Arg = usize;

                #[inline(always)]
                fn size(&self) -> usize {
                    self.0
                }

                #[inline(always)]
                fn from_size(size: usize) -> Option<Self> {
                    Some(Self(size))
                }

                #[inline(always)]
                fn from_arg(arg: Self::Arg) -> Self {
                    Self(arg)
                }
            }
        )+
    };
}

/// A mathematical product of two Dimensions `A` and `B`.
/// Preserves dimensionality statically across flatten operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProdDim<A, B>(pub usize, core::marker::PhantomData<(A, B)>);

impl<A: Dim, B: Dim> Dim for ProdDim<A, B> {
    type Arg = ();

    #[inline(always)]
    fn size(&self) -> usize {
        self.0
    }

    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        Some(Self(size, core::marker::PhantomData))
    }

    #[inline(always)]
    fn from_arg(_: Self::Arg) -> Self {
        Self(0, core::marker::PhantomData)
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
    type Arg = ();

    #[inline(always)]
    fn size(&self) -> usize {
        0
    }

    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        if size == 0 { Some(UTerm) } else { None }
    }

    #[inline(always)]
    fn from_arg(_: Self::Arg) -> Self {
        UTerm
    }
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
    type Arg = ();

    #[inline(always)]
    fn size(&self) -> usize {
        Self::USIZE
    }

    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        if size == Self::USIZE {
            Some(Default::default())
        } else {
            None
        }
    }

    #[inline(always)]
    fn from_arg(_: Self::Arg) -> Self {
        Default::default()
    }
}
