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

use typenum::{UTerm, UInt, Bit, Unsigned};

impl Dim for UTerm {
    type Arg = ();

    #[inline(always)]
    fn size(&self) -> usize { 0 }

    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        if size == 0 { Some(UTerm) } else { None }
    }

    #[inline(always)]
    fn from_arg(_: Self::Arg) -> Self { UTerm }
}

impl<U, B> Dim for UInt<U, B>
where
    U: Unsigned + Dim,
    B: Bit + Default + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq + 'static,
    UInt<U, B>: Unsigned + Default + Copy + Clone + core::fmt::Debug + Send + Sync + Eq + PartialEq + 'static,
{
    type Arg = ();

    #[inline(always)]
    fn size(&self) -> usize {
        Self::USIZE
    }

    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        if size == Self::USIZE { Some(Default::default()) } else { None }
    }

    #[inline(always)]
    fn from_arg(_: Self::Arg) -> Self { Default::default() }
}
