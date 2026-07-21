use crate::prelude::Dim;
use core::marker::PhantomData;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// Auto-generated documentation for NamedDyn.
pub struct NamedDyn<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> {
    /// Auto-generated documentation for size.
    pub size: usize,
    _marker: PhantomData<Tag>,
}

impl<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> NamedDyn<Tag> {
    #[inline(always)]
    /// Auto-generated documentation for new.
    pub fn new(size: usize) -> Self {
        Self {
            size,
            _marker: PhantomData,
        }
    }
}

impl<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> Dim
    for NamedDyn<Tag>
{
    /// Auto-generated documentation for Arg.
    type Arg = usize;

    #[inline(always)]
    /// Auto-generated documentation for size.
    fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    /// Auto-generated documentation for from_size.
    fn from_size(size: usize) -> Option<Self> {
        Some(Self::new(size))
    }

    #[inline(always)]
    /// Auto-generated documentation for from_arg.
    fn from_arg(arg: Self::Arg) -> Self {
        Self::new(arg)
    }

    #[inline(always)]
    /// Auto-generated documentation for arg.
    fn arg(&self) -> Self::Arg {
        self.size
    }
}
