use crate::prelude::Dim;
use core::marker::PhantomData;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// `NamedDyn`.
pub struct NamedDyn<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> {
    /// `size`.
    pub size: usize,
    _marker: PhantomData<Tag>,
}

impl<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> NamedDyn<Tag> {
    #[inline(always)]
    /// Creates a new instance with default (statically inferred) shape arguments.
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
    /// `Arg`.
    type Arg = usize;

    #[inline(always)]
    /// `size`.
    fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    /// `from_size`.
    fn from_size(size: usize) -> Option<Self> {
        Some(Self::new(size))
    }

    #[inline(always)]
    /// `from_arg`.
    fn from_arg(arg: Self::Arg) -> Self {
        Self::new(arg)
    }

    #[inline(always)]
    /// `arg`.
    fn arg(&self) -> Self::Arg {
        self.size
    }
}
