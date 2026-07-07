use crate::prelude::Dim;
use core::marker::PhantomData;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct NamedDyn<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> {
    pub size: usize,
    _marker: PhantomData<Tag>,
}

impl<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> NamedDyn<Tag> {
    #[inline(always)]
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
    type Arg = usize;

    #[inline(always)]
    fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    fn from_size(size: usize) -> Option<Self> {
        Some(Self::new(size))
    }

    #[inline(always)]
    fn from_arg(arg: Self::Arg) -> Self {
        Self::new(arg)
    }

    #[inline(always)]
    fn arg(&self) -> Self::Arg {
        self.size
    }
}
