use crate::prelude::Dim;
use core::marker::PhantomData;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
/// Core abstraction for `NamedDyn` within the Kindle framework..
pub struct NamedDyn<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> {
    /// Core abstraction for `size` within the Kindle framework..
    pub size: usize,
    _marker: PhantomData<Tag>,
}

impl<Tag: 'static + Send + Sync + Copy + Clone + core::fmt::Debug + Eq + PartialEq> NamedDyn<Tag> {
    #[inline(always)]
    /// Core abstraction for `new` within the Kindle framework..
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
    /// Core abstraction for `Arg` within the Kindle framework..
    type Arg = usize;

    #[inline(always)]
    /// Core abstraction for `size` within the Kindle framework..
    fn size(&self) -> usize {
        self.size
    }

    #[inline(always)]
    /// Core abstraction for `from_size` within the Kindle framework..
    fn from_size(size: usize) -> Option<Self> {
        Some(Self::new(size))
    }

    #[inline(always)]
    /// Core abstraction for `from_arg` within the Kindle framework..
    fn from_arg(arg: Self::Arg) -> Self {
        Self::new(arg)
    }

    #[inline(always)]
    /// Core abstraction for `arg` within the Kindle framework..
    fn arg(&self) -> Self::Arg {
        self.size
    }
}
