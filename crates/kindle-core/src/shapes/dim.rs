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
