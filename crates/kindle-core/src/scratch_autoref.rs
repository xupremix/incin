use core::marker::PhantomData;

pub trait Backend {
    type RawVar;
}
pub struct B1;
impl Backend for B1 { type RawVar = i32; }

pub trait AutorefFallback<B: Backend> {
    fn maybe_extend(&self, _phantom: PhantomData<B>) {}
}
impl<T, B: Backend> AutorefFallback<B> for &&T {}

pub trait Autoref<B: Backend> {
    fn maybe_extend(&self, _phantom: PhantomData<B>) {}
}
pub struct Conv2d<B: Backend> {
    _b: PhantomData<B>,
}
impl<B: Backend> Autoref<B> for &Conv2d<B> {}

pub struct CNN {
    conv1: Conv2d<B1>,
}

impl CNN {
    pub fn test(&self) {
        let f = 5.0f32;
        (&&self.conv1).maybe_extend(PhantomData::<B1>); // Works!
        (&&f).maybe_extend(PhantomData::<B1>); // Works!
    }
}
