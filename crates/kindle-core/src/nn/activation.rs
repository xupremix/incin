use crate::prelude::*;
use crate::nn::module::{Module, Parameters};
use alloc::vec::Vec;

#[derive(Debug, Clone, Default)]
pub struct ReLU;

impl<B: Backend<Dyn>> Parameters<B> for ReLU {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape, B: Backend<Dyn> + Backend<S>> Module<Tensor<S, B>> for ReLU {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.relu()
    }
}

#[derive(Debug, Clone, Default)]
pub struct GELU;

impl<B: Backend<Dyn>> Parameters<B> for GELU {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape, B: Backend<Dyn> + Backend<S>> Module<Tensor<S, B>> for GELU {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.gelu()
    }
}
