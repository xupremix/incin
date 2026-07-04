use crate::nn::module::{Module, Parameters};
use crate::prelude::*;
use alloc::vec::Vec;

#[derive(Debug, Clone, Default)]
pub struct ReLU;

impl<B: Backend> Parameters<B> for ReLU {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for ReLU {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.relu()
    }
}

#[derive(Debug, Clone, Default)]
pub struct GELU;

impl<B: Backend> Parameters<B> for GELU {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for GELU {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.gelu()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Swish;

impl<B: Backend> Parameters<B> for Swish {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Swish {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.swish()
    }
}

#[derive(Debug, Clone)]
pub struct Softmax {
    pub dim: usize,
}

impl Softmax {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
}

impl<B: Backend> Parameters<B> for Softmax {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Softmax {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.softmax(self.dim)
    }
}

#[derive(Debug, Clone, Default)]
pub struct Sigmoid;

impl<B: Backend> Parameters<B> for Sigmoid {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Sigmoid {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.sigmoid()
    }
}

#[derive(Debug, Clone, Default)]
pub struct Tanh;

impl<B: Backend> Parameters<B> for Tanh {
    fn parameters(&self) -> Vec<B::RawVar> {
        Vec::new()
    }
}

impl<S: Shape + DynShape, B: Backend> Module<Tensor<S, B>> for Tanh {
    type Output = Tensor<S, B>;
    type Error = Error;

    #[inline]
    fn forward(&self, x: Tensor<S, B>) -> core::result::Result<Tensor<S, B>, Error> {
        x.tanh()
    }
}
