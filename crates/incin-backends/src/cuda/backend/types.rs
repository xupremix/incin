//! `CudaBackendImpl`, `CudaVar`, and the tape-gradient-set alias every
//! other module in this split attaches its impls to.

use super::*;

/// CUDA compute backend implementation for Incin.
#[derive(Clone)]
pub struct CudaBackendImpl<D = Cuda>(core::marker::PhantomData<D>);

impl<D> CudaBackendImpl<D> {
    /// Construct the stateless CUDA executor.
    #[must_use]
    pub const fn new() -> Self {
        Self(core::marker::PhantomData)
    }
}

impl<D> Default for CudaBackendImpl<D> {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone)]
pub struct CudaVar {
    pub storage: CudaStorage,
}

pub type CudaGrads = crate::cuda::tape::CudaGrads;
