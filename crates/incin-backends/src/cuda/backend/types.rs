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

/// A trainable variable whose bytes live in CUDA memory.
#[derive(Clone)]
pub struct CudaVar {
    /// CUDA-resident storage backing this variable.
    pub storage: CudaStorage,
}

/// Gradient map type for the CUDA tape.
pub type CudaGrads = crate::cuda::tape::CudaGrads;
