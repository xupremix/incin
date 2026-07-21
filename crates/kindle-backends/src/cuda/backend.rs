use kindle_core::prelude::*;
use std::fmt::Debug;

/// A dedicated Cuda backend (Stub for now to allow separation).
#[derive(Debug, Clone, Copy)]
pub struct CudaBackend<K: DType, const N: usize = 0> {
    _marker: core::marker::PhantomData<K>,
}
