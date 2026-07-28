// The implementation type and associated types must be public because they
// appear when the public `IncinBackend` alias is normalized. They are not
// re-exported from the public prelude.
pub(crate) mod backend;
pub(crate) mod executor;
pub(crate) mod gpu;
pub(crate) mod ops;
pub(crate) mod storage;
pub(crate) mod tape;

pub use backend::{CudaBackendImpl, CudaGrads, CudaVar};
