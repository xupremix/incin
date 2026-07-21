// Only `CudaBackend` and its associated types (`CudaVar`, `CudaGrads`) are
// intentional public API. Everything else (dispatch helpers, storage
// internals, tape) is an implementation detail and is `pub(crate)` only.
pub(crate) mod backend;
pub(crate) mod gpu;
pub(crate) mod ops;
pub(crate) mod storage;
pub(crate) mod tape;

pub use backend::{CudaBackend, CudaGrads, CudaVar};
