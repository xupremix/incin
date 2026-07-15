#[macro_use]
extern crate alloc;
// Only `WgpuBackend` and its associated types (`WgpuVar`, `WgpuGrads`) are
// intentional public API.  Everything else (dispatch helpers, pipeline cache,
// device state, raw buffer types) is an implementation detail and is
// `pub(crate)` only.
pub(crate) mod device;
pub(crate) mod storage;
pub(crate) mod pipeline;
pub(crate) mod backend;
pub(crate) mod dispatch;

// The three types a downstream crate legitimately needs:
//   - `WgpuBackend<T, D>` to parameterise `Tensor`
//   - `WgpuVar` returned by `CreationOps::var_*`
//   - `WgpuGrads` as `Backend::Grads`
pub use backend::{WgpuBackend, WgpuVar, WgpuGrads};

mod tests;
