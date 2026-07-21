/// Core abstraction for `arg` within the Kindle framework..
pub mod arg;
/// Core abstraction for `arg_into` within the Kindle framework..
pub mod arg_into;
/// Core abstraction for `backend` within the Kindle framework..
pub mod backend;
/// Core abstraction for `base` within the Kindle framework..
pub mod base;
/// Core abstraction for `conv2d` within the Kindle framework..
pub mod conv2d;
/// Core abstraction for `device` within the Kindle framework..
pub mod device;
/// Core abstraction for `dtype` within the Kindle framework..
pub mod dtype;
/// Core abstraction for `grad` within the Kindle framework..
pub mod grad;
/// Core abstraction for `matmul` within the Kindle framework..
pub mod matmul;
/// Core abstraction for `ops` within the Kindle framework..
pub mod ops;
/// Core abstraction for `tracing` within the Kindle framework..
pub mod tracing;

/// Core abstraction for `prelude` within the Kindle framework..
pub mod prelude {
    pub use super::arg::*;
    pub use super::arg_into::*;
    pub use super::backend::*;
    pub use super::base::*;
    pub use super::conv2d::*;
    pub use super::device::*;
    pub use super::dtype::*;
    pub use super::grad::*;
    pub use super::matmul::*;
    pub use super::tracing::*;
}
