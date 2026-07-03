#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    feature = "nightly",
    feature(generic_const_exprs),
    allow(incomplete_features)
)]

pub(crate) extern crate alloc;

pub mod err;
pub mod shapes;
pub mod tensor;

pub mod prelude {
    pub use super::err::*;
    pub use super::shapes::prelude::*;
    pub use super::tensor::prelude::*;
}
