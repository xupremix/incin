#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    feature = "nightly",
    feature(generic_const_exprs),
    allow(incomplete_features)
)]

pub(crate) extern crate alloc;

pub mod err;
pub mod nn;
pub mod optim;
pub mod shapes;
pub mod tensor;

pub mod prelude {
    pub use super::err::*;
    pub use crate::nn::{
        activation::{GELU, ReLU, Sigmoid, Softmax, Swish, Tanh},
        conv2d::{Conv2d, Conv2dShape},
        linear::{Linear, LinearShape},
        module::{Module, Parameters, Sequential},
        param::Param,
    };
    pub use crate::seq;

    pub use super::shapes::prelude::*;
    pub use super::tensor::prelude::*;
    pub use crate::optim::{Gradients, Optimizer, SGD};
    pub use crate::shapes::dim::Dim;
    pub use crate::shapes::shape::{ConstShape, DynShape, PartialDynShape, Shape};
    pub use crate::symbolic_dim;
    pub use crate::tensor::backend::Backend;
    pub use typenum::{self, B0, B1, Bit, Diff, Prod, Quot, Sum, UInt, UTerm, Unsigned, consts::*};
}
