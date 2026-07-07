//! # Kindle Core
//!
//! `kindle-core` provides the fundamental abstractions, traits, and types for the Kindle framework.
//! It encapsulates the tensor representation, neural network layers, shape systems, optimization traits, and serialization support.
//!
//! ## Architecture
//!
//! The core is divided into several essential modules:
//! * **`shapes`**: Implements the type-level shapes using `typenum`. This allows Kindle to perform compile-time shape verification for all common tensor operations, ensuring operations like matrix multiplication, convolution, and concatenation are mathematically sound before the code even runs.
//! * **`tensor`**: Defines the central `Tensor<S: Shape, B: Backend>` abstraction. It defines how tensors interact with their underlying compute backends and defines all the mathematical operations available.
//! * **`nn`**: Provides high-level neural network components (Modules, Linear, Conv2d, BatchNorm2d) which can be composed to build larger models.
//! * **`optim`**: Interfaces for optimization (e.g., SGD).
//! * **`serialize`**: Defines `Serializer` and `Deserializer` traits for loading weights from disk (e.g., SafeTensors).
//!
//! ## Shapes Overview
//!
//! Kindle uses type-level lists (tuples of `typenum::Unsigned`) to represent static shapes, along with specialized types like `Dyn` for dynamic shapes.
//!
//! ```rust,ignore
//! use kindle_core::prelude::*;
//!
//! // A fully static 3D shape: [2, 3, 224]
//! type MyShape = (typenum::U2, typenum::U3, typenum::U224);
//!
//! // The same shape constructed via the `s![]` macro (provided by `kindle-macros`):
//! // type MyShape = s![2, 3, 224];
//! ```
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(
    feature = "nightly",
    feature(generic_const_exprs),
    allow(incomplete_features)
)]

pub(crate) extern crate alloc;

pub mod err;
pub mod graph;
pub mod nn;
pub mod onnx_exporter;
pub mod onnx_pb;
pub mod optim;
pub mod serialize;
pub mod shapes;
pub mod tensor;

pub mod prelude {
    pub use super::err::*;
    pub use crate::nn::{
        activation::{GELU, ReLU, Sigmoid, Softmax, Swish, Tanh},
        avg_pool2d::AvgPool2d,
        batch_norm::BatchNorm2d,
        conv1d::Conv1d,
        conv2d::Conv2d,
        layer_norm::LayerNorm,
        linear::{Linear, LinearShape},
        max_pool2d::MaxPool2d,
        module::{Module, Parameters, Sequential},
        param::Param,
    };
    pub use crate::seq;

    pub use super::shapes::prelude::*;
    pub use super::tensor::prelude::*;
    pub use crate::onnx_exporter::{OnnxExporter, OnnxImporter};
    pub use crate::optim::{Gradients, Optimizer, SGD};
    pub use crate::serialize::{Deserializer, Format, ModelExt, Serializer};
    pub use crate::shapes::dim::Dim;
    pub use crate::shapes::shape::{ConstShape, DynShape, PartialDynShape, Shape};
    pub use crate::symbolic_dim;
    pub use crate::tensor::backend::Backend;
    pub use typenum::{self, B0, B1, Bit, Diff, Prod, Quot, Sum, UInt, UTerm, Unsigned, consts::*};
}
