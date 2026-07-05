// #![cfg_attr(not(feature = "std"), no_std)]
// #![cfg_attr(
//     feature = "nightly",
//     feature(generic_const_exprs),
//     allow(incomplete_features)
// )]

pub use kindle_backends::*;
pub use kindle_core::*;

pub use kindle_macros::{forward, module, import_model};

pub mod hub {
    pub use kindle_data::hub::*;
}

// We define a type alias to restore the default Backend behavior without cyclical dependencies
#[cfg(feature = "cuda")]
pub type DefaultDevice = crate::prelude::Cuda;
#[cfg(all(not(feature = "cuda"), feature = "metal"))]
pub type DefaultDevice = crate::prelude::Metal;
#[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
pub type DefaultDevice = crate::prelude::Cpu;

#[cfg(feature = "candle")]
pub type Tensor<
    S,
    B = kindle_backends::candle::CandleBackend<f32, DefaultDevice>,
    G = kindle_core::prelude::Grad,
> = kindle_core::prelude::Tensor<S, B, G>;

#[cfg(not(feature = "candle"))]
pub type Tensor<
    S,
    B, // User must specify backend if Candle is disabled
    G = kindle_core::prelude::Grad,
> = kindle_core::prelude::Tensor<S, B, G>;

pub mod macros {
    pub use kindle_macros::{idx, impl_arg_into, s};
}

pub mod prelude {
    pub use kindle_backends::prelude::*;
    pub use kindle_core::prelude::*;
    pub use kindle_macros::*;

    // We intentionally overshadow kindle_core::Tensor with our aliased version
    pub use super::Tensor;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_export() {
        // Just verify types are properly exported and accessible
        #[cfg(feature = "candle")]
        {
            // Verify our alias correctly injects CandleBackend
            let _t: Tensor<Dyn> = Tensor::zeros(std::vec![2, 2]).unwrap();
        }
    }
}
