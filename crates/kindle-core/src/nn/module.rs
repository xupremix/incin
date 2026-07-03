use crate::prelude::*;
use alloc::vec::Vec;

/// A trait implemented by all Neural Network modules.
/// Usually automatically derived via `#[kindle::module]`.
pub trait Module<B: Backend<Dyn>> {
    /// Recursively extract all trainable parameters from this module.
    /// The parameters are returned as a list of backend-specific raw variables,
    /// which can be passed to an optimizer (e.g., `candle_nn::optim::SGD`).
    fn parameters(&self) -> Vec<B::RawVar>;
}
