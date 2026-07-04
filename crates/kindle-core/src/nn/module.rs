use crate::prelude::*;
use alloc::vec::Vec;

/// A trait implemented by all Neural Network modules.
/// Usually automatically derived via `#[kindle::module]`.
pub trait Parameters<B: Backend<Dyn>> {
    /// Recursively extract all trainable parameters from this module.
    /// The parameters are returned as a list of backend-specific raw variables,
    /// which can be passed to an optimizer (e.g., `candle_nn::optim::SGD`).
    fn parameters(&self) -> Vec<B::RawVar>;
}

/// A generic Neural Network Layer or Module.
/// Capable of taking an input and returning an output or error.
pub trait Module<Input> {
    type Output;
    type Error;

    fn forward(&self, input: Input) -> core::result::Result<Self::Output, Self::Error>;
}

/// A sequential container for composing two modules.
/// `Sequential` automatically implements `Module` if the inner modules are compatible.
#[derive(Debug, Clone)]
pub struct Sequential<L1, L2>(pub L1, pub L2);

impl<I, L1, L2> Module<I> for Sequential<L1, L2>
where
    L1: Module<I>,
    L2: Module<L1::Output, Error = L1::Error>,
{
    type Output = L2::Output;
    type Error = L1::Error;

    #[inline]
    fn forward(&self, input: I) -> core::result::Result<Self::Output, Self::Error> {
        let out1 = self.0.forward(input)?;
        self.1.forward(out1)
    }
}

impl<B: Backend<Dyn>, L1, L2> Parameters<B> for Sequential<L1, L2>
where
    L1: Parameters<B>,
    L2: Parameters<B>,
{
    fn parameters(&self) -> Vec<B::RawVar> {
        let mut p = self.0.parameters();
        p.extend(self.1.parameters());
        p
    }
}

/// A macro to easily build Sequential models with many layers.
/// `seq!(L1, L2, L3)` expands to `Sequential(L1, Sequential(L2, L3))`.
#[macro_export]
macro_rules! seq {
    ($l1:expr, $l2:expr) => {
        $crate::nn::Sequential($l1, $l2)
    };
    ($l1:expr, $l2:expr, $($tail:expr),+ $(,)?) => {
        $crate::nn::Sequential($l1, $crate::seq!($l2, $($tail),+))
    };
}
