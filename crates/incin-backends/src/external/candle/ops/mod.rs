//! Operation-family implementations for the Candle adapter.
//!
//! One module per `Backend` operation family, mirroring the layout of
//! `cpu/ops/` and `cuda/ops/`.

pub(crate) mod creation;
mod float;
mod loss;
mod module;
pub(crate) mod numeric;
mod quant;
mod reduce;
mod tensor;
