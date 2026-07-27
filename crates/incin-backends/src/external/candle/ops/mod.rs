//! Operation-family implementations for the Candle adapter.
//!
//! One module per `Backend` operation family, mirroring the layout of
//! `cpu/ops/` and `cuda/ops/`.

mod creation;
mod float;
mod loss;
mod module;
mod numeric;
mod optimizer;
mod quant;
mod reduce;
mod tensor;
