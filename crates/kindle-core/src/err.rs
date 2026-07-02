use crate::candle;
use alloc::string::String;
use core::fmt::Debug;

pub type Result<T> = core::result::Result<T, Error>;

#[derive(thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Candle(#[from] candle::Error),
    #[error("Invalid device provided: expected {expected}, got {got}")]
    DeviceInitializationError { expected: String, got: String },
}
// #[error("Invalid function call '{fn_call}' on Dyn parameter: expected {expected}, got {got}")]
// InvalidFunctionCallOnDynParameter {
//     fn_call: &'static str,
//     expected: String,
//     got: String,
// },

impl Debug for Error {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{self}")
    }
}
