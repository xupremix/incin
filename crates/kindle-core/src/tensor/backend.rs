use crate::prelude::{KindleDevice, KindleDType, Shape, Error};

/// A trait that abstracts the runtime computational engine (Candle, Burn, Ndarray, etc.).
/// It provides the raw, dynamic memory buffer used by this specific backend.
pub trait Backend<S: Shape> {
    type RawTensor: Clone;

    // Constructors
    fn zeros(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor, Error>;
    fn ones(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor, Error>;
    fn rand(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor, Error>;
    fn randn(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor, Error>;

    // Unary Ops
    fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;

    // Binary Ops
    fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn matmul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;

    // Tensor ops
    fn reshape(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor, Error>;
}
