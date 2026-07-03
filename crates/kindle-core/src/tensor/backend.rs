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
    fn neg(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn sqrt(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn exp(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn log(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn tanh(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn sigmoid(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;

    // Scalar Ops
    fn mul_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor, Error>;
    fn add_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor, Error>;

    // Reductions (0D Scalar output)
    fn sum_all(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn mean_all(t: &Self::RawTensor) -> Result<Self::RawTensor, Error>;

    // Binary Ops
    fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;
    fn matmul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor, Error>;

    // Tensor ops
    fn reshape(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor, Error>;
    
    // Slicing primitives
    fn narrow(t: &Self::RawTensor, dim: usize, start: usize, len: usize) -> Result<Self::RawTensor, Error>;
    fn squeeze(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor, Error>;
}
