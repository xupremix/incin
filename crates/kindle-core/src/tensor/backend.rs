use crate::prelude::{KindleDevice, KindleDType, Shape, Result};

/// A trait that abstracts the runtime computational engine (Candle, Burn, Ndarray, etc.).
/// It provides the raw, dynamic memory buffer used by this specific backend.
pub trait Backend<S: Shape> {
    type RawTensor: Clone;

    // Constructors
    fn zeros(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor>;
    fn ones(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor>;
    fn rand(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor>;
    fn randn(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor>;

    // Unary Ops
    fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn neg(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn sqrt(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn exp(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn log(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn tanh(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn sigmoid(t: &Self::RawTensor) -> Result<Self::RawTensor>;

    // Scalar Ops
    fn mul_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor>;
    fn add_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor>;

    // Reductions (0D Scalar output)
    fn sum_all(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn mean_all(t: &Self::RawTensor) -> Result<Self::RawTensor>;

    // Binary Ops
    fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn matmul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;

    // Tensor ops
    fn reshape(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor>;
    
    // Slicing primitives
    fn narrow(t: &Self::RawTensor, dim: usize, start: usize, len: usize) -> Result<Self::RawTensor>;
    fn squeeze(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;
    
    // Convolutions
    fn conv2d(
        t: &Self::RawTensor,
        weight: &Self::RawTensor,
        bias: Option<&Self::RawTensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self::RawTensor>;
}
