use crate::prelude::{KindleDType, KindleDevice, Result, Shape};

/// A trait that abstracts the runtime computational engine (Candle, Burn, Ndarray, etc.).
/// It provides the raw, dynamic memory buffer used by this specific backend.
pub trait Backend<S: Shape> {
    type RawTensor: Clone;
    type RawVar: Clone;

    fn shape(t: &Self::RawTensor) -> alloc::vec::Vec<usize>;

    // Var operations
    fn var_as_tensor(var: &Self::RawVar) -> Result<Self::RawTensor>;
    fn var_from_tensor(t: &Self::RawTensor) -> Result<Self::RawVar>;

    // Constructors
    fn zeros(shape: &[usize], dtype: KindleDType, device: &KindleDevice)
    -> Result<Self::RawTensor>;
    fn ones(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor>;
    fn rand(shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor>;
    fn randn(shape: &[usize], dtype: KindleDType, device: &KindleDevice)
    -> Result<Self::RawTensor>;

    fn var_zeros(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<Self::RawVar>;
    fn var_ones(shape: &[usize], dtype: KindleDType, device: &KindleDevice)
    -> Result<Self::RawVar>;
    fn var_rand(shape: &[usize], dtype: KindleDType, device: &KindleDevice)
    -> Result<Self::RawVar>;

    // Device Management
    fn tensor_to_device(t: &Self::RawTensor, device: &KindleDevice) -> Result<Self::RawTensor>;
    fn var_to_device(var: &Self::RawVar, device: &KindleDevice) -> Result<Self::RawVar>;
    fn var_randn(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<Self::RawVar>;

    // Unary Ops
    fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn exp(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn neg(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn sqrt(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn log(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn tanh(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn sigmoid(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn swish(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn softmax(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;

    // Scalar Ops
    fn mul_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor>;
    fn add_scalar(t: &Self::RawTensor, scalar: f64) -> Result<Self::RawTensor>;

    // Reductions (0D Scalar output)
    // Reductions (0D Scalar output)
    fn sum_all(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn mean_all(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn max_all(t: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn min_all(t: &Self::RawTensor) -> Result<Self::RawTensor>;

    // Reductions along dimension
    fn sum_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;
    fn sum_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;
    fn mean_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;
    fn mean_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;
    fn max_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;
    fn max_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;
    fn min_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;
    fn min_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;

    // Binary Ops
    fn add(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn sub(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn mul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn div(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn matmul(lhs: &Self::RawTensor, rhs: &Self::RawTensor) -> Result<Self::RawTensor>;

    // Type Casting
    fn to_dtype(t: &Self::RawTensor, dtype: KindleDType) -> Result<Self::RawTensor>;

    // Advanced Tensor Ops
    fn stack(t: &[&Self::RawTensor], dim: usize) -> Result<Self::RawTensor>;
    fn concat(t: &[&Self::RawTensor], dim: usize) -> Result<Self::RawTensor>;
    fn layer_norm(
        t: &Self::RawTensor,
        weight: &Self::RawTensor,
        bias: &Self::RawTensor,
        eps: f32,
    ) -> Result<Self::RawTensor>;
    fn batch_norm(
        t: &Self::RawTensor,
        weight: &Self::RawTensor,
        bias: &Self::RawTensor,
        running_mean: &Self::RawTensor,
        running_var: &Self::RawTensor,
        eps: f32,
    ) -> Result<Self::RawTensor>;

    // Tensor ops
    fn broadcast_as(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor>;
    fn broadcast_left(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor>;
    fn reshape(t: &Self::RawTensor, shape: &[usize]) -> Result<Self::RawTensor>;
    fn transpose(t: &Self::RawTensor, dim1: usize, dim2: usize) -> Result<Self::RawTensor>;
    fn flatten(t: &Self::RawTensor, start_dim: usize, end_dim: usize) -> Result<Self::RawTensor>;

    // Slicing primitives
    fn narrow(t: &Self::RawTensor, dim: usize, start: usize, len: usize)
    -> Result<Self::RawTensor>;
    fn squeeze(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor>;

    // Convolutions
    fn conv1d(
        t: &Self::RawTensor,
        w: &Self::RawTensor,
        b: Option<&Self::RawTensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self::RawTensor>;

    fn conv2d(
        t: &Self::RawTensor,
        w: &Self::RawTensor,
        b: Option<&Self::RawTensor>,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self::RawTensor>;

    fn conv_transpose2d(
        t: &Self::RawTensor,
        w: &Self::RawTensor,
        b: Option<&Self::RawTensor>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> Result<Self::RawTensor>;

    // Pooling
    fn max_pool2d(
        t: &Self::RawTensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self::RawTensor>;

    fn avg_pool2d(
        t: &Self::RawTensor,
        kernel_size: (usize, usize),
        stride: (usize, usize),
    ) -> Result<Self::RawTensor>;

    type Grads;

    // Optimization
    fn backward(loss: &Self::RawTensor) -> Result<Self::Grads>;
    fn step_sgd(params: &mut [Self::RawVar], grads: &Self::Grads, lr: f64) -> Result<()>;
    fn step_adamw(params: &mut [Self::RawVar], grads: &Self::Grads, lr: f64) -> Result<()>;
}
