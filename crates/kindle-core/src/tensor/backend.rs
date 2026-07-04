use crate::prelude::{KindleDType, KindleDevice, Result, Shape};

/// A trait that abstracts the runtime computational engine (Candle, Burn, Ndarray, etc.).
/// It provides the raw, dynamic memory buffer used by this specific backend.
pub trait Backend: Clone + 'static {
    type Device: super::device::Device;
    type DType: super::dtype::DType;
    type BackendWithDType<NewT: super::dtype::DType>: Backend<
            DType = NewT,
            Device = Self::Device,
            RawTensor = Self::RawTensor,
            RawVar = Self::RawVar,
            Grads = Self::Grads,
        >;
    type BackendWithDevice<NewD: super::device::Device>: Backend<
            Device = NewD,
            DType = Self::DType,
            RawTensor = Self::RawTensor,
            RawVar = Self::RawVar,
            Grads = Self::Grads,
        >;

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
        w: &Self::RawTensor,
        b: &Self::RawTensor,
        rm: &Self::RawTensor,
        rv: &Self::RawTensor,
        e: f32,
    ) -> Result<Self::RawTensor>;

    fn embedding(t: &Self::RawTensor, w: &Self::RawTensor) -> Result<Self::RawTensor>;

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

    fn adaptive_avg_pool2d(
        t: &Self::RawTensor,
        output_size: (usize, usize),
    ) -> Result<Self::RawTensor>;

    type Grads;

    /// Computes the backward pass starting from this tensor, returning the gradients.
    fn backward(loss: &Self::RawTensor) -> Result<Self::Grads>;
    fn step_sgd(params: &mut [Self::RawVar], grads: &Self::Grads, lr: f64) -> Result<()>;
    fn step_adamw(params: &mut [Self::RawVar], grads: &Self::Grads, lr: f64) -> Result<()>;
    fn step_adam(params: &mut [Self::RawVar], grads: &Self::Grads, lr: f64) -> Result<()>;

    // Loss Functions
    fn mse_loss(pred: &Self::RawTensor, target: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn cross_entropy_loss(pred: &Self::RawTensor, target: &Self::RawTensor) -> Result<Self::RawTensor>;
}

pub mod dummy {
    use super::*;
    use crate::prelude::*;

    #[derive(Clone, Debug, Default)]
    pub struct DummyBackend<T: DType = f32, D: Device = Cpu>(core::marker::PhantomData<(T, D)>);
    
    impl<T: DType, D: Device> Backend for DummyBackend<T, D> {
        type Device = D;
        type DType = T;
        type BackendWithDType<NewT: DType> = DummyBackend<NewT, D>;
        type BackendWithDevice<NewD: Device> = DummyBackend<T, NewD>;

        type RawTensor = ();
        type RawVar = ();
        type Grads = ();

        fn shape(_t: &Self::RawTensor) -> alloc::vec::Vec<usize> { alloc::vec![6] }
        fn var_as_tensor(_var: &Self::RawVar) -> Result<Self::RawTensor> { Ok(()) }
        fn var_from_tensor(_t: &Self::RawTensor) -> Result<Self::RawVar> { Ok(()) }
        fn zeros(_s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn ones(_s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn rand(_s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn randn(_s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn var_zeros(_s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }
        fn var_ones(_s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }
        fn var_rand(_s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }
        fn var_randn(_s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }
        fn tensor_to_device(_t: &Self::RawTensor, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
        fn var_to_device(_v: &Self::RawVar, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }
        fn relu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn softmax(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn swish(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn abs(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn mul_scalar(_t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> { Ok(()) }
        fn add_scalar(_t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> { Ok(()) }
        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn max_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn min_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sum_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn sum_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn mean_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn mean_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn max_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn max_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn min_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn min_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn to_dtype(_t: &Self::RawTensor, _d: KindleDType) -> Result<Self::RawTensor> { Ok(()) }
        fn broadcast_as(_t: &Self::RawTensor, _s: &[usize]) -> Result<Self::RawTensor> { Ok(()) }
        fn broadcast_left(_t: &Self::RawTensor, _s: &[usize]) -> Result<Self::RawTensor> { Ok(()) }
        fn add(_t1: &Self::RawTensor, _t2: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn sub(_t1: &Self::RawTensor, _t2: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn mul(_t1: &Self::RawTensor, _t2: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn div(_t1: &Self::RawTensor, _t2: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn matmul(_t1: &Self::RawTensor, _t2: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn reshape(_t: &Self::RawTensor, _s: &[usize]) -> Result<Self::RawTensor> { Ok(()) }
        fn transpose(_t: &Self::RawTensor, _d1: usize, _d2: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn flatten(_t: &Self::RawTensor, _s: usize, _e: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn narrow(_t: &Self::RawTensor, _d: usize, _s: usize, _l: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn squeeze(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn conv1d(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn conv2d(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn conv_transpose2d(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _op: usize, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn max_pool2d(_t: &Self::RawTensor, _k: (usize, usize), _s: (usize, usize)) -> Result<Self::RawTensor> { Ok(()) }
        fn avg_pool2d(_t: &Self::RawTensor, _k: (usize, usize), _s: (usize, usize)) -> Result<Self::RawTensor> { Ok(()) }
        fn adaptive_avg_pool2d(_t: &Self::RawTensor, _out: (usize, usize)) -> Result<Self::RawTensor> { Ok(()) }
        fn backward(_loss: &Self::RawTensor) -> Result<Self::Grads> { Ok(()) }
        fn step_sgd(_p: &mut [Self::RawVar], _g: &Self::Grads, _lr: f64) -> Result<()> { Ok(()) }
        fn step_adamw(_p: &mut [Self::RawVar], _g: &Self::Grads, _lr: f64) -> Result<()> { Ok(()) }
        fn step_adam(_p: &mut [Self::RawVar], _g: &Self::Grads, _lr: f64) -> Result<()> { Ok(()) }

        fn mse_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn cross_entropy_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
        fn stack(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn concat(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> { Ok(()) }
        fn layer_norm(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: &Self::RawTensor, _e: f32) -> Result<Self::RawTensor> { Ok(()) }
        fn batch_norm(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: &Self::RawTensor, _rm: &Self::RawTensor, _rv: &Self::RawTensor, _e: f32) -> Result<Self::RawTensor> { Ok(()) }
        fn embedding(_t: &Self::RawTensor, _w: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    }
}
