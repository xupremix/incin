use crate::prelude::{KindleDType, KindleDevice, Result};

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

    /// Formats the underlying tensor's data (if available) into a string
    fn format_tensor(t: &Self::RawTensor) -> alloc::string::String;

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
    fn l1_loss(pred: &Self::RawTensor, target: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn bce_with_logits_loss(pred: &Self::RawTensor, target: &Self::RawTensor) -> Result<Self::RawTensor>;
    fn cross_entropy_loss(pred: &Self::RawTensor, target: &Self::RawTensor) -> Result<Self::RawTensor>;

    // Serialization
    fn to_bytes(t: &Self::RawTensor) -> Result<alloc::vec::Vec<u8>>;
    fn from_bytes(bytes: &[u8], shape: &[usize], dtype: KindleDType, device: &KindleDevice) -> Result<Self::RawTensor>;
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

        type RawTensor = alloc::vec::Vec<usize>;
        type RawVar = alloc::vec::Vec<usize>;
        type Grads = ();

        fn shape(t: &Self::RawTensor) -> alloc::vec::Vec<usize> { t.clone() }
        fn format_tensor(t: &Self::RawTensor) -> alloc::string::String { alloc::format!("Tensor(shape={:?})", t) }
        fn var_as_tensor(var: &Self::RawVar) -> Result<Self::RawTensor> { Ok(var.clone()) }
        fn var_from_tensor(t: &Self::RawTensor) -> Result<Self::RawVar> { Ok(t.clone()) }
        fn zeros(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(s.to_vec()) }
        fn ones(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(s.to_vec()) }
        fn rand(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(s.to_vec()) }
        fn randn(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(s.to_vec()) }
        fn var_zeros(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(s.to_vec()) }
        fn var_ones(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(s.to_vec()) }
        fn var_rand(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(s.to_vec()) }
        fn var_randn(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(s.to_vec()) }
        fn tensor_to_device(t: &Self::RawTensor, _d: &KindleDevice) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn var_to_device(v: &Self::RawVar, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(v.clone()) }
        fn relu(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn gelu(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn softmax(t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn swish(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn abs(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn neg(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn sqrt(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn exp(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn log(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn tanh(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn sigmoid(t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn mul_scalar(t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn add_scalar(t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(alloc::vec![]) }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(alloc::vec![]) }
        fn max_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(alloc::vec![]) }
        fn min_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(alloc::vec![]) }
        fn sum_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> { let mut s = t.clone(); s.remove(dim); Ok(s) }
        fn sum_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> { let mut s = t.clone(); s[dim] = 1; Ok(s) }
        fn mean_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> { let mut s = t.clone(); s.remove(dim); Ok(s) }
        fn mean_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> { let mut s = t.clone(); s[dim] = 1; Ok(s) }
        fn max_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> { let mut s = t.clone(); s.remove(dim); Ok(s) }
        fn max_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> { let mut s = t.clone(); s[dim] = 1; Ok(s) }
        fn min_dim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> { let mut s = t.clone(); s.remove(dim); Ok(s) }
        fn min_keepdim(t: &Self::RawTensor, dim: usize) -> Result<Self::RawTensor> { let mut s = t.clone(); s[dim] = 1; Ok(s) }
        fn to_dtype(t: &Self::RawTensor, _d: KindleDType) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn broadcast_as(_t: &Self::RawTensor, s: &[usize]) -> Result<Self::RawTensor> { Ok(s.to_vec()) }
        fn broadcast_left(_t: &Self::RawTensor, s: &[usize]) -> Result<Self::RawTensor> { Ok(s.to_vec()) }
        fn add(t1: &Self::RawTensor, _t2: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t1.clone()) }
        fn sub(t1: &Self::RawTensor, _t2: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t1.clone()) }
        fn mul(t1: &Self::RawTensor, _t2: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t1.clone()) }
        fn div(t1: &Self::RawTensor, _t2: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(t1.clone()) }
        fn matmul(t1: &Self::RawTensor, t2: &Self::RawTensor) -> Result<Self::RawTensor> { 
            let mut out = t1.clone(); 
            *out.last_mut().unwrap() = *t2.last().unwrap(); 
            Ok(out) 
        }
        fn reshape(_t: &Self::RawTensor, s: &[usize]) -> Result<Self::RawTensor> { Ok(s.to_vec()) }
        fn transpose(t: &Self::RawTensor, d1: usize, d2: usize) -> Result<Self::RawTensor> { 
            let mut out = t.clone(); 
            out.swap(d1, d2); 
            Ok(out) 
        }
        fn flatten(t: &Self::RawTensor, s: usize, e: usize) -> Result<Self::RawTensor> { 
            let mut out = alloc::vec![];
            for i in 0..s { out.push(t[i]); }
            out.push(t[s..=e].iter().product());
            for i in e+1..t.len() { out.push(t[i]); }
            Ok(out) 
        }
        fn narrow(t: &Self::RawTensor, d: usize, _s: usize, l: usize) -> Result<Self::RawTensor> { 
            let mut out = t.clone(); 
            out[d] = l; 
            Ok(out) 
        }
        fn squeeze(t: &Self::RawTensor, d: usize) -> Result<Self::RawTensor> { 
            let mut out = t.clone(); 
            out.remove(d); 
            Ok(out) 
        }
        fn conv1d(t: &Self::RawTensor, w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor> { 
            Ok(alloc::vec![t[0], w[0], t[2]]) 
        }
        fn conv2d(t: &Self::RawTensor, w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor> { 
            Ok(alloc::vec![t[0], w[0], t[2], t[3]]) 
        }
        fn conv_transpose2d(t: &Self::RawTensor, w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _op: usize, _d: usize) -> Result<Self::RawTensor> { 
            Ok(alloc::vec![t[0], w[1], t[2], t[3]]) 
        }
        fn max_pool2d(t: &Self::RawTensor, _k: (usize, usize), _s: (usize, usize)) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn avg_pool2d(t: &Self::RawTensor, _k: (usize, usize), _s: (usize, usize)) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn adaptive_avg_pool2d(t: &Self::RawTensor, out: (usize, usize)) -> Result<Self::RawTensor> { 
            Ok(alloc::vec![t[0], t[1], out.0, out.1]) 
        }
        fn backward(_loss: &Self::RawTensor) -> Result<Self::Grads> { Ok(()) }
        fn step_sgd(_p: &mut [Self::RawVar], _g: &Self::Grads, _lr: f64) -> Result<()> { Ok(()) }
        fn step_adamw(_p: &mut [Self::RawVar], _g: &Self::Grads, _lr: f64) -> Result<()> { Ok(()) }
        fn step_adam(_p: &mut [Self::RawVar], _g: &Self::Grads, _lr: f64) -> Result<()> { Ok(()) }

        fn mse_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(alloc::vec![]) }
        fn l1_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(alloc::vec![]) }
        fn bce_with_logits_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(alloc::vec![]) }
        fn cross_entropy_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(alloc::vec![]) }
        fn stack(t: &[&Self::RawTensor], d: usize) -> Result<Self::RawTensor> { 
            let mut out = t[0].clone(); 
            out.insert(d, t.len()); 
            Ok(out) 
        }
        fn concat(t: &[&Self::RawTensor], d: usize) -> Result<Self::RawTensor> { 
            let mut out = t[0].clone(); 
            out[d] = t.iter().map(|x| x[d]).sum(); 
            Ok(out) 
        }
        fn layer_norm(t: &Self::RawTensor, _w: &Self::RawTensor, _b: &Self::RawTensor, _e: f32) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn batch_norm(t: &Self::RawTensor, _w: &Self::RawTensor, _b: &Self::RawTensor, _rm: &Self::RawTensor, _rv: &Self::RawTensor, _e: f32) -> Result<Self::RawTensor> { Ok(t.clone()) }
        fn embedding(_t: &Self::RawTensor, _w: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(alloc::vec![]) }

        fn to_bytes(_t: &Self::RawTensor) -> Result<alloc::vec::Vec<u8>> { Ok(alloc::vec::Vec::new()) }
        fn from_bytes(_bytes: &[u8], shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(shape.to_vec()) }
    }

}
