use crate::prelude::{KindleDType, KindleDevice, Result};
use crate::tensor::device::Device;
use crate::tensor::dtype::{DType, FloatDType, QuantDType};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    Float(f64),
    Int(i64),
}

impl ScalarValue {
    pub fn to_f64(&self) -> f64 {
        match self {
            ScalarValue::Float(f) => *f,
            ScalarValue::Int(i) => *i as f64,
        }
    }

    pub fn to_i64(&self) -> i64 {
        match self {
            ScalarValue::Float(f) => *f as i64,
            ScalarValue::Int(i) => *i,
        }
    }
}

impl From<f32> for ScalarValue {
    fn from(v: f32) -> Self {
        ScalarValue::Float(v as f64)
    }
}
impl From<f64> for ScalarValue {
    fn from(v: f64) -> Self {
        ScalarValue::Float(v)
    }
}
impl From<i32> for ScalarValue {
    fn from(v: i32) -> Self {
        ScalarValue::Int(v as i64)
    }
}
impl From<i64> for ScalarValue {
    fn from(v: i64) -> Self {
        ScalarValue::Int(v)
    }
}

pub trait SupportsDType<K: DType> {}

pub trait Backend:
    Sized
    + Clone
    + Send
    + Sync
    + 'static
    + TensorOps<Self>
    + NumericOps<Self>
    + FloatOps<Self>
    + CreationOps<Self>
    + ReductionOps<Self>
    + QuantizedOps<Self>
    + OptimizerOps<Self>
{
    type Device: Device;
    type FloatElem: DType;
    type IntElem: DType;

    type Storage<K: DType>: Clone;
    type RawVar: Clone;
    type Grads;

    type InnerBackend: Backend;

    type BackendWithDevice<NewD: Device>: Backend<Device = NewD, RawVar = Self::RawVar, FloatElem = Self::FloatElem, IntElem = Self::IntElem>;

    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize>;
    fn format_tensor_display<K: DType>(t: &Self::Storage<K>) -> alloc::string::String;
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String;

    fn backward<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads>;
    fn backward_with_nan_check<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads>;
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>>;

    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>>;
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<Self::Storage<K>>;

    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>>;
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar>;
    fn var_to_device(var: &Self::RawVar, device: &KindleDevice) -> Result<Self::RawVar>;
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()>;
}

// FloatOps only requires Backend, operates on FloatTensorPrimitive
pub trait FloatOps<B: Backend> {
    fn relu<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn gelu<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn abs<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn exp<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn neg<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn sqrt<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn log<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn tanh<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn sigmoid<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn swish<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn softmax<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn add_scalar_float<K: DType>(t: &B::Storage<K>, scalar: f64) -> Result<B::Storage<K>>;
    fn mul_scalar_float<K: DType>(t: &B::Storage<K>, scalar: f64) -> Result<B::Storage<K>>;
}

// NumericOps operates generically over any TensorKind!
pub trait NumericOps<B: Backend> {
    fn add<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn sub<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn mul<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn div<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
}

pub trait TensorOps<B: Backend> {
    fn reshape<K: DType>(t: &B::Storage<K>, shape: &[usize]) -> Result<B::Storage<K>>;
    fn transpose<K: DType>(t: &B::Storage<K>, dim1: usize, dim2: usize) -> Result<B::Storage<K>>;
    fn matmul<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn broadcast_as<K: DType>(t: &B::Storage<K>, shape: &[usize]) -> Result<B::Storage<K>>;
    fn narrow<K: DType>(
        t: &B::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<B::Storage<K>>;
    fn squeeze<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn stack<K: DType>(t: &[&B::Storage<K>], dim: usize) -> Result<B::Storage<K>>;
    fn concat<K: DType>(t: &[&B::Storage<K>], dim: usize) -> Result<B::Storage<K>>;
    fn slice<K: DType>(t: &B::Storage<K>, ranges: &[(usize, usize)]) -> Result<B::Storage<K>>;
    fn flatten<K: DType>(
        t: &B::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<B::Storage<K>>;
    fn broadcast_left<K: DType>(t: &B::Storage<K>, shape: &[usize]) -> Result<B::Storage<K>>;

    fn float_to_scalar<K: DType>(t: &B::Storage<K>) -> Result<f64>;
    fn float_to_vec1<K: DType>(t: &B::Storage<K>) -> Result<alloc::vec::Vec<f64>>;

    fn int_to_scalar<K: DType>(t: &B::Storage<K>) -> Result<i64>;
    fn int_to_vec1<K: DType>(t: &B::Storage<K>) -> Result<alloc::vec::Vec<i64>>;

    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &B::Storage<K>,
        dtype: KindleDType,
    ) -> Result<B::Storage<K2>>;
}

pub trait CreationOps<B: Backend> {
    fn zeros<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;
    fn ones<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;
    fn rand<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;
    fn randn<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;

    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::RawVar>;
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::RawVar>;
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::RawVar>;
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::RawVar>;

    fn tensor_to_device<K: DType>(
        t: &B::Storage<K>,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;
}

pub trait ReductionOps<B: Backend> {
    fn sum_all<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn mean_all<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn max_all<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn min_all<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    fn sum_dim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn sum_keepdim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn mean_dim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn mean_keepdim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn max_dim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn max_keepdim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn min_dim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn min_keepdim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    fn argmax<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: Option<usize>,
    ) -> Result<B::Storage<KInt>>;
    fn argmin<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: Option<usize>,
    ) -> Result<B::Storage<KInt>>;
}

pub trait ModuleOps<B: Backend> {
    fn layer_norm<K: DType>(
        t: &B::Storage<K>,
        weight: &B::Storage<K>,
        bias: Option<&B::Storage<K>>,
        eps: f32,
    ) -> Result<B::Storage<K>>;
    fn batch_norm<K: DType>(
        t: &B::Storage<K>,
        w: Option<&B::Storage<K>>,
        b: Option<&B::Storage<K>>,
        rm: Option<&B::Storage<K>>,
        rv: Option<&B::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<B::Storage<K>>;
    fn embedding<K: DType, KInt: DType>(
        t: &B::Storage<KInt>,
        w: &B::Storage<K>,
    ) -> Result<B::Storage<K>>;
    fn conv1d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    fn conv2d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    fn conv_transpose2d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    fn max_pool2d<K: DType>(
        t: &B::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<B::Storage<K>>;
    fn avg_pool2d<K: DType>(
        t: &B::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<B::Storage<K>>;
    fn adaptive_avg_pool2d<K: DType>(
        t: &B::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<B::Storage<K>>;
}

pub trait LossOps<B: Backend>: NumericOps<B> + FloatOps<B> + ReductionOps<B> {
    fn mse_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<B::Storage<K>> {
        let diff = <B as NumericOps<B>>::sub::<K>(pred, target)?;
        let sq = <B as NumericOps<B>>::mul::<K>(&diff, &diff)?;
        match reduction {
            crate::nn::loss::Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&sq),
            crate::nn::loss::Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&sq),
            crate::nn::loss::Reduction::None => Ok(sq),
        }
    }

    fn l1_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<B::Storage<K>> {
        let diff = <B as NumericOps<B>>::sub::<K>(pred, target)?;
        let abs_diff = <B as FloatOps<B>>::abs::<K>(&diff)?;
        match reduction {
            crate::nn::loss::Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&abs_diff),
            crate::nn::loss::Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&abs_diff),
            crate::nn::loss::Reduction::None => Ok(abs_diff),
        }
    }

    fn bce_with_logits_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<B::Storage<K>> {
        let max_x_0 = <B as FloatOps<B>>::relu::<K>(pred)?;
        let x_times_z = <B as NumericOps<B>>::mul::<K>(pred, target)?;
        let term1 = <B as NumericOps<B>>::sub::<K>(&max_x_0, &x_times_z)?;

        let abs_x = <B as FloatOps<B>>::abs::<K>(pred)?;
        let neg_abs_x = <B as FloatOps<B>>::neg::<K>(&abs_x)?;
        let exp_neg_abs_x = <B as FloatOps<B>>::exp::<K>(&neg_abs_x)?;
        let one_plus = <B as FloatOps<B>>::add_scalar_float::<K>(&exp_neg_abs_x, 1.0)?;
        let term2 = <B as FloatOps<B>>::log::<K>(&one_plus)?;

        let loss_elem = <B as NumericOps<B>>::add::<K>(&term1, &term2)?;

        match reduction {
            crate::nn::loss::Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&loss_elem),
            crate::nn::loss::Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&loss_elem),
            crate::nn::loss::Reduction::None => Ok(loss_elem),
        }
    }

    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<KInt>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<B::Storage<K>>;
}

pub trait QuantizedOps<B: Backend> {
    fn quantize<K: FloatDType, Q: QuantDType>(t: &B::Storage<K>) -> Result<B::Storage<Q>>;
    fn dequantize<Q: QuantDType, K: FloatDType>(t: &B::Storage<Q>) -> Result<B::Storage<K>>;
    fn quantized_matmul<Q: QuantDType>(lhs: &B::Storage<Q>, rhs: &B::Storage<Q>) -> Result<B::Storage<f32>>;
}

pub trait OptimizerOps<B: Backend> {
    fn adamw_step<K: DType>(
        var: &mut B::RawVar,
        grad: &B::Storage<K>,
        m: &mut B::Storage<K>,
        v: &mut B::Storage<K>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        step: usize,
    ) -> Result<()> {
        let mut t = B::var_as_tensor::<K>(var)?;
        let t_step = step as f64;
        let bias_correction1 = 1.0 - beta1.powf(t_step);
        let bias_correction2 = 1.0 - beta2.powf(t_step);

        if weight_decay > 0.0 {
            let decay = B::mul_scalar_float::<K>(&t, weight_decay * lr)?;
            t = B::sub::<K>(&t, &decay)?;
        }

        let term1_m = B::mul_scalar_float::<K>(m, beta1)?;
        let term2_m = B::mul_scalar_float::<K>(grad, 1.0 - beta1)?;
        let m_t = B::add::<K>(&term1_m, &term2_m)?;

        let grad_sq = B::mul::<K>(grad, grad)?;
        let term1_v = B::mul_scalar_float::<K>(v, beta2)?;
        let term2_v = B::mul_scalar_float::<K>(&grad_sq, 1.0 - beta2)?;
        let v_t = B::add::<K>(&term1_v, &term2_v)?;

        *m = m_t.clone();
        *v = v_t.clone();

        let m_hat = B::mul_scalar_float::<K>(&m_t, 1.0 / bias_correction1)?;
        let v_hat = B::mul_scalar_float::<K>(&v_t, 1.0 / bias_correction2)?;

        let denom = B::add_scalar_float::<K>(&B::sqrt::<K>(&v_hat)?, eps)?;
        let step_val = B::mul_scalar_float::<K>(&B::div::<K>(&m_hat, &denom)?, lr)?;

        let updated = B::sub::<K>(&t, &step_val)?;
        B::assign_var::<K>(var, &updated)?;
        Ok(())
    }
}
pub mod dummy {
    use super::*;
    use crate::nn::Reduction;
    use crate::prelude::Result;
    use crate::tensor::device::Device;
    use crate::tensor::device::KindleDevice;
    use crate::tensor::dtype::DType;

    pub struct DummyBackend<T, D> {
        _marker: core::marker::PhantomData<(T, D)>,
    }

    impl<T: DType, D: Device + Clone + 'static> Clone for DummyBackend<T, D> {
        fn clone(&self) -> Self {
            DummyBackend {
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<T: DType, D: Device + Clone + 'static> Backend for DummyBackend<T, D> {
        type Device = D;
        type FloatElem = f32;
        type IntElem = i64;
        type RawVar = alloc::vec::Vec<usize>;
        type Grads = ();
        type Storage<K: DType> = alloc::vec::Vec<usize>;
        type InnerBackend = Self;
        type BackendWithDevice<NewD: Device> = DummyBackend<T, NewD>;

        fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
            t.clone()
        }
        fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        fn format_tensor_debug<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        fn backward<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }
        fn backward_with_nan_check<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }
        fn get_grad<K: DType>(
            _t: &Self::Storage<K>,
            _grads: &Self::Grads,
        ) -> Result<Option<Self::Storage<K>>> {
            Ok(None)
        }
        fn to_bytes<K: DType>(_t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
            Ok(alloc::vec::Vec::new())
        }
        fn from_bytes<K: DType>(
            _bytes: &[u8],
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::Storage<K>> {
            Ok(shape.to_vec())
        }
        fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
            Ok(var.clone())
        }
        fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
            Ok(t.clone())
        }
        fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
            Ok(var.clone())
        }
        fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
            *var = tensor.clone();
            Ok(())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> CreationOps<Self> for DummyBackend<T, D> {
        fn zeros<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        fn ones<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        fn rand<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        fn randn<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        fn var_zeros<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        fn var_ones<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        fn var_rand<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        fn var_randn<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        fn tensor_to_device<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> NumericOps<Self> for DummyBackend<T, D> {
        fn add<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        fn sub<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        fn mul<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        fn div<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> FloatOps<Self> for DummyBackend<T, D> {
        fn add_scalar_float<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn mul_scalar_float<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn relu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn gelu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn abs<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn exp<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn neg<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn sqrt<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn log<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn tanh<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn sigmoid<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn swish<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn softmax<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> ReductionOps<Self> for DummyBackend<T, D> {
        fn sum_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        fn mean_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        fn max_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        fn min_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        fn sum_dim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        fn sum_keepdim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        fn mean_dim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        fn mean_keepdim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        fn max_dim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        fn max_keepdim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        fn min_dim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        fn min_keepdim<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        fn argmax<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        fn argmin<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
    }

    impl<T: DType, D: Device + Clone + 'static> TensorOps<Self> for DummyBackend<T, D> {
        fn matmul<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        fn float_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<f64> {
            Ok(0.0)
        }
        fn float_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<f64>> {
            Ok(alloc::vec![0.0])
        }
        fn int_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<i64> {
            Ok(0)
        }
        fn int_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<i64>> {
            Ok(alloc::vec![0])
        }

        fn broadcast_as<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn broadcast_left<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn reshape<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn transpose<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            d1: usize,
            d2: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            if d1 < out.len() && d2 < out.len() {
                out.swap(d1, d2);
            }
            Ok(out)
        }
        fn flatten<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _s: usize,
            _e: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn slice<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _ranges: &[(usize, usize)],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn narrow<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _d: usize,
            _s: usize,
            _l: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn squeeze<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            d: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            if d < out.len() {
                out.remove(d);
            }
            Ok(out)
        }
        fn stack<K: DType>(
            t: &[&<Self as Backend>::Storage<K>],
            _d: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t[0].clone())
        }
        fn concat<K: DType>(
            t: &[&<Self as Backend>::Storage<K>],
            _d: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t[0].clone())
        }
        fn tensor_to_dtype<K: DType, K2: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dtype: KindleDType,
        ) -> Result<<Self as Backend>::Storage<K2>> {
            Ok(alloc::vec![])
        }
    }

    impl<T: DType, D: Device + Clone + 'static> ModuleOps<Self> for DummyBackend<T, D> {
        fn layer_norm<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            _e: f32,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn batch_norm<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _w: Option<&<Self as Backend>::Storage<K>>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            _rm: Option<&<Self as Backend>::Storage<K>>,
            _rv: Option<&<Self as Backend>::Storage<K>>,
            _e: f32,
            _m: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn embedding<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<KInt>,
            _w: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        fn conv1d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            _s: usize,
            _p: usize,
            _d: usize,
            _groups: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn conv2d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            _s: usize,
            _p: usize,
            _d: usize,
            _groups: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn conv_transpose2d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            _s: usize,
            _p: usize,
            _op: usize,
            _d: usize,
            _groups: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn max_pool2d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _k: (usize, usize),
            _s: (usize, usize),
            _p: (usize, usize),
            _d: (usize, usize),
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn avg_pool2d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _k: (usize, usize),
            _s: (usize, usize),
            _p: (usize, usize),
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        fn adaptive_avg_pool2d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            out: (usize, usize),
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut shape = t.clone();
            let len = shape.len();
            if len >= 2 {
                shape[len - 2] = out.0;
                shape[len - 1] = out.1;
            }
            Ok(shape)
        }
    }

    impl<T: DType, D: Device + Clone + 'static> LossOps<Self> for DummyBackend<T, D> {
        fn mse_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        fn l1_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        fn bce_with_logits_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        fn cross_entropy_loss<K: DType, KInt: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<KInt>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
    }

    impl<T: DType, D: Device + Clone + 'static> QuantizedOps<Self> for DummyBackend<T, D> {
        fn quantize<K: FloatDType, Q: QuantDType>(_t: &<Self as Backend>::Storage<K>) -> Result<<Self as Backend>::Storage<Q>> {
            Ok(alloc::vec![])
        }
        fn dequantize<Q: QuantDType, K: FloatDType>(_t: &<Self as Backend>::Storage<Q>) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        fn quantized_matmul<Q: QuantDType>(_lhs: &<Self as Backend>::Storage<Q>, _rhs: &<Self as Backend>::Storage<Q>) -> Result<<Self as Backend>::Storage<f32>> {
            Ok(alloc::vec![])
        }
    }
    impl<T: DType, D: Device + Clone + 'static> OptimizerOps<Self> for DummyBackend<T, D> {}
}
