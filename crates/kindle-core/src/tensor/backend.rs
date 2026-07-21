use crate::prelude::{KindleDType, KindleDevice, Result};
use crate::tensor::device::Device;
use crate::tensor::dtype::{DType, FloatDType, QuantDType};

#[derive(Debug, Clone, Copy, PartialEq)]
/// Auto-generated documentation for ScalarValue.
pub enum ScalarValue {
    /// Auto-generated documentation for Float.
    Float(f64),
    /// Auto-generated documentation for Int.
    Int(i64),
}

impl ScalarValue {
    /// Auto-generated documentation for to_f64.
    pub fn to_f64(&self) -> f64 {
        match self {
            ScalarValue::Float(f) => *f,
            ScalarValue::Int(i) => *i as f64,
        }
    }

    /// Auto-generated documentation for to_i64.
    pub fn to_i64(&self) -> i64 {
        match self {
            ScalarValue::Float(f) => *f as i64,
            ScalarValue::Int(i) => *i,
        }
    }
}

impl From<f32> for ScalarValue {
    /// Auto-generated documentation for from.
    fn from(v: f32) -> Self {
        ScalarValue::Float(v as f64)
    }
}
impl From<f64> for ScalarValue {
    /// Auto-generated documentation for from.
    fn from(v: f64) -> Self {
        ScalarValue::Float(v)
    }
}
impl From<i32> for ScalarValue {
    /// Auto-generated documentation for from.
    fn from(v: i32) -> Self {
        ScalarValue::Int(v as i64)
    }
}
impl From<i64> for ScalarValue {
    /// Auto-generated documentation for from.
    fn from(v: i64) -> Self {
        ScalarValue::Int(v)
    }
}

/// Auto-generated documentation for SupportsDType.
pub trait SupportsDType<K: DType> {}

/// Auto-generated documentation for Backend.
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
    + crate::tensor::backend::ModuleOps<Self>
    + crate::tensor::backend::LossOps<Self>
{
    /// Auto-generated documentation for Device.
    type Device: Device;
    /// Auto-generated documentation for FloatElem.
    type FloatElem: DType;
    /// Auto-generated documentation for IntElem.
    type IntElem: DType;

    /// Auto-generated documentation for Storage.
    type Storage<K: DType>: Clone;
    /// Auto-generated documentation for RawVar.
    type RawVar: Clone;
    /// Auto-generated documentation for Grads.
    type Grads;

    /// Auto-generated documentation for InnerBackend.
    type InnerBackend: Backend;

    /// Auto-generated documentation for BackendWithDevice.
    type BackendWithDevice<NewD: Device>: Backend<
            Device = NewD,
            RawVar = Self::RawVar,
            FloatElem = Self::FloatElem,
            IntElem = Self::IntElem,
        >;

    /// Auto-generated documentation for shape.
    fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize>;
    /// Auto-generated documentation for format_tensor_display.
    fn format_tensor_display<K: DType>(t: &Self::Storage<K>) -> alloc::string::String;
    /// Auto-generated documentation for format_tensor_debug.
    fn format_tensor_debug<K: DType>(t: &Self::Storage<K>) -> alloc::string::String;

    /// Auto-generated documentation for backward.
    fn backward<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads>;
    /// Auto-generated documentation for backward_with_nan_check.
    fn backward_with_nan_check<K: DType>(t: &Self::Storage<K>) -> Result<Self::Grads>;
    /// Auto-generated documentation for get_grad.
    fn get_grad<K: DType>(
        t: &Self::Storage<K>,
        grads: &Self::Grads,
    ) -> Result<Option<Self::Storage<K>>>;

    /// Auto-generated documentation for to_bytes.
    fn to_bytes<K: DType>(t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>>;
    /// Auto-generated documentation for from_bytes.
    fn from_bytes<K: DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<Self::Storage<K>>;

    /// Auto-generated documentation for var_as_tensor.
    fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>>;
    /// Auto-generated documentation for var_from_tensor.
    fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar>;
    /// Auto-generated documentation for var_to_device.
    fn var_to_device(var: &Self::RawVar, device: &KindleDevice) -> Result<Self::RawVar>;
    /// Auto-generated documentation for assign_var.
    fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()>;
}

// FloatOps only requires Backend, operates on FloatTensorPrimitive
/// Auto-generated documentation for FloatOps.
pub trait FloatOps<B: Backend> {
    /// Auto-generated documentation for relu.
    fn relu<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for step.
    fn step<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for mish.
    fn mish<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for elu.
    fn elu<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for gelu.
    fn gelu<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for abs.
    fn abs<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for exp.
    fn exp<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for neg.
    fn neg<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for sqrt.
    fn sqrt<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for log.
    fn log<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for tanh.
    fn tanh<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for sigmoid.
    fn sigmoid<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for swish.
    fn swish<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for softmax.
    fn softmax<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for add_scalar_float.
    fn add_scalar_float<K: DType>(t: &B::Storage<K>, scalar: f64) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for mul_scalar_float.
    fn mul_scalar_float<K: DType>(t: &B::Storage<K>, scalar: f64) -> Result<B::Storage<K>>;
}

// NumericOps operates generically over any TensorKind!
/// Auto-generated documentation for NumericOps.
pub trait NumericOps<B: Backend> {
    /// Auto-generated documentation for add.
    fn add<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for sub.
    fn sub<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for mul.
    fn mul<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for div.
    fn div<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
}

/// Auto-generated documentation for TensorOps.
pub trait TensorOps<B: Backend> {
    /// Auto-generated documentation for reshape.
    fn reshape<K: DType>(t: &B::Storage<K>, shape: &[usize]) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for transpose.
    fn transpose<K: DType>(t: &B::Storage<K>, dim1: usize, dim2: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for matmul.
    fn matmul<K: DType>(lhs: &B::Storage<K>, rhs: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for broadcast_as.
    fn broadcast_as<K: DType>(t: &B::Storage<K>, shape: &[usize]) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for narrow.
    fn narrow<K: DType>(
        t: &B::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for squeeze.
    fn squeeze<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for stack.
    fn stack<K: DType>(t: &[&B::Storage<K>], dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for concat.
    fn concat<K: DType>(t: &[&B::Storage<K>], dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for slice.
    fn slice<K: DType>(t: &B::Storage<K>, ranges: &[(usize, usize)]) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for flatten.
    fn flatten<K: DType>(
        t: &B::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for broadcast_left.
    fn broadcast_left<K: DType>(t: &B::Storage<K>, shape: &[usize]) -> Result<B::Storage<K>>;

    /// Auto-generated documentation for float_to_scalar.
    fn float_to_scalar<K: DType>(t: &B::Storage<K>) -> Result<f64>;
    /// Auto-generated documentation for float_to_vec1.
    fn float_to_vec1<K: DType>(t: &B::Storage<K>) -> Result<alloc::vec::Vec<f64>>;

    /// Auto-generated documentation for int_to_scalar.
    fn int_to_scalar<K: DType>(t: &B::Storage<K>) -> Result<i64>;
    /// Auto-generated documentation for int_to_vec1.
    fn int_to_vec1<K: DType>(t: &B::Storage<K>) -> Result<alloc::vec::Vec<i64>>;

    /// Auto-generated documentation for tensor_to_dtype.
    fn tensor_to_dtype<K: DType, K2: DType>(
        t: &B::Storage<K>,
        dtype: KindleDType,
    ) -> Result<B::Storage<K2>>;
}

/// Auto-generated documentation for CreationOps.
pub trait CreationOps<B: Backend> {
    /// Auto-generated documentation for zeros.
    fn zeros<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for ones.
    fn ones<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for rand.
    fn rand<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for randn.
    fn randn<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;

    /// Auto-generated documentation for var_zeros.
    fn var_zeros<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::RawVar>;
    /// Auto-generated documentation for var_ones.
    fn var_ones<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::RawVar>;
    /// Auto-generated documentation for var_rand.
    fn var_rand<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::RawVar>;
    /// Auto-generated documentation for var_randn.
    fn var_randn<K: DType>(
        shape: &[usize],
        dtype: KindleDType,
        device: &KindleDevice,
    ) -> Result<B::RawVar>;

    /// Auto-generated documentation for tensor_to_device.
    fn tensor_to_device<K: DType>(
        t: &B::Storage<K>,
        device: &KindleDevice,
    ) -> Result<B::Storage<K>>;
}

/// Auto-generated documentation for ReductionOps.
pub trait ReductionOps<B: Backend> {
    /// Auto-generated documentation for sum_all.
    fn sum_all<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for mean_all.
    fn mean_all<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for max_all.
    fn max_all<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for min_all.
    fn min_all<K: DType>(t: &B::Storage<K>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for sum_dim.
    fn sum_dim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for sum_keepdim.
    fn sum_keepdim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for mean_dim.
    fn mean_dim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for mean_keepdim.
    fn mean_keepdim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for max_dim.
    fn max_dim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for max_keepdim.
    fn max_keepdim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for min_dim.
    fn min_dim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for min_keepdim.
    fn min_keepdim<K: DType>(t: &B::Storage<K>, dim: usize) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for argmax.
    fn argmax<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: Option<usize>,
    ) -> Result<B::Storage<KInt>>;
    /// Auto-generated documentation for argmin.
    fn argmin<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: Option<usize>,
    ) -> Result<B::Storage<KInt>>;
    /// Auto-generated documentation for topk.
    fn topk<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(B::Storage<K>, B::Storage<KInt>)>;
    /// Auto-generated documentation for argsort.
    fn argsort<K: DType, KInt: DType>(
        t: &B::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<B::Storage<KInt>>;
}

/// Auto-generated documentation for ModuleOps.
pub trait ModuleOps<B: Backend> {
    /// Auto-generated documentation for layer_norm.
    fn layer_norm<K: DType>(
        t: &B::Storage<K>,
        weight: &B::Storage<K>,
        bias: Option<&B::Storage<K>>,
        eps: f32,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for batch_norm.
    fn batch_norm<K: DType>(
        t: &B::Storage<K>,
        w: Option<&B::Storage<K>>,
        b: Option<&B::Storage<K>>,
        rm: Option<&B::Storage<K>>,
        rv: Option<&B::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for embedding.
    fn embedding<K: DType, KInt: DType>(
        t: &B::Storage<KInt>,
        w: &B::Storage<K>,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for conv1d.
    fn conv1d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for conv2d.
    fn conv2d<K: DType>(
        t: &B::Storage<K>,
        w: &B::Storage<K>,
        b: Option<&B::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for conv_transpose2d.
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
    /// Auto-generated documentation for max_pool2d.
    fn max_pool2d<K: DType>(
        t: &B::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for avg_pool2d.
    fn avg_pool2d<K: DType>(
        t: &B::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for adaptive_avg_pool2d.
    fn adaptive_avg_pool2d<K: DType>(
        t: &B::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<B::Storage<K>>;
}

/// Auto-generated documentation for LossOps.
pub trait LossOps<B: Backend>: NumericOps<B> + FloatOps<B> + ReductionOps<B> {
    /// Auto-generated documentation for mse_loss.
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

    /// Auto-generated documentation for l1_loss.
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

    /// Auto-generated documentation for bce_with_logits_loss.
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

    /// Auto-generated documentation for cross_entropy_loss.
    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<KInt>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<B::Storage<K>>;
}

/// Auto-generated documentation for QuantizedOps.
pub trait QuantizedOps<B: Backend> {
    /// Auto-generated documentation for quantize.
    fn quantize<K: FloatDType, Q: QuantDType>(t: &B::Storage<K>) -> Result<B::Storage<Q>>;
    /// Auto-generated documentation for dequantize.
    fn dequantize<Q: QuantDType, K: FloatDType>(t: &B::Storage<Q>) -> Result<B::Storage<K>>;
    /// Auto-generated documentation for quantized_matmul.
    fn quantized_matmul<Q: QuantDType>(
        lhs: &B::Storage<Q>,
        rhs: &B::Storage<Q>,
    ) -> Result<B::Storage<f32>>;
}

/// Auto-generated documentation for OptimizerOps.
pub trait OptimizerOps<B: Backend> {
    /// Auto-generated documentation for adamw_step.
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
/// Auto-generated documentation for dummy.
pub mod dummy {
    use super::*;
    use crate::nn::Reduction;
    use crate::prelude::Result;
    use crate::tensor::device::Device;
    use crate::tensor::device::KindleDevice;
    use crate::tensor::dtype::DType;

    /// Auto-generated documentation for DummyBackend.
    pub struct DummyBackend<T, D> {
        _marker: core::marker::PhantomData<(T, D)>,
    }

    impl<T: DType, D: Device + Clone + 'static> Clone for DummyBackend<T, D> {
        /// Auto-generated documentation for clone.
        fn clone(&self) -> Self {
            DummyBackend {
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<T: DType, D: Device + Clone + 'static> Backend for DummyBackend<T, D> {
        /// Auto-generated documentation for Device.
        type Device = D;
        /// Auto-generated documentation for FloatElem.
        type FloatElem = T;
        /// Auto-generated documentation for IntElem.
        type IntElem = i64;
        /// Auto-generated documentation for RawVar.
        type RawVar = alloc::vec::Vec<usize>;
        /// Auto-generated documentation for Grads.
        type Grads = ();
        /// Auto-generated documentation for Storage.
        type Storage<K: DType> = alloc::vec::Vec<usize>;
        /// Auto-generated documentation for InnerBackend.
        type InnerBackend = Self;
        /// Auto-generated documentation for BackendWithDevice.
        type BackendWithDevice<NewD: Device> = DummyBackend<T, NewD>;

        /// Auto-generated documentation for shape.
        fn shape<K: DType>(t: &Self::Storage<K>) -> alloc::vec::Vec<usize> {
            t.clone()
        }
        /// Auto-generated documentation for format_tensor_display.
        fn format_tensor_display<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        /// Auto-generated documentation for format_tensor_debug.
        fn format_tensor_debug<K: DType>(_t: &Self::Storage<K>) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }
        /// Auto-generated documentation for backward.
        fn backward<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }
        /// Auto-generated documentation for backward_with_nan_check.
        fn backward_with_nan_check<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }
        /// Auto-generated documentation for get_grad.
        fn get_grad<K: DType>(
            _t: &Self::Storage<K>,
            _grads: &Self::Grads,
        ) -> Result<Option<Self::Storage<K>>> {
            Ok(None)
        }
        /// Auto-generated documentation for to_bytes.
        fn to_bytes<K: DType>(_t: &Self::Storage<K>) -> Result<alloc::vec::Vec<u8>> {
            Ok(alloc::vec::Vec::new())
        }
        /// Auto-generated documentation for from_bytes.
        fn from_bytes<K: DType>(
            _bytes: &[u8],
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Auto-generated documentation for var_as_tensor.
        fn var_as_tensor<K: DType>(var: &Self::RawVar) -> Result<Self::Storage<K>> {
            Ok(var.clone())
        }
        /// Auto-generated documentation for var_from_tensor.
        fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::RawVar> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for var_to_device.
        fn var_to_device(var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> {
            Ok(var.clone())
        }
        /// Auto-generated documentation for assign_var.
        fn assign_var<K: DType>(var: &mut Self::RawVar, tensor: &Self::Storage<K>) -> Result<()> {
            *var = tensor.clone();
            Ok(())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> CreationOps<Self> for DummyBackend<T, D> {
        /// Auto-generated documentation for zeros.
        fn zeros<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Auto-generated documentation for ones.
        fn ones<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Auto-generated documentation for rand.
        fn rand<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Auto-generated documentation for randn.
        fn randn<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Auto-generated documentation for var_zeros.
        fn var_zeros<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Auto-generated documentation for var_ones.
        fn var_ones<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Auto-generated documentation for var_rand.
        fn var_rand<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Auto-generated documentation for var_randn.
        fn var_randn<K: DType>(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::RawVar> {
            Ok(shape.to_vec())
        }
        /// Auto-generated documentation for tensor_to_device.
        fn tensor_to_device<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _device: &KindleDevice,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> NumericOps<Self> for DummyBackend<T, D> {
        /// Auto-generated documentation for add.
        fn add<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        /// Auto-generated documentation for sub.
        fn sub<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        /// Auto-generated documentation for mul.
        fn mul<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
        /// Auto-generated documentation for div.
        fn div<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            _rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(lhs.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> FloatOps<Self> for DummyBackend<T, D> {
        /// Auto-generated documentation for add_scalar_float.
        fn add_scalar_float<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for mul_scalar_float.
        fn mul_scalar_float<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for relu.
        fn relu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for step.
        fn step<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for mish.
        fn mish<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for elu.
        fn elu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for gelu.
        fn gelu<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for abs.
        fn abs<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for exp.
        fn exp<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for neg.
        fn neg<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for sqrt.
        fn sqrt<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for log.
        fn log<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for tanh.
        fn tanh<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for sigmoid.
        fn sigmoid<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for swish.
        fn swish<K: DType>(
            t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for softmax.
        fn softmax<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> ReductionOps<Self> for DummyBackend<T, D> {
        /// Auto-generated documentation for sum_all.
        fn sum_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for mean_all.
        fn mean_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for max_all.
        fn max_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for min_all.
        fn min_all<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for sum_dim.
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
        /// Auto-generated documentation for sum_keepdim.
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
        /// Auto-generated documentation for mean_dim.
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
        /// Auto-generated documentation for mean_keepdim.
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
        /// Auto-generated documentation for max_dim.
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
        /// Auto-generated documentation for max_keepdim.
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
        /// Auto-generated documentation for min_dim.
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
        /// Auto-generated documentation for min_keepdim.
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
        /// Auto-generated documentation for argmax.
        fn argmax<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for argmin.
        fn argmin<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for topk.
        fn topk<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _k: usize,
            _dim: usize,
            _largest: bool,
        ) -> Result<(<Self as Backend>::Storage<K>, <Self as Backend>::Storage<KInt>)> {
            Ok((alloc::vec![], alloc::vec![]))
        }
        /// Auto-generated documentation for argsort.
        fn argsort<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<K>,
            _dim: usize,
            _descending: bool,
        ) -> Result<<Self as Backend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
    }

    impl<T: DType, D: Device + Clone + 'static> TensorOps<Self> for DummyBackend<T, D> {
        /// Auto-generated documentation for matmul.
        fn matmul<K: DType>(
            lhs: &<Self as Backend>::Storage<K>,
            rhs: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = lhs.clone();
            if out.len() >= 2 && rhs.len() >= 2 {
                let len = out.len();
                out[len - 1] = rhs[rhs.len() - 1];
            }
            Ok(out)
        }
        /// Auto-generated documentation for float_to_scalar.
        fn float_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<f64> {
            Ok(0.0)
        }
        /// Auto-generated documentation for float_to_vec1.
        fn float_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<f64>> {
            Ok(alloc::vec![0.0])
        }
        /// Auto-generated documentation for int_to_scalar.
        fn int_to_scalar<K: DType>(_t: &<Self as Backend>::Storage<K>) -> Result<i64> {
            Ok(0)
        }
        /// Auto-generated documentation for int_to_vec1.
        fn int_to_vec1<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<i64>> {
            Ok(alloc::vec![0])
        }

        /// Auto-generated documentation for broadcast_as.
        fn broadcast_as<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Auto-generated documentation for broadcast_left.
        fn broadcast_left<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = s.to_vec();
            out.extend_from_slice(t);
            Ok(out)
        }
        /// Auto-generated documentation for reshape.
        fn reshape<K: DType>(
            _t: &<Self as Backend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Auto-generated documentation for transpose.
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
        /// Auto-generated documentation for flatten.
        fn flatten<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            s: usize,
            e: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            if s > e || e >= t.len() {
                return Ok(t.clone());
            }
            let mut out = t[..s].to_vec();
            out.push(t[s..=e].iter().product());
            out.extend_from_slice(&t[e + 1..]);
            Ok(out)
        }
        /// Auto-generated documentation for slice.
        fn slice<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            ranges: &[(usize, usize)],
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            for (dim, &(start, end)) in ranges.iter().enumerate() {
                if dim < out.len() {
                    out[dim] = end.saturating_sub(start);
                }
            }
            Ok(out)
        }
        /// Auto-generated documentation for narrow.
        fn narrow<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            d: usize,
            _s: usize,
            l: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            if d < out.len() {
                out[d] = l;
            }
            Ok(out)
        }
        /// Auto-generated documentation for squeeze.
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
        /// Auto-generated documentation for stack.
        fn stack<K: DType>(
            t: &[&<Self as Backend>::Storage<K>],
            d: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            if t.is_empty() {
                return Ok(alloc::vec![]);
            }
            let mut out = t[0].clone();
            if d <= out.len() {
                out.insert(d, t.len());
            }
            Ok(out)
        }
        /// Auto-generated documentation for concat.
        fn concat<K: DType>(
            t: &[&<Self as Backend>::Storage<K>],
            d: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            if t.is_empty() {
                return Ok(alloc::vec![]);
            }
            let mut out = t[0].clone();
            if d < out.len() {
                out[d] = t.iter().map(|s| s.get(d).copied().unwrap_or(0)).sum();
            }
            Ok(out)
        }
        /// Auto-generated documentation for tensor_to_dtype.
        fn tensor_to_dtype<K: DType, K2: DType>(
            t: &<Self as Backend>::Storage<K>,
            _dtype: KindleDType,
        ) -> Result<<Self as Backend>::Storage<K2>> {
            Ok(t.clone())
        }
    }

    impl<T: DType, D: Device + Clone + 'static> ModuleOps<Self> for DummyBackend<T, D> {
        /// Auto-generated documentation for layer_norm.
        fn layer_norm<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            _w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            _e: f32,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Auto-generated documentation for batch_norm.
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
        /// Auto-generated documentation for embedding.
        fn embedding<K: DType, KInt: DType>(
            _t: &<Self as Backend>::Storage<KInt>,
            _w: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for conv1d.
        fn conv1d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            s: usize,
            p: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 3 && w.len() >= 3 {
                let l_in = out[len - 1];
                let k = w[w.len() - 1];
                let c_out = w[0]; // Assuming [C_out, C_in / groups, K]
                out[len - 2] = c_out;
                out[len - 1] = (l_in + 2 * p - d * (k - 1) - 1) / s + 1;
            }
            Ok(out)
        }
        /// Auto-generated documentation for conv2d.
        fn conv2d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            s: usize,
            p: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 4 && w.len() >= 4 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                let k_h = w[w.len() - 2];
                let k_w = w[w.len() - 1];
                let c_out = w[0]; // [C_out, C_in / groups, K_H, K_W]
                out[len - 3] = c_out;
                out[len - 2] = (h_in + 2 * p - d * (k_h - 1) - 1) / s + 1;
                out[len - 1] = (w_in + 2 * p - d * (k_w - 1) - 1) / s + 1;
            }
            Ok(out)
        }
        /// Auto-generated documentation for conv_transpose2d.
        fn conv_transpose2d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            w: &<Self as Backend>::Storage<K>,
            _b: Option<&<Self as Backend>::Storage<K>>,
            s: usize,
            p: usize,
            op: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 4 && w.len() >= 4 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                let k_h = w[w.len() - 2];
                let k_w = w[w.len() - 1];
                let c_out = w[1]; // [C_in, C_out / groups, K_H, K_W]
                out[len - 3] = c_out;
                out[len - 2] = (h_in - 1) * s - 2 * p + d * (k_h - 1) + op + 1;
                out[len - 1] = (w_in - 1) * s - 2 * p + d * (k_w - 1) + op + 1;
            }
            Ok(out)
        }
        /// Auto-generated documentation for max_pool2d.
        fn max_pool2d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            k: (usize, usize),
            s: (usize, usize),
            p: (usize, usize),
            d: (usize, usize),
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 2 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                out[len - 2] = (h_in + 2 * p.0 - d.0 * (k.0 - 1) - 1) / s.0 + 1;
                out[len - 1] = (w_in + 2 * p.1 - d.1 * (k.1 - 1) - 1) / s.1 + 1;
            }
            Ok(out)
        }
        /// Auto-generated documentation for avg_pool2d.
        fn avg_pool2d<K: DType>(
            t: &<Self as Backend>::Storage<K>,
            k: (usize, usize),
            s: (usize, usize),
            p: (usize, usize),
        ) -> Result<<Self as Backend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 2 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                out[len - 2] = (h_in + 2 * p.0 - (k.0 - 1) - 1) / s.0 + 1;
                out[len - 1] = (w_in + 2 * p.1 - (k.1 - 1) - 1) / s.1 + 1;
            }
            Ok(out)
        }
        /// Auto-generated documentation for adaptive_avg_pool2d.
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
        /// Auto-generated documentation for mse_loss.
        fn mse_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for l1_loss.
        fn l1_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for bce_with_logits_loss.
        fn bce_with_logits_loss<K: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<K>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for cross_entropy_loss.
        fn cross_entropy_loss<K: DType, KInt: DType>(
            _pred: &<Self as Backend>::Storage<K>,
            _target: &<Self as Backend>::Storage<KInt>,
            _r: Reduction,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
    }

    impl<T: DType, D: Device + Clone + 'static> QuantizedOps<Self> for DummyBackend<T, D> {
        /// Auto-generated documentation for quantize.
        fn quantize<K: FloatDType, Q: QuantDType>(
            _t: &<Self as Backend>::Storage<K>,
        ) -> Result<<Self as Backend>::Storage<Q>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for dequantize.
        fn dequantize<Q: QuantDType, K: FloatDType>(
            _t: &<Self as Backend>::Storage<Q>,
        ) -> Result<<Self as Backend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Auto-generated documentation for quantized_matmul.
        fn quantized_matmul<Q: QuantDType>(
            _lhs: &<Self as Backend>::Storage<Q>,
            _rhs: &<Self as Backend>::Storage<Q>,
        ) -> Result<<Self as Backend>::Storage<f32>> {
            Ok(alloc::vec![])
        }
    }
    impl<T: DType, D: Device + Clone + 'static> OptimizerOps<Self> for DummyBackend<T, D> {}
}
