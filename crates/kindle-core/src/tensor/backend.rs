use crate::prelude::{KindleDType, KindleDevice, Result};

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
}

impl From<f32> for ScalarValue { fn from(v: f32) -> Self { ScalarValue::Float(v as f64) } }
impl From<f64> for ScalarValue { fn from(v: f64) -> Self { ScalarValue::Float(v) } }
impl From<i32> for ScalarValue { fn from(v: i32) -> Self { ScalarValue::Int(v as i64) } }
impl From<i64> for ScalarValue { fn from(v: i64) -> Self { ScalarValue::Int(v) } }


pub trait Backend:
    Clone
    + 'static
    + CreationOps<Self>
    + NumericOps<Self>
    + TensorOps<Self>
{
    type Device: super::device::Device;
    type FloatElem: super::dtype::FloatDType;
    type IntElem: super::dtype::IntDType;
    type BoolElem: super::dtype::BoolDType;

    type FloatTensorPrimitive: Clone;
    type IntTensorPrimitive: Clone;
    type BoolTensorPrimitive: Clone;

    type RawVar: Clone;
    type Grads;
    type InnerBackend: Backend;

    fn shape<K: super::kind::TensorKind>(t: &K::Primitive<Self>) -> alloc::vec::Vec<usize>;
    fn format_tensor_display<K: super::kind::TensorKind>(t: &K::Primitive<Self>) -> alloc::string::String;
    fn format_tensor_debug<K: super::kind::TensorKind>(t: &K::Primitive<Self>) -> alloc::string::String;

    fn from_inner<K: super::kind::TensorKind>(
        inner: K::Primitive<Self>,
        shape: alloc::vec::Vec<usize>,
        device: Self::Device,
    ) -> crate::tensor::base::Tensor<crate::shapes::Dyn, Self, crate::tensor::grad::NoneGrad>
    where
        Self: Sized;

    fn backward<K: super::kind::TensorKind>(t: &K::Primitive<Self>) -> Result<Self::Grads>;
    fn get_grad<K: super::kind::TensorKind>(t: &K::Primitive<Self>, grads: &Self::Grads) -> Result<Option<K::Primitive<Self>>>;

    fn to_bytes<K: super::kind::TensorKind>(t: &K::Primitive<Self>) -> Result<std::vec::Vec<u8>>;
    
    // We provide from_bytes separated by primitive
    fn float_from_bytes(bytes: &[u8], shape: &[usize], device: &Self::Device) -> Result<Self::FloatTensorPrimitive>;
    fn int_from_bytes(bytes: &[u8], shape: &[usize], device: &Self::Device) -> Result<Self::IntTensorPrimitive>;
    fn bool_from_bytes(bytes: &[u8], shape: &[usize], device: &Self::Device) -> Result<Self::BoolTensorPrimitive>;
}

// FloatOps only requires Backend, operates on FloatTensorPrimitive
pub trait FloatOps<B: Backend> {
    fn abs(t: &B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive>;
    fn exp(t: &B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive>;
    fn neg(t: &B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive>;
    fn sqrt(t: &B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive>;
    fn log(t: &B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive>;
    fn tanh(t: &B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive>;
    fn sigmoid(t: &B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive>;
    fn swish(t: &B::FloatTensorPrimitive) -> Result<B::FloatTensorPrimitive>;
    fn softmax(t: &B::FloatTensorPrimitive, dim: usize) -> Result<B::FloatTensorPrimitive>;
    fn add_scalar_float(t: &B::FloatTensorPrimitive, scalar: f64) -> Result<B::FloatTensorPrimitive>;
    fn mul_scalar_float(t: &B::FloatTensorPrimitive, scalar: f64) -> Result<B::FloatTensorPrimitive>;
}

// NumericOps operates generically over any TensorKind! 
pub trait NumericOps<B: Backend> {
    fn add<K: super::kind::TensorKind>(lhs: &K::Primitive<B>, rhs: &K::Primitive<B>) -> Result<K::Primitive<B>>;
    fn sub<K: super::kind::TensorKind>(lhs: &K::Primitive<B>, rhs: &K::Primitive<B>) -> Result<K::Primitive<B>>;
    fn mul<K: super::kind::TensorKind>(lhs: &K::Primitive<B>, rhs: &K::Primitive<B>) -> Result<K::Primitive<B>>;
    fn div<K: super::kind::TensorKind>(lhs: &K::Primitive<B>, rhs: &K::Primitive<B>) -> Result<K::Primitive<B>>;
}

pub trait TensorOps<B: Backend> {
    fn reshape<K: super::kind::TensorKind>(t: &K::Primitive<B>, shape: &[usize]) -> Result<K::Primitive<B>>;
    fn transpose<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim1: usize, dim2: usize) -> Result<K::Primitive<B>>;
    fn matmul<K: super::kind::TensorKind>(lhs: &K::Primitive<B>, rhs: &K::Primitive<B>) -> Result<K::Primitive<B>>;
    fn broadcast_as<K: super::kind::TensorKind>(t: &K::Primitive<B>, shape: &[usize]) -> Result<K::Primitive<B>>;
    fn narrow<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize, start: usize, len: usize) -> Result<K::Primitive<B>>;
    fn squeeze<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize) -> Result<K::Primitive<B>>;
    fn stack<K: super::kind::TensorKind>(t: &[&K::Primitive<B>], dim: usize) -> Result<K::Primitive<B>>;
    fn concat<K: super::kind::TensorKind>(t: &[&K::Primitive<B>], dim: usize) -> Result<K::Primitive<B>>;
}

pub trait CreationOps<B: Backend> {
    fn float_zeros(shape: &[usize], device: &B::Device) -> Result<B::FloatTensorPrimitive>;
    fn float_ones(shape: &[usize], device: &B::Device) -> Result<B::FloatTensorPrimitive>;
    fn float_rand(shape: &[usize], device: &B::Device) -> Result<B::FloatTensorPrimitive>;
    fn float_randn(shape: &[usize], device: &B::Device) -> Result<B::FloatTensorPrimitive>;
    
    fn int_zeros(shape: &[usize], device: &B::Device) -> Result<B::IntTensorPrimitive>;
    fn int_ones(shape: &[usize], device: &B::Device) -> Result<B::IntTensorPrimitive>;
    
    fn bool_zeros(shape: &[usize], device: &B::Device) -> Result<B::BoolTensorPrimitive>;
    fn bool_ones(shape: &[usize], device: &B::Device) -> Result<B::BoolTensorPrimitive>;

    fn tensor_to_device<K: super::kind::TensorKind>(t: &K::Primitive<B>, device: &B::Device) -> Result<K::Primitive<B>>;
}

pub trait ReductionOps<B: Backend> {
    fn sum_all<K: super::kind::TensorKind>(t: &K::Primitive<B>) -> Result<K::Primitive<B>>;
    fn mean_all<K: super::kind::TensorKind>(t: &K::Primitive<B>) -> Result<K::Primitive<B>>;
    fn max_all<K: super::kind::TensorKind>(t: &K::Primitive<B>) -> Result<K::Primitive<B>>;
    fn min_all<K: super::kind::TensorKind>(t: &K::Primitive<B>) -> Result<K::Primitive<B>>;
    fn sum_dim<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize) -> Result<K::Primitive<B>>;
    fn sum_keepdim<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize) -> Result<K::Primitive<B>>;
    fn mean_dim<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize) -> Result<K::Primitive<B>>;
    fn mean_keepdim<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize) -> Result<K::Primitive<B>>;
    fn max_dim<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize) -> Result<K::Primitive<B>>;
    fn max_keepdim<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize) -> Result<K::Primitive<B>>;
    fn min_dim<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize) -> Result<K::Primitive<B>>;
    fn min_keepdim<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: usize) -> Result<K::Primitive<B>>;
    fn argmax<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: Option<usize>) -> Result<B::IntTensorPrimitive>;
    fn argmin<K: super::kind::TensorKind>(t: &K::Primitive<B>, dim: Option<usize>) -> Result<B::IntTensorPrimitive>;
}

pub trait ModuleOps<B: Backend> {
    fn conv2d(
        x: &B::FloatTensorPrimitive,
        weight: &B::FloatTensorPrimitive,
        bias: Option<&B::FloatTensorPrimitive>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<B::FloatTensorPrimitive>;
}

pub trait LossOps<B: Backend> {
    fn mse_loss(
        pred: &B::FloatTensorPrimitive,
        target: &B::FloatTensorPrimitive,
        reduction: crate::tensor::ops::loss::Reduction,
    ) -> Result<B::FloatTensorPrimitive>;

    fn cross_entropy_loss(
        pred: &B::FloatTensorPrimitive,
        target: &B::IntTensorPrimitive, // target is typically Int!
        reduction: crate::tensor::ops::loss::Reduction,
    ) -> Result<B::FloatTensorPrimitive>;
}
pub trait ModuleOps<B: Backend> {
    fn layer_norm(t: &B::RawTensor, weight: &B::RawTensor, bias: &B::RawTensor, eps: f32) -> Result<B::RawTensor>;
    fn batch_norm(t: &B::RawTensor, w: &B::RawTensor, b: &B::RawTensor, rm: &B::RawTensor, rv: &B::RawTensor, e: f32) -> Result<B::RawTensor>;
    fn embedding(t: &B::RawTensor, w: &B::RawTensor) -> Result<B::RawTensor>;
    fn conv1d(t: &B::RawTensor, w: &B::RawTensor, b: Option<&B::RawTensor>, stride: usize, padding: usize, dilation: usize) -> Result<B::RawTensor>;
    fn conv2d(t: &B::RawTensor, w: &B::RawTensor, b: Option<&B::RawTensor>, stride: usize, padding: usize, dilation: usize) -> Result<B::RawTensor>;
    fn conv_transpose2d(t: &B::RawTensor, w: &B::RawTensor, b: Option<&B::RawTensor>, stride: usize, padding: usize, output_padding: usize, dilation: usize) -> Result<B::RawTensor>;
    fn max_pool2d(t: &B::RawTensor, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize), dilation: (usize, usize)) -> Result<B::RawTensor>;
    fn avg_pool2d(t: &B::RawTensor, kernel_size: (usize, usize), stride: (usize, usize), padding: (usize, usize)) -> Result<B::RawTensor>;
    fn adaptive_avg_pool2d(t: &B::RawTensor, output_size: (usize, usize)) -> Result<B::RawTensor>;
}

pub trait LossOps<B: Backend> {
    fn mse_loss(pred: &B::RawTensor, target: &B::RawTensor, reduction: crate::nn::loss::Reduction) -> Result<B::RawTensor>;
    fn l1_loss(pred: &B::RawTensor, target: &B::RawTensor, reduction: crate::nn::loss::Reduction) -> Result<B::RawTensor>;
    fn bce_with_logits_loss(pred: &B::RawTensor, target: &B::RawTensor, reduction: crate::nn::loss::Reduction) -> Result<B::RawTensor>;
    fn cross_entropy_loss(pred: &B::RawTensor, target: &B::RawTensor, reduction: crate::nn::loss::Reduction) -> Result<B::RawTensor>;
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
        type InnerBackend = Self;

        fn shape(t: &<Self as Backend>::RawTensor) -> alloc::vec::Vec<usize> { t.clone() }
        fn format_tensor_display(t: &<Self as Backend>::RawTensor) -> alloc::string::String { alloc::format!("{:?}", t) }
        fn format_tensor_debug(t: &<Self as Backend>::RawTensor) -> alloc::string::String { alloc::format!("Tensor(shape={:?})", t) }
        fn var_as_tensor(var: &Self::RawVar) -> Result<<Self as Backend>::RawTensor> { Ok(var.clone()) }
        fn var_from_tensor(t: &<Self as Backend>::RawTensor) -> Result<Self::RawVar> { Ok(t.clone()) }
        fn assign_var(var: &mut Self::RawVar, tensor: &<Self as Backend>::RawTensor) -> Result<()> {
            *var = tensor.clone();
            Ok(())
        }
        fn tensor_to_device(t: &<Self as Backend>::RawTensor, _d: &KindleDevice) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn var_to_device(v: &Self::RawVar, _d: &KindleDevice) -> Result<Self::RawVar> { Ok(v.clone()) }
        fn to_dtype(t: &<Self as Backend>::RawTensor, _d: KindleDType) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn backward(_loss: &<Self as Backend>::RawTensor) -> Result<Self::Grads> { Ok(()) }
        fn get_grad(_var: &Self::RawVar, _grads: &Self::Grads) -> Result<Option<Self::RawTensor>> { Ok(None) }
        fn to_bytes(_t: &<Self as Backend>::RawTensor) -> Result<alloc::vec::Vec<u8>> { Ok(alloc::vec::Vec::new()) }
        fn from_bytes(_bytes: &[u8], shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<<Self as Backend>::RawTensor> { Ok(shape.to_vec()) }
    }

    impl<T: DType, D: Device> CreationOps<Self> for DummyBackend<T, D> {
        fn zeros(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<<Self as Backend>::RawTensor> { Ok(s.to_vec()) }
        fn ones(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<<Self as Backend>::RawTensor> { Ok(s.to_vec()) }
        fn rand(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<<Self as Backend>::RawTensor> { Ok(s.to_vec()) }
        fn randn(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<<Self as Backend>::RawTensor> { Ok(s.to_vec()) }
        fn var_zeros(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<<Self as Backend>::RawVar> { Ok(s.to_vec()) }
        fn var_ones(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<<Self as Backend>::RawVar> { Ok(s.to_vec()) }
        fn var_rand(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<<Self as Backend>::RawVar> { Ok(s.to_vec()) }
        fn var_randn(s: &[usize], _dt: KindleDType, _d: &KindleDevice) -> Result<<Self as Backend>::RawVar> { Ok(s.to_vec()) }
    }

    impl<T: DType, D: Device> NumericOps<Self> for DummyBackend<T, D> {
        fn add(t1: &<Self as Backend>::RawTensor, _t2: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t1.clone()) }
        fn sub(t1: &<Self as Backend>::RawTensor, _t2: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t1.clone()) }
        fn mul(t1: &<Self as Backend>::RawTensor, _t2: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t1.clone()) }
        fn div(t1: &<Self as Backend>::RawTensor, _t2: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t1.clone()) }
        fn matmul(t1: &<Self as Backend>::RawTensor, t2: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> {
            let mut out = t1.clone();
            if let (Some(last_out), Some(last_t2)) = (out.last_mut(), t2.last()) {
                *last_out = *last_t2;
            }
            Ok(out)
        }
        fn mul_scalar(t: &<Self as Backend>::RawTensor, _s: ScalarValue) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn add_scalar(t: &<Self as Backend>::RawTensor, _s: ScalarValue) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
    }

    impl<T: DType, D: Device> FloatOps<Self> for DummyBackend<T, D> {
        fn relu(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn gelu(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn abs(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn exp(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn neg(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn sqrt(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn log(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn tanh(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn sigmoid(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn swish(t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn softmax(t: &<Self as Backend>::RawTensor, _dim: usize) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
    }

    impl<T: DType, D: Device> ReductionOps<Self> for DummyBackend<T, D> {
        fn sum_all(_t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
        fn mean_all(_t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
        fn max_all(_t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
        fn min_all(_t: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
        fn sum_dim(t: &<Self as Backend>::RawTensor, dim: usize) -> Result<<Self as Backend>::RawTensor> { let mut s = t.clone(); if dim < s.len() { s.remove(dim); } Ok(s) }
        fn sum_keepdim(t: &<Self as Backend>::RawTensor, dim: usize) -> Result<<Self as Backend>::RawTensor> { let mut s = t.clone(); if dim < s.len() { s[dim] = 1; } Ok(s) }
        fn mean_dim(t: &<Self as Backend>::RawTensor, dim: usize) -> Result<<Self as Backend>::RawTensor> { let mut s = t.clone(); if dim < s.len() { s.remove(dim); } Ok(s) }
        fn mean_keepdim(t: &<Self as Backend>::RawTensor, dim: usize) -> Result<<Self as Backend>::RawTensor> { let mut s = t.clone(); if dim < s.len() { s[dim] = 1; } Ok(s) }
        fn max_dim(t: &<Self as Backend>::RawTensor, dim: usize) -> Result<<Self as Backend>::RawTensor> { let mut s = t.clone(); if dim < s.len() { s.remove(dim); } Ok(s) }
        fn max_keepdim(t: &<Self as Backend>::RawTensor, dim: usize) -> Result<<Self as Backend>::RawTensor> { let mut s = t.clone(); if dim < s.len() { s[dim] = 1; } Ok(s) }
        fn min_dim(t: &<Self as Backend>::RawTensor, dim: usize) -> Result<<Self as Backend>::RawTensor> { let mut s = t.clone(); if dim < s.len() { s.remove(dim); } Ok(s) }
        fn min_keepdim(t: &<Self as Backend>::RawTensor, dim: usize) -> Result<<Self as Backend>::RawTensor> { let mut s = t.clone(); if dim < s.len() { s[dim] = 1; } Ok(s) }
        fn argmax(_t: &<Self as Backend>::RawTensor, _dim: usize) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
        fn argmin(_t: &<Self as Backend>::RawTensor, _dim: usize) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
    }

    impl<T: DType, D: Device> TensorOps<Self> for DummyBackend<T, D> {
        fn broadcast_as(_t: &<Self as Backend>::RawTensor, s: &[usize]) -> Result<<Self as Backend>::RawTensor> { Ok(s.to_vec()) }
        fn broadcast_left(_t: &<Self as Backend>::RawTensor, s: &[usize]) -> Result<<Self as Backend>::RawTensor> { Ok(s.to_vec()) }
        fn reshape(_t: &<Self as Backend>::RawTensor, s: &[usize]) -> Result<<Self as Backend>::RawTensor> { Ok(s.to_vec()) }
        fn transpose(t: &<Self as Backend>::RawTensor, d1: usize, d2: usize) -> Result<<Self as Backend>::RawTensor> {
            let mut out = t.clone();
            if d1 < out.len() && d2 < out.len() { out.swap(d1, d2); }
            Ok(out)
        }
        fn flatten(t: &<Self as Backend>::RawTensor, s: usize, e: usize) -> Result<<Self as Backend>::RawTensor> {
            let mut out = alloc::vec![];
            for i in 0..s.min(t.len()) { out.push(t[i]); }
            if s <= e && s < t.len() { out.push(t[s..=e.min(t.len() - 1)].iter().product()); }
            for i in (e + 1)..t.len() { out.push(t[i]); }
            Ok(out)
        }
        fn slice(t: &<Self as Backend>::RawTensor, _ranges: &[(usize, usize)]) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn narrow(t: &<Self as Backend>::RawTensor, d: usize, _s: usize, l: usize) -> Result<<Self as Backend>::RawTensor> {
            let mut out = t.clone();
            if d < out.len() { out[d] = l; }
            Ok(out)
        }
        fn squeeze(t: &<Self as Backend>::RawTensor, d: usize) -> Result<<Self as Backend>::RawTensor> {
            let mut out = t.clone();
            if d < out.len() { out.remove(d); }
            Ok(out)
        }
        fn stack(t: &[&<Self as Backend>::RawTensor], d: usize) -> Result<<Self as Backend>::RawTensor> {
            let mut out = t[0].clone();
            if d <= out.len() { out.insert(d, t.len()); }
            Ok(out)
        }
        fn concat(t: &[&<Self as Backend>::RawTensor], d: usize) -> Result<<Self as Backend>::RawTensor> {
            let mut out = t[0].clone();
            if d < out.len() { out[d] = t.iter().map(|x| x[d]).sum(); }
            Ok(out)
        }
    }

    impl<T: DType, D: Device> ModuleOps<Self> for DummyBackend<T, D> {
        fn layer_norm(t: &<Self as Backend>::RawTensor, _w: &<Self as Backend>::RawTensor, _b: &<Self as Backend>::RawTensor, _e: f32) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn batch_norm(t: &<Self as Backend>::RawTensor, _w: &<Self as Backend>::RawTensor, _b: &<Self as Backend>::RawTensor, _rm: &<Self as Backend>::RawTensor, _rv: &<Self as Backend>::RawTensor, _e: f32) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn embedding(_t: &<Self as Backend>::RawTensor, _w: &<Self as Backend>::RawTensor) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
        fn conv1d(t: &<Self as Backend>::RawTensor, w: &<Self as Backend>::RawTensor, _b: Option<&<Self as Backend>::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<<Self as Backend>::RawTensor> {
            let v1 = t.get(0).copied().unwrap_or(0);
            let v2 = w.get(0).copied().unwrap_or(0);
            let v3 = t.get(2).copied().unwrap_or(0);
            Ok(alloc::vec![v1, v2, v3])
        }
        fn conv2d(t: &<Self as Backend>::RawTensor, w: &<Self as Backend>::RawTensor, _b: Option<&<Self as Backend>::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<<Self as Backend>::RawTensor> {
            let v1 = t.get(0).copied().unwrap_or(0);
            let v2 = w.get(0).copied().unwrap_or(0);
            let v3 = t.get(2).copied().unwrap_or(0);
            let v4 = t.get(3).copied().unwrap_or(0);
            Ok(alloc::vec![v1, v2, v3, v4])
        }
        fn conv_transpose2d(t: &<Self as Backend>::RawTensor, w: &<Self as Backend>::RawTensor, _b: Option<&<Self as Backend>::RawTensor>, _s: usize, _p: usize, _op: usize, _d: usize) -> Result<<Self as Backend>::RawTensor> {
            let v1 = t.get(0).copied().unwrap_or(0);
            let v2 = w.get(1).copied().unwrap_or(0);
            let v3 = t.get(2).copied().unwrap_or(0);
            let v4 = t.get(3).copied().unwrap_or(0);
            Ok(alloc::vec![v1, v2, v3, v4])
        }
        fn max_pool2d(t: &<Self as Backend>::RawTensor, _k: (usize, usize), _s: (usize, usize), _p: (usize, usize), _d: (usize, usize)) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn avg_pool2d(t: &<Self as Backend>::RawTensor, _k: (usize, usize), _s: (usize, usize), _p: (usize, usize)) -> Result<<Self as Backend>::RawTensor> { Ok(t.clone()) }
        fn adaptive_avg_pool2d(t: &<Self as Backend>::RawTensor, out: (usize, usize)) -> Result<<Self as Backend>::RawTensor> {
            let v1 = t.get(0).copied().unwrap_or(0);
            let v2 = t.get(1).copied().unwrap_or(0);
            Ok(alloc::vec![v1, v2, out.0, out.1])
        }
    }

    impl<T: DType, D: Device> LossOps<Self> for DummyBackend<T, D> {
        fn mse_loss(_pred: &<Self as Backend>::RawTensor, _target: &<Self as Backend>::RawTensor, _r: crate::nn::loss::Reduction) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
        fn l1_loss(_pred: &<Self as Backend>::RawTensor, _target: &<Self as Backend>::RawTensor, _r: crate::nn::loss::Reduction) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
        fn bce_with_logits_loss(_pred: &<Self as Backend>::RawTensor, _target: &<Self as Backend>::RawTensor, _r: crate::nn::loss::Reduction) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
        fn cross_entropy_loss(_pred: &<Self as Backend>::RawTensor, _target: &<Self as Backend>::RawTensor, _r: crate::nn::loss::Reduction) -> Result<<Self as Backend>::RawTensor> { Ok(alloc::vec![]) }
    }
}
