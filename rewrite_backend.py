import re

with open("crates/kindle-core/src/tensor/backend.rs", "r") as f:
    content = f.read()

new_backend = """
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
"""

content = re.sub(r'pub trait Backend:.*?(?=pub trait ModuleOps)', new_backend, content, flags=re.DOTALL)

with open("crates/kindle-core/src/tensor/backend.rs", "w") as f:
    f.write(content)
