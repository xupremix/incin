//! Element-wise tensor operations with compile-time shape checking.
//!
//! Operations require matching Shape, DType, Device, and RequiresGrad.
//! This ensures at compile time that you can't accidentally add tensors
//! of different shapes, dtypes, or on different devices.

use crate::prelude::{Backend, DType, Device, Dyn, DynShape, RequiresGrad, Result, Shape, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexSpec {
    All,
    Range(usize, usize),
    RangeFrom(usize),
    RangeTo(usize),
    Index(usize),
}

impl From<usize> for IndexSpec {
    fn from(idx: usize) -> Self {
        IndexSpec::Index(idx)
    }
}
impl From<core::ops::Range<usize>> for IndexSpec {
    fn from(r: core::ops::Range<usize>) -> Self {
        IndexSpec::Range(r.start, r.end)
    }
}
impl From<core::ops::RangeFrom<usize>> for IndexSpec {
    fn from(r: core::ops::RangeFrom<usize>) -> Self {
        IndexSpec::RangeFrom(r.start)
    }
}
impl From<core::ops::RangeTo<usize>> for IndexSpec {
    fn from(r: core::ops::RangeTo<usize>) -> Self {
        IndexSpec::RangeTo(r.end)
    }
}
impl From<core::ops::RangeFull> for IndexSpec {
    fn from(_: core::ops::RangeFull) -> Self {
        IndexSpec::All
    }
}

pub trait ShapeEq<Other> {
    const SHAPES_EQUAL: bool;
    const ASSERT_SHAPES_MATCH: ();
}

impl<S> ShapeEq<S> for S {
    const SHAPES_EQUAL: bool = true;
    const ASSERT_SHAPES_MATCH: () = assert!(
        Self::SHAPES_EQUAL,
        "Shape Mismatch: Attempted to operate on tensors of incompatible shapes."
    );
}

pub trait DTypeEq<Other> {
    const DTYPES_EQUAL: bool;
    const ASSERT_DTYPES_MATCH: ();
}

impl<T> DTypeEq<T> for T {
    const DTYPES_EQUAL: bool = true;
    const ASSERT_DTYPES_MATCH: () = assert!(
        Self::DTYPES_EQUAL,
        "DType Mismatch: Attempted to operate on tensors of incompatible datatypes."
    );
}

macro_rules! impl_binary_op {
    ($trait_name:ident, $method:ident, $backend_method:ident) => {
        // Tensor op Tensor → Result<Tensor> (owned)
        impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
            pub fn $method<S2: Shape, G2: RequiresGrad>(
                &self,
                rhs: &Tensor<S2, B, G2>,
            ) -> Result<Self>
            where
                S: ShapeEq<S2>,
            {
                let _ = <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
                let inner = B::$backend_method(&self.inner, &rhs.inner)?;
                Ok(Tensor::from_parts(
                    inner,
                    self._shape.clone(),
                    self._dtype.clone(),
                    self._device.clone(),
                    self._grad.clone(),
                ))
            }
        }
    };
}

impl_binary_op!(Add, add, add);
impl_binary_op!(Sub, sub, sub);
impl_binary_op!(Mul, mul, mul);
impl_binary_op!(Div, div, div);

macro_rules! impl_unary_op {
    ($method:ident, $backend_method:ident) => {
        pub fn $method(&self) -> Result<Self> {
            let inner = B::$backend_method(&self.inner)?;
            Ok(Tensor::from_parts(
                inner,
                self._shape.clone(),
                self._dtype.clone(),
                self._device.clone(),
                self._grad.clone(),
            ))
        }
    };
}

impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    impl_unary_op!(abs, abs);
    impl_unary_op!(relu, relu);
    impl_unary_op!(gelu, gelu);
    impl_unary_op!(swish, swish);

    #[inline]
    pub fn softmax(&self, dim: usize) -> Result<Tensor<S, B, G>> {
        let inner = B::softmax(&self.inner, dim)?;
        Ok(Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
    impl_unary_op!(neg, neg);
    impl_unary_op!(sqrt, sqrt);
    impl_unary_op!(exp, exp);
    impl_unary_op!(log, log);
    impl_unary_op!(tanh, tanh);
    impl_unary_op!(sigmoid, sigmoid);

    pub fn mul_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = B::mul_scalar(&self.inner, scalar)?;
        Ok(Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn add_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = B::add_scalar(&self.inner, scalar)?;
        Ok(Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

macro_rules! impl_reduction_op {
    ($method:ident, $backend_method:ident) => {
        pub fn $method(self) -> Result<Tensor<(), B, G>> {
            let inner = B::$backend_method(&self.inner)?;
            Ok(Tensor::from_parts(
                inner,
                (), // Scalar shape field
                self._dtype,
                self._device,
                self._grad,
            ))
        }
    };
}

macro_rules! impl_reduction_dim_op {
    ($method:ident, $backend_method:ident, $trait_bound:ident) => {
        pub fn $method<const DIM: usize>(&self) -> Result<Tensor<S::Output, B, G>>
        where
            S: DynShape + crate::shapes::$trait_bound<DIM>,
        {
            let inner = B::$backend_method(&self.inner, DIM)?;

            // We just use from_dyn to construct the resulting shape field dynamically,
            // since we know it's a dimensional reduction.
            let mut out_dims = S::dims(&self._shape).into();
            if stringify!($trait_bound) == "ReduceDim" {
                out_dims.remove(DIM);
            } else {
                out_dims[DIM] = 1;
            }

            Ok(Tensor::from_parts(
                inner,
                S::Output::from_dyn(&out_dims).unwrap(),
                self._dtype.clone(),
                self._device.clone(),
                self._grad.clone(),
            ))
        }
    };
}

impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    impl_reduction_op!(sum_all, sum_all);
    impl_reduction_op!(mean_all, mean_all);
    impl_reduction_op!(max_all, max_all);
    impl_reduction_op!(min_all, min_all);

    impl_reduction_dim_op!(sum_dim, sum_dim, ReduceDim);
    impl_reduction_dim_op!(sum_keepdim, sum_keepdim, ReduceKeepDim);
    impl_reduction_dim_op!(mean_dim, mean_dim, ReduceDim);
    impl_reduction_dim_op!(mean_keepdim, mean_keepdim, ReduceKeepDim);
    impl_reduction_dim_op!(max_dim, max_dim, ReduceDim);
    impl_reduction_dim_op!(max_keepdim, max_keepdim, ReduceKeepDim);
    impl_reduction_dim_op!(min_dim, min_dim, ReduceDim);
    impl_reduction_dim_op!(min_keepdim, min_keepdim, ReduceKeepDim);
}

impl<S: Shape + DynShape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    pub fn dyn_slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, G>> {
        let mut inner = self.inner.clone();
        for (dim, spec) in specs.iter().enumerate() {
            match spec {
                IndexSpec::All => {}
                IndexSpec::Range(start, end) => {
                    inner = B::narrow(&inner, dim, *start, *end - *start)?;
                }
                IndexSpec::RangeFrom(start) => {
                    let current_dims = S::dims(&self._shape);
                    let len = current_dims.as_ref()[dim] - start;
                    inner = B::narrow(&inner, dim, *start, len)?;
                }
                IndexSpec::RangeTo(end) => {
                    inner = B::narrow(&inner, dim, 0, *end)?;
                }
                IndexSpec::Index(idx) => {
                    let narrowed = B::narrow(&inner, dim, *idx, 1)?;
                    inner = B::squeeze(&narrowed, dim)?;
                }
            }
        }

        Ok(Tensor::<Dyn, B, G>::from_parts(
            inner,
            alloc::vec![],
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

// -------------------------------------------------------------
// Structural Ops (Reshape, Broadcast, Transpose, Flatten)
// -------------------------------------------------------------

impl<S: Shape + DynShape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    /// Reshape this tensor into explicitly provided shape `S2`.
    /// The number of elements must be strictly equal.
    pub fn reshape<S2: Shape + DynShape>(&self, args: S2::Arg) -> Result<Tensor<S2, B, G>> {
        let new_shape_field = S2::init(args);
        let new_dims = S2::dims(&new_shape_field);

        // Runtime boundaries checking
        assert_eq!(
            S::numel(&self._shape),
            S2::numel(&new_shape_field),
            "Reshape failed: source numel != target numel"
        );

        let inner = B::reshape(&self.inner, new_dims.as_ref())?;
        Ok(Tensor::<S2, B, G>::from_parts(
            inner,
            new_shape_field,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Broadcast the tensor to the specific shape `S2`.
    pub fn broadcast_to<S2: Shape + DynShape>(&self, args: S2::Arg) -> Result<Tensor<S2, B, G>> {
        let new_shape_field = S2::init(args);
        let new_dims = S2::dims(&new_shape_field);
        let inner = B::broadcast_as(&self.inner, new_dims.as_ref())?;
        Ok(Tensor::<S2, B, G>::from_parts(
            inner,
            new_shape_field,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn to_dtype<T2: crate::prelude::ConstDType>(
        &self,
    ) -> Result<Tensor<S, B::BackendWithDType<T2>, G>> {
        let inner = B::to_dtype(&self.inner, T2::DTYPE)?;
        Ok(Tensor::from_parts(
            inner,
            self._shape.clone(),
            T2::init(()), // Initialize DType field
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Permute the tensor's dimensions by swapping `D1` and `D2`.
    /// Strongly typed output shape via `Transpose<D1, D2>`.
    pub fn transpose<const D1: usize, const D2: usize>(&self) -> Result<Tensor<S::Output, B, G>>
    where
        S: crate::shapes::Transpose<D1, D2>,
    {
        let inner = B::transpose(&self.inner, D1, D2)?;
        let mut out_dims = S::dims(&self._shape).into();
        out_dims.swap(D1, D2);

        Ok(Tensor::from_parts(
            inner,
            S::Output::from_dyn(&out_dims).unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Flattens dimensions from `START` to `END` inclusive.
    /// Uses `ProdDim` algebraically to track shapes.
    pub fn flatten<const START: usize, const END: usize>(&self) -> Result<Tensor<S::Output, B, G>>
    where
        S: crate::shapes::Flatten<START, END>,
    {
        let inner = B::flatten(&self.inner, START, END)?;
        let in_dims = S::dims(&self._shape).into();
        let mut out_dims = Vec::new();

        for i in 0..START {
            out_dims.push(in_dims[i]);
        }

        let mut prod = 1;
        for i in START..=END {
            prod *= in_dims[i];
        }
        out_dims.push(prod);

        for i in (END + 1)..in_dims.len() {
            out_dims.push(in_dims[i]);
        }

        Ok(Tensor::from_parts(
            inner,
            S::Output::from_dyn(&out_dims).unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    #[inline]
    pub fn layer_norm(
        &self,
        weight: &Tensor<Dyn, B, G>,
        bias: &Tensor<Dyn, B, G>,
        eps: f32,
    ) -> Result<Tensor<S, B, G>> {
        // weight and bias should technically be 1D tensors matching the last dimension, but we use DynShape for now
        let inner = B::layer_norm(&self.inner, &weight.inner, &bias.inner, eps)?;
        Ok(Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    #[inline]
    pub fn batch_norm(
        &self,
        weight: &Tensor<Dyn, B, G>,
        bias: &Tensor<Dyn, B, G>,
        running_mean: &Tensor<Dyn, B, G>,
        running_var: &Tensor<Dyn, B, G>,
        eps: f32,
    ) -> Result<Tensor<S, B, G>> {
        let inner = B::batch_norm(
            &self.inner,
            &weight.inner,
            &bias.inner,
            &running_mean.inner,
            &running_var.inner,
            eps,
        )?;
        Ok(Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{KindleDType, KindleDevice};
    use alloc::vec;

    #[derive(Clone)]
    pub struct DummyOpsBackend<T: DType, D: Device>(core::marker::PhantomData<(T, D)>);
    impl<T: DType, D: Device> Backend for DummyOpsBackend<T, D> {
        fn shape(_t: &Self::RawTensor) -> alloc::vec::Vec<usize> {
            unimplemented!()
        }

        fn conv1d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _stride: usize,
            _padding: usize,
            _dilation: usize,
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn conv_transpose2d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _stride: usize,
            _padding: usize,
            _output_padding: usize,
            _dilation: usize,
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn max_pool2d(
            _t: &Self::RawTensor,
            _kernel_size: (usize, usize),
            _stride: (usize, usize),
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn avg_pool2d(
            _t: &Self::RawTensor,
            _kernel_size: (usize, usize),
            _stride: (usize, usize),
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn embedding(_t: &Self::RawTensor, _w: &Self::RawTensor) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        type Device = D;
        type DType = T;
        type BackendWithDType<NewT: DType> = DummyOpsBackend<NewT, D>; // Mock, won't actually change types
        type BackendWithDevice<NewD: Device> = DummyOpsBackend<T, NewD>;

        type RawTensor = ();
        type RawVar = ();
        type Grads = ();

        fn var_as_tensor(_var: &Self::RawVar) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn var_from_tensor(_t: &Self::RawTensor) -> Result<Self::RawVar> {
            Ok(())
        }
        fn var_to_device(_var: &Self::RawVar, _dev: &KindleDevice) -> Result<Self::RawVar> {
            Ok(())
        }
        fn var_zeros(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(())
        }
        fn var_ones(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(())
        }
        fn var_rand(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(())
        }
        fn var_randn(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawVar> {
            Ok(())
        }

        fn zeros(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn ones(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn rand(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn randn(
            _shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn abs(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn relu(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn swish(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn softmax(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn mul_scalar(_t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn add_scalar(_t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn max_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn min_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn sum_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn mean_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn max_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn min_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn sum_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn mean_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn max_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn min_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn stack(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn concat(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn layer_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn batch_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _rm: &Self::RawTensor,
            _rv: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn tensor_to_device(_t: &Self::RawTensor, _dev: &KindleDevice) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn to_dtype(_t: &Self::RawTensor, _dt: KindleDType) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn broadcast_as(_t: &Self::RawTensor, _s: &[usize]) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn broadcast_left(_t: &Self::RawTensor, _s: &[usize]) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn add(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn sub(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn mul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn div(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn transpose(_t: &Self::RawTensor, _d1: usize, _d2: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn flatten(_t: &Self::RawTensor, _s: usize, _e: usize) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn narrow(
            _t: &Self::RawTensor,
            _dim: usize,
            _s: usize,
            _l: usize,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
        fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(())
        }

        fn backward(_loss: &Self::RawTensor) -> Result<Self::Grads> {
            Ok(())
        }
        fn step_sgd(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> {
            Ok(())
        }
        fn step_adamw(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> {
            Ok(())
        }
        fn conv2d(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: Option<&Self::RawTensor>,
            _s: usize,
            _p: usize,
            _d: usize,
        ) -> Result<Self::RawTensor> {
            Ok(())
        }
    }

    #[test]
    fn test_tensor_ops() {
        let t1: Tensor<Dyn, DummyOpsBackend<f32, crate::prelude::Cpu>> =
            Tensor::zeros(vec![2, 2]).unwrap();
        let t2: Tensor<Dyn, DummyOpsBackend<f32, crate::prelude::Cpu>> =
            Tensor::ones(vec![2, 2]).unwrap();

        // Binary ops
        let _res_add = t1.add(&t2).unwrap();
        let _res_sub = t1.sub(&t2).unwrap();
        let _res_mul = t1.mul(&t2).unwrap();
        let _res_div = t1.div(&t2).unwrap();

        // Unary ops
        let _res_abs = t1.abs().unwrap();
        let _res_relu = t1.relu().unwrap();
        let _res_exp = t1.exp().unwrap();

        // Scalar ops
        let _res_muls = t1.mul_scalar(2.0).unwrap();

        // Slicing
        let _res_slice = t1
            .dyn_slice(&[IndexSpec::All, IndexSpec::Index(0)])
            .unwrap();
    }
}
