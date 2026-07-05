//! Element-wise tensor operations with compile-time shape checking.
//!
//! Operations require matching Shape, DType, Device, and RequiresGrad.
//! This ensures at compile time that you can't accidentally add tensors
//! of different shapes, dtypes, or on different devices.

use crate::prelude::{Backend, Dyn, DynShape, RequiresGrad, Result, Shape, Tensor};

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
                Ok(Tensor::from_parts_unchecked(
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

macro_rules! impl_broadcast_binary_op {
    ($trait_name:ident, $method:ident, $backend_method:ident) => {
        impl<S1: Shape + crate::shapes::DynShape, B: Backend, G: RequiresGrad> Tensor<S1, B, G> {
            #[inline]
            pub fn $method<S2>(&self, rhs: &Tensor<S2, B, G>) -> Result<Tensor<<S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output, B, G>>
            where
                S2: Shape + crate::shapes::DynShape,
                S1: crate::shapes::broadcast::BroadcastShape<S2>,
                <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::Output: Shape,
            {
                let b_shape = <S1 as crate::shapes::broadcast::BroadcastShape<S2>>::output_shape(self.shape_field(), rhs.shape_field());

                let inner = B::$backend_method(&self.inner, &rhs.inner)?;
                Ok(Tensor::from_parts_unchecked(
                    inner,
                    b_shape,
                    self._dtype.clone(),
                    self._device.clone(),
                    self._grad.clone(),
                ))
            }
        }
    };
}

impl_broadcast_binary_op!(BroadcastAdd, broadcast_add, add);
impl_broadcast_binary_op!(BroadcastSub, broadcast_sub, sub);
impl_broadcast_binary_op!(BroadcastMul, broadcast_mul, mul);
impl_broadcast_binary_op!(BroadcastDiv, broadcast_div, div);


macro_rules! impl_unary_op {
    ($method:ident, $backend_method:ident) => {
        pub fn $method(&self) -> Result<Self> {
            let inner = B::$backend_method(&self.inner)?;
            Ok(Tensor::from_parts_unchecked(
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
        Ok(Tensor::from_parts_unchecked(
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
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn add_scalar(&self, scalar: f64) -> Result<Self> {
        let inner = B::add_scalar(&self.inner, scalar)?;
        Ok(Tensor::from_parts_unchecked(
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
            Ok(Tensor::from_parts_unchecked(
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
    ($method:ident, $backend_method:ident, $trait_bound:ident, $keep_dim:expr) => {
        pub fn $method<const DIM: usize>(&self) -> Result<Tensor<S::Output, B, G>>
        where
            S: DynShape + crate::shapes::$trait_bound<DIM>,
        {
            let inner = B::$backend_method(&self.inner, DIM)?;

            // We just use from_dyn to construct the resulting shape field dynamically,
            // since we know it's a dimensional reduction.
            let mut out_dims = S::dims(&self._shape).into();
            if $keep_dim {
                out_dims[DIM] = 1;
            } else {
                out_dims.remove(DIM);
            }

            Ok(Tensor::from_parts_unchecked(
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

    impl_reduction_dim_op!(sum_dim, sum_dim, ReduceDim, false);
    impl_reduction_dim_op!(sum_keepdim, sum_keepdim, ReduceKeepDim, true);
    impl_reduction_dim_op!(mean_dim, mean_dim, ReduceDim, false);
    impl_reduction_dim_op!(mean_keepdim, mean_keepdim, ReduceKeepDim, true);
    impl_reduction_dim_op!(max_dim, max_dim, ReduceDim, false);
    impl_reduction_dim_op!(max_keepdim, max_keepdim, ReduceKeepDim, true);
    impl_reduction_dim_op!(min_dim, min_dim, ReduceDim, false);
    impl_reduction_dim_op!(min_keepdim, min_keepdim, ReduceKeepDim, true);
}

impl<S: Shape + DynShape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    pub fn dyn_slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, G>> {
        let current_dims = S::dims(&self._shape);
        if specs.len() > current_dims.as_ref().len() {
             return Err(crate::err::Error::Msg(alloc::format!(
                 "Too many slicing specs ({}) for tensor of rank {}",
                 specs.len(),
                 current_dims.as_ref().len()
             )));
        }

        let mut inner = self.inner.clone();
        for (dim, spec) in specs.iter().enumerate() {
            match spec {
                IndexSpec::All => {}
                IndexSpec::Range(start, end) => {
                    inner = B::narrow(&inner, dim, *start, *end - *start)?;
                }
                IndexSpec::RangeFrom(start) => {
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

        let out_shape = B::shape(&inner);

        Ok(Tensor::<Dyn, B, G>::from_parts_unchecked(
            inner,
            out_shape,
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
    /// This is guaranteed at compile-time to have matching elements.
    pub fn reshape<S2>(&self, args: S2::Arg) -> Result<Tensor<S2, B, G>>
    where
        S2: Shape + DynShape,
        S: crate::shapes::reshape::ReshapeShape<S2>,
    {
        let new_shape_field = S2::init(args);
        let new_dims = S2::dims(&new_shape_field);

        let inner = B::reshape(&self.inner, new_dims.as_ref())?;
        Ok(Tensor::<S2, B, G>::from_parts_unchecked(
            inner,
            new_shape_field,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Try to reshape this tensor into the provided shape `S2`.
    /// This falls back to a runtime verification for dynamic shapes.
    
    pub fn try_narrow(self, dim: usize, start: usize, len: usize) -> Result<Tensor<Dyn, B, G>> {
        let inner = B::narrow(&self.inner, dim, start, len)?;
        let mut shape = S::dims(&self._shape).as_ref().to_vec();
        shape[dim] = len;
        Ok(Tensor {
            inner,
            _shape: shape,
            _dtype: self._dtype,
            _device: self._device,
            _grad: self._grad.clone(),
        })
    }

    pub fn try_squeeze(self, dim: usize) -> Result<Tensor<Dyn, B, G>> {
        let inner = B::squeeze(&self.inner, dim)?;
        let mut shape = S::dims(&self._shape).as_ref().to_vec();
        shape.remove(dim);
        Ok(Tensor {
            inner,
            _shape: shape,
            _dtype: self._dtype,
            _device: self._device,
            _grad: self._grad.clone(),
        })
    }
pub fn try_reshape<S2>(&self, args: S2::Arg) -> Result<Tensor<S2, B, G>>
    where
        S2: Shape + DynShape,
        S: crate::shapes::reshape::TryReshape<S2>,
    {
        let new_shape_field = S2::init(args);
        let new_dims = S2::dims(&new_shape_field);

        // Runtime boundaries checking
        let source_numel = S::numel(&self._shape);
        let target_numel = S2::numel(&new_shape_field);
        if source_numel != target_numel {
            return Err(crate::err::Error::ShapeMismatch {
                op: "try_reshape",
                expected: alloc::vec![source_numel], // We use numels here
                got: alloc::vec![target_numel],
                msg: alloc::format!("Reshape failed: source numel ({}) != target numel ({})", source_numel, target_numel),
            });
        }

        let inner = B::reshape(&self.inner, new_dims.as_ref())?;
        Ok(Tensor::<S2, B, G>::from_parts_unchecked(
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
        Ok(Tensor::<S2, B, G>::from_parts_unchecked(
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
        Ok(Tensor::from_parts_unchecked(
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

        Ok(Tensor::from_parts_unchecked(
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

        Ok(Tensor::from_parts_unchecked(
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
        Ok(Tensor::from_parts_unchecked(
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
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}


impl<S: Shape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    /// Dynamically concatenates a slice of tensors along `dim`.
    /// This is fallible at runtime if shapes mismatch or dim is out of bounds.
    pub fn try_concat_slice(tensors: &[&Tensor<S, B, G>], dim: usize) -> Result<Tensor<Dyn, B, G>> {
        let raw_tensors: alloc::vec::Vec<&B::RawTensor> = tensors.iter().map(|t| &t.inner).collect();
        if raw_tensors.is_empty() {
            return Err(crate::err::Error::Msg("Cannot concat empty list".to_string()));
        }
        let inner = B::concat(&raw_tensors, dim)?;
        let mut out_shape = B::shape(&tensors[0].inner);
        out_shape[dim] = tensors.iter().map(|t| B::shape(&t.inner)[dim]).sum();
        Ok(Tensor::from_parts_unchecked(
            inner,
            <Dyn as Shape>::from_dyn(&out_shape).unwrap(),
            tensors[0]._dtype.clone(),
            tensors[0]._device.clone(),
            tensors[0]._grad.clone(),
        ))
    }

    /// Statically concatenates `self` with `other` along `Axis`.
    pub fn concat<S2, Axis>(&self, other: &Tensor<S2, B, G>) -> Result<Tensor<<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output, B, G>>
    where
        S2: Shape,
        Axis: typenum::Unsigned,
        S: crate::shapes::concat::ConcatShape<S2, Axis>,
        <<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output as Shape>::Field: core::default::Default,
    {
        let dim = Axis::USIZE;
        let inner = B::concat(&[&self.inner, &other.inner], dim)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            core::default::Default::default(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
    
    /// Dynamically concatenates `self` with `other` along `dim`.
    pub fn try_concat<S2>(&self, other: &Tensor<S2, B, G>, dim: usize) -> Result<Tensor<Dyn, B, G>>
    where
        S2: Shape,
    {
        let inner = B::concat(&[&self.inner, &other.inner], dim)?;
        let mut out_shape = B::shape(&self.inner);
        out_shape[dim] += B::shape(&other.inner)[dim];
        Ok(Tensor::from_parts_unchecked(
            inner,
            <Dyn as Shape>::from_dyn(&out_shape).unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Dynamically stacks a slice of tensors along `dim`.
    pub fn try_stack_slice(tensors: &[&Tensor<S, B, G>], dim: usize) -> Result<Tensor<Dyn, B, G>> {
        let raw_tensors: alloc::vec::Vec<&B::RawTensor> = tensors.iter().map(|t| &t.inner).collect();
        if raw_tensors.is_empty() {
            return Err(crate::err::Error::Msg("Cannot stack empty list".to_string()));
        }
        let inner = B::stack(&raw_tensors, dim)?;
        let mut out_shape = B::shape(&tensors[0].inner);
        out_shape.insert(dim, tensors.len());
        Ok(Tensor::from_parts_unchecked(
            inner,
            <Dyn as Shape>::from_dyn(&out_shape).unwrap(),
            tensors[0]._dtype.clone(),
            tensors[0]._device.clone(),
            tensors[0]._grad.clone(),
        ))
    }

    /// Statically stacks `self` with `other` along `Axis`.
    pub fn stack<Axis>(&self, other: &Tensor<S, B, G>) -> Result<Tensor<<S as crate::shapes::stack::StackShape<Axis>>::Output, B, G>>
    where
        Axis: typenum::Unsigned,
        S: crate::shapes::stack::StackShape<Axis>,
        <<S as crate::shapes::stack::StackShape<Axis>>::Output as Shape>::Field: core::default::Default,
    {
        let dim = Axis::USIZE;
        let inner = B::stack(&[&self.inner, &other.inner], dim)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            core::default::Default::default(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
    
    /// Dynamically stacks `self` with `other` along `dim`.
    pub fn try_stack(&self, other: &Tensor<S, B, G>, dim: usize) -> Result<Tensor<Dyn, B, G>>
    {
        let inner = B::stack(&[&self.inner, &other.inner], dim)?;
        let mut out_shape = B::shape(&self.inner);
        out_shape.insert(dim, 2);
        Ok(Tensor::from_parts_unchecked(
            inner,
            <Dyn as Shape>::from_dyn(&out_shape).unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::{DType, Device};
    use crate::prelude::{KindleDType, KindleDevice};
    use alloc::vec;

    #[derive(Clone)]
    pub struct DummyOpsBackend<T: DType, D: Device>(core::marker::PhantomData<(T, D)>);
    impl<T: DType, D: Device> Backend for DummyOpsBackend<T, D> {
        fn shape(t: &Self::RawTensor) -> alloc::vec::Vec<usize> {
            t.clone()
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

        type RawTensor = alloc::vec::Vec<usize>;
        type RawVar = ();
        type Grads = ();

        fn var_as_tensor(_var: &Self::RawVar) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
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
        
        fn to_bytes(_t: &Self::RawTensor) -> Result<alloc::vec::Vec<u8>> {
            Ok(alloc::vec::Vec::new())
        }
        
        fn from_bytes(_bytes: &[u8], shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }

        fn zeros(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }
        fn ones(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }
        fn rand(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }
        fn randn(
            shape: &[usize],
            _dtype: KindleDType,
            _device: &KindleDevice,
        ) -> Result<Self::RawTensor> {
            Ok(shape.to_vec())
        }

        fn abs(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn relu(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn swish(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn softmax(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn mul_scalar(t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> {
            Ok(t.clone())
        }
        fn add_scalar(t: &Self::RawTensor, _s: f64) -> Result<Self::RawTensor> {
            Ok(t.clone())
        }

        fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn max_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn min_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn sum_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn mean_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn max_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn min_dim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn sum_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn mean_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn max_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn min_keepdim(_t: &Self::RawTensor, _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn stack(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn concat(_t: &[&Self::RawTensor], _d: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn layer_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn batch_norm(
            _t: &Self::RawTensor,
            _w: &Self::RawTensor,
            _b: &Self::RawTensor,
            _rm: &Self::RawTensor,
            _rv: &Self::RawTensor,
            _e: f32,
        ) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn tensor_to_device(_t: &Self::RawTensor, _dev: &KindleDevice) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn to_dtype(_t: &Self::RawTensor, _dt: KindleDType) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn broadcast_as(_t: &Self::RawTensor, _s: &[usize]) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn broadcast_left(_t: &Self::RawTensor, _s: &[usize]) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }

        fn add(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn sub(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn mul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn div(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn transpose(_t: &Self::RawTensor, _d1: usize, _d2: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn flatten(_t: &Self::RawTensor, _s: usize, _e: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn narrow(
            _t: &Self::RawTensor,
            _dim: usize,
            _s: usize,
            _l: usize,
        ) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
        }
        fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> {
            Ok(alloc::vec::Vec::new())
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
            Ok(alloc::vec::Vec::new())
        }

        fn format_tensor(t: &Self::RawTensor) -> alloc::string::String {
            alloc::format!("{:?}", t)
        }

        fn adaptive_avg_pool2d(
            _t: &Self::RawTensor,
            _output_size: (usize, usize),
        ) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn step_adam(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> {
            unimplemented!()
        }

        fn mse_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn l1_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn bce_with_logits_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> {
            unimplemented!()
        }

        fn cross_entropy_loss(_pred: &Self::RawTensor, _target: &Self::RawTensor) -> Result<Self::RawTensor> {
            unimplemented!()
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

pub fn try_stack_tensors<S: Shape + DynShape, B: Backend, G: crate::tensor::grad::RequiresGrad>(tensors: &[&Tensor<S, B, G>], dim: usize) -> Result<Tensor<Dyn, B, G>> where G::Field: Clone {
    if tensors.is_empty() {
        return Err(crate::prelude::Error::ShapeMismatch {
            op: "stack_tensors",
            expected: alloc::vec![],
            got: alloc::vec![],
            msg: alloc::string::String::from("Cannot stack empty list of tensors"),
        });
    }
    let raw_tensors: alloc::vec::Vec<&B::RawTensor> = tensors.iter().map(|t| &t.inner).collect();
    let inner = B::stack(&raw_tensors, dim)?;
    let mut shape = S::dims(&tensors[0]._shape).as_ref().to_vec();
    shape.insert(dim, tensors.len());
    Ok(Tensor {
        inner,
        _shape: shape,
        _dtype: tensors[0]._dtype.clone(),
        _device: tensors[0]._device.clone(),
        _grad: tensors[0]._grad.clone(),
    })
}
