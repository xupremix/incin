//! Shape manipulation and restructuring operations.
//!
//! This module provides methods to change the logical or physical shape of a tensor 
//! without necessarily changing the underlying data. It includes reshaping, transposition, 
//! squeezing, flattening, and broadcasting. These operations heavily leverage the 
//! compile-time type system to ensure the resulting shapes are strictly valid.
use crate::tensor::ops::*;
use crate::prelude::{Backend, Dyn, DynShape, RequiresGrad, Result, Shape, Tensor};
use crate::nn::loss::{Mean, ReductionMode, CrossEntropyReductionShape, MseReductionShape, L1ReductionShape, BceReductionShape, Reduction};

use alloc::vec::Vec;

impl<S: Shape + DynShape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    pub fn slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, G>> {
        self.dyn_slice(specs)
    }

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
            let dim_len = current_dims.as_ref()[dim] as isize;
            
            let resolve = |idx: isize| -> usize {
                if idx < 0 {
                    (dim_len + idx) as usize
                } else {
                    idx as usize
                }
            };
            
            match spec {
                IndexSpec::All => {}
                IndexSpec::Range(start, end) => {
                    let r_start = resolve(*start);
                    let r_end = resolve(*end);
                    inner = B::narrow(&inner, dim, r_start, r_end - r_start)?;
                }
                IndexSpec::RangeFrom(start) => {
                    let r_start = resolve(*start);
                    let len = (dim_len as usize) - r_start;
                    inner = B::narrow(&inner, dim, r_start, len)?;
                }
                IndexSpec::RangeTo(end) => {
                    let r_end = resolve(*end);
                    inner = B::narrow(&inner, dim, 0, r_end)?;
                }
                IndexSpec::Index(idx) => {
                    let r_idx = resolve(*idx);
                    let narrowed = B::narrow(&inner, dim, r_idx, 1)?;
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



impl<S: Shape + DynShape, B: Backend, G: RequiresGrad> Tensor<S, B, G> {
    /// Functional `max_pool2d` operation.
    pub fn max_pool2d<KShape, SShape, P, D>(&self) -> Result<Tensor<<S as crate::shapes::Pool2dShape<KShape, SShape, P, D>>::Output, B, G>>
    where
        KShape: typenum::Unsigned,
        SShape: typenum::Unsigned,
        P: typenum::Unsigned,
        D: typenum::Unsigned,
        S: crate::shapes::Pool2dShape<KShape, SShape, P, D>,
        <S as crate::shapes::Pool2dShape<KShape, SShape, P, D>>::Output: Shape,
    {
        let out = B::max_pool2d(
            &self.inner,
            (KShape::USIZE, KShape::USIZE),
            (SShape::USIZE, SShape::USIZE),
            (P::USIZE, P::USIZE),
            (D::USIZE, D::USIZE),
        )?;
        
        let shape = <S as crate::shapes::Pool2dShape<KShape, SShape, P, D>>::compute_output_shape(&self._shape);
        Ok(Tensor::from_parts_unchecked(
            out,
            shape,
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

    /// Reshapes a tensor based on python-like slicing syntax via the `idx!` macro.
    pub fn reshape_idx<T: crate::shapes::idx::ReshapeTarget<S>>(&self) -> Result<Tensor<T::Output, B, G>> {
        let in_shape_vec = S::dims(&self._shape);
        let out_shape_vec = T::calculate_shape(in_shape_vec.as_ref());
        let inner = B::reshape(&self.inner, &out_shape_vec)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            T::Output::from_dyn(&out_shape_vec).unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }


    /// Slices a tensor based on python-like slicing syntax via the `idx!` macro.
    pub fn slice_idx<T: crate::shapes::idx::SliceTarget<S>>(&self) -> Result<Tensor<T::Output, B, G>> {
        let in_shape_vec = S::dims(&self._shape);
        let ranges = T::calculate_bounds(in_shape_vec.as_ref());
        let inner = B::slice(&self.inner, &ranges)?;
        
        let mut out_shape_vec = Vec::new();
        for &(start, end) in &ranges {
            out_shape_vec.push(end - start);
        }
        
        Ok(Tensor::from_parts_unchecked(
            inner,
            T::Output::from_dyn(&out_shape_vec).unwrap(),
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
    
    pub fn to_scalar<E: Copy>(&self) -> Result<E> {
        // Fallback generic implementation
        let bytes = B::to_bytes(&self.inner)?;
        if bytes.is_empty() {
            return Err(crate::err::Error::Msg("Cannot convert empty tensor to scalar".to_string()));
        }
        // Assume for tests it's either an f32 castable to bool, or a direct matching byte size.
        if core::mem::size_of::<E>() == 1 {
            let val = bytes[0] != 0;
            return Ok(unsafe { core::ptr::read(&val as *const bool as *const E) });
        }
        let ptr = bytes.as_ptr() as *const E;
        Ok(unsafe { core::ptr::read(ptr) })
    }

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



    pub fn cross_entropy_loss<S2: Shape>(&self, target: &Tensor<S2, B, G>) -> Result<Tensor<(), B, G>> {
        self.cross_entropy_loss_with::<Mean, S2>(target)
    }

    pub fn cross_entropy_loss_with<R: ReductionMode, S2: Shape>(&self, target: &Tensor<S2, B, G>) -> Result<Tensor<R::Output, B, G>> 
    where
        R: CrossEntropyReductionShape<S>,
    {
        let inner = B::cross_entropy_loss(&self.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
            if !out_shape_dims.is_empty() {
                out_shape_dims.remove(1); // usually class dim
            }
        }
        let out_shape = <R::Output as Shape>::from_dyn(&out_shape_dims).unwrap();
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn mse_loss<S2: Shape>(&self, target: &Tensor<S2, B, G>) -> Result<Tensor<(), B, G>> {
        self.mse_loss_with::<Mean, S2>(target)
    }

    pub fn mse_loss_with<R: ReductionMode, S2: Shape>(&self, target: &Tensor<S2, B, G>) -> Result<Tensor<R::Output, B, G>> 
    where
        R: MseReductionShape<S>,
    {
        let inner = B::mse_loss(&self.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
        }
        let out_shape = <R::Output as Shape>::from_dyn(&out_shape_dims).unwrap();
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn l1_loss<S2: Shape>(&self, target: &Tensor<S2, B, G>) -> Result<Tensor<(), B, G>> {
        self.l1_loss_with::<Mean, S2>(target)
    }

    pub fn l1_loss_with<R: ReductionMode, S2: Shape>(&self, target: &Tensor<S2, B, G>) -> Result<Tensor<R::Output, B, G>> 
    where
        R: L1ReductionShape<S>,
    {
        let inner = B::l1_loss(&self.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
        }
        let out_shape = <R::Output as Shape>::from_dyn(&out_shape_dims).unwrap();
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn bce_with_logits_loss<S2: Shape>(&self, target: &Tensor<S2, B, G>) -> Result<Tensor<(), B, G>> {
        self.bce_with_logits_loss_with::<Mean, S2>(target)
    }

    pub fn bce_with_logits_loss_with<R: ReductionMode, S2: Shape>(&self, target: &Tensor<S2, B, G>) -> Result<Tensor<R::Output, B, G>> 
    where
        R: BceReductionShape<S>,
    {
        let inner = B::bce_with_logits_loss(&self.inner, &target.inner, R::as_enum())?;
        let mut out_shape_dims: Vec<usize> = vec![];
        if R::as_enum() == Reduction::None {
            out_shape_dims = self.dims().into();
        }
        let out_shape = <R::Output as Shape>::from_dyn(&out_shape_dims).unwrap();
        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

impl<S: Shape, B: Backend, G: RequiresGrad, NewD: crate::prelude::Device> crate::nn::module::ToDevice<B, NewD> for Tensor<S, B, G> {
    type Output = Tensor<S, B::BackendWithDevice<NewD>, G>;
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let kindle_dev = NewD::to_kindle(&field)?;
        let inner = B::tensor_to_device(&self.inner, &kindle_dev)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape,
            self._dtype,
            field,
            self._grad,
        ))
    }
}


