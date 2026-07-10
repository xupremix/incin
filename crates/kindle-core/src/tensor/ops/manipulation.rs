//! Shape manipulation and restructuring operations.
//!
//! This module provides methods to change the logical or physical shape of a tensor
//! without necessarily changing the underlying data. It includes reshaping, transposition,
//! squeezing, flattening, and broadcasting. These operations heavily leverage the
//! compile-time type system to ensure the resulting shapes are strictly valid.
use crate::nn::loss::{
    BceReductionShape, CrossEntropyReductionShape, L1ReductionShape, Mean, MseReductionShape,
    Reduction, ReductionMode,
};
use crate::prelude::{Backend, Dyn, DynShape, RequiresGrad, Result, Shape, Tensor};
use crate::tensor::ops::*;

use alloc::vec::Vec;

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, D: crate::tensor::device::Device, G: RequiresGrad> Tensor<S, B, K, D, G>
{
    /// Slices a tensor dynamically based on a slice of `IndexSpec` configurations.
    /// Returns a dynamically shaped tensor (`Dyn`).
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![3, 3], DefaultBackend>::ones(()).unwrap();
    /// let s = t.slice(&[IndexSpec::All, IndexSpec::Index(0)]).unwrap();
    /// ```
    pub fn slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, K, D, G>> {
        self.dyn_slice(specs)
    }

    /// Ergonomic slicing and indexing API using `IndexArgs`.
    /// 
    /// # Examples
    /// ```rust,ignore
    /// let sliced = tensor.get((0, 1..3, ..))?;
    /// ```
    pub fn get<I: crate::tensor::ops::index::IndexArgs>(&self, index: I) -> Result<Tensor<Dyn, B, K, D, G>> {
        self.dyn_slice(&index.into_specs())
    }

    /// Internal alias for `slice`.
    pub fn dyn_slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, K, D, G>> {
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

        Ok(Tensor::from_parts_unchecked(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }
}

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, D: crate::tensor::device::Device, G: RequiresGrad> Tensor<S, B, K, D, G>
{
    /// Functional `max_pool2d` operation.
    pub fn max_pool2d<KShape, SShape, P, Dilation>(
        &self,
    ) -> Result<Tensor<<S as crate::shapes::Pool2dShape<KShape, SShape, P, Dilation>>::Output, B, K, D, G>>
    where
        KShape: typenum::Unsigned,
        SShape: typenum::Unsigned,
        P: typenum::Unsigned,
        Dilation: typenum::Unsigned,
        S: crate::shapes::Pool2dShape<KShape, SShape, P, Dilation>,
        <S as crate::shapes::Pool2dShape<KShape, SShape, P, Dilation>>::Output: Shape,
    {
        let out = B::max_pool2d::<K>(
            &self.inner,
            (KShape::USIZE, KShape::USIZE),
            (SShape::USIZE, SShape::USIZE),
            (P::USIZE, P::USIZE),
            (Dilation::USIZE, Dilation::USIZE),
        )?;

        let shape = <S as crate::shapes::Pool2dShape<KShape, SShape, P, Dilation>>::compute_output_shape(
            &self._shape,
        );
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

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, D: crate::tensor::device::Device, G: RequiresGrad> Tensor<S, B, K, D, G>
{
    /// Reshape this tensor into explicitly provided shape `S2`.
    /// This is guaranteed at compile-time to have matching elements.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
    /// let r = t.reshape::<s![6]>(()).unwrap();
    /// ```
    pub fn reshape<S2>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, D, G>>
    where
        S2: Shape + DynShape,
        S: crate::shapes::reshape::ReshapeShape<S2>,
    {
        let new_shape_field = S2::init(args);
        let new_dims = S2::dims(&new_shape_field);

        let inner = B::reshape(&self.inner, new_dims.as_ref())?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            new_shape_field,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Reshapes a tensor based on python-like slicing syntax via the `idx!` macro.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
    /// let r = t.reshape_idx::<idx![6]>().unwrap();
    /// ```
    pub fn reshape_idx<T: crate::shapes::idx::ReshapeTarget<S>>(
        &self,
    ) -> Result<Tensor<T::Output, B, K, D, G>> {
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
    pub fn slice_idx<T: crate::shapes::idx::SliceTarget<S>>(
        &self,
    ) -> Result<Tensor<T::Output, B, K, D, G>> {
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

    /// Narrows the tensor dynamically, returning a tensor with `Dyn` shape.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![10], DefaultBackend>::ones(()).unwrap();
    /// let n = t.try_narrow(0, 2, 5).unwrap(); // shape [5]
    /// ```
    pub fn try_narrow(self, dim: usize, start: usize, len: usize) -> Result<Tensor<Dyn, B, K, D, G>> {
        let inner = B::narrow(&self.inner, dim, start, len)?;
        let mut shape = S::dims(&self._shape).as_ref().to_vec();
        shape[dim] = len;
        Ok(Tensor {
            inner,
            _shape: shape,
            _dtype: self._dtype,
            _device: self._device,
            _grad: self._grad,
        })
    }

    /// Squeezes the tensor dynamically by removing the dimension `dim` if its size is 1.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![1, 5], DefaultBackend>::ones(()).unwrap();
    /// let sq = t.try_squeeze(0).unwrap(); // shape [5]
    /// ```
    pub fn try_squeeze(self, dim: usize) -> Result<Tensor<Dyn, B, K, D, G>> {
        let inner = B::squeeze(&self.inner, dim)?;
        let mut shape = S::dims(&self._shape).as_ref().to_vec();
        shape.remove(dim);
        Ok(Tensor {
            inner,
            _shape: shape,
            _dtype: self._dtype,
            _device: self._device,
            _grad: self._grad,
        })
    }
    pub fn try_reshape<S2>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, D, G>>
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
                msg: alloc::format!(
                    "Reshape failed: source numel ({}) != target numel ({})",
                    source_numel,
                    target_numel
                ),
            });
        }

        let inner = B::reshape(&self.inner, new_dims.as_ref())?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            new_shape_field,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Broadcast the tensor to the specific shape `S2`.
    pub fn broadcast_to<S2: Shape + DynShape>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, D, G>> {
        let new_shape_field = S2::init(args);
        let new_dims = S2::dims(&new_shape_field);
        let inner = B::broadcast_as(&self.inner, new_dims.as_ref())?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            new_shape_field,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    pub fn to_dtype<T2: crate::tensor::dtype::DType<Arg = ()>>(
        &self,
    ) -> Result<Tensor<S, B, T2, D, G>> {
        let field = T2::init(());
        let kindle_dtype = T2::to_kindle(&field);
        let inner = B::tensor_to_dtype::<K, T2>(&self.inner, kindle_dtype)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            field,
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Permute the tensor's dimensions by swapping `D1` and `D2`.
    /// Strongly typed output shape via `Transpose<D1, D2>`.

    /// Extracts a single scalar value from a 0D or 1D tensor.
    /// This will bring the tensor data to the CPU and read the bytes.
    pub fn to_scalar<E: Copy>(&self) -> Result<E> {
        let bytes = B::to_bytes(&self.inner)?;
        if bytes.len() != core::mem::size_of::<E>() {
            // Attempt to dynamically cast f32 -> bool if requested, to keep old fallback working
            if core::mem::size_of::<E>() == 1 && bytes.len() > 0 {
                let val = bytes[0] != 0;
                return Ok(unsafe { core::ptr::read_unaligned(&val as *const bool as *const E) });
            }
            return Err(crate::err::Error::Msg(alloc::format!(
                "Size mismatch when converting to scalar. Tensor dtype bytes: {}, expected: {}",
                bytes.len(),
                core::mem::size_of::<E>()
            )));
        }
        let val = unsafe { core::ptr::read_unaligned(bytes.as_ptr() as *const E) };
        Ok(val)
    }

    /// Extracts a 1D vector of scalars from this tensor.
    pub fn to_vec1<E: Copy>(&self) -> Result<alloc::vec::Vec<E>> {
        let bytes = B::to_bytes(&self.inner)?;
        let num_elements = S::numel(&self._shape);
        let expected_bytes = num_elements * core::mem::size_of::<E>();
        if bytes.len() != expected_bytes {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Size mismatch when converting to vec. Tensor dtype bytes: {}, expected: {}",
                bytes.len(),
                expected_bytes
            )));
        }
        let mut out = alloc::vec::Vec::with_capacity(num_elements);
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr() as *const E,
                out.as_mut_ptr(),
                num_elements,
            );
            out.set_len(num_elements);
        }
        Ok(out)
    }

    /// Permutes the tensor's dimensions by swapping `D1` and `D2`.
    /// Strongly typed output shape via `Transpose<D1, D2>`.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
    /// let tr = t.transpose::<0, 1>().unwrap(); // shape [3, 2]
    /// ```
    pub fn transpose<const D1: usize, const D2: usize>(&self) -> Result<Tensor<S::Output, B, K, D, G>>
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
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![2, 3, 4], DefaultBackend>::ones(()).unwrap();
    /// let f = t.flatten::<1, 2>().unwrap(); // shape [2, 12]
    /// ```
    pub fn flatten<const START: usize, const END: usize>(&self) -> Result<Tensor<S::Output, B, K, D, G>>
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
        weight: &Tensor<Dyn, B, K, D, G>,
        bias: &Tensor<Dyn, B, K, D, G>,
        eps: f32,
    ) -> Result<Tensor<S, B, K, D, G>> {
        // weight and bias should technically be 1D tensors matching the last dimension, but we use DynShape for now
        let inner = B::layer_norm::<K>(&self.inner, &weight.inner, Some(&bias.inner), eps)?;
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
        weight: &Tensor<Dyn, B, K, D, G>,
        bias: &Tensor<Dyn, B, K, D, G>,
        running_mean: &Tensor<Dyn, B, K, D, G>,
        running_var: &Tensor<Dyn, B, K, D, G>,
        eps: f32,
    ) -> Result<Tensor<S, B, K, D, G>> {
        let inner = B::batch_norm::<K>(
            &self.inner,
            Some(&weight.inner),
            Some(&bias.inner),
            Some(&running_mean.inner),
            Some(&running_var.inner),
            eps,
            0.1,
        )?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Computes the Cross Entropy loss between predictions and target labels.
    /// Uses the default `Mean` reduction.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let pred = Tensor::<s![2, 10], DefaultBackend>::zeros(()).unwrap();
    /// let target = Tensor::<s![2], DefaultBackend>::zeros(()).unwrap();
    /// let loss = pred.cross_entropy_loss(&target).unwrap();
    /// ```
    pub fn cross_entropy_loss<S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, D, G>,
    ) -> Result<Tensor<(), B, K, D, G>> {
        self.cross_entropy_loss_with::<Mean, S2>(target)
    }

    pub fn cross_entropy_loss_with<R: ReductionMode, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, D, G>,
    ) -> Result<Tensor<R::Output, B, K, D, G>>
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

    /// Computes the Mean Squared Error (MSE) loss.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let pred = Tensor::<s![2], DefaultBackend>::ones(()).unwrap();
    /// let target = Tensor::<s![2], DefaultBackend>::zeros(()).unwrap();
    /// let loss = pred.mse_loss(&target).unwrap();
    /// ```
    pub fn mse_loss<S2: Shape>(&self, target: &Tensor<S2, B, K, D, G>) -> Result<Tensor<(), B, K, D, G>> {
        self.mse_loss_with::<Mean, S2>(target)
    }

    pub fn mse_loss_with<R: ReductionMode, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, D, G>,
    ) -> Result<Tensor<R::Output, B, K, D, G>>
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

    pub fn l1_loss<S2: Shape>(&self, target: &Tensor<S2, B, K, D, G>) -> Result<Tensor<(), B, K, D, G>> {
        self.l1_loss_with::<Mean, S2>(target)
    }

    pub fn l1_loss_with<R: ReductionMode, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, D, G>,
    ) -> Result<Tensor<R::Output, B, K, D, G>>
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

    pub fn bce_with_logits_loss<S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, D, G>,
    ) -> Result<Tensor<(), B, K, D, G>> {
        self.bce_with_logits_loss_with::<Mean, S2>(target)
    }

    pub fn bce_with_logits_loss_with<R: ReductionMode, S2: Shape>(
        &self,
        target: &Tensor<S2, B, K, D, G>,
    ) -> Result<Tensor<R::Output, B, K, D, G>>
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

impl<S: Shape, B: Backend, K: crate::tensor::dtype::DType, D: crate::tensor::device::Device, G: RequiresGrad, NewD: crate::prelude::Device>
    crate::nn::module::ToDevice<B, NewD> for Tensor<S, B, K, D, G>
where
    B: Backend<BackendWithDevice<NewD> = B>,
{
    type Output = Tensor<S, B, K, NewD, G>;
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let kindle_dev = NewD::to_kindle(&field)?;
        let inner = B::tensor_to_device::<K>(&self.inner, &kindle_dev)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape,
            self._dtype,
            field,
            self._grad,
        ))
    }
}
