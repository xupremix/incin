//! Shape manipulation and restructuring operations.
//!
//! This module provides methods to change the logical or physical shape of a tensor
//! without necessarily changing the underlying data. It includes reshaping, transposition,
//! squeezing, flattening, and broadcasting. These operations heavily leverage the
//! compile-time type system to ensure the resulting shapes are strictly valid.
use crate::prelude::{
    Backend, Dyn, DynShape, RequiresGrad, Result, Shape, SupportsDType, Tensor, TransferTo,
};
use crate::tensor::ops::*;

use alloc::string::ToString;
use alloc::vec::Vec;

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G>
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
    pub fn slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, K, G>> {
        self.dyn_slice(specs)
    }

    /// Ergonomic slicing and indexing API using `IndexArgs`.
    ///
    /// # Examples
    /// ```rust,ignore
    /// let sliced = tensor.get((0, 1..3, ..))?;
    /// ```
    pub fn get<I: crate::tensor::ops::index::IndexArgs>(
        &self,
        index: I,
    ) -> Result<Tensor<Dyn, B, K, G>> {
        self.dyn_slice(&index.into_specs())
    }

    /// Internal alias for `slice`.
    pub fn dyn_slice(&self, specs: &[IndexSpec]) -> Result<Tensor<Dyn, B, K, G>> {
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

impl<
    S: Shape + DynShape,
    B: Backend + crate::tensor::backend::ModuleOps<B>,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
> Tensor<S, B, K, G>
{
    /// Functional `max_pool2d` operation.
    pub fn max_pool2d<KShape, SShape, P, Dilation>(
        &self,
    ) -> Result<
        Tensor<<S as crate::shapes::Pool2dShape<KShape, SShape, P, Dilation>>::Output, B, K, G>,
    >
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

        let shape =
            <S as crate::shapes::Pool2dShape<KShape, SShape, P, Dilation>>::compute_output_shape(
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

impl<S: Shape + DynShape, B: Backend, K: crate::tensor::dtype::DType, G: RequiresGrad>
    Tensor<S, B, K, G>
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
    pub fn reshape<S2>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G>>
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
    ) -> Result<Tensor<T::Output, B, K, G>> {
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
    ) -> Result<Tensor<T::Output, B, K, G>> {
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

    /// Narrows the tensor dynamically, returning a tensor with `Dyn` shape.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use kindle::prelude::*;
    /// let t = Tensor::<s![10], DefaultBackend>::ones(()).unwrap();
    /// let n = t.try_narrow(0, 2, 5).unwrap(); // shape [5]
    /// ```
    pub fn try_narrow(self, dim: usize, start: usize, len: usize) -> Result<Tensor<Dyn, B, K, G>> {
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
    pub fn try_squeeze(self, dim: usize) -> Result<Tensor<Dyn, B, K, G>> {
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
    /// `try_reshape`.
    pub fn try_reshape<S2>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G>>
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
    pub fn broadcast_to<S2: Shape + DynShape>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G>> {
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

    /// `to_dtype`.
    pub fn to_dtype<T2: crate::tensor::dtype::DType<Arg = ()>>(
        &self,
    ) -> Result<Tensor<S, B, T2, G>> {
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

    /// Extracts a single scalar value from a 0D or 1D tensor.
    /// This will bring the tensor data to the CPU and read the bytes.
    ///
    /// `bool` is handled as a truthy (any-nonzero-byte) conversion rather
    /// than a raw reinterpret, regardless of whether the tensor's actual
    /// dtype element size happens to match `size_of::<bool>()`: `bool` has
    /// only two valid bit patterns (`0x00`/`0x01`), and there is no
    /// `DTypeId::Bool` (ONNX-style boolean tensors are stored as another
    /// dtype, typically `U8`, and read out via this truthy conversion), so
    /// reinterpreting an arbitrary stored byte as `bool` via
    /// `read_unaligned` would be undefined behavior whenever that byte
    /// isn't `0` or `1`.
    pub fn to_scalar<E: Copy + 'static>(&self) -> Result<E> {
        let bytes = B::to_bytes(&self.inner)?;
        let dtype = K::to_kindle(&self._dtype);

        if core::any::TypeId::of::<E>() == core::any::TypeId::of::<bool>() {
            if bytes.is_empty() {
                return Err(crate::err::Error::Msg(
                    "cannot convert an empty tensor to a bool scalar".into(),
                ));
            }
            let val = bytes.iter().any(|&byte| byte != 0);
            // SAFETY: `E` is verified to be exactly `bool` above, so this
            // reinterprets a genuine, valid `bool` as itself.
            return Ok(unsafe { core::ptr::read_unaligned(&val as *const bool as *const E) });
        }

        let elem_size = core::mem::size_of::<E>();
        let expected_size = dtype.element_size();
        if bytes.len() != elem_size || elem_size != expected_size {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Size mismatch when converting to scalar. Tensor dtype {:?} ({} bytes) vs requested type ({} bytes)",
                dtype,
                bytes.len(),
                elem_size
            )));
        }
        let val = unsafe { core::ptr::read_unaligned(bytes.as_ptr() as *const E) };
        Ok(val)
    }

    /// Extracts a 1D vector of scalars from this tensor.
    ///
    /// See `to_scalar`'s doc comment for why `bool` is handled as a
    /// per-element truthy conversion rather than a raw reinterpret.
    pub fn to_vec1<E: Copy + 'static>(&self) -> Result<alloc::vec::Vec<E>> {
        let bytes = B::to_bytes(&self.inner)?;
        let num_elements = S::numel(&self._shape);
        let dtype = K::to_kindle(&self._dtype);

        if core::any::TypeId::of::<E>() == core::any::TypeId::of::<bool>() {
            let elem_size = dtype.element_size();
            let expected_bytes = num_elements * elem_size;
            if bytes.len() != expected_bytes {
                return Err(crate::err::Error::Msg(alloc::format!(
                    "Size mismatch when converting to vec. Tensor dtype bytes: {}, expected: {}",
                    bytes.len(),
                    expected_bytes
                )));
            }
            let mut out = alloc::vec::Vec::with_capacity(num_elements);
            for chunk in bytes.chunks_exact(elem_size) {
                let val = chunk.iter().any(|&byte| byte != 0);
                // SAFETY: `E` is verified to be exactly `bool` above.
                out.push(unsafe { core::ptr::read_unaligned(&val as *const bool as *const E) });
            }
            return Ok(out);
        }

        let elem_size = core::mem::size_of::<E>();
        let expected_elem_size = dtype.element_size();
        if elem_size != expected_elem_size {
            return Err(crate::err::Error::Msg(alloc::format!(
                "Element size mismatch converting to vec: Tensor dtype {:?} element size {} vs requested type size {}",
                dtype,
                expected_elem_size,
                elem_size
            )));
        }
        let expected_bytes = num_elements * elem_size;
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
    pub fn transpose<const D1: usize, const D2: usize>(&self) -> Result<Tensor<S::Output, B, K, G>>
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
    pub fn flatten<const START: usize, const END: usize>(
        &self,
    ) -> Result<Tensor<S::Output, B, K, G>>
    where
        S: crate::shapes::Flatten<START, END>,
    {
        let inner = B::flatten(&self.inner, START, END)?;
        let in_dims: Vec<usize> = S::dims(&self._shape).into();
        let mut out_dims = Vec::new();

        out_dims.extend_from_slice(&in_dims[0..START]);

        let prod: usize = in_dims[START..=END].iter().product();
        out_dims.push(prod);

        out_dims.extend_from_slice(&in_dims[(END + 1)..]);

        Ok(Tensor::from_parts_unchecked(
            inner,
            S::Output::from_dyn(&out_dims).unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        ))
    }

    /// Dynamically concatenates a slice of tensors along `dim`.
    /// This is fallible at runtime if shapes mismatch or dim is out of bounds.
    pub fn try_concat_slice(
        tensors: &[&Tensor<S, B, K, G>],
        dim: usize,
    ) -> Result<Tensor<Dyn, B, K, G>> {
        let raw_tensors: alloc::vec::Vec<&B::Storage<K>> =
            tensors.iter().map(|t| &t.inner).collect();
        if raw_tensors.is_empty() {
            return Err(crate::err::Error::Msg(
                "Cannot concat empty list".to_string(),
            ));
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
    pub fn concat<S2, Axis>(
        &self,
        other: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output, B, K, G>>
    where
        S2: Shape,
        Axis: typenum::Unsigned,
        S: crate::shapes::concat::ConcatShape<S2, Axis>,
        <<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output as Shape>::Field:
            core::default::Default,
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
    pub fn try_concat<S2>(
        &self,
        other: &Tensor<S2, B, K, G>,
        dim: usize,
    ) -> Result<Tensor<Dyn, B, K, G>>
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
    pub fn try_stack_slice(
        tensors: &[&Tensor<S, B, K, G>],
        dim: usize,
    ) -> Result<Tensor<Dyn, B, K, G>> {
        let raw_tensors: alloc::vec::Vec<&B::Storage<K>> =
            tensors.iter().map(|t| &t.inner).collect();
        if raw_tensors.is_empty() {
            return Err(crate::err::Error::Msg(
                "Cannot stack empty list".to_string(),
            ));
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
    pub fn stack<Axis>(
        &self,
        other: &Tensor<S, B, K, G>,
    ) -> Result<Tensor<<S as crate::shapes::stack::StackShape<Axis>>::Output, B, K, G>>
    where
        Axis: typenum::Unsigned,
        S: crate::shapes::stack::StackShape<Axis>,
        <<S as crate::shapes::stack::StackShape<Axis>>::Output as Shape>::Field:
            core::default::Default,
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
    pub fn try_stack(
        &self,
        other: &Tensor<S, B, K, G>,
        dim: usize,
    ) -> Result<Tensor<Dyn, B, K, G>> {
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

/// `try_stack_tensors`.
pub fn try_stack_tensors<
    S: Shape + DynShape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: crate::tensor::grad::RequiresGrad,
>(
    tensors: &[&Tensor<S, B, K, G>],
    dim: usize,
) -> Result<Tensor<Dyn, B, K, G>>
where
    G::Field: Clone,
{
    if tensors.is_empty() {
        return Err(crate::prelude::Error::ShapeMismatch {
            op: "stack_tensors",
            expected: alloc::vec![],
            got: alloc::vec![],
            msg: alloc::string::String::from("Cannot stack empty list of tensors"),
        });
    }
    let raw_tensors: alloc::vec::Vec<&B::Storage<K>> = tensors.iter().map(|t| &t.inner).collect();
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

impl<
    S: Shape,
    B: Backend,
    K: crate::tensor::dtype::DType,
    G: RequiresGrad,
    NewD: crate::prelude::Device,
> crate::nn::module::ToDevice<B, NewD> for Tensor<S, B, K, G>
where
    B: Backend + TransferTo<NewD>,
    <B as TransferTo<NewD>>::Output: SupportsDType<K>,
{
    /// The output tensor type produced by this module's forward pass.
    type Output = Tensor<S, <B as TransferTo<NewD>>::Output, K, G>;
    /// `to_device`.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let inner = B::transfer_storage(&self.inner, &self._dtype, &field)?;
        Ok(Tensor::from_parts_unchecked(
            inner,
            self._shape,
            self._dtype,
            field,
            self._grad,
        ))
    }
}
