//! Shape manipulation and restructuring operations.
//!
//! This module provides methods to change the logical or physical shape of a tensor
//! without necessarily changing the underlying data. It includes reshaping, transposition,
//! squeezing, flattening, and broadcasting. These operations heavily leverage the
//! compile-time type system to ensure the resulting shapes are strictly valid.
use crate::prelude::{
    Backend, Dyn, DynShape, RequiresGrad, Result, Shape, SupportsDType, Tensor, TransferTo,
};
use crate::shapes::error::OperationKind;
use crate::shapes::shape::field_from_dims;
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
    /// use incin::prelude::*;
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

        Tensor::from_parts(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
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
            )?;
        Tensor::from_parts(
            out,
            shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
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
    /// use incin::prelude::*;
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
        Tensor::from_parts(
            inner,
            new_shape_field,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Reshapes a tensor based on python-like slicing syntax via the `idx!` macro.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use incin::prelude::*;
    /// let t = Tensor::<s![2, 3], DefaultBackend>::ones(()).unwrap();
    /// let r = t.reshape_idx::<idx![6]>().unwrap();
    /// ```
    pub fn reshape_idx<T: crate::shapes::idx::ReshapeTarget<S>>(
        &self,
    ) -> Result<Tensor<T::Output, B, K, G>> {
        let in_shape_vec = S::dims(&self._shape);
        let out_shape_vec = T::calculate_shape(in_shape_vec.as_ref());
        let inner = B::reshape(&self.inner, &out_shape_vec)?;
        Tensor::from_parts(
            inner,
            field_from_dims::<T::Output>(OperationKind::Reshape, &out_shape_vec)?,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
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

        Tensor::from_parts(
            inner,
            field_from_dims::<T::Output>(OperationKind::Reshape, &out_shape_vec)?,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Narrows the tensor dynamically, returning a tensor with `Dyn` shape.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use incin::prelude::*;
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
    /// use incin::prelude::*;
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
        Tensor::from_parts(
            inner,
            new_shape_field,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Broadcast the tensor to the specific shape `S2`.
    pub fn broadcast_to<S2: Shape + DynShape>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G>> {
        let new_shape_field = S2::init(args);
        let new_dims = S2::dims(&new_shape_field);
        let inner = B::broadcast_as(&self.inner, new_dims.as_ref())?;
        Tensor::from_parts(
            inner,
            new_shape_field,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// `to_dtype`.
    pub fn to_dtype<T2: crate::tensor::dtype::DType<Arg = ()>>(
        &self,
    ) -> Result<Tensor<S, B, T2, G>> {
        let field = T2::init(());
        let incin_dtype = T2::to_incin(&field);
        let inner = B::tensor_to_dtype::<K, T2>(&self.inner, incin_dtype)?;
        Tensor::from_parts(
            inner,
            self._shape.clone(),
            field,
            self._device.clone(),
            self._grad.clone(),
        )
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
        let dtype = K::to_incin(&self._dtype);

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
        let dtype = K::to_incin(&self._dtype);

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
    /// use incin::prelude::*;
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

        Tensor::from_parts(
            inner,
            field_from_dims::<S::Output>(OperationKind::Reshape, &out_dims)?,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Flattens dimensions from `START` to `END` inclusive.
    ///
    /// The range uses const generics because its values are axis indices, not
    /// dimension sizes. The shape dimensions remain type-level `Dim` values,
    /// and `ProdDim` computes the flattened output dimension in the type system.
    ///
    /// # Examples
    /// ```rust,ignore
    /// use incin::prelude::*;
    /// let t = Tensor::<s![2, 3, 4], DefaultBackend>::ones(()).unwrap();
    /// let f = t.flatten::<1, 2>().unwrap(); // shape [2, 12]
    /// ```
    pub fn flatten<const START: usize, const END: usize>(
        &self,
    ) -> Result<Tensor<S::Output, B, K, G>>
    where
        S: crate::shapes::Flatten<START, END>,
    {
        let in_dims: Vec<usize> = S::dims(&self._shape).into();
        if START > END || END >= in_dims.len() {
            return Err(crate::shapes::ShapeError::InvalidAxisRange {
                operation: OperationKind::Flatten,
                start: START,
                end: END,
                rank: in_dims.len(),
            }
            .into());
        }

        let inner = B::flatten(&self.inner, START, END)?;
        let mut out_dims = Vec::new();

        out_dims.extend_from_slice(&in_dims[0..START]);

        let prod = in_dims[START..=END]
            .iter()
            .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
            .ok_or(crate::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Flatten,
                expression: "flattened dimension product",
            })?;
        out_dims.push(prod);

        out_dims.extend_from_slice(&in_dims[(END + 1)..]);

        Tensor::from_parts(
            inner,
            field_from_dims::<S::Output>(OperationKind::Reshape, &out_dims)?,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
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
        Tensor::from_parts(
            inner,
            field_from_dims::<Dyn>(OperationKind::Reshape, &out_shape)?,
            tensors[0]._dtype.clone(),
            tensors[0]._device.clone(),
            tensors[0]._grad.clone(),
        )
    }

    /// Statically concatenates `self` with `other` along `Axis`.
    pub fn concat<S2, Axis>(
        &self,
        other: &Tensor<S2, B, K, G>,
    ) -> Result<Tensor<<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output, B, K, G>>
    where
        S2: Shape + DynShape,
        Axis: typenum::Unsigned,
        S: crate::shapes::concat::ConcatShape<S2, Axis>,
    {
        let dim = Axis::USIZE;
        let inner = B::concat(&[&self.inner, &other.inner], dim)?;

        // Built from the operands' real dims (not `Default::default()`):
        // for purely-typenum shapes the output `Field` is a zero-sized
        // `PhantomData` either way, but any runtime-carrying dimension
        // (a plain `usize` axis, or a `symbolic_dim!` name) needs its
        // actual value copied through, or the result's declared shape
        // would silently report `0`/the wrapped type's default instead of
        // the tensor's real size.
        let mut out_dims: Vec<usize> = S::dims(&self._shape).into();
        let other_dims: Vec<usize> = S2::dims(&other._shape).into();
        out_dims[dim] += other_dims[dim];

        Tensor::from_parts(
            inner,
            <<S as crate::shapes::concat::ConcatShape<S2, Axis>>::Output as Shape>::from_dyn(
                &out_dims,
            )
            .unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
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
        Tensor::from_parts(
            inner,
            field_from_dims::<Dyn>(OperationKind::Reshape, &out_shape)?,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
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
        Tensor::from_parts(
            inner,
            field_from_dims::<Dyn>(OperationKind::Reshape, &out_shape)?,
            tensors[0]._dtype.clone(),
            tensors[0]._device.clone(),
            tensors[0]._grad.clone(),
        )
    }

    /// Statically stacks `self` with `other` along `Axis`.
    pub fn stack<Axis>(
        &self,
        other: &Tensor<S, B, K, G>,
    ) -> Result<Tensor<<S as crate::shapes::stack::StackShape<Axis>>::Output, B, K, G>>
    where
        Axis: typenum::Unsigned,
        S: crate::shapes::stack::StackShape<Axis>,
    {
        let dim = Axis::USIZE;
        let inner = B::stack(&[&self.inner, &other.inner], dim)?;

        // Built from `self`'s real dims (not `Default::default()`) — see
        // the identical fix and rationale on `concat` above: any
        // runtime-carrying dimension (a plain `usize` axis, or a
        // `symbolic_dim!` name) needs its actual value copied through.
        let mut out_dims: Vec<usize> = S::dims(&self._shape).into();
        out_dims.insert(dim, 2);

        Tensor::from_parts(
            inner,
            <<S as crate::shapes::stack::StackShape<Axis>>::Output as Shape>::from_dyn(&out_dims)
                .unwrap(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
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
        Tensor::from_parts(
            inner,
            field_from_dims::<Dyn>(OperationKind::Reshape, &out_shape)?,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Conditional selection: picks elements from `on_true` where `self` is non-zero, and `on_false` elsewhere.
    pub fn where_cond<S2: Shape, G2: RequiresGrad, G3: RequiresGrad>(
        &self,
        on_true: &Tensor<S2, B, K, G2>,
        on_false: &Tensor<S2, B, K, G3>,
    ) -> Result<Tensor<S2, B, K, G2>>
    where
        S: ShapeEq<S2>,
    {
        <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
        let inner = B::where_cond::<K, K>(&self.inner, &on_true.inner, &on_false.inner)?;
        Tensor::from_parts(
            inner,
            on_true._shape.clone(),
            on_true._dtype.clone(),
            on_true._device.clone(),
            on_true._grad.clone(),
        )
    }

    /// Fills elements where `mask` is non-zero with `value`.
    pub fn masked_fill<
        S2: Shape,
        KMask: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        Sc: Into<crate::tensor::backend::ScalarValue>,
    >(
        &self,
        mask: &Tensor<S2, B, KMask, G2>,
        value: Sc,
    ) -> Result<Self>
    where
        S: ShapeEq<S2>,
    {
        <S as ShapeEq<S2>>::ASSERT_SHAPES_MATCH;
        let val_f64 = value.into().to_f64();
        let inner = B::masked_fill::<K, KMask>(&self.inner, &mask.inner, val_f64)?;
        Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Gathers values along `dim` specified by `index`.
    pub fn gather<S2: Shape, KInt: crate::tensor::dtype::DType, G2: RequiresGrad>(
        &self,
        dim: usize,
        index: &Tensor<S2, B, KInt, G2>,
    ) -> Result<Tensor<S2, B, K, G>> {
        let inner = B::gather::<K, KInt>(&self.inner, dim, &index.inner)?;
        Tensor::from_parts(
            inner,
            index._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Scatters `src` values along `dim` into `self` using `index`.
    pub fn scatter<
        S2: Shape,
        S3: Shape,
        KInt: crate::tensor::dtype::DType,
        G2: RequiresGrad,
        G3: RequiresGrad,
    >(
        &self,
        dim: usize,
        index: &Tensor<S2, B, KInt, G2>,
        src: &Tensor<S3, B, K, G3>,
    ) -> Result<Self>
    where
        S2: ShapeEq<S3>,
    {
        <S2 as ShapeEq<S3>>::ASSERT_SHAPES_MATCH;
        let inner = B::scatter::<K, KInt>(&self.inner, dim, &index.inner, &src.inner)?;
        Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Selects slices along `dim` given 1D `index`.
    pub fn index_select<S2: Shape, KInt: crate::tensor::dtype::DType, G2: RequiresGrad>(
        &self,
        dim: usize,
        index: &Tensor<S2, B, KInt, G2>,
    ) -> Result<Tensor<Dyn, B, K, G>> {
        let inner = B::index_select::<K, KInt>(&self.inner, dim, &index.inner)?;
        let out_shape = B::shape(&inner);
        Tensor::from_parts(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Inserts a 1-sized dimension at position `dim`.
    pub fn unsqueeze(&self, dim: usize) -> Result<Tensor<Dyn, B, K, G>> {
        let inner = B::unsqueeze::<K>(&self.inner, dim)?;
        let out_shape = B::shape(&inner);
        Tensor::from_parts(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Repeats tensor data along each dimension according to `repeats`.
    pub fn repeat(&self, repeats: &[usize]) -> Result<Tensor<Dyn, B, K, G>> {
        let inner = B::repeat::<K>(&self.inner, repeats)?;
        let out_shape = B::shape(&inner);
        Tensor::from_parts(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Pads tensor according to `padding` (before, after) pairs per dimension with `val`.
    pub fn pad<Sc: Into<crate::tensor::backend::ScalarValue>>(
        &self,
        padding: &[(usize, usize)],
        val: Sc,
    ) -> Result<Tensor<Dyn, B, K, G>> {
        let val_f64 = val.into().to_f64();
        let inner = B::pad::<K>(&self.inner, padding, val_f64)?;
        let out_shape = B::shape(&inner);
        Tensor::from_parts(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Returns upper triangular part of matrix.
    pub fn triu(&self, k: i64) -> Result<Self> {
        let inner = B::triu::<K>(&self.inner, k)?;
        Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Returns lower triangular part of matrix.
    pub fn tril(&self, k: i64) -> Result<Self> {
        let inner = B::tril::<K>(&self.inner, k)?;
        Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Extracts or constructs diagonal tensor.
    pub fn diag(&self, k: i64) -> Result<Tensor<Dyn, B, K, G>> {
        let inner = B::diag::<K>(&self.inner, k)?;
        let out_shape = B::shape(&inner);
        Tensor::from_parts(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Splits tensor into `chunks` equal parts along `dim`.
    pub fn chunk(
        &self,
        chunks: usize,
        dim: usize,
    ) -> Result<alloc::vec::Vec<Tensor<Dyn, B, K, G>>> {
        let dim_size = S::dims(&self._shape).as_ref()[dim];
        if chunks == 0 {
            return Err(crate::err::Error::Msg(
                "chunk expects positive number of chunks".into(),
            ));
        }
        let chunk_size = dim_size.div_ceil(chunks);
        let mut out = alloc::vec::Vec::with_capacity(chunks);
        for i in 0..chunks {
            let start = i * chunk_size;
            if start >= dim_size {
                break;
            }
            let len = (dim_size - start).min(chunk_size);
            out.push(self.clone().try_narrow(dim, start, len)?);
        }
        Ok(out)
    }

    /// Splits tensor into sections of size `split_size` along `dim`.
    pub fn split(
        &self,
        split_size: usize,
        dim: usize,
    ) -> Result<alloc::vec::Vec<Tensor<Dyn, B, K, G>>> {
        let dim_size = S::dims(&self._shape).as_ref()[dim];
        if split_size == 0 {
            return Err(crate::err::Error::Msg(
                "split expects positive split_size".into(),
            ));
        }
        let chunks = dim_size.div_ceil(split_size);
        let mut out = alloc::vec::Vec::with_capacity(chunks);
        for i in 0..chunks {
            let start = i * split_size;
            let len = (dim_size - start).min(split_size);
            out.push(self.clone().try_narrow(dim, start, len)?);
        }
        Ok(out)
    }

    /// Expands the tensor to target shape `S2`.
    pub fn expand<S2: Shape + DynShape>(&self, args: S2::Arg) -> Result<Tensor<S2, B, K, G>> {
        self.broadcast_to::<S2>(args)
    }

    /// Extracts sliding window slices along `dim`.
    pub fn unfold(&self, dim: usize, size: usize, step: usize) -> Result<Tensor<Dyn, B, K, G>> {
        let inner = B::unfold::<K>(&self.inner, dim, size, step)?;
        let out_shape = B::shape(&inner);
        Tensor::from_parts(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Rearranges elements in a 4D tensor of shape (N, C, H, W) to (N, C / r^2, H * r, W * r).
    pub fn pixel_shuffle(&self, upscale_factor: usize) -> Result<Tensor<Dyn, B, K, G>> {
        let inner = B::pixel_shuffle::<K>(&self.inner, upscale_factor)?;
        let out_shape = B::shape(&inner);
        Tensor::from_parts(
            inner,
            out_shape,
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Group normalization across `groups`.
    pub fn group_norm(&self, groups: usize, eps: f64) -> Result<Self> {
        let inner = B::group_norm::<K>(&self.inner, groups, eps)?;
        Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
    }

    /// Instance normalization for 4D (N, C, H, W) tensors.
    pub fn instance_norm(&self, eps: f64) -> Result<Self> {
        let inner = B::instance_norm::<K>(&self.inner, eps)?;
        Tensor::from_parts(
            inner,
            self._shape.clone(),
            self._dtype.clone(),
            self._device.clone(),
            self._grad.clone(),
        )
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
    /// The same tensor, rebuilt on backend `NewD`.
    type Output = Tensor<S, <B as TransferTo<NewD>>::Output, K, G>;
    /// Transfers storage to device `arg`, keeping shape/dtype/grad-tracking.
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let inner = B::transfer_storage(&self.inner, &self._dtype, &field)?;
        Tensor::from_parts(inner, self._shape, self._dtype, field, self._grad)
    }
}
