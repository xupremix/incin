use super::*;


    use super::*;
    use crate::exec::spec::ExecutionDescriptor;
    use crate::tensor::reduction::Reduction;
    use crate::prelude::Result;
    use crate::tensor::device::Device;
    use crate::tensor::device::DeviceId;
    use crate::tensor::dtype::DType;

    /// Test-only stand-in `Backend` used by `tensor/base.rs`'s unit tests to
    /// exercise `Tensor`'s generic-over-`Backend` machinery without pulling
    /// in a real compute backend. Its `Storage<K>` is literally the shape
    /// (`Vec<usize>`) --- every op below tracks how an operation would
    /// transform the *shape*, using the same arithmetic real backends use,
    /// but performs no actual data computation and holds no element values.
    ///
    /// Dtype is not part of the backend's identity --- a single `DummyBackend<D>`
    /// can hold f32, f64, i64, etc. tensors, all represented as shape-only `Vec<usize>`.
    pub struct DummyBackend<D> {
        _marker: core::marker::PhantomData<D>,
    }

    impl<D: Device + Clone + 'static> Default for DummyBackend<D> {
        fn default() -> Self {
            DummyBackend {
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<D: Device + Clone + 'static> Capabilities for DummyBackend<D> {
        fn support(&self, _query: &crate::exec::CapabilityQuery) -> crate::exec::SupportLevel {
            crate::exec::SupportLevel::Native
        }
    }

    impl<D: Device + Clone + 'static> Clone for DummyBackend<D> {
        /// Cheap: the type carries no state beyond its `PhantomData` markers.
        fn clone(&self) -> Self {
            DummyBackend {
                _marker: core::marker::PhantomData,
            }
        }
    }

    impl<D: Device + Clone + 'static> StorageBackend for DummyBackend<D> {
        const BACKEND_NAME: &'static str = "dummy";
        /// The device type this stand-in is parameterized over.
        type Device = D;
        /// Shape-only storage: `Storage<K>` is the tensor's shape, not its
        /// values, regardless of `K`.
        type Storage<K: DType> = alloc::vec::Vec<usize>;

        fn metadata<K: DType>(storage: &<Self as StorageBackend>::Storage<K>) -> &TensorMeta {
            let shape_buf = crate::shapes::ShapeBuf::from_slice(storage);
            let numel = shape_buf.numel().unwrap_or(0);
            let dtype = K::descriptor(&K::Field::default());
            alloc::boxed::Box::leak(alloc::boxed::Box::new(
                TensorMeta::contiguous(
                    shape_buf,
                    dtype,
                    DeviceId::cpu(),
                    crate::exec::meta::Alignment::of::<f32>(),
                    numel,
                )
                .expect("valid dummy metadata"),
            ))
        }
    }

    impl<D: Device + Clone + 'static, O: crate::exec::catalog::Operation> Execute<O>
        for DummyBackend<D>
    {
        type Output = alloc::vec::Vec<usize>;

        fn execute(
            &self,
            request: ExecutionRequest<'_, O, Self>,
        ) -> core::result::Result<Self::Output, BackendError> {
            if let Some(output) = request.operation.descriptor().output_shape() {
                return Ok(output.dims().to_vec());
            }
            if let Some(input) = request.inputs.first() {
                Ok(input.metadata().shape.dims().to_vec())
            } else {
                Ok(alloc::vec![])
            }
        }
    }

    impl<D: Device + Clone + 'static> Backend for DummyBackend<D> {
        /// No dispatch wrapper --- this stand-in is always its own inner backend.
        type InnerBackend = Self;

    }

    impl<D: Device + Clone + 'static> crate::tensor::backend::HostInterop
        for DummyBackend<D>
    {
        fn host_format_display<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }

        fn host_format_debug<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> alloc::string::String {
            alloc::string::String::from("dummy")
        }

        /// Always empty: there are no element values to serialize.
        fn to_bytes<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<u8>> {
            Ok(alloc::vec::Vec::new())
        }

        /// Ignores `bytes` entirely and reconstructs storage from `shape`.
        fn from_bytes<K: DType>(
            _bytes: &[u8],
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
    }

    impl<D: Device + Clone + 'static> AutogradBackend for DummyBackend<D> {
        type Grads = ();

        fn backward<K: DType>(_t: &Self::Storage<K>) -> Result<Self::Grads> {
            Ok(())
        }

        fn backward_with<K: DType>(
            _t: &Self::Storage<K>,
            _seed: &Self::Storage<K>,
        ) -> Result<Self::Grads> {
            Ok(())
        }

        fn get_grad<K: DType>(
            _t: &Self::Storage<K>,
            _grads: &Self::Grads,
        ) -> Result<Option<Self::Storage<K>>> {
            Ok(None)
        }
    }

    impl<D: Device + Clone + 'static> VariableBackend for DummyBackend<D> {
        type Var<K: DType> = alloc::vec::Vec<usize>;

        fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> Result<Self::Storage<K>> {
            Ok(var.clone())
        }

        fn var_from_tensor<K: DType>(t: &Self::Storage<K>) -> Result<Self::Var<K>> {
            Ok(t.clone())
        }

        fn assign_var<K: DType>(var: &mut Self::Var<K>, tensor: &Self::Storage<K>) -> Result<()> {
            *var = tensor.clone();
            Ok(())
        }
    }

    impl<D: Device + Clone + 'static, K: DType> SupportsDType<K> for DummyBackend<D> {
        fn resolve_dtype(field: &K::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
            Ok(K::descriptor(field))
        }
    }

    /// Output spatial size for conv/pool shape math:
    /// `(in + 2*pad - dilation*(kernel-1) - 1) / stride + 1`. Uses saturating
    /// arithmetic throughout (never panics/wraps on pathological inputs ---
    /// small `in` with a large `kernel`/`dilation`/`padding` would otherwise
    /// underflow the `usize` subtraction), matching the CPU backend's own
    /// `out_size` (`cpu/ops/pool.rs`), which already uses the same
    /// saturate-rather-than-error convention for this exact case. This is
    /// shape-only bookkeeping for `DummyBackend` (a test-only stand-in with
    /// no real storage), so a saturated/degenerate size is the appropriate
    /// "can't compute a real answer" response, not an error.
    fn conv_out_size(
        len: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> usize {
        let padded = len.saturating_add(2 * padding);
        let effective_kernel = dilation
            .saturating_mul(kernel_size.saturating_sub(1))
            .saturating_add(1);
        padded.saturating_sub(effective_kernel) / stride.max(1) + 1
    }

    /// Output spatial size for `conv_transpose2d` shape math:
    /// `(in - 1) * stride - 2*pad + dilation*(kernel-1) + output_padding + 1`.
    /// Same saturating-arithmetic rationale as `conv_out_size`.
    fn conv_transpose_out_size(
        len: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
    ) -> usize {
        let strided = len.saturating_sub(1).saturating_mul(stride);
        let effective_kernel = dilation.saturating_mul(kernel_size.saturating_sub(1));
        strided
            .saturating_sub(2 * padding)
            .saturating_add(effective_kernel)
            .saturating_add(output_padding)
            .saturating_add(1)
    }

    impl<D: Device + Clone + 'static, NewD: Device + Clone + 'static> StorageTransfer<NewD>
        for DummyBackend<D>
    {
        type Output = DummyBackend<NewD>;

        fn transfer_storage<K: DType>(
            storage: &<Self as StorageBackend>::Storage<K>,
            _dtype: &K::Field,
            _device: &NewD::Field,
        ) -> Result<<Self::Output as StorageBackend>::Storage<K>>
        where
            Self::Output: SupportsDType<K>,
        {
            Ok(storage.clone())
        }

    }

    impl<D: Device + Clone + 'static, NewD: Device + Clone + 'static> TransferTo<NewD>
        for DummyBackend<D>
    {
        fn transfer_var<K: DType>(
            variable: &Self::Var<K>,
            _dtype: &K::Field,
            _device: &NewD::Field,
        ) -> Result<<Self::Output as crate::tensor::backend::VariableBackend>::Var<K>>
        where
            Self::Output: SupportsDType<K>,
        {
            Ok(variable.clone())
        }
    }

    /// Shape is preserved by every allocation, since it's the only thing
    /// `Storage`/`Var<K>` track --- no real fill value is ever written.
    impl<D: Device + Clone + 'static> CreationOps<Self> for DummyBackend<D> {
        /// Returns `shape` verbatim as the storage handle.
        fn zeros<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn full<K: DType>(
            _val: f64,
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn arange<K: DType>(
            _start: f64,
            _step: f64,
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn linspace<K: DType>(
            _start: f64,
            _end: f64,
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn ones<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn rand<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the storage handle.
        fn randn<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_zeros<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as crate::tensor::backend::VariableBackend>::Var<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_ones<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as crate::tensor::backend::VariableBackend>::Var<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_rand<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as crate::tensor::backend::VariableBackend>::Var<K>> {
            Ok(shape.to_vec())
        }
        /// Returns `shape` verbatim as the variable handle.
        fn var_randn<K: DType>(
            shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<<Self as crate::tensor::backend::VariableBackend>::Var<K>> {
            Ok(shape.to_vec())
        }
    }

    /// Every binary op broadcasts its two shapes the way a real backend does.
    ///
    /// These returned `lhs`'s shape unchanged until `UX-013`, which is wrong for
    /// the same reason it was invisible: `add` is reached with differently
    /// shaped operands only through `broadcast_add` and friends, which then
    /// hand the result to `Tensor::from_parts` against the *broadcast* type. A
    /// stand-in whose shape arithmetic disagrees with every real backend's is
    /// not a stand-in, and this crate's own documented examples of
    /// `broadcast_add` were the first thing to run into it.
    impl<D: Device + Clone + 'static> NumericOps<Self> for DummyBackend<D> {
        /// Returns the two operands' broadcast shape.
        fn add<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(crate::shapes::broadcast::broadcast_dim_slices(lhs, rhs)?)
        }
        /// Returns the two operands' broadcast shape.
        fn sub<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(crate::shapes::broadcast::broadcast_dim_slices(lhs, rhs)?)
        }
        /// Returns the two operands' broadcast shape.
        fn mul<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(crate::shapes::broadcast::broadcast_dim_slices(lhs, rhs)?)
        }
        /// Returns the two operands' broadcast shape.
        fn div<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(crate::shapes::broadcast::broadcast_dim_slices(lhs, rhs)?)
        }
    }

    /// The remaining float operations, all of which preserve their input's
    /// shape and so are the same clone as the ones written out above.
    ///
    /// `DummyBackend` exists to exercise shape behavior, so covering these by
    /// hand would add a hundred lines that all say `Ok(t.clone())`. They are
    /// listed rather than inherited because `FloatOps` no longer supplies a
    /// default body: an operation this backend does not model has to be
    /// visible here.
    macro_rules! shape_preserving_float_ops {
        (
            unary: $($unary:ident),* $(,)?;
            exponent: $($exponent:ident),* $(,)?;
            bounds: $($bounds:ident),* $(,)?;
            binary: $($binary:ident),* $(,)?;
        ) => {
            $(
                fn $unary<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $exponent<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                    _exponent: f64,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $bounds<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                    _min: f64,
                    _max: f64,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $binary<K: DType>(
                    lhs: &<Self as StorageBackend>::Storage<K>,
                    _rhs: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(lhs.clone())
                }
            )*
        };
    }

    /// Every activation and scalar op is shape-preserving, so each is a
    /// plain clone of the input shape.
    impl<D: Device + Clone + 'static> FloatOps<Self> for DummyBackend<D> {
        /// Returns `t`'s shape unchanged.
        fn add_scalar_float<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn mul_scalar_float<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _scalar: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn relu<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn step<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn mish<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn elu<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn gelu<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn abs<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn exp<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn neg<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn sqrt<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn log<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn tanh<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn sigmoid<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn swish<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged (`dim` is not validated).
        fn softmax<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }

        shape_preserving_float_ops! {
            unary: sign, floor, ceil, round, log2, log10, sin, cos, tan, asin,
                   acos, atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt,
                   trunc, frac;
            exponent: powf;
            bounds: clamp;
            binary: atan2, fmod, remainder;
        }
    }

    /// `_all` reductions collapse to an (empty) scalar shape; `_dim`
    /// reductions either remove `dim` or clamp it to size 1 (`_keepdim`),
    /// exactly like a real reduction's shape effect --- again with no real
    /// values behind either result.
    impl<D: Device + Clone + 'static> ReductionOps<Self> for DummyBackend<D> {
        /// Collapses to an empty (scalar) shape.
        fn sum_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn mean_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn max_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn min_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Collapses to an empty (scalar) shape.
        fn prod_all<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Removes `dim` from the shape.
        fn prod_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// A running sum along `dim` leaves the shape unchanged.
        fn cumsum<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Removes `dim` from the shape.
        fn sum_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn sum_keepdim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Removes `dim` from the shape.
        fn mean_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn mean_keepdim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Removes `dim` from the shape.
        fn max_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn max_keepdim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Removes `dim` from the shape.
        fn min_dim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s.remove(dim);
            }
            Ok(s)
        }
        /// Sets `dim`'s size to 1, keeping the dimension in the shape.
        fn min_keepdim<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            dim: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut s = t.clone();
            if dim < s.len() {
                s[dim] = 1;
            }
            Ok(s)
        }
        /// Always an empty shape --- no indices are actually computed.
        fn argmax<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Always an empty shape --- no indices are actually computed.
        fn argmin<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: Option<usize>,
        ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
        /// Always an empty `(values, indices)` pair.
        fn topk<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _k: usize,
            _dim: usize,
            _largest: bool,
        ) -> Result<(
            <Self as StorageBackend>::Storage<K>,
            <Self as StorageBackend>::Storage<KInt>,
        )> {
            Ok((alloc::vec![], alloc::vec![]))
        }
        /// Always an empty shape --- no indices are actually computed.
        fn argsort<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
            _descending: bool,
        ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
            Ok(alloc::vec![])
        }
    }

    /// The `TensorOps` members whose output shape equals an input's, split by
    /// which operand supplies it. These mirror `NumericOps`' convention above,
    /// where a binary op reports `lhs`'s shape.
    macro_rules! shape_preserving_tensor_ops {
        (
            unary: $($unary:ident),* $(,)?;
            scalar: $($scalar:ident),* $(,)?;
            diagonal: $($diagonal:ident),* $(,)?;
            binary: $($binary:ident),* $(,)?;
        ) => {
            $(
                fn $unary<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $scalar<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                    _val: f64,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $diagonal<K: DType>(
                    t: &<Self as StorageBackend>::Storage<K>,
                    _k: i64,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(t.clone())
                }
            )*
            $(
                fn $binary<K: DType>(
                    lhs: &<Self as StorageBackend>::Storage<K>,
                    _rhs: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Ok(lhs.clone())
                }
            )*
        };
    }

    /// The `TensorOps` members whose output shape this stand-in does not
    /// model. Returning a plausible-looking wrong shape would be worse than
    /// refusing: shape is the only thing `DummyBackend` asserts, and a test
    /// reading a fabricated one would pass for the wrong reason.
    macro_rules! unmodeled_tensor_ops {
        (
            indexed: $($indexed:ident),* $(,)?;
            dim: $($dim:ident),* $(,)?;
            binary: $($binary:ident),* $(,)?;
        ) => {
            $(
                fn $indexed<K: DType, KInt: DType>(
                    _t: &<Self as StorageBackend>::Storage<K>,
                    _dim: usize,
                    _index: &<Self as StorageBackend>::Storage<KInt>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Err(crate::err::Error::UnsupportedBackendOperation {
                        op: stringify!($indexed),
                        backend: core::any::type_name::<Self>(),
                    })
                }
            )*
            $(
                fn $dim<K: DType>(
                    _t: &<Self as StorageBackend>::Storage<K>,
                    _dim: usize,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Err(crate::err::Error::UnsupportedBackendOperation {
                        op: stringify!($dim),
                        backend: core::any::type_name::<Self>(),
                    })
                }
            )*
            $(
                fn $binary<K: DType>(
                    _lhs: &<Self as StorageBackend>::Storage<K>,
                    _rhs: &<Self as StorageBackend>::Storage<K>,
                ) -> Result<<Self as StorageBackend>::Storage<K>> {
                    Err(crate::err::Error::UnsupportedBackendOperation {
                        op: stringify!($binary),
                        backend: core::any::type_name::<Self>(),
                    })
                }
            )*
        };
    }

    /// Each op tracks its real shape-transformation logic (matmul's last
    /// dim, transpose's swap, flatten's dimension collapse, etc.) since
    /// shape *is* everything this stand-in's storage represents --- but
    /// still no element values exist behind any of it.
    impl<D: Device + Clone + 'static> TensorOps<Self> for DummyBackend<D> {
        shape_preserving_tensor_ops! {
            unary: ;
            scalar: sub_scalar, div_scalar, instance_norm;
            diagonal: triu, tril;
            binary: maximum, minimum, abs_diff;
        }

        fn cmp_eq<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_ne<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_lt<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_le<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_gt<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn cmp_ge<K: DType>(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn logical_and(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn logical_or(
            lhs: &alloc::vec::Vec<usize>,
            _rhs: &alloc::vec::Vec<usize>,
        ) -> Result<alloc::vec::Vec<usize>> {
            Ok(lhs.clone())
        }
        fn logical_not(t: &alloc::vec::Vec<usize>) -> Result<alloc::vec::Vec<usize>> {
            Ok(t.clone())
        }

        unmodeled_tensor_ops! {
            indexed: gather, index_select;
            dim: unsqueeze, pixel_shuffle;
            binary: bmm;
        }

        /// Returns `on_true`'s shape, which is the branch the output takes.
        fn where_cond<K: DType>(
            _mask: &<Self as StorageBackend>::Storage<bool>,
            on_true: &<Self as StorageBackend>::Storage<K>,
            _on_false: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(on_true.clone())
        }

        /// Filling masked positions leaves the shape untouched.
        fn masked_fill<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _mask: &<Self as StorageBackend>::Storage<bool>,
            _value: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }

        /// Interpolating between two tensors keeps `start`'s shape.
        fn lerp<K: DType>(
            start: &<Self as StorageBackend>::Storage<K>,
            _end: &<Self as StorageBackend>::Storage<K>,
            _weight: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(start.clone())
        }

        /// Normalizing over groups leaves the shape untouched.
        fn group_norm<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _groups: usize,
            _eps: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }

        /// Not modeled: the output tiles each axis by its own factor.
        fn repeat<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _repeats: &[usize],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "repeat",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: the output grows by the padding on each axis.
        fn pad<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _padding: &[(usize, usize)],
            _val: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "pad",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: `diag` both extracts and constructs, changing rank
        /// in opposite directions depending on the input.
        fn diag<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _k: i64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "diag",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: writes into a copy of the target, whose shape this
        /// stand-in would have to reconcile against the index and source.
        fn scatter<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
            _index: &<Self as StorageBackend>::Storage<KInt>,
            _src: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "scatter",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: the fused product's shape follows `mat1 @ mat2`
        /// broadcast against `mat`.
        fn addmm<K: DType>(
            _mat: &<Self as StorageBackend>::Storage<K>,
            _mat1: &<Self as StorageBackend>::Storage<K>,
            _mat2: &<Self as StorageBackend>::Storage<K>,
            _beta: f64,
            _alpha: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "addmm",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: the output takes its trailing axis from `v`, not `q`.
        fn scaled_dot_product_attention<K: DType>(
            _q: &<Self as StorageBackend>::Storage<K>,
            _k: &<Self as StorageBackend>::Storage<K>,
            _v: &<Self as StorageBackend>::Storage<K>,
            _mask: Option<&<Self as StorageBackend>::Storage<K>>,
            _scale: Option<f64>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "scaled_dot_product_attention",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Not modeled: sliding windows replace one axis with two.
        fn unfold<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            _dim: usize,
            _size: usize,
            _step: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Err(crate::err::Error::UnsupportedBackendOperation {
                op: "unfold",
                backend: core::any::type_name::<Self>(),
            })
        }

        /// Broadcasts leading batch axes and applies the trailing matrix
        /// contraction, mirroring real matmul's output shape.
        fn matmul<K: DType>(
            lhs: &<Self as StorageBackend>::Storage<K>,
            rhs: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            if lhs.len() < 2 || rhs.len() < 2 || lhs[lhs.len() - 1] != rhs[rhs.len() - 2] {
                return Err(crate::err::Error::ShapeMismatch {
                    op: "matmul",
                    expected: lhs.clone(),
                    got: rhs.clone(),
                    msg: "matmul requires rank >= 2 and equal contraction dimensions".into(),
                });
            }

            let lhs_batch = &lhs[..lhs.len() - 2];
            let rhs_batch = &rhs[..rhs.len() - 2];
            let rank = lhs_batch.len().max(rhs_batch.len());
            let mut out = alloc::vec::Vec::with_capacity(rank + 2);
            for axis in 0..rank {
                let from_end = rank - axis;
                let l = lhs_batch
                    .len()
                    .checked_sub(from_end)
                    .map_or(1, |index| lhs_batch[index]);
                let r = rhs_batch
                    .len()
                    .checked_sub(from_end)
                    .map_or(1, |index| rhs_batch[index]);
                if l != r && l != 1 && r != 1 {
                    return Err(crate::err::Error::ShapeMismatch {
                        op: "matmul",
                        expected: lhs.clone(),
                        got: rhs.clone(),
                        msg: "matmul batch dimensions are not broadcast-compatible".into(),
                    });
                }
                out.push(if l == 1 { r } else { l });
            }
            out.extend_from_slice(&[lhs[lhs.len() - 2], rhs[rhs.len() - 1]]);
            Ok(out)
        }
        /// Always `0.0` --- there is no real element value to read.
        fn float_to_scalar<K: DType>(_t: &<Self as StorageBackend>::Storage<K>) -> Result<f64> {
            Ok(0.0)
        }
        /// Always a single `0.0` --- there are no real element values to read.
        fn float_to_vec1<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<f64>> {
            Ok(alloc::vec![0.0])
        }
        /// Always `0` --- there is no real element value to read.
        fn int_to_scalar<K: DType>(_t: &<Self as StorageBackend>::Storage<K>) -> Result<i64> {
            Ok(0)
        }
        /// Always a single `0` --- there are no real element values to read.
        fn int_to_vec1<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<alloc::vec::Vec<i64>> {
            Ok(alloc::vec![0])
        }

        /// Returns the target `shape` verbatim (broadcast compatibility is
        /// not validated).
        fn broadcast_as<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Prepends the target `shape`'s dimensions ahead of `t`'s own.
        fn broadcast_left<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = s.to_vec();
            out.extend_from_slice(t);
            Ok(out)
        }
        /// Returns the target `shape` verbatim (numel compatibility is not
        /// validated).
        fn reshape<K: DType>(
            _t: &<Self as StorageBackend>::Storage<K>,
            s: &[usize],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(s.to_vec())
        }
        /// Swaps dimensions `d1`/`d2` in the shape.
        fn transpose<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            d1: usize,
            d2: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            if d1 < out.len() && d2 < out.len() {
                out.swap(d1, d2);
            }
            Ok(out)
        }
        /// Collapses dimensions `[s, e]` into their product.
        fn flatten<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            s: usize,
            e: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            if s > e || e >= t.len() {
                return Ok(t.clone());
            }
            let mut out = t[..s].to_vec();
            out.push(
                crate::shapes::ShapeBuf::from_slice(&t[s..=e])
                    .checked_numel(crate::shapes::error::OperationKind::Flatten)?,
            );
            out.extend_from_slice(&t[e + 1..]);
            Ok(out)
        }
        /// Sets each dimension's size to its `(start, end)` window length.
        fn slice<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            ranges: &[(usize, usize)],
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            for (dim, &(start, end)) in ranges.iter().enumerate() {
                if dim < out.len() {
                    out[dim] = end.saturating_sub(start);
                }
            }
            Ok(out)
        }
        /// Sets dimension `d`'s size to the requested window length `l`.
        fn narrow<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            d: usize,
            _s: usize,
            l: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            if d < out.len() {
                out[d] = l;
            }
            Ok(out)
        }
        /// Removes dimension `d` from the shape.
        fn squeeze<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            d: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            if d < out.len() {
                out.remove(d);
            }
            Ok(out)
        }
        /// Inserts a new dimension at `d`, sized to the number of stacked
        /// tensors.
        fn stack<K: DType>(
            t: &[&<Self as StorageBackend>::Storage<K>],
            d: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            if t.is_empty() {
                return Ok(alloc::vec![]);
            }
            let mut out = t[0].clone();
            if d <= out.len() {
                out.insert(d, t.len());
            }
            Ok(out)
        }
        /// Sets dimension `d`'s size to the sum of every input's size
        /// along `d`.
        fn concat<K: DType>(
            t: &[&<Self as StorageBackend>::Storage<K>],
            d: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            if t.is_empty() {
                return Ok(alloc::vec![]);
            }
            let mut out = t[0].clone();
            if d < out.len() {
                out[d] = t.iter().map(|s| s.get(d).copied().unwrap_or(0)).sum();
            }
            Ok(out)
        }
        /// Returns `t`'s shape unchanged --- no element values exist to cast.
        fn tensor_to_dtype<K: DType, K2: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _dtype: DTypeDescriptor,
        ) -> Result<<Self as StorageBackend>::Storage<K2>> {
            Ok(t.clone())
        }
    }

    /// Normalization ops are shape-preserving no-ops; conv/pool ops
    /// compute their real output spatial size via `conv_out_size`/
    /// `conv_transpose_out_size` (the saturating helpers above) so tests
    /// can assert on shape correctness even though no data is computed.
    impl<D: Device + Clone + 'static> ModuleOps<Self> for DummyBackend<D> {
        /// Returns `t`'s shape unchanged.
        fn layer_norm<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _w: &<Self as StorageBackend>::Storage<K>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            _e: f32,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Returns `t`'s shape unchanged.
        fn batch_norm<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            _w: Option<&<Self as StorageBackend>::Storage<K>>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            _rm: Option<&<Self as StorageBackend>::Storage<K>>,
            _rv: Option<&<Self as StorageBackend>::Storage<K>>,
            _e: f32,
            _m: f64,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(t.clone())
        }
        /// Always an empty shape --- no real gather is performed.
        fn embedding<K: DType, KInt: DType>(
            _t: &<Self as StorageBackend>::Storage<KInt>,
            _w: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Computes the real output shape: channel dim from `w[0]`, spatial
        /// dim via `conv_out_size`.
        fn conv1d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            w: &<Self as StorageBackend>::Storage<K>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            s: usize,
            p: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 3 && w.len() >= 3 {
                let l_in = out[len - 1];
                let k = w[w.len() - 1];
                let c_out = w[0]; // Assuming [C_out, C_in / groups, K]
                out[len - 2] = c_out;
                out[len - 1] = conv_out_size(l_in, k, s, p, d);
            }
            Ok(out)
        }
        /// Computes the real output shape: channel dim from `w[0]`, spatial
        /// dims via `conv_out_size`.
        fn conv2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            w: &<Self as StorageBackend>::Storage<K>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            s: usize,
            p: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 4 && w.len() >= 4 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                let k_h = w[w.len() - 2];
                let k_w = w[w.len() - 1];
                let c_out = w[0]; // [C_out, C_in / groups, K_H, K_W]
                out[len - 3] = c_out;
                out[len - 2] = conv_out_size(h_in, k_h, s, p, d);
                out[len - 1] = conv_out_size(w_in, k_w, s, p, d);
            }
            Ok(out)
        }
        /// Computes the real output shape: channel dim from `w[1]`
        /// (transposed conv's weight layout), spatial dims via
        /// `conv_transpose_out_size`.
        fn conv_transpose2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            w: &<Self as StorageBackend>::Storage<K>,
            _b: Option<&<Self as StorageBackend>::Storage<K>>,
            s: usize,
            p: usize,
            op: usize,
            d: usize,
            _groups: usize,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 4 && w.len() >= 4 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                let k_h = w[w.len() - 2];
                let k_w = w[w.len() - 1];
                let c_out = w[1]; // [C_in, C_out / groups, K_H, K_W]
                out[len - 3] = c_out;
                out[len - 2] = conv_transpose_out_size(h_in, k_h, s, p, op, d);
                out[len - 1] = conv_transpose_out_size(w_in, k_w, s, p, op, d);
            }
            Ok(out)
        }
        /// Computes the real output spatial shape via `conv_out_size`.
        fn max_pool2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            k: (usize, usize),
            s: (usize, usize),
            p: (usize, usize),
            d: (usize, usize),
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 2 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                out[len - 2] = conv_out_size(h_in, k.0, s.0, p.0, d.0);
                out[len - 1] = conv_out_size(w_in, k.1, s.1, p.1, d.1);
            }
            Ok(out)
        }
        /// Computes the real output spatial shape via `conv_out_size`
        /// (dilation fixed to 1, matching `avg_pool2d` having no dilation
        /// parameter).
        fn avg_pool2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            k: (usize, usize),
            s: (usize, usize),
            p: (usize, usize),
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut out = t.clone();
            let len = out.len();
            if len >= 2 {
                let h_in = out[len - 2];
                let w_in = out[len - 1];
                out[len - 2] = conv_out_size(h_in, k.0, s.0, p.0, 1);
                out[len - 1] = conv_out_size(w_in, k.1, s.1, p.1, 1);
            }
            Ok(out)
        }
        /// Sets the trailing two dimensions directly to `out` (adaptive
        /// pooling's whole point is that the output size is exact,
        /// regardless of input size).
        fn adaptive_avg_pool2d<K: DType>(
            t: &<Self as StorageBackend>::Storage<K>,
            out: (usize, usize),
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            let mut shape = t.clone();
            let len = shape.len();
            if len >= 2 {
                shape[len - 2] = out.0;
                shape[len - 1] = out.1;
            }
            Ok(shape)
        }
    }

    /// All quantization ops are no-ops returning an empty shape --- there is
    /// no real data to (de)quantize.
    impl<D: Device + Clone + 'static> QuantizedOps<Self> for DummyBackend<D> {
        /// Always an empty shape.
        fn quantize<K: FloatDType, Q: QuantDType>(
            _t: &<Self as StorageBackend>::Storage<K>,
        ) -> Result<<Self as StorageBackend>::Storage<Q>> {
            Ok(alloc::vec![])
        }
        /// Always an empty shape.
        fn dequantize<Q: QuantDType, K: FloatDType>(
            _t: &<Self as StorageBackend>::Storage<Q>,
        ) -> Result<<Self as StorageBackend>::Storage<K>> {
            Ok(alloc::vec![])
        }
        /// Always an empty shape.
        fn quantized_matmul<Q: QuantDType>(
            _lhs: &<Self as StorageBackend>::Storage<Q>,
            _rhs: &<Self as StorageBackend>::Storage<Q>,
        ) -> Result<<Self as StorageBackend>::Storage<f32>> {
            Ok(alloc::vec![])
        }
    }
