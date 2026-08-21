use crate::backend_authoring::{Capabilities, Execute, Operation};
use crate::dist::Local;
use crate::err::{Error, ErrorMessage, Result};
use crate::exec::catalog::{CreationAttributes, FullAttributes, ScalarAttributes, op};
use crate::exec::dispatch;
use crate::exec::request::TensorHandle;
use crate::nn::init::{InitContext, InitPlan, ParameterRole};
use crate::shapes::Dyn;
use crate::shapes::{DynShape, Shape, ShapeBuf, ShapeValue};
use crate::tensor::arg::TensorArgs;
use crate::tensor::arg_into::ArgInto;
use crate::tensor::backend::{HostInterop, SupportsDType, VariableBackend, VariableTransfer};
use crate::tensor::base::Tensor;
use crate::tensor::device::{Device, DeviceId};
use crate::tensor::dtype::{DType, DTypeDescriptor};
use crate::tensor::grad::{Grad, NoGrad, RequiresGrad};
use alloc::vec::Vec;
use core::marker::PhantomData;

/// Marker struct for trainable parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Trainable;

/// Marker struct for frozen (non-trainable) parameters.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Frozen;

/// Trait defining parameter trainability typestate.
pub trait TrainState: 'static + Send + Sync + core::fmt::Debug + Clone + Copy {
    /// The tensor gradient requirement type.
    type TensorGrad: RequiresGrad<Arg = ()>;
    /// Whether parameters in this state receive optimizer updates.
    const TRAINABLE: bool;
}

impl TrainState for Trainable {
    type TensorGrad = Grad;
    const TRAINABLE: bool = true;
}

impl TrainState for Frozen {
    type TensorGrad = NoGrad;
    const TRAINABLE: bool = false;
}

/// Canonical parameter initialization capability.
///
/// This is implemented automatically for backends that provide the exact
/// variable-creation and scalar operation descriptors used by `InitPlan`.
pub trait ParameterInit<K: DType>: VariableBackend + SupportsDType<K> {
    fn execute_plan_raw(
        dims: &[usize],
        dtype_field: &<K as DType>::Field,
        device_field: &<Self::Device as Device>::Field,
        plan: crate::nn::init::InitPlan,
    ) -> Result<Self::Var<K>>;
}

fn validate_initialized_var<B, K>(
    raw_var: &<B as crate::tensor::backend::VariableBackend>::Var<K>,
    shape: &ShapeBuf,
    dtype: &K::Field,
    device: &<B::Device as Device>::Field,
    operation: &'static str,
) -> Result<()>
where
    B: VariableBackend + SupportsDType<K>,
    K: DType,
{
    let storage = B::var_as_tensor::<K>(raw_var)?;
    let meta = B::metadata(&storage);
    if meta.shape.as_ref() != shape.as_ref() {
        return Err(Error::ShapeMismatch {
            op: operation,
            expected: shape.as_ref().to_vec(),
            got: meta.shape.as_ref().to_vec(),
            msg: "Initialized variable shape mismatch".into(),
        });
    }
    let incin_device = <B::Device as Device>::to_incin(device)?;
    let expected_dtype = B::resolve_dtype(dtype, &incin_device)?;
    if meta.dtype != expected_dtype {
        return Err(Error::DTypeStorageMismatch {
            expected: expected_dtype,
            got: meta.dtype,
        });
    }
    if meta.device != incin_device {
        return Err(Error::DeviceStorageMismatch {
            expected: incin_device,
            got: meta.device,
        });
    }
    Ok(())
}

impl<B, K: DType> ParameterInit<K> for B
where
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + Capabilities
        + Execute<op::VariableZeros>
        + Execute<op::VariableOnes>
        + Execute<op::Full>
        + Execute<op::UniformRandom>
        + Execute<op::NormalRandom>
        + Execute<op::MulScalar>
        + Execute<op::AddScalar>,
    <B as Execute<op::VariableZeros>>::Output:
        Into<<B as crate::tensor::backend::VariableBackend>::Var<K>>,
    <B as Execute<op::VariableOnes>>::Output:
        Into<<B as crate::tensor::backend::VariableBackend>::Var<K>>,
    <B as Execute<op::Full>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::UniformRandom>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::NormalRandom>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::MulScalar>>::Output: Into<B::Storage<K>>,
    <B as Execute<op::AddScalar>>::Output: Into<B::Storage<K>>,
{
    fn execute_plan_raw(
        dims: &[usize],
        dtype_field: &<K as DType>::Field,
        device_field: &<B::Device as Device>::Field,
        plan: crate::nn::init::InitPlan,
    ) -> Result<<B as crate::tensor::backend::VariableBackend>::Var<K>> {
        let device = B::Device::to_incin(device_field)?;
        let dtype = B::resolve_dtype(dtype_field, &device)?;
        match plan {
            InitPlan::Zeros => execute_variable::<op::VariableZeros, B, K>(dims, dtype, device),
            InitPlan::Ones => execute_variable::<op::VariableOnes, B, K>(dims, dtype, device),
            InitPlan::Constant(value) => {
                let storage = execute_storage::<op::Full, B, K>(
                    FullAttributes {
                        shape: dims.to_vec(),
                        dtype,
                        device,
                        value,
                    },
                    dims,
                )?;
                B::var_from_tensor(&storage)
            }
            InitPlan::Uniform { low, high } => {
                let storage = execute_storage::<op::UniformRandom, B, K>(
                    CreationAttributes {
                        shape: dims.to_vec(),
                        dtype,
                        device,
                    },
                    dims,
                )?;
                let storage = execute_scalar::<op::MulScalar, B, K>(&storage, high - low)?;
                let storage = execute_scalar::<op::AddScalar, B, K>(&storage, low)?;
                B::var_from_tensor(&storage)
            }
            InitPlan::Normal { mean, std } => {
                let storage = execute_storage::<op::NormalRandom, B, K>(
                    CreationAttributes {
                        shape: dims.to_vec(),
                        dtype,
                        device,
                    },
                    dims,
                )?;
                let storage = execute_scalar::<op::MulScalar, B, K>(&storage, std)?;
                if mean != 0.0 {
                    B::var_from_tensor(&execute_scalar::<op::AddScalar, B, K>(&storage, mean)?)
                } else {
                    B::var_from_tensor(&storage)
                }
            }
        }
    }
}

pub fn execute_plan_raw<B, K: DType>(
    dims: &[usize],
    dtype_field: &<K as DType>::Field,
    device_field: &<B::Device as Device>::Field,
    plan: crate::nn::init::InitPlan,
) -> Result<<B as crate::tensor::backend::VariableBackend>::Var<K>>
where
    B: ParameterInit<K>,
{
    B::execute_plan_raw(dims, dtype_field, device_field, plan)
}

fn execute_variable<O, B, K: DType>(
    dims: &[usize],
    dtype: DTypeDescriptor,
    device: DeviceId,
) -> Result<<B as crate::tensor::backend::VariableBackend>::Var<K>>
where
    O: Operation<Attributes = CreationAttributes>,
    B: crate::tensor::backend::VariableBackend + Execute<O> + Capabilities,
    <B as Execute<O>>::Output: Into<<B as crate::tensor::backend::VariableBackend>::Var<K>>,
{
    // Parameter allocation is outside the differentiable graph. The returned
    // variable becomes a graph input only when the caller uses it later.
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute::<O, B>(
        &context,
        CreationAttributes {
            shape: dims.to_vec(),
            dtype,
            device,
        },
        &[],
    )
    .map(Into::into)
    .map_err(crate::err::Error::from)
}

fn execute_storage<O, B, K>(attributes: O::Attributes, dims: &[usize]) -> Result<B::Storage<K>>
where
    O: Operation,
    B: crate::tensor::backend::VariableBackend + Execute<O> + Capabilities,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let shape = ShapeBuf::from_slice(dims);
    let expected = ShapeValue::<Dyn>::try_new(shape).map_err(Error::Shape)?;
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute_shaped::<O, B, Dyn>(&context, attributes, &[], &expected)
        .map(Into::into)
        .map_err(crate::err::Error::from)
}

fn execute_scalar<O, B, K>(storage: &B::Storage<K>, value: f64) -> Result<B::Storage<K>>
where
    O: Operation<Attributes = ScalarAttributes>,
    B: crate::tensor::backend::VariableBackend + Execute<O> + Capabilities,
    K: DType,
    <B as Execute<O>>::Output: Into<B::Storage<K>>,
{
    let shape = B::shape(storage);
    let expected = ShapeValue::<Dyn>::try_new(shape).map_err(Error::Shape)?;
    let handle = TensorHandle::from_storage::<B, K, Local>(storage);
    let context = crate::exec::ExecutionContext::from_scope(B::default())
        .with_grad_mode(crate::exec::GradMode::Disabled);
    dispatch::execute_shaped::<O, B, Dyn>(
        &context,
        ScalarAttributes { value },
        &[handle],
        &expected,
    )
    .map(Into::into)
    .map_err(crate::err::Error::from)
}

fn execute_initializer<B, K: DType>(
    dims: &[usize],
    dtype_field: &<K as DType>::Field,
    device_field: &<B::Device as Device>::Field,
    init: crate::nn::init::Init,
) -> Result<<B as crate::tensor::backend::VariableBackend>::Var<K>>
where
    B: ParameterInit<K>,
{
    let context = InitContext::new(ParameterRole::Other);
    let plan = init.plan(context)?;
    execute_plan_raw::<B, K>(dims, dtype_field, device_field, plan)
}

/// A parameter storing an underlying backend variable that supports gradient computation.
pub struct Param<
    S: Shape,
    B: crate::tensor::backend::VariableBackend,
    K: DType = f32,
    Train: TrainState = Trainable,
> {
    pub(crate) inner: <B as crate::tensor::backend::VariableBackend>::Var<K>,
    pub(crate) pending_state: Option<(B::Storage<K>, B::Storage<K>)>,
    /// Private identity copied by `Clone` only when the backend explicitly
    /// declares cloned handles to share a mutable variable slot.
    pub(crate) state_slot: Option<usize>,
    pub(crate) state_committed: bool,
    pub(crate) _shape: ShapeValue<S>,
    pub(crate) _dtype: K::Field,
    pub(crate) _device: <B::Device as Device>::Field,
    pub(crate) _train: PhantomData<Train>,
}

impl<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState> Clone
    for Param<S, B, K, Train>
{
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            pending_state: None,
            state_slot: self.state_slot,
            state_committed: false,
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
            _train: PhantomData,
        }
    }
}

impl<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    core::fmt::Debug for Param<S, B, K, Train>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Param").field("inner", &"...").finish()
    }
}

impl<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    Param<S, B, K, Train>
{
    /// Returns the runtime descriptor carried by this parameter.
    pub(crate) fn dtype_descriptor(&self) -> crate::tensor::dtype::DTypeDescriptor {
        K::descriptor(&self._dtype)
    }

    /// Exposes the backend variable only to optimizer-owned collection code.
    pub(crate) fn variable_any(&self) -> &dyn core::any::Any {
        &self.inner
    }

    /// Checked constructor boundary from parts.
    pub fn from_parts_checked(
        raw_var: <B as crate::tensor::backend::VariableBackend>::Var<K>,
        shape: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
    ) -> Result<Self>
    where
        B: SupportsDType<K>,
    {
        let shape_value = ShapeValue::<S>::try_new(shape.clone()).map_err(Error::Shape)?;
        let storage = B::var_as_tensor::<K>(&raw_var)?;
        let meta = B::metadata(&storage);

        let expected_shape = shape.as_ref().to_vec();
        if meta.shape.as_ref() != expected_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                op: "Param::from_parts_checked",
                expected: expected_shape,
                got: meta.shape.as_ref().to_vec(),
                msg: alloc::string::ToString::to_string("Parameter shape mismatch"),
            });
        }

        let incin_device = <B::Device as Device>::to_incin(&device)?;
        let expected_dtype = B::resolve_dtype(&dtype, &incin_device)?;
        if meta.dtype != expected_dtype {
            return Err(Error::DTypeStorageMismatch {
                expected: expected_dtype,
                got: meta.dtype,
            });
        }

        if meta.device != incin_device {
            return Err(Error::DeviceStorageMismatch {
                expected: incin_device,
                got: meta.device,
            });
        }

        let state_slot = B::var_slot_identity(&raw_var);
        Ok(Self {
            inner: raw_var,
            pending_state: None,
            state_slot,
            state_committed: false,
            _shape: shape_value,
            _dtype: dtype,
            _device: device,
            _train: PhantomData,
        })
    }

    /// Extract a functional Tensor from this variable for forward passes.
    pub fn as_tensor(&self) -> Result<Tensor<S, B, K, Train::TensorGrad>> {
        let inner_tensor = B::var_as_tensor(&self.inner)?;
        Ok(Tensor {
            inner: inner_tensor,
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
            _grad: <Train::TensorGrad as RequiresGrad>::init(()),
            _placement: PhantomData,
        })
    }

    /// Freezes this parameter so it does not participate in optimizer updates.
    pub fn freeze(self) -> Param<S, B, K, Frozen> {
        Param {
            inner: self.inner,
            pending_state: None,
            state_slot: self.state_slot,
            state_committed: false,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: self._device,
            _train: PhantomData,
        }
    }

    /// Unfreezes this parameter so it participates in optimizer updates.
    pub fn unfreeze(self) -> Param<S, B, K, Trainable> {
        Param {
            inner: self.inner,
            pending_state: None,
            state_slot: self.state_slot,
            state_committed: false,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: self._device,
            _train: PhantomData,
        }
    }
}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    Param<S, B, K, Train>
{
    /// Returns the shape dimensions of this parameter.
    pub fn shape_dims(&self) -> Vec<usize> {
        self._shape.shape_buf().as_ref().to_vec()
    }
}

impl<S, B, K, Train> Param<S, B, K, Train>
where
    S: Shape,
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + crate::exec::Capabilities
        + HostInterop,
    K: DType<Arg = ()>,
    Train: TrainState,
{
    pub(crate) fn snapshot_state_value(
        &self,
        path: &crate::nn::StatePath,
    ) -> Result<crate::nn::StateValue> {
        let storage = B::var_as_tensor::<K>(&self.inner)?;
        let bytes = B::to_bytes::<K>(&storage)?;
        crate::nn::StateValue::new(
            self._shape.shape_buf().clone(),
            K::descriptor(&self._dtype),
            bytes,
            crate::nn::StateRole::Parameter,
        )
        .map_err(|error| Error::InvalidModuleState {
            operation: "snapshot parameter",
            reason: ErrorMessage::new(format!("{path}: {error}")),
        })
    }

    pub(crate) fn prepare_state_value(
        &mut self,
        path: &crate::nn::StatePath,
        snapshot: &crate::nn::StateSnapshot,
    ) -> Result<()> {
        let value = snapshot
            .get(path)
            .ok_or_else(|| Error::InvalidModuleState {
                operation: "prepare parameter",
                reason: ErrorMessage::new(format!("missing state path {path}")),
            })?;
        let expected_dtype = K::descriptor(&self._dtype);
        if value.role() != crate::nn::StateRole::Parameter
            || value.shape() != self._shape.shape_buf()
            || value.dtype() != expected_dtype
        {
            return Err(Error::InvalidModuleState {
                operation: "prepare parameter",
                reason: ErrorMessage::new(format!("shape or dtype mismatch at {path}")),
            });
        }
        let device = <B::Device as Device>::to_incin(&self._device)?;
        let storage =
            B::from_bytes::<K>(value.bytes(), value.shape().dims(), expected_dtype, &device)?;
        let original = B::var_as_tensor::<K>(&self.inner)?;
        self.pending_state = Some((original, storage));
        Ok(())
    }

    pub(crate) fn commit_prepared_state(&mut self) -> Result<()> {
        if let Some((_, candidate)) = &self.pending_state {
            B::assign_var::<K>(&mut self.inner, candidate)?;
            self.state_committed = true;
        }
        Ok(())
    }

    pub(crate) fn rollback_prepared_state(&mut self) -> Result<()> {
        if self.state_committed {
            if let Some((original, _)) = &self.pending_state {
                B::assign_var::<K>(&mut self.inner, original)?;
            }
            self.state_committed = false;
        }
        Ok(())
    }

    pub(crate) fn clear_prepared_state(&mut self) {
        self.pending_state = None;
        self.state_committed = false;
    }

    /// Clears staging only after a successful rollback (or before any commit).
    /// A failed restore must retain its original snapshot and committed marker
    /// so transaction failure reporting never claims the leaf was restored.
    pub(crate) fn clear_prepared_state_if_rolled_back(&mut self) {
        if !self.state_committed {
            self.clear_prepared_state();
        }
    }

    pub(crate) fn state_slot_identity(&self) -> Option<usize> {
        self.state_slot
    }
}

impl<
    S: Shape,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    Train: TrainState,
    NewD: crate::tensor::device::Device,
> crate::tensor::transfer::ToDevice<B, NewD> for Param<S, B, K, Train>
where
    B: VariableTransfer<NewD>,
    <B as VariableTransfer<NewD>>::VariableOutput: SupportsDType<K>,
{
    type Output = Param<S, <B as VariableTransfer<NewD>>::VariableOutput, K, Train>;
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let inner = B::transfer_var::<K>(&self.inner, &self._dtype, &field)?;
        let state_slot = <B as VariableTransfer<NewD>>::VariableOutput::var_slot_identity(&inner);
        Ok(Param {
            inner,
            pending_state: None,
            state_slot,
            state_committed: false,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: field,
            _train: PhantomData,
        })
    }
}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + SupportsDType<K> + ParameterInit<K>,
    K: DType,
    Train: TrainState,
> Param<S, B, K, Train>
where
    (S, K, B::Device, Grad): TensorArgs<S, K, B::Device, Grad>,
{
    /// Allocates storage of the given shape/dtype/device and fills it according to `init`.
    #[allow(clippy::type_complexity)]
    pub fn new_init_raw(
        args: <(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args,
        init: crate::nn::init::Init,
    ) -> Result<Self>
    where
        B: SupportsDType<K>,
    {
        let (_shape, _dtype, _device, _) = <(S, K, B::Device, Grad)>::construct(args)?;
        let dims = _shape.clone();
        let inner = execute_initializer::<B, K>(dims.as_ref(), &_dtype, &_device, init)?;
        validate_initialized_var::<B, K>(&inner, &dims, &_dtype, &_device, "Param::new_init_raw")?;
        let state_slot = B::var_slot_identity(&inner);

        Ok(Self {
            inner,
            pending_state: None,
            state_slot,
            state_committed: false,
            _shape: ShapeValue::from_validated(_shape),
            _dtype,
            _device,
            _train: PhantomData,
        })
    }

    /// Same as `new_init_raw`, but accepts any argument type convertible via `ArgInto`.
    pub fn new_init<A>(args: A, init: crate::nn::init::Init) -> Result<Self>
    where
        B: SupportsDType<K>,
        A: ArgInto<<(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args>,
    {
        Self::new_init_raw(args.into_arg(), init)
    }

    /// Allocates storage of the given shape/dtype/device, filled with zero.
    #[allow(clippy::type_complexity)]
    pub fn zeros_raw(
        args: <(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args,
    ) -> Result<Self> {
        let (_shape, _dtype, _device, _) = <(S, K, B::Device, Grad)>::construct(args)?;
        let dims = _shape.clone();
        let inner = execute_plan_raw::<B, K>(
            dims.as_ref(),
            &_dtype,
            &_device,
            crate::nn::init::InitPlan::Zeros,
        )?;
        validate_initialized_var::<B, K>(&inner, &dims, &_dtype, &_device, "Param::zeros_raw")?;
        let state_slot = B::var_slot_identity(&inner);
        Ok(Self {
            inner,
            pending_state: None,
            state_slot,
            state_committed: false,
            _shape: ShapeValue::from_validated(_shape),
            _dtype,
            _device,
            _train: PhantomData,
        })
    }

    /// Same as `zeros_raw`, but accepts any argument type convertible via `ArgInto`.
    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args>,
    {
        Self::zeros_raw(args.into_arg())
    }

    /// Allocates storage of the given shape/dtype/device, filled with standard normal samples.
    #[allow(clippy::type_complexity)]
    pub fn randn<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args>,
    {
        let (_shape, _dtype, _device, _) = <(S, K, B::Device, Grad)>::construct(args.into_arg())?;
        let dims = _shape.clone();
        let inner = execute_plan_raw::<B, K>(
            dims.as_ref(),
            &_dtype,
            &_device,
            crate::nn::init::InitPlan::Normal {
                mean: 0.0,
                std: 1.0,
            },
        )?;
        validate_initialized_var::<B, K>(&inner, &dims, &_dtype, &_device, "Param::randn")?;
        let state_slot = B::var_slot_identity(&inner);
        Ok(Self {
            inner,
            pending_state: None,
            state_slot,
            state_committed: false,
            _shape: ShapeValue::from_validated(_shape),
            _dtype,
            _device,
            _train: PhantomData,
        })
    }

    /// Allocates storage of the given shape/dtype/device, filled with one.
    #[allow(clippy::type_complexity)]
    pub fn ones_raw(
        args: <(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args,
    ) -> Result<Self> {
        let (_shape, _dtype, _device, _) = <(S, K, B::Device, Grad)>::construct(args)?;
        let dims = _shape.clone();
        let inner = execute_plan_raw::<B, K>(
            dims.as_ref(),
            &_dtype,
            &_device,
            crate::nn::init::InitPlan::Ones,
        )?;
        validate_initialized_var::<B, K>(&inner, &dims, &_dtype, &_device, "Param::ones_raw")?;
        let state_slot = B::var_slot_identity(&inner);
        Ok(Self {
            inner,
            pending_state: None,
            state_slot,
            state_committed: false,
            _shape: ShapeValue::from_validated(_shape),
            _dtype,
            _device,
            _train: PhantomData,
        })
    }

    /// Same as `ones_raw`, but accepts any argument type convertible via `ArgInto`.
    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args>,
    {
        Self::ones_raw(args.into_arg())
    }

    /// Construct a Param directly from a backend's `Var<K>`.
    pub fn from_raw<A>(
        inner: <B as crate::tensor::backend::VariableBackend>::Var<K>,
        args: A,
    ) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args>,
        B: SupportsDType<K>,
    {
        let (_shape, _dtype, _device, _) = <(S, K, B::Device, Grad)>::construct(args.into_arg())?;
        let shape_value = ShapeValue::<S>::try_new(_shape.clone()).map_err(Error::Shape)?;
        let storage = B::var_as_tensor::<K>(&inner)?;
        let meta = B::metadata(&storage);
        let actual = B::shape(&storage);
        if actual.as_ref() != _shape.as_ref() {
            return Err(Error::ShapeMismatch {
                op: "Param::from_raw",
                expected: _shape.as_ref().to_vec(),
                got: actual.as_ref().to_vec(),
                msg: "Parameter storage shape mismatch".into(),
            });
        }
        let incin_device = <B::Device as Device>::to_incin(&_device)?;
        let expected_dtype = B::resolve_dtype(&_dtype, &incin_device)?;
        if meta.dtype != expected_dtype {
            return Err(Error::DTypeStorageMismatch {
                expected: expected_dtype,
                got: meta.dtype,
            });
        }
        if meta.device != incin_device {
            return Err(Error::DeviceStorageMismatch {
                expected: incin_device,
                got: meta.device,
            });
        }
        let state_slot = B::var_slot_identity(&inner);
        Ok(Self {
            inner,
            pending_state: None,
            state_slot,
            state_committed: false,
            _shape: shape_value,
            _dtype,
            _device,
            _train: PhantomData,
        })
    }

    /// Moves this parameter to the specified device, returning a new Param.
    pub fn to_device<D2: Device>(
        self,
        _device: &D2::Field,
    ) -> Result<Param<S, <B as VariableTransfer<D2>>::VariableOutput, K, Train>>
    where
        B: VariableTransfer<D2>,
        <B as VariableTransfer<D2>>::VariableOutput: SupportsDType<K>,
    {
        let new_inner = B::transfer_var::<K>(&self.inner, &self._dtype, _device)?;
        let state_slot = <B as VariableTransfer<D2>>::VariableOutput::var_slot_identity(&new_inner);
        Ok(Param {
            inner: new_inner,
            pending_state: None,
            state_slot,
            state_committed: false,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: _device.clone(),
            _train: PhantomData,
        })
    }
}

impl<
    S: Shape,
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + crate::exec::Capabilities
        + HostInterop,
    K: DType<Arg = ()>,
    Train: TrainState,
> crate::nn::VisitState<B> for Param<S, B, K, Train>
{
    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        visitor.visit_param(path, self)
    }
}

impl<
    S: Shape,
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + crate::exec::Capabilities
        + HostInterop,
    K: DType<Arg = ()>,
    Train: TrainState,
> crate::nn::VisitStateMut<B> for Param<S, B, K, Train>
{
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        visitor.visit_param(path, self)
    }
}

impl<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    crate::nn::VisitParameters<B> for Param<S, B, K, Train>
{
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        if Train::TRAINABLE {
            visitor.visit_param(path, self)?;
        }
        Ok(())
    }
}

impl<S1: DynShape, B: crate::tensor::backend::VariableBackend, K: DType, Train: TrainState>
    Param<S1, B, K, Train>
{
    /// Reinterprets a dynamically-shaped parameter as a statically-shaped `S2`.
    pub fn into_shape<S2: Shape>(self) -> Result<Param<S2, B, K, Train>>
    where
        B: crate::tensor::backend::VariableBackend,
    {
        let current_dims = self._shape.shape_buf();
        let new_shape = S2::try_from_dims(current_dims.as_ref()).map_err(Error::Shape)?;

        let shape_value = ShapeValue::<S2>::try_new(new_shape).map_err(Error::Shape)?;

        Ok(Param::<S2, B, K, Train> {
            inner: self.inner,
            pending_state: None,
            state_slot: self.state_slot,
            state_committed: false,
            _shape: shape_value,
            _dtype: self._dtype,
            _device: self._device,
            _train: PhantomData,
        })
    }
}

/// A non-trainable state buffer.
pub struct Buffer<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType = f32> {
    pub(crate) inner: <B as crate::tensor::backend::VariableBackend>::Var<K>,
    pub(crate) pending_state: Option<(B::Storage<K>, B::Storage<K>)>,
    pub(crate) state_committed: bool,
    pub(crate) _shape: ShapeValue<S>,
    pub(crate) _dtype: K::Field,
    pub(crate) _device: <B::Device as Device>::Field,
}

impl<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType> Clone for Buffer<S, B, K> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            pending_state: None,
            state_committed: false,
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
        }
    }
}

impl<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType> core::fmt::Debug
    for Buffer<S, B, K>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Buffer").field("inner", &"...").finish()
    }
}

impl<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType> Buffer<S, B, K> {
    /// Checked constructor boundary from parts.
    pub fn from_parts_checked(
        raw_var: <B as crate::tensor::backend::VariableBackend>::Var<K>,
        shape: ShapeBuf,
        dtype: K::Field,
        device: <B::Device as Device>::Field,
    ) -> Result<Self>
    where
        B: SupportsDType<K>,
    {
        let shape_value = ShapeValue::<S>::try_new(shape.clone()).map_err(Error::Shape)?;
        let storage = B::var_as_tensor::<K>(&raw_var)?;
        let meta = B::metadata(&storage);

        let expected_shape = shape.as_ref().to_vec();
        if meta.shape.as_ref() != expected_shape.as_slice() {
            return Err(Error::ShapeMismatch {
                op: "Buffer::from_parts_checked",
                expected: expected_shape,
                got: meta.shape.as_ref().to_vec(),
                msg: alloc::string::ToString::to_string("Buffer shape mismatch"),
            });
        }

        let incin_device = <B::Device as Device>::to_incin(&device)?;
        let expected_dtype = B::resolve_dtype(&dtype, &incin_device)?;
        if meta.dtype != expected_dtype {
            return Err(Error::DTypeStorageMismatch {
                expected: expected_dtype,
                got: meta.dtype,
            });
        }

        if meta.device != incin_device {
            return Err(Error::DeviceStorageMismatch {
                expected: incin_device,
                got: meta.device,
            });
        }

        Ok(Self {
            inner: raw_var,
            pending_state: None,
            state_committed: false,
            _shape: shape_value,
            _dtype: dtype,
            _device: device,
        })
    }

    pub fn as_tensor(&self) -> Result<Tensor<S, B, K, NoGrad>> {
        let inner_tensor = B::var_as_tensor(&self.inner)?;
        Ok(Tensor {
            inner: inner_tensor,
            _shape: self._shape.clone(),
            _dtype: self._dtype.clone(),
            _device: self._device.clone(),
            _grad: PhantomData,
            _placement: PhantomData,
        })
    }
}

impl<S: Shape + DynShape, B: crate::tensor::backend::VariableBackend, K: DType> Buffer<S, B, K> {
    pub fn shape_dims(&self) -> Vec<usize> {
        self._shape.shape_buf().as_ref().to_vec()
    }
}

impl<S, B, K> Buffer<S, B, K>
where
    S: Shape,
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + crate::exec::Capabilities
        + HostInterop,
    K: DType<Arg = ()>,
{
    pub(crate) fn prepare_state_value(
        &mut self,
        path: &crate::nn::StatePath,
        snapshot: &crate::nn::StateSnapshot,
    ) -> Result<()> {
        let value = snapshot
            .get(path)
            .ok_or_else(|| Error::InvalidModuleState {
                operation: "prepare buffer",
                reason: ErrorMessage::new(format!("missing state path {path}")),
            })?;
        let expected_dtype = K::descriptor(&self._dtype);
        if value.role() != crate::nn::StateRole::Buffer
            || value.shape() != self._shape.shape_buf()
            || value.dtype() != expected_dtype
        {
            return Err(Error::InvalidModuleState {
                operation: "prepare buffer",
                reason: ErrorMessage::new(format!("shape or dtype mismatch at {path}")),
            });
        }
        let device = <B::Device as Device>::to_incin(&self._device)?;
        let storage =
            B::from_bytes::<K>(value.bytes(), value.shape().dims(), expected_dtype, &device)?;
        let original = B::var_as_tensor::<K>(&self.inner)?;
        self.pending_state = Some((original, storage));
        Ok(())
    }

    pub(crate) fn commit_prepared_state(&mut self) -> Result<()> {
        if let Some((_, candidate)) = &self.pending_state {
            B::assign_var::<K>(&mut self.inner, candidate)?;
            self.state_committed = true;
        }
        Ok(())
    }

    pub(crate) fn rollback_prepared_state(&mut self) -> Result<()> {
        if self.state_committed {
            if let Some((original, _)) = &self.pending_state {
                B::assign_var::<K>(&mut self.inner, original)?;
            }
            self.state_committed = false;
        }
        Ok(())
    }

    pub(crate) fn clear_prepared_state(&mut self) {
        self.pending_state = None;
        self.state_committed = false;
    }

    /// See [`Param::clear_prepared_state_if_rolled_back`].
    pub(crate) fn clear_prepared_state_if_rolled_back(&mut self) {
        if !self.state_committed {
            self.clear_prepared_state();
        }
    }
}

impl<
    S: Shape,
    B: crate::tensor::backend::VariableBackend,
    K: DType,
    NewD: crate::tensor::device::Device,
> crate::tensor::transfer::ToDevice<B, NewD> for Buffer<S, B, K>
where
    B: VariableTransfer<NewD>,
    <B as VariableTransfer<NewD>>::VariableOutput: SupportsDType<K>,
{
    type Output = Buffer<S, <B as VariableTransfer<NewD>>::VariableOutput, K>;
    fn to_device(self, arg: &NewD::Arg) -> Result<Self::Output> {
        let field = NewD::init(arg.clone());
        let inner = B::transfer_var::<K>(&self.inner, &self._dtype, &field)?;
        Ok(Buffer {
            inner,
            pending_state: None,
            state_committed: false,
            _shape: self._shape,
            _dtype: self._dtype,
            _device: field,
        })
    }
}

impl<
    S: Shape + DynShape,
    B: crate::tensor::backend::VariableBackend + SupportsDType<K> + ParameterInit<K>,
    K: DType,
> Buffer<S, B, K>
where
    (S, K, B::Device, Grad): TensorArgs<S, K, B::Device, Grad>,
{
    #[allow(clippy::type_complexity)]
    pub fn new_init_raw(
        args: <(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args,
        init: crate::nn::init::Init,
    ) -> Result<Self>
    where
        B: SupportsDType<K>,
    {
        let (_shape, _dtype, _device, _) = <(S, K, B::Device, Grad)>::construct(args)?;
        let dims = _shape.clone();
        let inner = execute_initializer::<B, K>(dims.as_ref(), &_dtype, &_device, init)?;
        validate_initialized_var::<B, K>(&inner, &dims, &_dtype, &_device, "Buffer::new_init_raw")?;

        Ok(Self {
            inner,
            pending_state: None,
            state_committed: false,
            _shape: ShapeValue::from_validated(_shape),
            _dtype,
            _device,
        })
    }

    pub fn new_init<A>(args: A, init: crate::nn::init::Init) -> Result<Self>
    where
        B: SupportsDType<K>,
        A: ArgInto<<(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args>,
    {
        Self::new_init_raw(args.into_arg(), init)
    }

    #[allow(clippy::type_complexity)]
    pub fn zeros_raw(
        args: <(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args,
    ) -> Result<Self> {
        let (_shape, _dtype, _device, _) = <(S, K, B::Device, Grad)>::construct(args)?;
        let dims = _shape.clone();
        let inner = execute_plan_raw::<B, K>(
            dims.as_ref(),
            &_dtype,
            &_device,
            crate::nn::init::InitPlan::Zeros,
        )?;
        validate_initialized_var::<B, K>(&inner, &dims, &_dtype, &_device, "Buffer::zeros_raw")?;
        Ok(Self {
            inner,
            pending_state: None,
            state_committed: false,
            _shape: ShapeValue::from_validated(_shape),
            _dtype,
            _device,
        })
    }

    pub fn zeros<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args>,
    {
        Self::zeros_raw(args.into_arg())
    }

    #[allow(clippy::type_complexity)]
    pub fn ones_raw(
        args: <(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args,
    ) -> Result<Self> {
        let (_shape, _dtype, _device, _) = <(S, K, B::Device, Grad)>::construct(args)?;
        let dims = _shape.clone();
        let inner = execute_plan_raw::<B, K>(
            dims.as_ref(),
            &_dtype,
            &_device,
            crate::nn::init::InitPlan::Ones,
        )?;
        validate_initialized_var::<B, K>(&inner, &dims, &_dtype, &_device, "Buffer::ones_raw")?;
        Ok(Self {
            inner,
            pending_state: None,
            state_committed: false,
            _shape: ShapeValue::from_validated(_shape),
            _dtype,
            _device,
        })
    }

    pub fn ones<A>(args: A) -> Result<Self>
    where
        A: ArgInto<<(S, K, B::Device, Grad) as TensorArgs<S, K, B::Device, Grad>>::Args>,
    {
        Self::ones_raw(args.into_arg())
    }
}

impl<
    S: Shape,
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + crate::exec::Capabilities
        + HostInterop,
    K: DType<Arg = ()>,
> crate::nn::VisitState<B> for Buffer<S, B, K>
{
    fn visit_state<V: crate::nn::StateVisitor<B>>(
        &self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        visitor.visit_buffer(path, self)
    }
}

impl<
    S: Shape,
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + crate::exec::Capabilities
        + HostInterop,
    K: DType<Arg = ()>,
> crate::nn::VisitStateMut<B> for Buffer<S, B, K>
{
    fn visit_state_mut<V: crate::nn::StateMutVisitor<B>>(
        &mut self,
        path: &crate::nn::StatePath,
        visitor: &mut V,
    ) -> Result<()> {
        visitor.visit_buffer(path, self)
    }
}

impl<S, B, K> Buffer<S, B, K>
where
    S: Shape,
    B: crate::tensor::backend::VariableBackend
        + SupportsDType<K>
        + crate::exec::Capabilities
        + HostInterop,
    K: DType<Arg = ()>,
{
    pub(crate) fn snapshot_state_value(
        &self,
        path: &crate::nn::StatePath,
    ) -> Result<crate::nn::StateValue> {
        let storage = B::var_as_tensor::<K>(&self.inner)?;
        let bytes = B::to_bytes::<K>(&storage)?;
        crate::nn::StateValue::new(
            self._shape.shape_buf().clone(),
            K::descriptor(&self._dtype),
            bytes,
            crate::nn::StateRole::Buffer,
        )
        .map_err(|error| Error::InvalidModuleState {
            operation: "snapshot buffer",
            reason: ErrorMessage::new(format!("{path}: {error}")),
        })
    }
}

impl<S: Shape, B: crate::tensor::backend::VariableBackend, K: DType> crate::nn::VisitParameters<B>
    for Buffer<S, B, K>
{
    fn visit_parameters<V: crate::nn::ParameterVisitor<B>>(
        &self,
        _path: &crate::nn::StatePath,
        _visitor: &mut V,
    ) -> Result<()> {
        Ok(())
    }
}
