//! The `TensorTarget`/`DtypeTarget` abstraction: a place tensors can be
//! allocated, and rebinding it to a different dtype.

use super::*;

/// A place tensors and parameters can be allocated: a device selector plus the
/// float dtype generated allocations should use.
///
/// Implemented by the device values themselves at their default float
/// (`f32`), and by [`DtypeView`] for every other dtype.
pub trait TensorTarget {
    /// The dtype generated tensors are created at.
    type Dtype: DType;

    /// The dtype stored layer parameters are created at.
    type ParameterDtype: DType;

    /// The device selector this target allocates on.
    type Device: Device;

    /// The backend family owned by this target.
    type Backend: Backend<Device = Self::Device> + VariableBackend;

    /// The selector value the device's own constructor argument needs.
    fn device_arg(&self) -> <Self::Device as Device>::Arg;

    /// Generated dtype field.
    fn dtype_field(&self) -> <Self::Dtype as DType>::Field;

    /// Parameter dtype field.
    fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field;

    /// Precision policy.
    fn precision_policy(&self) -> RuntimePrecisionPolicy;
}

/// The backend a target resolves to for element type `K`. Users never write this type.
pub type TargetBackendFor<T> = <T as TensorTarget>::Backend;

/// The default backend a target resolves to for its default generated dtype. Users never write this type.
pub type TargetBackend<T> = <T as TensorTarget>::Backend;

/// A tensor allocated on target `T` from data of element type `K`.
///
/// The layout slot takes its default, so this names a tensor that claims
/// nothing about its memory order -- which is what the plain constructors
/// return. [`TargetTensorIn`] is the spelling for one that claims something.
pub type TargetTensor<T, S, K> = Tensor<S, TargetBackend<T>, K, NoGrad>;

/// A tensor allocated on target `T` in a named layout.
///
/// The counterpart to [`TargetTensor`] for the `*_in` constructors, which
/// allocate with the strides `L` asks for and so are entitled to return a
/// tensor typed `L`.
///
/// It is a separate alias rather than a defaulted parameter on `TargetTensor`
/// because the constructors have to stay inferrable. A generic parameter on a
/// function cannot have a default, so widening `zeros` to be generic over `L`
/// would make `target.zeros(..)` ambiguous at every existing call site -- the
/// same reason `scaled_dot_product_attention` had to tie its `q` operand to
/// the impl block's layout parameter.
pub type TargetTensorIn<T, S, K, L> =
    Tensor<S, TargetBackend<T>, K, NoGrad, incin_core::dist::Local, L>;

/// The same for a gradient-tracking parameter.
///
/// A parameter is allocated the same way a buffer is and then marked, and
/// marking carries the layout because it re-tags the autograd identity without
/// touching the storage. So a parameter is entitled to the same proof its
/// allocation earned.
pub type TargetTensorInGrad<T, S, K, L> =
    Tensor<S, TargetBackend<T>, K, incin_core::tensor::grad::Grad, incin_core::dist::Local, L>;

/// Rebinding the dtype a target generates.
pub trait DtypeTarget: TensorTarget + Sized + Clone {
    /// Rebinds this target to generate `K` instead of `Self::Dtype`.
    fn dtype<K: ConstDType>(&self) -> Result<DtypeView<Self, K>>
    where
        Self::Backend: SupportsDType<K>,
    {
        let device =
            <Self::Device as Device>::to_incin(&<Self::Device as Device>::init(self.device_arg()))?;
        let field = <K as DType>::init(());
        <Self::Backend as SupportsDType<K>>::resolve_dtype(&field, &device)?;
        Ok(DtypeView::new(self.clone(), field))
    }

    /// Rebinds this target to generate a dynamic runtime descriptor `Dyn`.
    fn dtype_dynamic(&self, descriptor: DTypeDescriptor) -> Result<DtypeView<Self, Dyn>>
    where
        Self::Backend: SupportsDType<Dyn>,
    {
        let device =
            <Self::Device as Device>::to_incin(&<Self::Device as Device>::init(self.device_arg()))?;
        let field = <Dyn as DType>::init(descriptor);
        <Self::Backend as SupportsDType<Dyn>>::resolve_dtype(&field, &device)?;
        Ok(DtypeView::new(self.clone(), field))
    }
}

impl<T: TensorTarget + Sized + Clone> DtypeTarget for T {}

/// A target bound to an explicit dtype `K`, delegating backend selection to the underlying target `T`.
#[derive(Debug, Clone)]
pub struct DtypeView<T, K: DType> {
    target: T,
    field: K::Field,
}

impl<T, K: DType> DtypeView<T, K> {
    pub(crate) const fn new(target: T, field: K::Field) -> Self {
        Self { target, field }
    }

    /// Borrows the placed backend value.
    pub fn target(&self) -> &T {
        &self.target
    }
}

impl<T: TensorTarget, K: DType> TensorTarget for DtypeView<T, K> {
    type Dtype = K;
    type ParameterDtype = T::ParameterDtype;
    type Device = T::Device;
    type Backend = T::Backend;

    fn device_arg(&self) -> <Self::Device as Device>::Arg {
        self.target.device_arg()
    }

    fn dtype_field(&self) -> <K as DType>::Field {
        self.field.clone()
    }

    fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
        self.target.parameter_dtype_field()
    }

    fn precision_policy(&self) -> RuntimePrecisionPolicy {
        self.target.precision_policy()
    }
}
