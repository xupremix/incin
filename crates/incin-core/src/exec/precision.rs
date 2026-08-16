//! Floating-point precision policies and loss scaling for mixed-precision training.

use crate::err::{Error, Result};
use crate::exec::{LayoutClass, MathMode};
use crate::shapes::Dyn;
use crate::shapes::error::OperationKind;
use crate::tensor::dtype::{BuiltinDType, ConstDType, DType, DTypeDescriptor, FloatDType};
use crate::tensor::dtype::{bf16, f16};

/// Role of precision choice being queried (compute or accumulator).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecisionRole {
    Compute,
    Accumulator,
}

/// Loss scale state tracking current multiplier and overflow history.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LossScaleState {
    policy: LossScaling,
    scale: f32,
    steps_since_last_overflow: usize,
}

impl LossScaleState {
    /// Creates a new loss scale state initialized from policy.
    #[must_use]
    pub fn new(policy: LossScaling) -> Self {
        let scale = policy.initial_scale();
        Self {
            policy,
            scale,
            steps_since_last_overflow: 0,
        }
    }

    /// Returns the underlying loss scaling configuration policy.
    #[must_use]
    pub const fn policy(&self) -> LossScaling {
        self.policy
    }

    /// Returns the current scale factor.
    #[must_use]
    pub const fn scale(&self) -> f32 {
        self.scale
    }

    /// Returns the number of consecutive finite steps recorded since last overflow.
    #[must_use]
    pub const fn steps_since_last_overflow(&self) -> usize {
        self.steps_since_last_overflow
    }

    /// Updates dynamic loss scaling based on whether non-finite (NaN/Inf) values were found.
    pub fn update(&mut self, found_nan_or_inf: bool) {
        if let LossScaling::Dynamic {
            growth_factor,
            backoff_factor,
            growth_interval,
            ..
        } = self.policy
        {
            if found_nan_or_inf {
                self.scale = (self.scale * backoff_factor).max(1.0);
                self.steps_since_last_overflow = 0;
            } else {
                self.steps_since_last_overflow += 1;
                if self.steps_since_last_overflow >= growth_interval {
                    self.scale *= growth_factor;
                    self.steps_since_last_overflow = 0;
                }
            }
        }
    }
}

/// Loss scaling configuration for mixed-precision numerical stability.
#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub enum LossScaling {
    /// No loss scaling applied.
    #[default]
    None,
    /// Static loss scaling with a fixed multiplier.
    Static(f32),
    /// Dynamic loss scaling that adjusts based on non-finite gradient detection.
    Dynamic {
        /// Initial scale factor.
        initial_scale: f32,
        /// Factor by which to multiply scale when no non-finite gradients occur.
        growth_factor: f32,
        /// Factor by which to multiply scale when a non-finite gradient occurs.
        backoff_factor: f32,
        /// Number of consecutive finite steps required before growing scale.
        growth_interval: usize,
    },
}

impl LossScaling {
    /// Creates a static loss scaling policy.
    #[must_use]
    pub const fn static_scale(scale: f32) -> Self {
        Self::Static(scale)
    }

    /// Creates a dynamic loss scaling policy with recommended defaults.
    #[must_use]
    pub const fn dynamic_default() -> Self {
        Self::Dynamic {
            initial_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
        }
    }

    /// Creates a custom dynamic loss scaling policy.
    #[must_use]
    pub const fn dynamic(
        initial_scale: f32,
        growth_factor: f32,
        backoff_factor: f32,
        growth_interval: usize,
    ) -> Self {
        Self::Dynamic {
            initial_scale,
            growth_factor,
            backoff_factor,
            growth_interval,
        }
    }

    /// Returns the initial scale factor.
    #[must_use]
    pub fn initial_scale(&self) -> f32 {
        match *self {
            Self::None => 1.0,
            Self::Static(scale) => scale,
            Self::Dynamic { initial_scale, .. } => initial_scale,
        }
    }
}

impl core::hash::Hash for LossScaling {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match *self {
            Self::None => {}
            Self::Static(s) => s.to_bits().hash(state),
            Self::Dynamic {
                initial_scale,
                growth_factor,
                backoff_factor,
                growth_interval,
            } => {
                initial_scale.to_bits().hash(state);
                growth_factor.to_bits().hash(state);
                backoff_factor.to_bits().hash(state);
                growth_interval.hash(state);
            }
        }
    }
}

impl Eq for LossScaling {}

/// Explicit precision choice for compute and accumulator attributes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PrecisionChoice {
    /// Use the backend's truthful native arithmetic.
    #[default]
    Native,
    /// Require exact DTypeDescriptor for this role.
    Exact(DTypeDescriptor),
}

/// Immutable runtime precision policy governing datatype defaults and choices.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RuntimePrecisionPolicy {
    pub(crate) generated: DTypeDescriptor,
    pub(crate) parameter: DTypeDescriptor,
    pub(crate) active_dtype: Option<DTypeDescriptor>,
    pub(crate) compute: PrecisionChoice,
    pub(crate) accumulator: PrecisionChoice,
    pub(crate) loss_scaling: LossScaling,
}

impl core::default::Default for RuntimePrecisionPolicy {
    fn default() -> Self {
        Self::fp32()
    }
}

impl RuntimePrecisionPolicy {
    /// Standard full-precision float32 policy.
    #[must_use]
    pub const fn fp32() -> Self {
        Self {
            generated: <f32 as ConstDType>::DESCRIPTOR,
            parameter: <f32 as ConstDType>::DESCRIPTOR,
            active_dtype: None,
            compute: PrecisionChoice::Native,
            accumulator: PrecisionChoice::Native,
            loss_scaling: LossScaling::None,
        }
    }

    /// Standard double-precision float64 policy.
    #[must_use]
    pub const fn fp64() -> Self {
        Self::exact_dtype(<f64 as ConstDType>::DESCRIPTOR)
    }

    /// Policy requiring exact dtype descriptor for all operations.
    #[must_use]
    pub const fn exact_dtype(dtype: DTypeDescriptor) -> Self {
        Self {
            generated: dtype,
            parameter: dtype,
            active_dtype: Some(dtype),
            compute: PrecisionChoice::Exact(dtype),
            accumulator: PrecisionChoice::Exact(dtype),
            loss_scaling: LossScaling::None,
        }
    }

    /// Policy requiring exact float dtype `K`.
    pub fn exact<K: ConstDType + FloatDType>() -> Self {
        Self::exact_dtype(K::DESCRIPTOR)
    }

    /// Mixed-precision float16 policy.
    #[must_use]
    pub const fn mixed_f16() -> Self {
        Self {
            generated: <f16 as ConstDType>::DESCRIPTOR,
            parameter: <f16 as ConstDType>::DESCRIPTOR,
            active_dtype: Some(<f16 as ConstDType>::DESCRIPTOR),
            compute: PrecisionChoice::Native,
            accumulator: PrecisionChoice::Exact(<f32 as ConstDType>::DESCRIPTOR),
            loss_scaling: LossScaling::dynamic_default(),
        }
    }

    /// Mixed-precision bfloat16 policy.
    #[must_use]
    pub const fn mixed_bf16() -> Self {
        Self {
            generated: <bf16 as ConstDType>::DESCRIPTOR,
            parameter: <bf16 as ConstDType>::DESCRIPTOR,
            active_dtype: Some(<bf16 as ConstDType>::DESCRIPTOR),
            compute: PrecisionChoice::Native,
            accumulator: PrecisionChoice::Exact(<f32 as ConstDType>::DESCRIPTOR),
            loss_scaling: LossScaling::None,
        }
    }

    /// Returns generated tensor default dtype descriptor.
    #[must_use]
    pub const fn generated(&self) -> DTypeDescriptor {
        self.generated
    }

    /// Returns parameter default dtype descriptor.
    #[must_use]
    pub const fn parameter(&self) -> DTypeDescriptor {
        self.parameter
    }

    /// Returns the target active dtype descriptor if constrained to a specific storage dtype.
    pub const fn active_dtype(&self) -> Option<DTypeDescriptor> {
        self.active_dtype
    }

    /// Returns compute precision choice.
    #[must_use]
    pub const fn compute(&self) -> PrecisionChoice {
        self.compute
    }

    /// Returns accumulator precision choice.
    #[must_use]
    pub const fn accumulator(&self) -> PrecisionChoice {
        self.accumulator
    }

    /// Returns loss scaling configuration.
    #[must_use]
    pub const fn loss_scaling(&self) -> LossScaling {
        self.loss_scaling
    }

    /// Sets compute precision choice.
    #[must_use]
    pub const fn with_compute(mut self, compute: PrecisionChoice) -> Self {
        self.compute = compute;
        self
    }

    /// Sets accumulator precision choice.
    #[must_use]
    pub const fn with_accumulator(mut self, accumulator: PrecisionChoice) -> Self {
        self.accumulator = accumulator;
        self
    }
}

/// Precision request passed during operation resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PrecisionRequest {
    pub operation: OperationKind,
    pub storage: DTypeDescriptor,
    pub output: DTypeDescriptor,
    pub layout: LayoutClass,
    pub rank: usize,
    pub training: bool,
    pub math_mode: MathMode,
}

impl PrecisionRequest {
    /// Creates a precision request for an operation.
    #[must_use]
    pub const fn new(
        operation: OperationKind,
        storage: DTypeDescriptor,
        output: DTypeDescriptor,
        layout: LayoutClass,
        rank: usize,
        training: bool,
        math_mode: MathMode,
    ) -> Self {
        Self {
            operation,
            storage,
            output,
            layout,
            rank,
            training,
            math_mode,
        }
    }
}

impl core::default::Default for PrecisionRequest {
    fn default() -> Self {
        Self {
            operation: OperationKind::Pointwise,
            storage: <f32 as ConstDType>::DESCRIPTOR,
            output: <f32 as ConstDType>::DESCRIPTOR,
            layout: LayoutClass::Contiguous,
            rank: 1,
            training: false,
            math_mode: MathMode::Fast,
        }
    }
}

/// Concrete resolved data types for storage, compute, accumulation, and output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResolvedPrecision {
    pub storage: DTypeDescriptor,
    pub compute: DTypeDescriptor,
    pub accumulator: DTypeDescriptor,
    pub output: DTypeDescriptor,
    pub loss_scaling: LossScaling,
}

impl ResolvedPrecision {
    /// Creates a resolved precision contract.
    #[must_use]
    pub const fn new(
        storage: DTypeDescriptor,
        compute: DTypeDescriptor,
        accumulator: DTypeDescriptor,
        output: DTypeDescriptor,
        loss_scaling: LossScaling,
    ) -> Self {
        Self {
            storage,
            compute,
            accumulator,
            output,
            loss_scaling,
        }
    }

    /// Storage dtype descriptor.
    #[must_use]
    pub const fn storage(&self) -> DTypeDescriptor {
        self.storage
    }

    /// Compute dtype descriptor.
    #[must_use]
    pub const fn compute(&self) -> DTypeDescriptor {
        self.compute
    }

    /// Accumulator dtype descriptor.
    #[must_use]
    pub const fn accumulator(&self) -> DTypeDescriptor {
        self.accumulator
    }

    /// Output dtype descriptor.
    #[must_use]
    pub const fn output(&self) -> DTypeDescriptor {
        self.output
    }

    /// Loss scaling configuration.
    #[must_use]
    pub const fn loss_scaling(&self) -> LossScaling {
        self.loss_scaling
    }
}

/// Trait implemented by concrete backends to describe native arithmetic capabilities.
pub trait PrecisionCapabilities {
    /// Queries the backend's truthful native precision for `request`.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedDType`] if the storage dtype is not supported.
    fn native_precision(&self, request: &PrecisionRequest) -> Result<ResolvedPrecision>;
}

/// Authoritative resolver applying a precision policy against backend capabilities.
///
/// # Errors
///
/// Returns [`Error::UnsupportedPrecision`] if an `Exact` choice cannot be honored,
/// or [`Error::UnsupportedDType`] if the storage dtype is unsupported.
pub fn resolve_precision<B>(
    backend: &B,
    policy: RuntimePrecisionPolicy,
    request: &PrecisionRequest,
) -> Result<ResolvedPrecision>
where
    B: PrecisionCapabilities + ?Sized,
{
    let native = backend.native_precision(request)?;

    if let Some(active) = policy.active_dtype
        && request.storage != active
    {
        return Ok(ResolvedPrecision {
            storage: request.storage,
            compute: native.compute,
            accumulator: native.accumulator,
            output: request.output,
            loss_scaling: LossScaling::None,
        });
    }

    let compute = match policy.compute {
        PrecisionChoice::Native => native.compute,
        PrecisionChoice::Exact(k) => {
            if native.compute == k {
                k
            } else {
                return Err(Error::UnsupportedPrecision {
                    operation: request.operation,
                    storage: request.storage,
                    requested: k,
                    role: PrecisionRole::Compute,
                    backend: core::any::type_name::<B>(),
                });
            }
        }
    };

    let accumulator = match policy.accumulator {
        PrecisionChoice::Native => native.accumulator,
        PrecisionChoice::Exact(k) => {
            if native.accumulator == k {
                k
            } else {
                return Err(Error::UnsupportedPrecision {
                    operation: request.operation,
                    storage: request.storage,
                    requested: k,
                    role: PrecisionRole::Accumulator,
                    backend: core::any::type_name::<B>(),
                });
            }
        }
    };

    let loss_scaling = match policy.active_dtype {
        Some(active) if request.storage == active => policy.loss_scaling,
        None => policy.loss_scaling,
        Some(_) => LossScaling::None,
    };

    Ok(ResolvedPrecision {
        storage: request.storage,
        compute,
        accumulator,
        output: request.output,
        loss_scaling,
    })
}

/// Type-level precision specification trait for targets.
pub trait PrecisionSpec:
    'static + Send + Sync + Copy + core::fmt::Debug + Eq + PartialEq + core::hash::Hash
{
    /// Field type stored in `Target`.
    type Field: Clone + Send + Sync + 'static + core::fmt::Debug;

    /// Generated tensor default dtype.
    type GeneratedDType: DType;
    /// Layer parameter default dtype.
    type ParameterDType: DType;

    /// Initializes field for target.
    fn init_field(&self) -> Self::Field;

    /// Extracts generated dtype field representation.
    fn generated_dtype_field(field: &Self::Field) -> <Self::GeneratedDType as DType>::Field;

    /// Extracts parameter dtype field representation.
    fn parameter_dtype_field(field: &Self::Field) -> <Self::ParameterDType as DType>::Field;

    /// Extracts runtime policy representation.
    fn runtime_policy(field: &Self::Field) -> RuntimePrecisionPolicy;
}

/// Static policy markers.
#[allow(clippy::module_inception)]
pub mod precision {
    use super::*;

    /// Default policy marker (f32 generated, f32 parameters).
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
    pub struct Default;

    impl PrecisionSpec for Default {
        type Field = ();
        type GeneratedDType = f32;
        type ParameterDType = f32;

        fn init_field(&self) -> Self::Field {}

        fn generated_dtype_field(_field: &Self::Field) -> <Self::GeneratedDType as DType>::Field {
            <Self::GeneratedDType as DType>::init(())
        }

        fn parameter_dtype_field(_field: &Self::Field) -> <Self::ParameterDType as DType>::Field {
            <Self::ParameterDType as DType>::init(())
        }

        fn runtime_policy(_field: &Self::Field) -> RuntimePrecisionPolicy {
            RuntimePrecisionPolicy::fp32()
        }
    }

    /// bfloat16 mixed precision policy marker.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
    pub struct Bf16Mixed;

    /// Legacy compatibility alias.
    pub type MixedBf16 = Bf16Mixed;

    impl PrecisionSpec for Bf16Mixed {
        type Field = ();
        type GeneratedDType = bf16;
        type ParameterDType = bf16;

        fn init_field(&self) -> Self::Field {}

        fn generated_dtype_field(_field: &Self::Field) -> <Self::GeneratedDType as DType>::Field {
            <Self::GeneratedDType as DType>::init(())
        }

        fn parameter_dtype_field(_field: &Self::Field) -> <Self::ParameterDType as DType>::Field {
            <Self::ParameterDType as DType>::init(())
        }

        fn runtime_policy(_field: &Self::Field) -> RuntimePrecisionPolicy {
            RuntimePrecisionPolicy::mixed_bf16()
        }
    }

    /// float16 mixed precision policy marker.
    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
    pub struct F16Mixed;

    /// Legacy compatibility alias.
    pub type MixedF16 = F16Mixed;

    impl PrecisionSpec for F16Mixed {
        type Field = ();
        type GeneratedDType = f16;
        type ParameterDType = f16;

        fn init_field(&self) -> Self::Field {}

        fn generated_dtype_field(_field: &Self::Field) -> <Self::GeneratedDType as DType>::Field {
            <Self::GeneratedDType as DType>::init(())
        }

        fn parameter_dtype_field(_field: &Self::Field) -> <Self::ParameterDType as DType>::Field {
            <Self::ParameterDType as DType>::init(())
        }

        fn runtime_policy(_field: &Self::Field) -> RuntimePrecisionPolicy {
            RuntimePrecisionPolicy::mixed_f16()
        }
    }

    /// Exact precision policy marker for floating dtype `K`.
    pub struct Exact<K: BuiltinDType + FloatDType>(core::marker::PhantomData<K>);

    impl<K: BuiltinDType + FloatDType> Clone for Exact<K> {
        fn clone(&self) -> Self {
            *self
        }
    }

    impl<K: BuiltinDType + FloatDType> Copy for Exact<K> {}

    impl<K: BuiltinDType + FloatDType> core::fmt::Debug for Exact<K> {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            f.debug_struct("Exact").finish()
        }
    }

    impl<K: BuiltinDType + FloatDType> core::default::Default for Exact<K> {
        fn default() -> Self {
            Self::new()
        }
    }

    impl<K: BuiltinDType + FloatDType> PartialEq for Exact<K> {
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }

    impl<K: BuiltinDType + FloatDType> Eq for Exact<K> {}

    impl<K: BuiltinDType + FloatDType> core::hash::Hash for Exact<K> {
        fn hash<H: core::hash::Hasher>(&self, _state: &mut H) {}
    }

    impl<K: BuiltinDType + FloatDType> Exact<K> {
        /// Creates a new `Exact<K>` policy marker.
        #[must_use]
        pub const fn new() -> Self {
            Self(core::marker::PhantomData)
        }
    }

    impl<K: BuiltinDType + FloatDType> PrecisionSpec for Exact<K> {
        type Field = ();
        type GeneratedDType = K;
        type ParameterDType = K;

        fn init_field(&self) -> Self::Field {}

        fn generated_dtype_field(_field: &Self::Field) -> <K as DType>::Field {
            K::init(())
        }

        fn parameter_dtype_field(_field: &Self::Field) -> <K as DType>::Field {
            K::init(())
        }

        fn runtime_policy(_field: &Self::Field) -> RuntimePrecisionPolicy {
            RuntimePrecisionPolicy::exact::<K>()
        }
    }
}

pub use precision::{Bf16Mixed, Default, Exact, F16Mixed, MixedBf16, MixedF16};

impl PrecisionSpec for Dyn {
    type Field = RuntimePrecisionPolicy;
    type GeneratedDType = Dyn;
    type ParameterDType = Dyn;

    fn init_field(&self) -> Self::Field {
        RuntimePrecisionPolicy::default()
    }

    fn generated_dtype_field(field: &Self::Field) -> DTypeDescriptor {
        field.generated
    }

    fn parameter_dtype_field(field: &Self::Field) -> DTypeDescriptor {
        field.parameter
    }

    fn runtime_policy(field: &Self::Field) -> RuntimePrecisionPolicy {
        *field
    }
}
