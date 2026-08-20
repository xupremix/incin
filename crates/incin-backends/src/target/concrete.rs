//! `Target<E, D, P>`: an engine-aware, physical-device-placed,
//! precision-configured tensor target value.

use super::*;

/// An engine-aware, physical-device-placed, precision-configured tensor target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Target<E, D, P = precision::Default>
where
    E: EngineSpec,
    D: Device,
    P: PrecisionSpec,
{
    pub(crate) engine: E::Field,
    pub(crate) device: D::Arg,
    pub(crate) precision: P::Field,
}

impl<E, D, P> Target<E, D, P>
where
    E: EngineSpec,
    D: Device,
    P: PrecisionSpec,
{
    /// Creates a new target from explicit engine, device argument, and precision policy fields.
    pub fn new(engine: E::Field, device: D::Arg, precision: P::Field) -> Self {
        Self {
            engine,
            device,
            precision,
        }
    }

    /// Rebinds the precision policy of this target, returning a new `Target`.
    pub fn with_precision<P2: PrecisionSpec>(self, policy: P2) -> Target<E, D, P2> {
        Target {
            engine: self.engine,
            device: self.device,
            precision: policy.init_field(),
        }
    }

    /// Rebinds the target to use a dynamic runtime precision policy.
    pub fn with_runtime_precision(self, policy: RuntimePrecisionPolicy) -> Target<E, D, Dyn> {
        Target {
            engine: self.engine,
            device: self.device,
            precision: policy,
        }
    }
}

impl<E, D, P> TensorTarget for Target<E, D, P>
where
    E: EngineOn<D>,
    D: Device,
    P: PrecisionSpec,
{
    type Dtype = P::GeneratedDType;
    type ParameterDtype = P::ParameterDType;
    type Device = D;
    type Backend = E::Backend;

    fn device_arg(&self) -> D::Arg {
        self.device.clone()
    }

    fn dtype_field(&self) -> <Self::Dtype as DType>::Field {
        P::generated_dtype_field(&self.precision)
    }

    fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
        P::parameter_dtype_field(&self.precision)
    }

    fn precision_policy(&self) -> RuntimePrecisionPolicy {
        P::runtime_policy(&self.precision)
    }
}

impl Native {
    /// Binds the `Native` engine to a physical device target.
    pub fn on<T: TensorTarget>(target: T) -> Target<Native, T::Device, precision::Default> {
        Target {
            engine: (),
            device: target.device_arg(),
            precision: (),
        }
    }
}

#[cfg(feature = "external-candle")]
impl Candle {
    /// Binds the `Candle` engine to a physical device target.
    pub fn on<T: TensorTarget>(target: T) -> Target<Candle, T::Device, precision::Default> {
        Target {
            engine: (),
            device: target.device_arg(),
            precision: (),
        }
    }
}
