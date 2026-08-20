//! Resolving an execution engine and a physical device to a backend type, and
//! wiring bare device values (`Cpu`, `Cuda`, ...) into `TensorTarget` through
//! the `Native` engine.

use super::*;

#[cfg(feature = "cpu")]
macro_rules! impl_unit_arg_target {
    ($($device:ty),* $(,)?) => {
        $(
            impl TensorTarget for $device {
                type Dtype = f32;
                type ParameterDtype = f32;
                type Device = Self;
                type Backend = <Native as EngineOn<Self>>::Backend;
                fn device_arg(&self) {}
                fn dtype_field(&self) -> <Self::Dtype as DType>::Field {
                    <Self::Dtype as DType>::init(())
                }
                fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
                    <Self::ParameterDtype as DType>::init(())
                }
                fn precision_policy(&self) -> RuntimePrecisionPolicy {
                    RuntimePrecisionPolicy::default()
                }
            }
        )*
    };
}

#[cfg(any(feature = "cuda", feature = "wgpu", feature = "metal"))]
macro_rules! impl_self_arg_target {
    ($($device:ty),* $(,)?) => {
        $(
            impl TensorTarget for $device {
                type Dtype = f32;
                type ParameterDtype = f32;
                type Device = Self;
                type Backend = <Native as EngineOn<Self>>::Backend;
                fn device_arg(&self) -> Self {
                    self.clone()
                }
                fn dtype_field(&self) -> <Self::Dtype as DType>::Field {
                    <Self::Dtype as DType>::init(())
                }
                fn parameter_dtype_field(&self) -> <Self::ParameterDtype as DType>::Field {
                    <Self::ParameterDtype as DType>::init(())
                }
                fn precision_policy(&self) -> RuntimePrecisionPolicy {
                    RuntimePrecisionPolicy::default()
                }
            }
        )*
    };
}

#[cfg(feature = "cpu")]
impl_unit_arg_target!(Cpu);

#[cfg(feature = "cuda")]
impl_self_arg_target!(Cuda);

#[cfg(feature = "wgpu")]
impl_self_arg_target!(Wgpu);

#[cfg(feature = "metal")]
impl_self_arg_target!(Metal);

/// Trait implemented by type-level execution engines (`Native`, `Candle`, `Dyn`).
pub trait EngineSpec: 'static + Send + Sync + Copy + Debug + Eq + PartialEq + Hash {
    /// Associated state carried by runtime instances of this engine.
    type Field: Clone + Send + Sync + 'static + Debug;
}

/// Execution engine marker for Incin's native backends (CPU, CUDA, WGPU, Metal).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Native;

impl EngineSpec for Native {
    type Field = ();
}

/// Execution engine marker for the Candle backend.
#[cfg(feature = "external-candle")]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, Hash)]
pub struct Candle;

#[cfg(feature = "external-candle")]
impl EngineSpec for Candle {
    type Field = ();
}

/// Runtime engine selection tag.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum RuntimeEngine {
    Native,
    #[cfg(feature = "external-candle")]
    Candle,
}

impl EngineSpec for Dyn {
    type Field = RuntimeEngine;
}

/// Maps an execution engine `E` and a physical device `D` to a backend family.
pub trait EngineOn<D: Device>: EngineSpec {
    type Backend: Backend<Device = D> + VariableBackend;
}

/// Backend type selected by engine `E` on physical device `D`.
pub type EngineBackend<E, D> = <E as EngineOn<D>>::Backend;

/// Backend type selected by the `Native` engine on physical device `D`.
pub type NativeBackend<D> = <Native as EngineOn<D>>::Backend;

impl EngineOn<Dyn> for Native {
    type Backend = crate::dispatch::DispatchBackend<Dyn>;
}

#[cfg(feature = "cpu")]
impl EngineOn<Cpu> for Native {
    type Backend = crate::cpu::CpuBackendImpl<Cpu>;
}

#[cfg(feature = "cuda")]
impl EngineOn<Cuda> for Native {
    type Backend = crate::cuda::CudaBackendImpl<Cuda>;
}

#[cfg(feature = "cuda")]
impl<O: typenum::Unsigned + Send + Sync + Eq + Debug + 'static> EngineOn<CudaN<O>> for Native {
    type Backend = crate::cuda::CudaBackendImpl<CudaN<O>>;
}

#[cfg(feature = "wgpu")]
impl EngineOn<Wgpu> for Native {
    type Backend = crate::wgpu::WgpuBackendImpl<Wgpu>;
}

#[cfg(feature = "wgpu")]
impl<O: typenum::Unsigned + Send + Sync + Eq + Debug + 'static> EngineOn<WgpuN<O>> for Native {
    type Backend = crate::wgpu::WgpuBackendImpl<WgpuN<O>>;
}

#[cfg(feature = "metal")]
impl EngineOn<Metal> for Native {
    type Backend = crate::metal::MetalBackendImpl<Metal>;
}

#[cfg(feature = "external-candle")]
impl EngineOn<Cpu> for Candle {
    type Backend = crate::external::candle::CandleBackend<Cpu>;
}

#[cfg(all(feature = "external-candle", feature = "cuda"))]
impl EngineOn<Cuda> for Candle {
    type Backend = crate::external::candle::CandleBackend<Cuda>;
}

#[cfg(all(feature = "external-candle", feature = "cuda"))]
impl<O: typenum::Unsigned + Send + Sync + Eq + Debug + 'static> EngineOn<CudaN<O>> for Candle {
    type Backend = crate::external::candle::CandleBackend<CudaN<O>>;
}
