//! Architecture regression tests enforcing concrete backend dtype decoupling.

use incin_backends::cpu::CpuBackendImpl;
use incin_backends::dispatch::DispatchBackend;
use incin_backends::target::{EngineOn, Native, TargetBackendFor, TensorTarget};
use incin_core::prelude::{Cpu, Dyn, ShapeBuf, StorageBackend};

trait Same<T> {}
impl<T> Same<T> for T {}

fn assert_same_type<A, B>()
where
    A: Same<B>,
    B: Same<A>,
{
}

#[test]
fn test_single_generic_backend_types() {
    // Assert CpuBackendImpl is single-generic: CpuBackendImpl<Cpu>
    let cpu_backend = CpuBackendImpl::<Cpu>::new();
    assert_eq!(<CpuBackendImpl<Cpu> as StorageBackend>::BACKEND_NAME, "Cpu");
    let _ = cpu_backend;

    // Assert DispatchBackend is single-generic: DispatchBackend<Dyn>
    let dispatch_backend = DispatchBackend::<Dyn>::new();
    assert_eq!(
        <DispatchBackend<Dyn> as StorageBackend>::BACKEND_NAME,
        "Dispatch"
    );
    let _ = dispatch_backend;

    #[cfg(feature = "wgpu")]
    {
        use incin_backends::wgpu::WgpuBackendImpl;
        use incin_core::prelude::Wgpu;
        let wgpu_backend = WgpuBackendImpl::<Wgpu>::new();
        assert_eq!(
            <WgpuBackendImpl<Wgpu> as StorageBackend>::BACKEND_NAME,
            "Wgpu"
        );
        let _ = wgpu_backend;
    }

    #[cfg(feature = "cuda")]
    {
        use incin_backends::cuda::CudaBackendImpl;
        use incin_core::prelude::Cuda;
        let cuda_backend = CudaBackendImpl::<Cuda>::new();
        assert_eq!(
            <CudaBackendImpl<Cuda> as StorageBackend>::BACKEND_NAME,
            "Cuda"
        );
        let _ = cuda_backend;
    }

    #[cfg(feature = "metal")]
    {
        use incin_backends::metal::MetalBackendImpl;
        use incin_core::prelude::Metal;
        let metal_backend = MetalBackendImpl::<Metal>::new();
        assert_eq!(
            <MetalBackendImpl<Metal> as StorageBackend>::BACKEND_NAME,
            "Metal"
        );
        let _ = metal_backend;
    }

    #[cfg(feature = "external-candle")]
    {
        use incin_backends::external::candle::CandleBackend;
        let candle_backend = CandleBackend::<Cpu>::new();
        assert_eq!(
            <CandleBackend<Cpu> as StorageBackend>::BACKEND_NAME,
            "Candle"
        );
        let _ = candle_backend;
    }
}

#[test]
fn test_tensor_target_single_generic_backend() {
    assert_same_type::<TargetBackendFor<Cpu>, CpuBackendImpl<Cpu>>();
}

#[test]
fn test_engine_on_single_generic_backend() {
    assert_same_type::<<Native as EngineOn<Cpu>>::Backend, CpuBackendImpl<Cpu>>();
}

#[test]
fn test_8a_same_native_cpu_backend_across_dtypes() {
    use incin_backends::target::{Target, precision};
    type NativeTarget = Target<Native, Cpu, precision::Default>;
    assert_same_type::<<NativeTarget as TensorTarget>::Backend, CpuBackendImpl<Cpu>>();
}

#[cfg(feature = "external-candle")]
#[test]
fn test_8b_same_candle_cpu_backend_across_dtypes() {
    use incin_backends::external::candle::CandleBackend;
    use incin_backends::target::{Candle, Target, precision};
    type CandleTarget = Target<Candle, Cpu, precision::Default>;
    assert_same_type::<<CandleTarget as TensorTarget>::Backend, CandleBackend<Cpu>>();
}

#[cfg(feature = "external-candle")]
#[test]
fn test_8c_dtype_view_preserves_engine() {
    use incin_backends::external::candle::CandleBackend;
    use incin_backends::target::{DtypeTarget, DtypeView};
    let target = incin_backends::target::Candle::on(Cpu);
    let view = target.dtype::<i64>().unwrap();
    assert_same_type::<
        <DtypeView<incin_backends::target::Target<incin_backends::target::Candle, Cpu>, i64> as TensorTarget>::Backend,
        CandleBackend<Cpu>,
    >();
    let _ = view;
}

#[test]
fn test_8d_parameter_dtype_is_independent() {
    use incin_core::nn::Param;
    use incin_core::prelude::Dyn;

    // Prove Param<Dyn, CpuBackendImpl<Cpu>, f32> and Param<Dyn, CpuBackendImpl<Cpu>, f64> are both valid
    fn check_param_types<B: incin_core::backend_authoring::Backend>() {}
    check_param_types::<CpuBackendImpl<Cpu>>();

    type ParamF32 = Param<Dyn, CpuBackendImpl<Cpu>, f32>;
    type ParamF64 = Param<Dyn, CpuBackendImpl<Cpu>, f64>;

    assert_same_type::<ParamF32, Param<Dyn, CpuBackendImpl<Cpu>, f32>>();
    assert_same_type::<ParamF64, Param<Dyn, CpuBackendImpl<Cpu>, f64>>();
}

#[test]
fn test_8e_minimal_backend_extension() {
    use incin_core::backend_authoring::{Backend, StorageBackend};
    use incin_core::exec::TensorMeta;
    use incin_core::prelude::{DType, DTypeDescriptor, DeviceId, Result};

    #[derive(Clone, Default)]
    struct MinimalBackend;

    impl incin_core::exec::Capabilities for MinimalBackend {
        fn support(
            &self,
            query: &incin_core::exec::CapabilityQuery,
        ) -> incin_core::exec::SupportLevel {
            let incin_core::exec::OperationIdentity::Builtin(operation) = &query.operation else {
                return incin_core::exec::SupportLevel::Unsupported(
                    incin_core::exec::UnsupportedReason::CustomOperation {
                        operation: match &query.operation {
                            incin_core::exec::OperationIdentity::Custom(operation) => {
                                operation.clone()
                            }
                            incin_core::exec::OperationIdentity::Builtin(_) => unreachable!(),
                        },
                    },
                );
            };
            incin_core::exec::SupportLevel::Unsupported(
                incin_core::exec::UnsupportedReason::Operation {
                    operation: *operation,
                },
            )
        }
    }

    impl StorageBackend for MinimalBackend {
        const BACKEND_NAME: &'static str = "Minimal";
        type Storage<K: DType> = ();
        type Device = Cpu;
        fn metadata<K: DType>(_storage: &Self::Storage<K>) -> &TensorMeta {
            unimplemented!()
        }
    }

    impl Backend for MinimalBackend {
        type Var<K: DType> = ();
        type Grads = ();
        type InnerBackend = Self;

        fn shape<K: DType>(_t: &()) -> ShapeBuf {
            ShapeBuf::scalar()
        }
        fn backward<K: DType>(_t: &()) -> Result<()> {
            Ok(())
        }
        fn get_grad<K: DType>(_t: &(), _grads: &()) -> Result<Option<()>> {
            Ok(None)
        }
        fn to_bytes<K: DType>(_t: &()) -> Result<std::vec::Vec<u8>> {
            Ok(vec![])
        }
        fn from_bytes<K: DType>(
            _bytes: &[u8],
            _shape: &[usize],
            _dtype: DTypeDescriptor,
            _device: &DeviceId,
        ) -> Result<()> {
            Ok(())
        }
        fn var_as_tensor<K: DType>(_var: &()) -> Result<()> {
            Ok(())
        }
        fn var_from_tensor<K: DType>(_t: &()) -> Result<()> {
            Ok(())
        }
        fn assign_var<K: DType>(_var: &mut (), _tensor: &()) -> Result<()> {
            Ok(())
        }
    }

    assert_eq!(<MinimalBackend as StorageBackend>::BACKEND_NAME, "Minimal");
}
