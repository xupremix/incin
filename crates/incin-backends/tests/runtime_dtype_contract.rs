//! Acceptance test suite for Runtime DType Abstraction (Backend contracts).
//! Tests built-in backend rejection of unsupported custom dtypes, public tensor dtype contract,
//! target dtype extensibility, and runtime dtype view.

use core::marker::PhantomData;
use incin_backends::cpu::{Cpu, CpuBackendImpl};
use incin_backends::target::{DtypeTarget, TargetExt};
use incin_core::backend_authoring::SupportsDType;
use incin_core::prelude::*;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct TestOpaque16;

impl DType for TestOpaque16 {
    type Arg = ();
    type Field = PhantomData<Self>;

    fn init(_: ()) -> Self::Field {
        PhantomData
    }

    fn descriptor(_: &Self::Field) -> DTypeDescriptor {
        Self::DESCRIPTOR
    }
}

impl ConstDType for TestOpaque16 {
    const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::new(
        DTypeKey::new("incin_test", "opaque16", 1),
        DTypeKind::Opaque,
        StorageEncoding::scalar(2, 2),
    );
}

/// 1G — Custom capability rejection from built-in backend.
#[test]
fn test_custom_capability_rejection_from_builtin_backend() {
    let descriptor = TestOpaque16::DESCRIPTOR;
    let field = <Dyn as DType>::init(descriptor);
    let device = DeviceId::CPU;
    let err =
        <CpuBackendImpl<Dyn> as SupportsDType<Dyn>>::resolve_dtype(&field, &device).unwrap_err();
    match err {
        Error::UnsupportedDType { dtype, .. } => {
            assert_eq!(dtype, descriptor);
        }
        _ => panic!(
            "expected UnsupportedDType error carrying custom descriptor, got {:?}",
            err
        ),
    }
}

/// 1I — Public tensor dtype contract.
#[test]
fn test_public_tensor_dtype_contract() {
    let tensor = Cpu.zeros([2, 3]).unwrap();
    assert_eq!(tensor.dtype(), <f32 as ConstDType>::DESCRIPTOR);
    assert_eq!(tensor.builtin_dtype_id(), Some(DTypeId::F32));
}

/// 1J — Target dtype extensibility contract.
#[test]
fn test_target_dtype_extensibility_contract() {
    let target_res = Cpu.dtype_dynamic(TestOpaque16::DESCRIPTOR);
    assert!(target_res.is_err());
    if let Err(Error::UnsupportedDType { dtype, .. }) = target_res {
        assert_eq!(dtype, TestOpaque16::DESCRIPTOR);
    }
}

/// 1K — Runtime DType view.
#[test]
#[allow(clippy::result_large_err)]
fn test_runtime_dtype_view() -> Result<()> {
    let target = Cpu.dtype_dynamic(<f32 as ConstDType>::DESCRIPTOR)?;
    let tensor = target.zeros([2, 3])?;
    assert_eq!(tensor.dtype(), <f32 as ConstDType>::DESCRIPTOR);
    Ok(())
}

#[cfg(feature = "cuda")]
#[test]
fn test_cuda_custom_capability_rejection() {
    use incin_backends::cuda::CudaBackendImpl;
    let descriptor = TestOpaque16::DESCRIPTOR;
    let field = <Dyn as DType>::init(descriptor);
    let device = DeviceId::cuda(0);
    let err =
        <CudaBackendImpl<Dyn> as SupportsDType<Dyn>>::resolve_dtype(&field, &device).unwrap_err();
    assert!(matches!(err, Error::UnsupportedDType { .. }));
}

#[cfg(feature = "wgpu")]
#[test]
fn test_wgpu_custom_capability_rejection() {
    use incin_backends::wgpu::WgpuBackendImpl;
    let descriptor = TestOpaque16::DESCRIPTOR;
    let field = <Dyn as DType>::init(descriptor);
    let device = DeviceId::wgpu(0);
    let err =
        <WgpuBackendImpl<Dyn> as SupportsDType<Dyn>>::resolve_dtype(&field, &device).unwrap_err();
    assert!(matches!(err, Error::UnsupportedDType { .. }));
}

#[cfg(feature = "metal")]
#[test]
fn test_metal_custom_capability_rejection() {
    use incin_backends::metal::MetalBackendImpl;
    let descriptor = TestOpaque16::DESCRIPTOR;
    let field = <Dyn as DType>::init(descriptor);
    let device = DeviceId::metal(0);
    let err =
        <MetalBackendImpl<Dyn> as SupportsDType<Dyn>>::resolve_dtype(&field, &device).unwrap_err();
    assert!(matches!(err, Error::UnsupportedDType { .. }));
}

#[cfg(feature = "external-candle")]
#[test]
fn test_candle_custom_capability_rejection() {
    use incin_backends::external::candle::CandleBackend;
    let descriptor = TestOpaque16::DESCRIPTOR;
    let field = <Dyn as DType>::init(descriptor);
    let device = DeviceId::CPU;
    let err =
        <CandleBackend<Cpu> as SupportsDType<Dyn>>::resolve_dtype(&field, &device).unwrap_err();
    assert!(matches!(err, Error::UnsupportedDType { .. }));
}
