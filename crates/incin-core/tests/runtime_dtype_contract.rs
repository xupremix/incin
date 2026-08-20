//! Acceptance test suite for Runtime DType Abstraction (Core contracts).
//! Tests third-party custom dtype definition, Dyn descriptor field, TensorMeta custom descriptor,
//! Capability queries with custom descriptors, PrecisionRequest with custom descriptors,
//! and Built-in compatibility.

use core::marker::PhantomData;
use incin_core::exec::{
    Alignment, Capabilities, CapabilityQuery, CapabilityRegistry, CapabilityRule,
    ImplementationKind, LayoutClass, MathMode, PrecisionRequest, TensorMeta,
};
use incin_core::prelude::*;
use incin_core::shapes::ShapeBuf;
use incin_core::tensor::dtype::StorageEncoding;

/// 1B - Custom DType definition (NO DTypeId variant).
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

/// 1C - Dyn DType contract.
#[test]
fn test_dyn_dtype_contract() {
    let descriptor = TestOpaque16::DESCRIPTOR;
    let field = <Dyn as DType>::init(descriptor);
    assert_eq!(<Dyn as DType>::descriptor(&field), descriptor);
}

/// 1D - TensorMeta contract.
#[test]
fn test_tensor_meta_contract() -> Result<()> {
    let meta = TensorMeta::contiguous(
        ShapeBuf::from_slice(&[2, 3]),
        TestOpaque16::DESCRIPTOR,
        DeviceId::CPU,
        Alignment::new(2)?,
        6,
    )?;

    assert_eq!(meta.dtype, TestOpaque16::DESCRIPTOR);
    Ok(())
}

/// 1E - Capability contract.
#[test]
fn test_capability_contract() {
    const CUSTOM: DTypeDescriptor = TestOpaque16::DESCRIPTOR;
    const DTYPES: &[DTypeDescriptor] = &[CUSTOM];
    const LAYOUTS: &[LayoutClass] = &[LayoutClass::Contiguous];
    const MODES: &[MathMode] = &[MathMode::Precise];

    const RULES: &[CapabilityRule] = &[CapabilityRule::new(
        OperationKind::Zeros,
        DTYPES,
        LAYOUTS,
        0,
        usize::MAX,
        false,
        MODES,
        ImplementationKind::Native,
    )];

    let query = CapabilityQuery {
        operation: incin_core::exec::OperationIdentity::Builtin(OperationKind::Zeros),
        dtype: CUSTOM,
        layout: LayoutClass::Contiguous,
        rank: 2,
        training: false,
        math_mode: MathMode::Precise,
    };

    assert!(
        CapabilityRegistry::new(RULES)
            .support(&query)
            .is_supported()
    );
}

/// 1F - Precision contract.
#[test]
fn test_precision_contract() {
    let request = PrecisionRequest::new(
        OperationKind::Zeros,
        TestOpaque16::DESCRIPTOR,
        TestOpaque16::DESCRIPTOR,
        LayoutClass::Contiguous,
        2,
        false,
        MathMode::Precise,
    );
    assert_eq!(request.storage, TestOpaque16::DESCRIPTOR);
    assert_eq!(request.output, TestOpaque16::DESCRIPTOR);
}

/// 1H - Built-in compatibility.
#[test]
fn test_builtin_compatibility() {
    assert_eq!(
        <f32 as ConstDType>::DESCRIPTOR.builtin_id(),
        Some(DTypeId::F32)
    );
    assert_eq!(TestOpaque16::DESCRIPTOR.builtin_id(), None);
}
