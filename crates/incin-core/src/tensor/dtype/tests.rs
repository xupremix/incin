use super::*;
use crate::shapes::error::OperationKind;

// --- StorageEncoding tests ---

#[test]
fn f32_encoding_is_scalar() {
    let enc = DTypeId::F32.encoding();
    assert_eq!(enc.logical_elements_per_block(), 1);
    assert_eq!(enc.bytes_per_block(), 4);
    assert_eq!(enc.alignment(), 4);
    assert!(enc.is_scalar());
    assert!(!enc.is_block());
    assert_eq!(enc.scalar_bytes(), Some(4));
    assert_eq!(enc.size_bytes(64, OperationKind::Storage).unwrap(), 256);
}

#[test]
fn f64_encoding() {
    let enc = DTypeId::F64.encoding();
    assert_eq!(enc.size_bytes(64, OperationKind::Storage).unwrap(), 512);
}

#[test]
fn q8_0_encoding_is_block() {
    let enc = DTypeId::Q8_0.encoding();
    assert_eq!(enc.logical_elements_per_block(), 32);
    assert_eq!(enc.bytes_per_block(), 34);
    assert_eq!(enc.alignment(), 2);
    assert!(!enc.is_scalar());
    assert!(enc.is_block());
    assert_eq!(enc.scalar_bytes(), None);
    assert_eq!(enc.size_bytes(32, OperationKind::Storage).unwrap(), 34);
    assert_eq!(enc.size_bytes(64, OperationKind::Storage).unwrap(), 68);
    assert_eq!(enc.size_bytes(0, OperationKind::Storage).unwrap(), 0);
    assert!(enc.size_bytes(33, OperationKind::Storage).is_err());
}

#[test]
fn scalar_overflow_returns_error() {
    let enc = StorageEncoding::scalar(8, 8);
    assert!(enc.size_bytes(usize::MAX, OperationKind::Storage).is_err());
}

#[test]
fn block_overflow_returns_error() {
    let enc = StorageEncoding::block(32, 34, 2);
    // usize::MAX is not a multiple of 32 in general, so this might
    // fail at the divisibility check or the overflow check.
    assert!(
        enc.size_bytes(usize::MAX - 1, OperationKind::Storage)
            .is_err()
    );
}

// --- DTypeDescriptor tests ---

#[test]
fn f32_descriptor() {
    let d = DTypeId::F32.descriptor();
    assert_eq!(d.key().namespace(), "incin");
    assert_eq!(d.key().name(), "f32");
    assert_eq!(d.key().version(), 1);
    assert_eq!(d.kind(), DTypeKind::Float);
    assert_eq!(d.builtin_id(), Some(DTypeId::F32));
}

#[test]
fn q8_0_descriptor() {
    let d = DTypeId::Q8_0.descriptor();
    assert_eq!(d.kind(), DTypeKind::Quantized);
    assert!(!d.encoding().is_scalar());
    assert_eq!(d.encoding().logical_elements_per_block(), 32);
    assert_eq!(d.encoding().bytes_per_block(), 34);
    assert_eq!(d.builtin_id(), Some(DTypeId::Q8_0));
}

#[test]
fn builtin_round_trip() {
    for id in [
        DTypeId::U8,
        DTypeId::U32,
        DTypeId::I64,
        DTypeId::BF16,
        DTypeId::F16,
        DTypeId::F32,
        DTypeId::F64,
        DTypeId::Q8_0,
    ] {
        assert_eq!(id.descriptor().builtin_id(), Some(id));
    }
}

#[test]
fn custom_dtype_has_no_builtin_id() {
    let key = DTypeKey::new("test", "packed", 1);
    let enc = StorageEncoding::block(3, 5, 1);
    let desc = DTypeDescriptor::new(key, DTypeKind::Opaque, enc);
    assert_eq!(desc.builtin_id(), None);
}

// --- DTypeId classification ---

#[test]
fn dtype_id_classification() {
    assert!(DTypeId::F32.is_float());
    assert!(DTypeId::F16.is_float());
    assert!(DTypeId::BF16.is_float());
    assert!(DTypeId::F64.is_float());
    assert!(DTypeId::U8.is_integer());
    assert!(DTypeId::U32.is_integer());
    assert!(DTypeId::I64.is_integer());
    assert!(DTypeId::Q8_0.is_quantized());
    assert!(!DTypeId::F32.is_integer());
    assert!(!DTypeId::U8.is_float());
    assert!(!DTypeId::Q8_0.is_float());
}

// --- Extensibility: custom dtype without DTypeId variant ---

#[test]
fn custom_dtype_compile_test() {
    #[derive(Clone, Copy, Debug, PartialEq)]
    struct TestPacked;

    impl DType for TestPacked {
        type Arg = ();
        type Field = core::marker::PhantomData<Self>;
        fn init(_: ()) -> Self::Field {
            core::marker::PhantomData
        }
        fn descriptor(_: &Self::Field) -> DTypeDescriptor {
            Self::DESCRIPTOR
        }
    }

    impl ConstDType for TestPacked {
        const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::new(
            DTypeKey::new("test", "packed", 1),
            DTypeKind::Opaque,
            StorageEncoding::block(3, 5, 1),
        );
    }

    // Verify it compiles and has no builtin ID.
    let desc = TestPacked::DESCRIPTOR;
    assert_eq!(desc.builtin_id(), None);
    assert_eq!(desc.key().namespace(), "test");
    assert_eq!(desc.key().name(), "packed");
    assert_eq!(desc.encoding().logical_elements_per_block(), 3);
    assert_eq!(desc.encoding().bytes_per_block(), 5);
}

// --- PlainDType trait bounds compile tests ---

fn _assert_plain<K: PlainDType>() {}
fn _assert_not_plain_compile_check() {
    _assert_plain::<f32>();
    _assert_plain::<f64>();
    _assert_plain::<i64>();
    // Q8_0 deliberately NOT listed — it must NOT implement PlainDType.
    // Uncomment the next line to verify it fails to compile:
    // _assert_plain::<Q8_0>();
}
