//! Tests for CUDA multi-dtype shape and movement operations.

use incin_core::tensor::dtype::DTypeId;

#[test]
fn test_shape_ops_supported_bitwidths() {
    let supported_dtypes = [
        DTypeId::Bool,
        DTypeId::U8,
        DTypeId::F16,
        DTypeId::BF16,
        DTypeId::F32,
        DTypeId::U32,
        DTypeId::F64,
        DTypeId::I64,
    ];

    for dtype in supported_dtypes {
        let size = dtype
            .descriptor()
            .size_bytes(1, incin_core::shapes::OperationKind::Reshape)
            .unwrap();
        assert!(matches!(size, 1 | 2 | 4 | 8));
    }
}
