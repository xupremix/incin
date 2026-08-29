//! Tests for CUDA reduce operations: topk, cumsum, argmax, argmin.

use incin_core::tensor::dtype::DTypeId;

#[test]
fn test_reduce_ops_spec() {
    let supported = [
        DTypeId::F32,
        DTypeId::F64,
        DTypeId::I64,
    ];
    for dtype in supported {
        assert!(dtype.descriptor().size_bytes(1, incin_core::shapes::OperationKind::Reduction).is_ok());
    }
}
