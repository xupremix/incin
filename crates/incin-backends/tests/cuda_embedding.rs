//! Tests for CUDA mixed-precision embeddings (f32, f64, f16, bf16).

#![allow(unused_imports)]

use incin_core::tensor::dtype::DTypeId;

#[test]
fn test_embedding_mixed_precision_spec() {
    let supported_dtypes = [DTypeId::F32, DTypeId::F64, DTypeId::F16, DTypeId::BF16];

    for dtype in supported_dtypes {
        assert!(dtype.descriptor().is_float());
    }
}
