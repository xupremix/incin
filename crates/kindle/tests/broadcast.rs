use kindle_core::prelude::*;
use typenum::{U2, U3};

/// B.
type B = kindle_backends::cpu::CpuBackendImpl;

#[test]
/// Test broadcast success.
fn test_broadcast_success() {
    let t1: Tensor<(U2,), B> = Tensor::zeros(()).unwrap();
    let t2: Tensor<(U3, U2), B> = Tensor::zeros(()).unwrap();

    // (2,) and (3, 2) should broadcast to (3, 2)
    let out = t1.broadcast_add(&t2).unwrap();

    // Type checking the output
    let _check: Tensor<(U3, U2), B> = out;
}
