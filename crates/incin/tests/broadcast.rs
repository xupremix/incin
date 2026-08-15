#![cfg(feature = "cpu")]
#![allow(clippy::type_complexity)]

use incin_core::prelude::*;

/// B.
type B = incin_backends::cpu::CpuBackendImpl;

#[test]
/// Test broadcast success.
fn test_broadcast_success() {
    let t1: Tensor<s![2], B> = Tensor::zeros(()).unwrap();
    let t2: Tensor<s![3, 2], B> = Tensor::zeros(()).unwrap();

    // (2,) and (3, 2) should broadcast to (3, 2)
    let out = t1.broadcast_add(&t2).unwrap();

    // Type checking the output
    let _check = out;
}
