use kindle::prelude::*;
use kindle::{Conv2d, Linear};
/// B.
type B = kindle_backends::cpu::CpuBackendImpl;

#[test]
/// Test linear mixed shapes.
fn test_linear_mixed_shapes() {
    // 1. Fully static
    let _ = Linear::<s![3, 4], B>::build(()).unwrap();

    // 2. Fully dynamic
    let _ = Linear::<s![dyn, dyn], B>::build((3, 4)).unwrap();

    // 3. Partially static (In static, Out dynamic)
    let _ = Linear::<s![3, dyn], B>::build(4).unwrap();

    // 4. Partially static (In dynamic, Out static)
    let _ = Linear::<s![dyn, 4], B>::build(3).unwrap();
}

#[test]
/// Test conv2d mixed shapes.
fn test_conv2d_mixed_shapes() {
    // Conv2d<S: Conv2dShape, B: Backend>
    // S = (OutC, InC, K, S, P, D)

    // 1. Fully static
    let _ = Conv2d::<s![3, 4, 3, 1, 1, 1], B>::build(()).unwrap();

    // 2. Fully dynamic
    let _ = Conv2d::<s![dyn, dyn, 3, 1, 1, 1], B>::build((3, 4)).unwrap();

    // 3. Partially static (Out dynamic, In static)
    let _ = Conv2d::<s![dyn, 4, 3, 1, 1, 1], B>::build(3).unwrap();

    // 4. Partially static (Out static, In dynamic)
    let _ = Conv2d::<s![3, dyn, 3, 1, 1, 1], B>::build(4).unwrap();
}

#[test]
/// Test tensor mixed shapes.
fn test_tensor_mixed_shapes() {
    let _ = Tensor::<s![3, 4], B>::zeros(()).unwrap();
    let _ = Tensor::<s![dyn, dyn], B>::zeros((3, 4)).unwrap();
}
