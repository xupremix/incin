use kindle::prelude::*;
use kindle::nn::{Linear, Conv2d};
use kindle_backends::candle::CandleBackend;

type B = CandleBackend<f32, kindle::prelude::Cpu>;

#[test]
fn test_linear_mixed_shapes() {
    // 1. Fully static
    let _ = Linear::<s![3, 4], B>::new().unwrap();

    // 2. Fully dynamic
    let _ = Linear::<s![dyn, dyn], B>::new(3, 4).unwrap();

    // 3. Partially static (In static, Out dynamic)
    let _ = Linear::<s![3, dyn], B>::new(4).unwrap();

    // 4. Partially static (In dynamic, Out static)
    let _ = Linear::<s![dyn, 4], B>::new(3).unwrap();
}

#[test]
fn test_conv2d_mixed_shapes() {
    // Conv2d<K, S, P, D, W, B>
    // K defaults to 3, S defaults to 1, P defaults to 1, D defaults to 1

    // 1. Fully static
    let _ = Conv2d::<U3, U1, U1, U1, s![3, 4, 3, 3], B>::new().unwrap();

    // 2. Fully dynamic
    let _ = Conv2d::<U3, U1, U1, U1, s![dyn, dyn, 3, 3], B>::new(4, 3).unwrap();

    // 3. Partially static (Out dynamic, In static)
    let _ = Conv2d::<U3, U1, U1, U1, s![dyn, 4, 3, 3], B>::new(3).unwrap();

    // 4. Partially static (Out static, In dynamic)
    let _ = Conv2d::<U3, U1, U1, U1, s![3, dyn, 3, 3], B>::new(4).unwrap();
}

#[test]
fn test_tensor_mixed_shapes() {
    let _ = Tensor::<s![3, 4], B>::zeros(()).unwrap();
    let _ = Tensor::<s![dyn, dyn], B>::zeros((3, 4)).unwrap();
}
