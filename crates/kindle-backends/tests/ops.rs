use kindle_backends::Backend;
use kindle_backends::candle::CandleBackend;
use kindle_core::prelude::*;

type CBackend = CandleBackend<f32, kindle_core::tensor::device::Cpu>;

#[test]
fn test_slice() {
    let t = CBackend::zeros::<f32>(&[4, 4], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let s = CBackend::slice::<f32>(&t, &[(0, 4), (1, 3)]).unwrap();
    assert_eq!(CBackend::shape::<f32>(&s), vec![4, 2]);
}

#[test]
fn test_reshape() {
    let t = CBackend::zeros::<f32>(&[2, 8], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let r = CBackend::reshape::<f32>(&t, &[4, 4]).unwrap();
    assert_eq!(CBackend::shape::<f32>(&r), vec![4, 4]);
}

#[test]
fn test_max_pool2d() {
    let t =
        CBackend::zeros::<f32>(&[1, 3, 16, 16], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let p = CBackend::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    assert_eq!(CBackend::shape::<f32>(&p), vec![1, 3, 8, 8]);
}

#[test]
fn test_avg_pool2d() {
    let t =
        CBackend::zeros::<f32>(&[1, 3, 16, 16], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let p = CBackend::avg_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0)).unwrap();
    assert_eq!(CBackend::shape::<f32>(&p), vec![1, 3, 8, 8]);
}

#[test]
fn test_conv2d() {
    let t =
        CBackend::zeros::<f32>(&[1, 3, 16, 16], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let w = CBackend::zeros::<f32>(&[8, 3, 3, 3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let c = CBackend::conv2d::<f32>(&t, &w, None, 1, 1, 1, 1).unwrap();
    assert_eq!(CBackend::shape::<f32>(&c), vec![1, 8, 16, 16]);
}

#[test]
fn test_batch_norm() {
    let t =
        CBackend::zeros::<f32>(&[1, 3, 16, 16], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let w = CBackend::ones::<f32>(&[3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let b = CBackend::zeros::<f32>(&[3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let rm = CBackend::zeros::<f32>(&[3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let rv = CBackend::ones::<f32>(&[3], KindleDType::F32, &KindleDevice::cpu()).unwrap();
    let bn = CBackend::batch_norm::<f32>(&t, Some(&w), Some(&b), Some(&rm), Some(&rv), 1e-5, 0.1)
        .unwrap();
    assert_eq!(CBackend::shape::<f32>(&bn), vec![1, 3, 16, 16]);
}
