extern crate kindle_core as kindle;

use kindle_core::prelude::Cpu;
use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::prelude::*;
use kindle_macros::s;

#[test]
fn test_rms_norm_static() {
    let _t1: Tensor<s![2, 3, 4], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();
    let norm: RMSNorm<s![4], DummyBackend<f32, Cpu>> = RMSNorm::new(0.001).unwrap();

    let _ = norm;
}

#[test]
fn test_dropout() {
    let mut dropout: Dropout = Dropout::new(0.5);
    // test properties
    assert!(dropout.is_training);
    dropout.is_training = false;
    assert!(!dropout.is_training);
}

#[test]
fn test_var_std_static() {
    let t1: Tensor<s![2, 3, 4], DummyBackend<f32, Cpu>> = Tensor::zeros(()).unwrap();
    let _var_all = t1.var_all(true).unwrap();
    let _std_all = t1.std_all(false).unwrap();

    let _var_dim = t1.var_dim::<1>(true).unwrap();
    let _std_dim = t1.std_dim::<2>(false).unwrap();
    let _var_keepdim = t1.var_keepdim::<0>(true).unwrap();
}

#[test]
fn test_lr_schedulers() {
    let mut linear = LinearLR::new(1.0, 0.1, 10);
    assert_eq!(linear.get_lr(), 1.0);
    linear.step();
    assert_eq!(linear.get_lr(), 0.91);
    for _ in 0..10 {
        linear.step();
    }
    assert_eq!(linear.get_lr(), 0.1);
}
