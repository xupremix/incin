//! Comprehensive verification test suite for Linear parameter initialization,
//! trainability typestate, target materialization, and backend integration.

#![cfg(feature = "target-api")]

extern crate incin_core as incin;

use incin_backends::nn_target::{InitOnTarget, LinearNewOnTarget};
use incin_backends::prelude::*;
use incin_backends::target::Native;
use incin_core::nn::init::{self, InitContext, InitPlan, ParameterRole};
use incin_core::nn::linear::{Linear, linear};
use incin_core::nn::module::Module;
use incin_core::nn::param::{Frozen, Trainable};
use incin_core::prelude::*;

#[test]
fn test_init_plan_lowering() {
    // 1. Zeros & Ones
    let p_zeros = init::zeros()
        .plan(InitContext::new(ParameterRole::Weight))
        .unwrap();
    assert_eq!(p_zeros, InitPlan::Zeros);

    let p_ones = init::ones()
        .plan(InitContext::new(ParameterRole::Bias))
        .unwrap();
    assert_eq!(p_ones, InitPlan::Ones);

    // 2. Constant
    let p_const = init::constant(3.14)
        .plan(InitContext::new(ParameterRole::Weight))
        .unwrap();
    assert_eq!(p_const, InitPlan::Constant(3.14));

    // 3. Xavier Uniform with Fan (in = 100, out = 400)
    let ctx = InitContext::new(ParameterRole::Weight).with_fan(100, 400);
    let p_xavier = init::xavier_uniform().plan(ctx).unwrap();
    if let InitPlan::Uniform { low, high } = p_xavier {
        let expected_bound = f64::sqrt(6.0 / (100.0 + 400.0));
        assert!((low - (-expected_bound)).abs() < 1e-6);
        assert!((high - expected_bound).abs() < 1e-6);
    } else {
        panic!("Expected Uniform plan for XavierUniform");
    }

    // 4. Kaiming Uniform with Fan (in = 100, out = 400, default a = sqrt(5))
    let p_kaiming = init::kaiming_uniform().plan(ctx).unwrap();
    if let InitPlan::Uniform { low, high } = p_kaiming {
        let a = f64::sqrt(5.0);
        let std = f64::sqrt(2.0 / ((1.0 + a * a) * 100.0));
        let expected_bound = f64::sqrt(3.0) * std;
        assert!((low - (-expected_bound)).abs() < 1e-6);
        assert!((high - expected_bound).abs() < 1e-6);
    } else {
        panic!("Expected Uniform plan for KaimingUniform");
    }
}

#[test]
fn test_linear_builder_typestate() {
    let target = Native::on(Cpu);

    // Default builder: Bias = True, Train = Trainable
    let b1 = linear(shape![128, 64]);
    let l1 = b1.init(&target).unwrap();
    assert_eq!(l1.weight.shape_dims(), vec![64, 128]);
    assert!(l1.bias.is_some());
    assert_eq!(l1.bias.as_ref().unwrap().shape_dims(), vec![64]);

    // Frozen builder: Train = Frozen
    let b2 = linear(shape![128, 64].resolve().unwrap()).frozen();
    let l2 = b2.init(&target).unwrap();
    assert_eq!(l2.weight.shape_dims(), vec![64, 128]);

    // No bias builder: Bias = False
    let b3 = linear(shape![128, 64].resolve().unwrap()).no_bias();
    let l3 = b3.init(&target).unwrap();
    assert!(l3.bias.is_none());

    // Custom init builder
    let b4 = linear(shape![128, 64].resolve().unwrap())
        .no_bias()
        .weight_init(init::zeros());
    let l4 = b4.init(&target).unwrap();
    assert!(l4.bias.is_none());
}

#[test]
fn test_target_parameter_dtype_deduction() {
    // 1. Native F32 (Default Exact)
    let t_f32 = Native::on(Cpu);
    let l_f32: Linear<_, _, True, f32, Trainable> = linear(shape![10, 20].resolve().unwrap())
        .init(&t_f32)
        .unwrap();
    assert_eq!(l_f32.weight.shape_dims(), vec![20, 10]);

    // 2. Native F64 (Exact<f64>)
    let t_f64 = Native::on(Cpu).with_precision(precision::Exact::<f64>::new());
    let l_f64: Linear<_, _, True, f64, Trainable> = linear(shape![10, 20].resolve().unwrap())
        .init(&t_f64)
        .unwrap();
    assert_eq!(l_f64.weight.shape_dims(), vec![20, 10]);
}

#[test]
fn test_linear_freeze_unfreeze() {
    let target = Native::on(Cpu);
    let layer: Linear<_, IncinBackend<Cpu>, True, f32, Trainable> =
        linear(shape![16, 8].resolve().unwrap())
            .init(&target)
            .unwrap();

    // Collect parameters of trainable layer via Module::parameters()
    let params_trainable = layer.parameters();
    assert_eq!(params_trainable.len(), 2); // weight and bias

    // Freeze layer
    let frozen_layer: Linear<_, IncinBackend<Cpu>, True, f32, Frozen> = layer.freeze();
    let params_frozen = frozen_layer.parameters();
    assert_eq!(params_frozen.len(), 0); // Frozen parameters insert nothing into optimizer!

    // Unfreeze layer
    let unfrozen_layer: Linear<_, IncinBackend<Cpu>, True, f32, Trainable> =
        frozen_layer.unfreeze();
    let params_unfrozen = unfrozen_layer.parameters();
    assert_eq!(params_unfrozen.len(), 2);
}

#[test]
fn test_linear_direct_new_probe() {
    let target = Native::on(Cpu);
    // Direct LinearNewOnTarget::new extension trait probe
    let layer: Linear<_, IncinBackend<Cpu>, True, f32, Trainable> = <Linear<
        _,
        IncinBackend<Cpu>,
        True,
        f32,
        Trainable,
    > as LinearNewOnTarget<_, _>>::new_on_target(
        shape![32, 16].resolve().unwrap(),
        &target,
    )
    .unwrap();
    assert_eq!(layer.weight.shape_dims(), vec![16, 32]);
    assert!(layer.bias.is_some());
}
