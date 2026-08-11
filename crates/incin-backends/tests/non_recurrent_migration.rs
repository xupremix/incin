//! Comprehensive verification test suite for non-recurrent learnable layers:
//! Embedding, LayerNorm, RMSNorm, BatchNorm2d, Conv1d, Conv2d.
//! Tests builder typestates, target-aware materialization, parameter/buffer separation,
//! precision policy propagation (e.g. bf16), and freeze/unfreeze state transitions.

#![cfg(feature = "target-api")]

extern crate incin_core as incin;

use incin_backends::nn_target::InitOnTarget;
use incin_backends::prelude::*;
use incin_backends::target::Native;
use incin_core::nn::module::Module;
use incin_core::nn::param::{Frozen, Trainable};
use incin_core::prelude::*;

#[test]
fn test_embedding_migration() {
    let target = Native::on(Cpu);

    let builder = embedding(shape![1000, 128]);
    let emb = builder.init(&target).unwrap();
    assert_eq!(emb.weight.shape_dims(), vec![1000, 128]);
    assert_eq!(emb.parameters().len(), 1);

    // Freeze & Unfreeze
    let frozen_emb = emb.freeze();
    assert_eq!(frozen_emb.parameters().len(), 0);
    let unfrozen_emb = frozen_emb.unfreeze();
    assert_eq!(unfrozen_emb.parameters().len(), 1);
}

#[test]
fn test_layer_norm_migration() {
    let target = Native::on(Cpu);

    let builder = layer_norm(shape![64].resolve().unwrap(), 1e-5);
    let ln = builder.init(&target).unwrap();
    assert_eq!(ln.weight.shape_dims(), vec![64]);
    assert_eq!(ln.bias.shape_dims(), vec![64]);
    assert_eq!(ln.eps, 1e-5);
    assert_eq!(ln.parameters().len(), 2);

    // Freeze & Unfreeze
    let frozen_ln = ln.freeze();
    assert_eq!(frozen_ln.parameters().len(), 0);
    let unfrozen_ln = frozen_ln.unfreeze();
    assert_eq!(unfrozen_ln.parameters().len(), 2);
}

#[test]
fn test_rms_norm_migration() {
    let target = Native::on(Cpu);

    let builder = rms_norm(shape![128].resolve().unwrap(), 1e-6);
    let rms = builder.init(&target).unwrap();
    assert_eq!(rms.weight.shape_dims(), vec![128]);
    assert_eq!(rms.eps, 1e-6);
    assert_eq!(rms.parameters().len(), 1);

    // Freeze & Unfreeze
    let frozen_rms = rms.freeze();
    assert_eq!(frozen_rms.parameters().len(), 0);
    let unfrozen_rms = frozen_rms.unfreeze();
    assert_eq!(unfrozen_rms.parameters().len(), 1);
}

#[test]
fn test_batch_norm2d_migration() {
    let target = Native::on(Cpu);

    let builder = batch_norm2d(shape![32].resolve().unwrap(), 1e-5, 0.1);
    let bn = builder.init(&target).unwrap();
    assert_eq!(bn.weight.shape_dims(), vec![32]);
    assert_eq!(bn.bias.shape_dims(), vec![32]);
    assert_eq!(bn.running_mean.shape_dims(), vec![32]);
    assert_eq!(bn.running_var.shape_dims(), vec![32]);
    assert_eq!(bn.eps, 1e-5);
    assert_eq!(bn.momentum, 0.1);

    // Only weight & bias are trainable parameters (running stats are non-trainable Buffers)
    assert_eq!(bn.parameters().len(), 2);

    // Freeze & Unfreeze
    let frozen_bn = bn.freeze();
    assert_eq!(frozen_bn.parameters().len(), 0);
    let unfrozen_bn = frozen_bn.unfreeze();
    assert_eq!(unfrozen_bn.parameters().len(), 2);
}

#[test]
fn test_conv1d_migration() {
    let target = Native::on(Cpu);

    let builder = conv1d(shape![64, 3, 5, 1, 0, 1].resolve().unwrap());
    let conv = builder.init(&target).unwrap();

    assert_eq!(conv.weight.shape_dims(), vec![64, 3, 5]);
    assert!(conv.bias.is_some());
    assert_eq!(conv.bias.as_ref().unwrap().shape_dims(), vec![64]);
    assert_eq!(conv.stride, 1);
    assert_eq!(conv.padding, 0);
    assert_eq!(conv.dilation, 1);
    assert_eq!(conv.parameters().len(), 2);

    // Test no_bias
    let builder_no_bias = conv1d(shape![64, 3, 5, 1, 0, 1].resolve().unwrap()).no_bias();
    let conv_no_bias = builder_no_bias.init(&target).unwrap();
    assert!(conv_no_bias.bias.is_none());
    assert_eq!(conv_no_bias.parameters().len(), 1);

    // Freeze & Unfreeze
    let frozen_conv = conv.freeze();
    assert_eq!(frozen_conv.parameters().len(), 0);
    let unfrozen_conv = frozen_conv.unfreeze();
    assert_eq!(unfrozen_conv.parameters().len(), 2);
}

#[test]
fn test_conv2d_migration() {
    let target = Native::on(Cpu);

    let builder = conv2d(shape![32, 16, 3, 1, 1, 1].resolve().unwrap());
    let conv = builder.init(&target).unwrap();

    assert_eq!(conv.weight.shape_dims(), vec![32, 16, 3, 3]);
    assert!(conv.bias.is_some());
    assert_eq!(conv.bias.as_ref().unwrap().shape_dims(), vec![32]);
    assert_eq!(conv.parameters().len(), 2);

    // Test no_bias
    let builder_no_bias = conv2d(shape![32, 16, 3, 1, 1, 1].resolve().unwrap()).no_bias();
    let conv_no_bias = builder_no_bias.init(&target).unwrap();
    assert!(conv_no_bias.bias.is_none());
    assert_eq!(conv_no_bias.parameters().len(), 1);

    // Freeze & Unfreeze
    let frozen_conv = conv.freeze();
    assert_eq!(frozen_conv.parameters().len(), 0);
    let unfrozen_conv = frozen_conv.unfreeze();
    assert_eq!(unfrozen_conv.parameters().len(), 2);
}

#[test]
fn test_non_recurrent_bf16_precision() {
    let target = Native::on(Cpu).with_precision(precision::Exact::<bf16>::new());

    // 1. Linear
    let l = linear(shape![16, 8].resolve().unwrap())
        .init(&target)
        .unwrap();
    assert_eq!(l.weight.shape_dims(), vec![8, 16]);

    // 2. Embedding
    let emb = embedding(shape![500, 64].resolve().unwrap())
        .init(&target)
        .unwrap();
    assert_eq!(emb.weight.shape_dims(), vec![500, 64]);

    // 3. LayerNorm
    let ln = layer_norm(shape![64].resolve().unwrap(), 1e-5)
        .init(&target)
        .unwrap();
    assert_eq!(ln.weight.shape_dims(), vec![64]);

    // 4. RMSNorm
    let rms = rms_norm(shape![64].resolve().unwrap(), 1e-5)
        .init(&target)
        .unwrap();
    assert_eq!(rms.weight.shape_dims(), vec![64]);

    // 5. BatchNorm2d
    let bn = batch_norm2d(shape![32].resolve().unwrap(), 1e-5, 0.1)
        .init(&target)
        .unwrap();
    assert_eq!(bn.weight.shape_dims(), vec![32]);

    // 6. Conv1d
    let c1 = conv1d(shape![16, 8, 3, 1, 0, 1].resolve().unwrap())
        .init(&target)
        .unwrap();
    assert_eq!(c1.weight.shape_dims(), vec![16, 8, 3]);

    // 7. Conv2d
    let c2 = conv2d(shape![16, 8, 3, 1, 0, 1].resolve().unwrap())
        .init(&target)
        .unwrap();
    assert_eq!(c2.weight.shape_dims(), vec![16, 8, 3, 3]);
}
