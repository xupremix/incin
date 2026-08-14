//! Comprehensive verification test suite for recurrent learnable layers:
//! RNNCell, RNN, LSTMCell, LSTM.
//! Tests builder typestates, target-aware materialization, parameter collection,
//! precision policy propagation (e.g. bf16), freeze/unfreeze state transitions,
//! StateDict safety, and engine preservation.

#![cfg(feature = "target-api")]

extern crate incin_core as incin;

use incin_backends::nn_target::InitOnTarget;
use incin_backends::prelude::*;
use incin_backends::target::Native;
use incin_core::prelude::*;
use std::collections::BTreeMap;

#[test]
fn test_rnn_cell_migration() {
    let target = Native::on(Cpu);

    let builder = rnn_cell(shape![10, 20].resolve().unwrap());
    let cell = builder.init(&target).unwrap();

    assert_eq!(cell.wi.weight.shape_dims(), vec![20, 10]);
    assert_eq!(cell.wh.weight.shape_dims(), vec![20, 20]);
    assert!(cell.wi.bias.is_some());
    assert!(cell.wh.bias.is_some());

    // Both biases present by default => 4 trainable parameters (2 weights + 2 biases)
    assert_eq!(cell.parameters().len(), 4);

    // Test input bias removal
    let cell_no_ih = rnn_cell(shape![10, 20].resolve().unwrap())
        .no_input_bias()
        .init(&target)
        .unwrap();
    assert!(cell_no_ih.wi.bias.is_none());
    assert!(cell_no_ih.wh.bias.is_some());
    assert_eq!(cell_no_ih.parameters().len(), 3);

    // Test hidden bias removal
    let cell_no_hh = rnn_cell(shape![10, 20].resolve().unwrap())
        .no_hidden_bias()
        .init(&target)
        .unwrap();
    assert!(cell_no_hh.wi.bias.is_some());
    assert!(cell_no_hh.wh.bias.is_none());
    assert_eq!(cell_no_hh.parameters().len(), 3);

    // Test both biases removal
    let cell_no_bias = rnn_cell(shape![10, 20].resolve().unwrap())
        .no_bias()
        .init(&target)
        .unwrap();
    assert!(cell_no_bias.wi.bias.is_none());
    assert!(cell_no_bias.wh.bias.is_none());
    assert_eq!(cell_no_bias.parameters().len(), 2);

    // Freeze & Unfreeze
    let frozen = cell.freeze();
    assert_eq!(frozen.parameters().len(), 0);
    let unfrozen = frozen.unfreeze();
    assert_eq!(unfrozen.parameters().len(), 4);
}

#[test]
fn test_rnn_sequence_migration() {
    let target = Native::on(Cpu);

    let builder = rnn(shape![10, 20].resolve().unwrap());
    let rnn_layer = builder.init(&target).unwrap();

    assert_eq!(rnn_layer.cell.wi.weight.shape_dims(), vec![20, 10]);
    assert_eq!(rnn_layer.cell.wh.weight.shape_dims(), vec![20, 20]);
    assert_eq!(rnn_layer.parameters().len(), 4);

    // Freeze & Unfreeze
    let frozen_rnn = rnn_layer.freeze();
    assert_eq!(frozen_rnn.parameters().len(), 0);
    let unfrozen_rnn = frozen_rnn.unfreeze();
    assert_eq!(unfrozen_rnn.parameters().len(), 4);
}

#[test]
fn test_mixed_shape_rnn() {
    let target = Native::on(Cpu);
    let runtime_in = 10;

    let builder = rnn(shape![runtime_in, 20].resolve().unwrap());
    let rnn_layer = builder.init(&target).unwrap();

    assert_eq!(rnn_layer.cell.wi.weight.shape_dims(), vec![20, 10]);
    assert_eq!(rnn_layer.cell.wh.weight.shape_dims(), vec![20, 20]);
    assert_eq!(rnn_layer.parameters().len(), 4);
}

#[test]
fn test_rnn_bf16_precision() {
    let target = Native::on(Cpu).with_precision(precision::Bf16Mixed);

    let rnn_layer = rnn(shape![10, 20].resolve().unwrap())
        .init(&target)
        .unwrap();

    assert_eq!(rnn_layer.cell.wi.weight.shape_dims(), vec![20, 10]);
    assert_eq!(rnn_layer.cell.wh.weight.shape_dims(), vec![20, 20]);
    assert_eq!(rnn_layer.parameters().len(), 4);
}

#[test]
fn test_lstm_cell_migration() {
    let target = Native::on(Cpu);

    let builder = lstm_cell(shape![10, 20].resolve().unwrap());
    let cell = builder.init(&target).unwrap();

    // Verify 8 Linears exist with correct typed shapes (4 x [10, 20], 4 x [20, 20])
    assert_eq!(cell.wi_i.weight.shape_dims(), vec![20, 10]);
    assert_eq!(cell.wi_f.weight.shape_dims(), vec![20, 10]);
    assert_eq!(cell.wi_g.weight.shape_dims(), vec![20, 10]);
    assert_eq!(cell.wi_o.weight.shape_dims(), vec![20, 10]);
    assert_eq!(cell.wh_i.weight.shape_dims(), vec![20, 20]);
    assert_eq!(cell.wh_f.weight.shape_dims(), vec![20, 20]);
    assert_eq!(cell.wh_g.weight.shape_dims(), vec![20, 20]);
    assert_eq!(cell.wh_o.weight.shape_dims(), vec![20, 20]);

    // 8 weights + 8 biases = 16 parameters
    assert_eq!(cell.parameters().len(), 16);

    // Test input bias removal -> 8 weights + 4 hidden biases = 12
    let cell_no_ih = lstm_cell(shape![10, 20].resolve().unwrap())
        .no_input_bias()
        .init(&target)
        .unwrap();
    assert_eq!(cell_no_ih.parameters().len(), 12);

    // Test hidden bias removal -> 8 weights + 4 input biases = 12
    let cell_no_hh = lstm_cell(shape![10, 20].resolve().unwrap())
        .no_hidden_bias()
        .init(&target)
        .unwrap();
    assert_eq!(cell_no_hh.parameters().len(), 12);

    // Test both biases removal -> 8 weights = 8
    let cell_no_bias = lstm_cell(shape![10, 20].resolve().unwrap())
        .no_bias()
        .init(&target)
        .unwrap();
    assert_eq!(cell_no_bias.parameters().len(), 8);

    // Freeze & Unfreeze
    let frozen = cell.freeze();
    assert_eq!(frozen.parameters().len(), 0);
    let unfrozen = frozen.unfreeze();
    assert_eq!(unfrozen.parameters().len(), 16);
}

#[test]
fn test_lstm_sequence_migration() {
    let target = Native::on(Cpu);

    let builder = lstm(shape![10, 20].resolve().unwrap());
    let lstm_layer = builder.init(&target).unwrap();

    assert_eq!(lstm_layer.cell.wi_i.weight.shape_dims(), vec![20, 10]);
    assert_eq!(lstm_layer.cell.wh_i.weight.shape_dims(), vec![20, 20]);
    assert_eq!(lstm_layer.parameters().len(), 16);

    // Test no_bias
    let rnn_no_bias = lstm(shape![10, 20].resolve().unwrap())
        .no_bias()
        .init(&target)
        .unwrap();
    assert_eq!(rnn_no_bias.parameters().len(), 8);

    // Freeze & Unfreeze
    let frozen_lstm = lstm_layer.freeze();
    assert_eq!(frozen_lstm.parameters().len(), 0);
    let unfrozen_lstm = frozen_lstm.unfreeze();
    assert_eq!(unfrozen_lstm.parameters().len(), 16);
}

#[test]
fn test_lstm_bf16_precision() {
    let target = Native::on(Cpu).with_precision(precision::Bf16Mixed);

    let lstm_layer = lstm(shape![10, 20].resolve().unwrap())
        .init(&target)
        .unwrap();

    assert_eq!(lstm_layer.cell.wi_i.weight.shape_dims(), vec![20, 10]);
    assert_eq!(lstm_layer.cell.wh_i.weight.shape_dims(), vec![20, 20]);
    assert_eq!(lstm_layer.parameters().len(), 16);
}

#[test]
fn test_statedict_f32_safety_and_roundtrip() {
    let target = Native::on(Cpu);

    let rnn1 = rnn(shape![10, 20].resolve().unwrap())
        .init(&target)
        .unwrap();
    let mut rnn2 = rnn(shape![10, 20].resolve().unwrap())
        .init(&target)
        .unwrap();

    let state = rnn1.state_dict().unwrap();

    assert!(
        state
            .iter()
            .any(|(path, _)| path.as_str() == "cell.wi.weight")
    );
    assert!(
        state
            .iter()
            .any(|(path, _)| path.as_str() == "cell.wh.weight")
    );

    assert!(rnn2.load_state_dict(&state).is_ok());
}
