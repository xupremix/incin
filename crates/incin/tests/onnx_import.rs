//! End-to-end integration tests for ONNX model import with dense initializers (Issue #11).

#![cfg(feature = "cpu")]

use incin::prelude::*;

incin::experimental::import_model!(
    "tests/fixtures/dense_initializers.onnx",
    DenseInitializerModel
);

#[test]
fn onnx_dense_initializer_model_instantiation_and_forward() {
    let model = DenseInitializerModel::<DefaultBackend>::new()
        .expect("DenseInitializerModel instantiation should succeed");

    // Check that state collection discovers the embedded parameters
    let state = incin::state::collect_state::<DefaultBackend, _>(&model)
        .expect("collect_state should succeed");

    assert_eq!(state.len(), 2);
    let paths: Vec<_> = state.iter().map(|(p, _)| p.as_str()).collect();
    assert!(paths.contains(&"initializer_000"));
    assert!(paths.contains(&"initializer_001"));

    // Verify forward execution:
    // Input X: [1.0, 2.0] (shape [1, 2])
    // Weight: [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]] (shape [2, 3])
    // MatMul: [1*2 + 2*5, 1*3 + 2*6, 1*4 + 2*7] = [12.0, 15.0, 18.0]
    // Bias: [0.5, 0.5, 0.5] (shape [1, 3])
    // Add: [12.5, 15.5, 18.5]
    let input = Tensor::<s![1, 2], DefaultBackend>::from_slice(&[1.0f32, 2.0], ()).unwrap();
    let output = model.forward(input).expect("forward pass should succeed");

    let output_data = output.to_vec1::<f32>().unwrap();
    assert_eq!(output_data.len(), 3);
    assert!((output_data[0] - 12.5).abs() < 1e-5);
    assert!((output_data[1] - 15.5).abs() < 1e-5);
    assert!((output_data[2] - 18.5).abs() < 1e-5);
}

#[test]
fn onnx_dense_initializer_model_state_collection_and_snapshot() {
    let model = DenseInitializerModel::<DefaultBackend>::new()
        .expect("DenseInitializerModel instantiation should succeed");

    let state = incin::state::collect_state::<DefaultBackend, _>(&model)
        .expect("collect_state should succeed");

    let weight_entry = state.iter().find(|(p, _)| p.as_str() == "initializer_000");
    assert!(weight_entry.is_some());
    let (_, val) = weight_entry.unwrap();
    assert_eq!(val.shape().as_ref(), &[2, 3]);

    let bias_entry = state.iter().find(|(p, _)| p.as_str() == "initializer_001");
    assert!(bias_entry.is_some());
    let (_, val) = bias_entry.unwrap();
    assert_eq!(val.shape().as_ref(), &[1, 3]);
}

#[test]
fn onnx_dense_initializer_model_backward_and_grad() {
    let model = DenseInitializerModel::<DefaultBackend>::new()
        .expect("DenseInitializerModel instantiation should succeed");

    let input = Tensor::<s![1, 2], DefaultBackend>::from_slice(&[1.0f32, 2.0], ()).unwrap();
    let output = model.forward(input).expect("forward pass should succeed");

    let sum = output.sum_all().expect("sum_all should succeed");
    let grads = sum.backward().expect("backward pass should succeed");

    let w_tensor = model
        .initializer_000
        .as_tensor()
        .expect("as_tensor should succeed");
    let w_grad = grads.get(&w_tensor).expect("get grad should succeed");
    assert!(w_grad.is_some());
}
