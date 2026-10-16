//! #150: a module parameter's gradient has a public read path.
//!
//! `Gradients::require_values` reads through whatever shape and layout a
//! module chose for its parameters, and `require_bytes` covers the
//! fully-generic visitor case where no element type can be named. The tests
//! below grad-check parameters inside a composed `Linear` module against
//! analytic values, which is what `gradcheck_model` cannot do: it only
//! checks gradients of hand-built `Tensor` operations.

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::nn::{Linear, Module, Param, ParameterVisitor, StatePath, TrainState, VisitParameters};
use incin_core::prelude::*;
use incin_core::tensor::dtype::DType;

type B = CpuBackendImpl;

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Collects every parameter's gradient through the fully-generic read path.
struct GradBytes {
    grads: Option<incin_core::autograd::Gradients<B>>,
    found: Vec<(String, Vec<u8>)>,
}

impl ParameterVisitor<B> for GradBytes {
    fn visit_param<S, K, Train>(
        &mut self,
        path: &StatePath,
        param: &Param<S, B, K, Train>,
    ) -> incin_core::error::Result<()>
    where
        S: incin_core::shapes::Shape,
        K: DType,
        Train: TrainState,
    {
        let bytes = self.grads.as_ref().unwrap().require_bytes(&param.as_tensor()?)?;
        self.found.push((format!("{path:?}"), bytes));
        Ok(())
    }
}

#[test]
fn module_parameter_gradients_read_against_analytic_values() {
    let model = Linear::<Dyn, B>::build((2, 2)).unwrap();
    let x = Tensor::<Dyn, B, f32, Grad, incin_core::dist::Local, Dyn>::ones(vec![2, 2]).unwrap();
    let grads = model.forward(x).unwrap().sum_all().unwrap().backward().unwrap();
    // Direct concrete read: weight gradient for a 2x2 ones input is 2.0
    // everywhere (each weight sees two unit inputs).
    let w_values = grads.require_values(&model.weight.as_tensor().unwrap()).unwrap();
    assert_eq!(w_values.len(), 4);
    assert!(w_values.iter().all(|v| (*v - 2.0).abs() < 1e-5), "{w_values:?}");
    // Generic visitor sweep: every parameter (weight and bias) reads 16
    // bytes decoding to four finite f32 values.
    let mut collector = GradBytes { grads: Some(grads), found: vec![] };
    model.visit_parameters(&StatePath::root(), &mut collector).unwrap();
    assert_eq!(collector.found.len(), 2, "weight and bias");
    for (path, bytes) in &collector.found {
        let values = decode_f32(bytes);
        // Weight is 4 entries, bias is 2; every entry is 2.0 (two unit
        // inputs per output) and finite.
        assert!(!values.is_empty(), "{path}");
        assert!(values.iter().all(|v| (*v - 2.0).abs() < 1e-5), "{path}: {values:?}");
    }
}

#[test]
fn missing_parameter_gradient_is_a_structured_error() {
    let model = Linear::<Dyn, B>::build((2, 2)).unwrap();
    let x = Tensor::<Dyn, B, f32, Grad, incin_core::dist::Local, Dyn>::ones(vec![2, 2]).unwrap();
    // No forward, no backward: nothing was ever recorded.
    let grads = x.sum_all().unwrap().backward().unwrap();
    let mut collector = GradBytes { grads: Some(grads), found: vec![] };
    let result = model.visit_parameters(&StatePath::root(), &mut collector);
    assert!(result.is_err(), "an unrecorded parameter must fail closed");
}
