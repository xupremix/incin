#![cfg(feature = "compiled")]

use incin_core::experimental::compiled::{CapturedGraph, CompileOptions, CompiledPlan, ShapeGuard};
use incin_core::graph::{Graph, OpType};
use incin_core::prelude::DTypeId;
use std::collections::BTreeMap;

#[test]
fn test_compiled_plan_construction_and_guards() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4, 8], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![2, 8], DTypeId::F32, Some("z".into()));

    graph.mark_input(x);
    graph.mark_input(y);
    graph.mark_output(z);
    graph.add_node(OpType::MatMul, vec![x, y], vec![z], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let options = CompileOptions::default();
    let plan = CompiledPlan::compile(captured, options).expect("plan should compile");

    assert_eq!(plan.input_guards.len(), 2);
    assert!(plan.verify_input(0, &[2, 4], DTypeId::F32).is_ok());
    assert!(plan.verify_input(1, &[4, 8], DTypeId::F32).is_ok());
    assert!(plan.verify_input(0, &[2, 5], DTypeId::F32).is_err());
    assert!(plan.verify_input(2, &[1], DTypeId::F32).is_err());
}

#[test]
fn test_shape_guard_verification() {
    let guard = ShapeGuard::new(0, vec![2, 4], DTypeId::F32);
    assert!(guard.check(&[2, 4], DTypeId::F32).is_ok());

    // Mismatched shape fails
    assert!(guard.check(&[2, 5], DTypeId::F32).is_err());

    // Mismatched dtype fails
    assert!(guard.check(&[2, 4], DTypeId::F16).is_err());
}
