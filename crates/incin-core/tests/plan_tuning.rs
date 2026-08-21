//! Preview tuning placeholders fail closed until a measurement hook exists.

#![cfg(feature = "compiled")]

use std::collections::BTreeMap;

use incin_core::experimental::compiled::{
    BoundedPlanTuner, CapturedGraph, CompileOptions, CompiledPlan,
};
use incin_core::graph::Graph;
use incin_core::prelude::DTypeId;
use incin_core::prelude::OperationKind;

fn make_test_plan_with_nodes(node_count: usize) -> CompiledPlan {
    let mut graph = Graph::new();
    let first = graph.add_value(vec![4], DTypeId::F32, Some("x0".into()));
    graph.mark_input(first);
    let mut prev = first;
    for i in 0..node_count {
        let next = graph.add_value(vec![4], DTypeId::F32, Some(format!("x{}", i + 1)));
        graph.add_node(OperationKind::Relu, vec![prev], vec![next], BTreeMap::new());
        prev = next;
    }
    graph.mark_output(prev);
    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    CompiledPlan::compile(captured, CompileOptions::new()).unwrap()
}

#[test]
fn bounded_plan_tuning_is_unavailable_for_nonempty_plan() {
    // A nonempty plan must not manufacture a proxy measurement or report.
    let plan = make_test_plan_with_nodes(10);

    let tuner = BoundedPlanTuner::new(10, 1.15);
    assert!(tuner.tune_plan(&plan).is_err());
}

#[test]
fn bounded_plan_tuning_is_unavailable_for_minimal_plan() {
    // A minimal plan has the same fail-closed result.
    let plan = make_test_plan_with_nodes(1);
    let tuner = BoundedPlanTuner::new(5, 1.0);
    assert!(tuner.tune_plan(&plan).is_err());
}
