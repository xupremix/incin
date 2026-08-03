#![cfg(feature = "compiled")]

use incin_core::experimental::compiled::{
    AllocationPlanner, CapturedGraph, LivenessMap, SavedTensorSet,
};
use incin_core::graph::{Graph, OpType};
use incin_core::prelude::DTypeId;
use std::collections::BTreeMap;

#[test]
fn test_liveness_analysis_basic() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4, 8], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![2, 8], DTypeId::F32, Some("z".into()));

    graph.mark_input(x);
    graph.mark_input(y);
    graph.mark_output(z);
    graph.add_node(OpType::MatMul, vec![x, y], vec![z], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let liveness = LivenessMap::compute(&captured);

    // x and y are graph inputs, defined at node 0
    assert!(liveness.get(x).is_some());
    assert!(liveness.get(y).is_some());

    // z is defined at node 0 (the MatMul) and is a graph output
    assert!(liveness.get(z).is_some());
}

#[test]
fn test_allocation_planner_peak_memory() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4, 8], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![2, 8], DTypeId::F32, Some("z".into()));

    graph.mark_input(x);
    graph.mark_input(y);
    graph.mark_output(z);
    graph.add_node(OpType::MatMul, vec![x, y], vec![z], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let liveness = LivenessMap::compute(&captured);
    let planner = AllocationPlanner;
    let plan = planner
        .plan(&liveness, &captured)
        .expect("planning should succeed");

    // All values should be assigned slots
    assert!(plan.assignments.contains_key(&x));
    assert!(plan.assignments.contains_key(&y));
    assert!(plan.assignments.contains_key(&z));
}

#[test]
fn test_saved_tensor_extends_liveness() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));

    graph.mark_input(x);
    graph.mark_output(y);
    graph.add_node(OpType::Relu, vec![x], vec![y], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let mut liveness = LivenessMap::compute(&captured);

    // x should be live at most up to node 0 (relu)
    let before = liveness.get(x).unwrap();
    assert!(before.last_use_node <= 1);

    // Mark x as saved for the backward pass (backward ends at node 5)
    let mut saved = SavedTensorSet::new();
    saved.save(x);
    liveness.extend_for_saved_tensors(&saved, 5);

    // Now x should be live until the backward end
    let after = liveness.get(x).unwrap();
    assert_eq!(after.last_use_node, 5);
}

#[test]
fn test_saved_tensor_does_not_shrink_liveness() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));

    graph.mark_input(x);
    graph.mark_output(y);
    graph.add_node(OpType::Relu, vec![x], vec![y], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let mut liveness = LivenessMap::compute(&captured);

    let before_last = liveness.get(x).unwrap().last_use_node;

    // extend with backward_end_node = 0 should not shrink anything
    let mut saved = SavedTensorSet::new();
    saved.save(x);
    liveness.extend_for_saved_tensors(&saved, 0);

    let after_last = liveness.get(x).unwrap().last_use_node;
    assert!(after_last >= before_last);
}
