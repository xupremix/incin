#![cfg(feature = "compiled")]

use incin_core::experimental::compiled::{
    AllocationPlanner, CapturedGraph, LivenessInterval, LivenessMap, MemoryPlan, SavedTensorSet,
};
use incin_core::graph::Graph;
use incin_core::prelude::DTypeId;
use incin_core::prelude::OperationKind;
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
    graph.add_node(OperationKind::MatMul, vec![x, y], vec![z], BTreeMap::new());

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
    graph.add_node(OperationKind::MatMul, vec![x, y], vec![z], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let liveness = LivenessMap::compute(&captured);
    let planner = AllocationPlanner;
    let plan = planner
        .plan(&liveness, &captured)
        .expect("planning should succeed");

    // All values should be assigned slots
    assert!(plan.assignments().contains_key(&x));
    assert!(plan.assignments().contains_key(&y));
    assert!(plan.assignments().contains_key(&z));
}

#[test]
fn allocation_planner_reuses_slots_after_an_intermediate_dies() {
    let mut graph = Graph::new();
    let input = graph.add_value(vec![4], DTypeId::F32, Some("input".into()));
    let first = graph.add_value(vec![4], DTypeId::F32, Some("first".into()));
    let output = graph.add_value(vec![4], DTypeId::F32, Some("output".into()));
    graph.mark_input(input);
    graph.mark_output(output);
    graph.add_node(
        OperationKind::Relu,
        vec![input],
        vec![first],
        BTreeMap::new(),
    );
    graph.add_node(
        OperationKind::Relu,
        vec![first],
        vec![output],
        BTreeMap::new(),
    );

    let captured = CapturedGraph::capture(&graph).unwrap();
    let liveness = LivenessMap::compute(&captured);
    let plan = AllocationPlanner.plan(&liveness, &captured).unwrap();

    assert_eq!(plan.assignments()[&input], plan.assignments()[&output]);
}

#[test]
fn test_saved_tensor_extends_liveness() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));

    graph.mark_input(x);
    graph.mark_output(y);
    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let mut liveness = LivenessMap::compute(&captured);

    // x should be live at most up to node 0 (relu)
    let before = liveness.get(x).unwrap();
    assert!(before.last_use_node() <= 1);

    // Mark x as saved for the backward pass (backward ends at node 5)
    let mut saved = SavedTensorSet::new();
    saved.save(x);
    liveness.extend_for_saved_tensors(&saved, 5);

    // Now x should be live until the backward end
    let after = liveness.get(x).unwrap();
    assert_eq!(after.last_use_node(), 5);
}

#[test]
fn test_saved_tensor_does_not_shrink_liveness() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));

    graph.mark_input(x);
    graph.mark_output(y);
    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let mut liveness = LivenessMap::compute(&captured);

    let before_last = liveness.get(x).unwrap().last_use_node();

    // extend with backward_end_node = 0 should not shrink anything
    let mut saved = SavedTensorSet::new();
    saved.save(x);
    liveness.extend_for_saved_tensors(&saved, 0);

    let after_last = liveness.get(x).unwrap().last_use_node();
    assert!(after_last >= before_last);
}

#[test]
fn liveness_deserialization_revalidates_interval_order_and_round_trips() {
    assert!(
        serde_json::from_str::<LivenessInterval>(r#"{"def_node":4,"last_use_node":3}"#).is_err()
    );

    let valid: LivenessInterval =
        serde_json::from_str(r#"{"def_node":3,"last_use_node":4}"#).unwrap();
    assert_eq!(valid.def_node(), 3);
    assert_eq!(valid.last_use_node(), 4);
    assert_eq!(
        serde_json::from_str::<LivenessInterval>(&serde_json::to_string(&valid).unwrap()).unwrap(),
        valid
    );
}

#[test]
fn memory_plan_deserialization_revalidates_slots_and_round_trips() {
    let mut graph = Graph::new();
    let input = graph.add_value(vec![2], DTypeId::F32, Some("input".into()));
    let output = graph.add_value(vec![2], DTypeId::F32, Some("output".into()));
    graph.mark_input(input);
    graph.mark_output(output);
    graph.add_node(
        OperationKind::Relu,
        vec![input],
        vec![output],
        BTreeMap::new(),
    );

    let captured = CapturedGraph::capture(&graph).unwrap();
    let liveness = LivenessMap::compute(&captured);
    let plan = AllocationPlanner.plan(&liveness, &captured).unwrap();
    assert!(
        plan.assignments()
            .values()
            .all(|slot| slot.index() < plan.peak_live_slots())
    );

    let encoded = serde_json::to_value(&plan).unwrap();
    assert_eq!(
        serde_json::from_value::<MemoryPlan>(encoded.clone()).unwrap(),
        plan
    );

    let mut malformed = encoded;
    malformed["peak_live_slots"] = serde_json::json!(0);
    assert!(serde_json::from_value::<MemoryPlan>(malformed).is_err());
}
