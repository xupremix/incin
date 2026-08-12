#![cfg(feature = "compiled")]

use incin_core::experimental::compiled::{CapturedGraph, FusionPass};
use incin_core::graph::Graph;
use incin_core::prelude::DTypeId;
use incin_core::prelude::OperationKind;
use std::collections::BTreeMap;

#[test]
fn test_fusion_detects_pointwise_chain_candidates() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![4], DTypeId::F32, Some("z".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::Add, vec![x, x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Relu, vec![y], vec![z], BTreeMap::new());
    graph.add_node(OperationKind::Mul, vec![z, x], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let pass = FusionPass;
    let candidates = pass.find_candidates(&captured);

    // Add->Relu should be a candidate (both pointwise, output not in graph outputs)
    assert!(!candidates.is_empty());
    assert_eq!(candidates[0].producer_op, OperationKind::Add);
    assert_eq!(candidates[0].consumer_op, OperationKind::Relu);
}

#[test]
fn test_fusion_apply_fails_closed_without_executable_lowering() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Neg, vec![y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let pass = FusionPass;
    let candidates = pass.find_candidates(&captured);
    assert!(pass.apply(&captured, &candidates).is_err());
}
