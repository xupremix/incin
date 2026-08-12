#![cfg(feature = "compiled")]

use incin_core::experimental::compiled::{CapturedGraph, FusionPass};
use incin_core::graph::{Graph, OpType};
use incin_core::prelude::DTypeId;
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

    graph.add_node(OpType::Add, vec![x, x], vec![y], BTreeMap::new());
    graph.add_node(OpType::Relu, vec![y], vec![z], BTreeMap::new());
    graph.add_node(OpType::Mul, vec![z, x], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let pass = FusionPass;
    let candidates = pass.find_candidates(&captured);

    assert!(candidates.is_empty());
}

#[test]
fn test_fusion_apply_reduces_node_count() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OpType::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OpType::Neg, vec![y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let pass = FusionPass;
    let candidates = pass.find_candidates(&captured);
    let (fused_graph, kernels) = pass
        .apply(&captured, &candidates)
        .expect("fusion should succeed");

    // Fusion remains disabled until composite operation semantics are represented.
    assert_eq!(captured.node_count(), 2);
    assert_eq!(fused_graph.node_count(), captured.node_count());
    assert!(kernels.is_empty());
}

#[test]
fn test_fusion_rejects_consumer_with_unrelated_input() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));
    graph.mark_input(x);
    graph.mark_output(out);
    graph.add_node(OpType::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OpType::Add, vec![y, x], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    assert!(FusionPass.find_candidates(&captured).is_empty());
}

#[test]
fn test_fusion_rejects_producer_with_multiple_consumers() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![4], DTypeId::F32, Some("z".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));
    graph.mark_input(x);
    graph.mark_output(out);
    graph.add_node(OpType::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OpType::Neg, vec![y], vec![z], BTreeMap::new());
    graph.add_node(OpType::Abs, vec![y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    assert!(FusionPass.find_candidates(&captured).is_empty());
}
