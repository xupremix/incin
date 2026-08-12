#![cfg(feature = "compiled")]

use incin_core::experimental::compiled::CapturedGraph;
use incin_core::graph::{Graph, OpType};
use incin_core::prelude::DTypeId;
use std::collections::BTreeMap;

#[test]
fn test_eager_graph_capture_and_validation() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4, 8], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![2, 8], DTypeId::F32, Some("z".into()));

    graph.mark_input(x);
    graph.mark_input(y);
    graph.mark_output(z);

    graph.add_node(OpType::MatMul, vec![x, y], vec![z], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    assert_eq!(captured.node_count(), 1);
    assert_eq!(captured.value_count(), 3);
    assert_eq!(captured.inputs, vec![x, y]);
    assert_eq!(captured.outputs, vec![z]);
    assert_eq!(captured.nodes[0].op, OpType::MatMul);
    assert_eq!(captured.value(x).expect("captured input").shape, vec![2, 4]);
}

#[test]
fn test_capture_preserves_node_metadata() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2], DTypeId::F32, Some("x".into()));
    let out = graph.add_value(vec![2], DTypeId::F32, Some("out".into()));
    graph.mark_input(x);
    graph.mark_output(out);
    let mut attributes = BTreeMap::new();
    attributes.insert("axis".into(), incin_core::graph::AttributeValue::Int(1));
    graph.add_node(OpType::Relu, vec![x], vec![out], attributes.clone());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    assert_eq!(captured.nodes[0].attributes, attributes);
    assert_eq!(captured.nodes[0].identity, None);
}

#[test]
fn test_eager_graph_capture_rejects_undefined_output() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 4], DTypeId::F32, Some("x".into()));
    graph.mark_input(x);
    graph.mark_output(999); // Undefined value ID 999

    assert!(CapturedGraph::capture(&graph).is_err());
}

#[test]
fn test_capture_rejects_forward_reference() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2], DTypeId::F32, Some("x".into()));
    let later = graph.add_value(vec![2], DTypeId::F32, Some("later".into()));
    let out = graph.add_value(vec![2], DTypeId::F32, Some("out".into()));
    graph.mark_input(x);
    graph.mark_output(out);
    graph.add_node(OpType::Add, vec![later, x], vec![out], BTreeMap::new());
    graph.add_node(OpType::Relu, vec![x], vec![later], BTreeMap::new());

    assert!(CapturedGraph::capture(&graph).is_err());
}

#[test]
fn test_capture_rejects_duplicate_producer() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2], DTypeId::F32, Some("x".into()));
    let out = graph.add_value(vec![2], DTypeId::F32, Some("out".into()));
    graph.mark_input(x);
    graph.mark_output(out);
    graph.add_node(OpType::Relu, vec![x], vec![out], BTreeMap::new());
    graph.add_node(OpType::Neg, vec![x], vec![out], BTreeMap::new());

    assert!(CapturedGraph::capture(&graph).is_err());
}

#[test]
fn test_captured_graph_rejects_duplicate_node_ids() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2], DTypeId::F32, Some("x".into()));
    let out = graph.add_value(vec![2], DTypeId::F32, Some("out".into()));
    graph.mark_input(x);
    graph.mark_output(out);
    graph.add_node(OpType::Relu, vec![x], vec![out], BTreeMap::new());
    let mut captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    captured.nodes.push(captured.nodes[0].clone());

    assert!(captured.validate().is_err());
}
