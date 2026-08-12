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
}

#[test]
fn test_eager_graph_capture_rejects_undefined_output() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 4], DTypeId::F32, Some("x".into()));
    graph.mark_input(x);
    graph.mark_output(999); // Undefined value ID 999

    assert!(CapturedGraph::capture(&graph).is_err());
}
