//! Integration coverage for `test_fusion_detects_pointwise_chain_candidates` on the documented public surface.
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

/// A value with two readers must not be fused away.
///
/// This is the case the positional scan got wrong. `y` feeds both the `Relu`
/// immediately after it and a `Mul` further down. Fusing `Add` into `Relu`
/// makes `y` cease to exist, and the `Mul` is then reading a value nothing
/// produces. The old scan paired node 0 with node 1 by position and never
/// counted readers, so it offered exactly this fusion.
#[test]
fn a_value_with_two_consumers_is_not_a_fusion_candidate() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![4], DTypeId::F32, Some("z".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::Add, vec![x, x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Relu, vec![y], vec![z], BTreeMap::new());
    // Second reader of `y`, further down the graph.
    graph.add_node(OperationKind::Mul, vec![z, y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let candidates = FusionPass.find_candidates(&captured);

    assert!(
        !candidates
            .iter()
            .any(|candidate| candidate.producer_op == OperationKind::Add),
        "Add produces a value with two readers and must not be fused: {candidates:?}"
    );
}

/// A producer and its consumer need not be adjacent.
///
/// The old scan only paired node `i` with node `i + 1`, so an intervening
/// unrelated node hid a perfectly legal fusion. Following the edge finds it.
#[test]
fn a_non_adjacent_consumer_is_still_a_candidate() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let unrelated = graph.add_value(vec![4], DTypeId::F32, Some("unrelated".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);
    graph.mark_output(unrelated);

    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    // Independent of `y`, sitting between producer and consumer.
    graph.add_node(
        OperationKind::Neg,
        vec![x],
        vec![unrelated],
        BTreeMap::new(),
    );
    graph.add_node(OperationKind::Neg, vec![y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let candidates = FusionPass.find_candidates(&captured);

    let relu = candidates
        .iter()
        .find(|candidate| candidate.producer_op == OperationKind::Relu)
        .expect("Relu's single consumer is two nodes away but still exclusive");
    assert_eq!(relu.producer_idx, 0);
    assert_eq!(relu.consumer_idx, 2);
}

/// Two unrelated neighbours must not be mistaken for a producer and consumer.
///
/// Both nodes read the graph input and neither reads the other's output, so
/// there is no intermediate to eliminate. Position alone made them look paired.
#[test]
fn unrelated_neighbours_are_not_a_fusion_candidate() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let a = graph.add_value(vec![4], DTypeId::F32, Some("a".into()));
    let b = graph.add_value(vec![4], DTypeId::F32, Some("b".into()));

    graph.mark_input(x);
    graph.mark_output(a);
    graph.mark_output(b);

    graph.add_node(OperationKind::Relu, vec![x], vec![a], BTreeMap::new());
    graph.add_node(OperationKind::Neg, vec![x], vec![b], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let candidates = FusionPass.find_candidates(&captured);

    assert!(
        candidates.is_empty(),
        "neither node consumes the other's output: {candidates:?}"
    );
}

/// A value read twice by one node is still exclusively consumed.
///
/// `y + y` needs `y` to exist for exactly one node, and a fused body can name
/// its operand as many times as it likes. Counting mentions rather than reader
/// nodes would refuse this fusion for no reason.
#[test]
fn a_value_read_twice_by_one_node_is_still_fusable() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Add, vec![y, y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let candidates = FusionPass.find_candidates(&captured);

    assert!(
        candidates
            .iter()
            .any(|candidate| candidate.producer_op == OperationKind::Relu),
        "one reader node is exclusive even when it reads twice: {candidates:?}"
    );
}
