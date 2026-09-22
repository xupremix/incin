//! Integration coverage for CMP-005 fusion discovery, link gates, and group
//! planning on the documented public surface.
#![cfg(feature = "compiled")]

use incin_core::compiled::{
    FusionBlocker, FusionPatternKind, GroupRefusal, SavedTensorSet, fusion_pattern_kind,
};
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

/// A three-node exclusive path plans as one fusion group.
///
/// `plan_groups` must chain accepted edges into a maximal path: first-wins on
/// consumers may not treat the junction node (Relu's consumer, Add's producer)
/// as a conflict, or every chain longer than two nodes would shatter.
#[test]
fn plan_groups_chains_an_exclusive_three_node_path() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let c = graph.add_value(vec![4], DTypeId::F32, Some("c".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![4], DTypeId::F32, Some("z".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_input(c);
    graph.mark_output(out);

    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Add, vec![y, c], vec![z], BTreeMap::new());
    graph.add_node(OperationKind::Mul, vec![z, x], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let candidates = FusionPass.find_candidates(&captured);
    let planned = FusionPass.plan_groups(&captured, &candidates, &SavedTensorSet::new());

    assert_eq!(planned.groups.len(), 1, "refused: {:?}", planned.refused);
    assert_eq!(planned.groups[0].source_node_indices, vec![0, 1, 2]);
    assert_eq!(planned.groups[0].primary_op, OperationKind::Relu);
    assert!(planned.refused.is_empty(), "{:?}", planned.refused);
}

/// An intermediate on the backward tape cannot disappear into a fused group.
#[test]
fn plan_groups_refuses_an_intermediate_saved_for_backward() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Neg, vec![y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let candidates = FusionPass.find_candidates(&captured);
    assert_eq!(candidates.len(), 1, "{candidates:?}");

    let mut saved = SavedTensorSet::new();
    saved.save(y);
    let planned = FusionPass.plan_groups(&captured, &candidates, &saved);

    assert!(planned.groups.is_empty());
    assert_eq!(
        planned.refused,
        vec![GroupRefusal::Link {
            producer_idx: 0,
            consumer_idx: 1,
            blocker: FusionBlocker::SavedForBackward,
        }]
    );
}

/// When two edges compete for the same consumer, the first wins and the loser
/// is refused; the winner's path still forms a valid group.
#[test]
fn plan_groups_first_wins_when_two_edges_compete_for_one_consumer() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![4], DTypeId::F32, Some("z".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Neg, vec![x], vec![z], BTreeMap::new());
    graph.add_node(OperationKind::Add, vec![y, z], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let candidates = FusionPass.find_candidates(&captured);
    let planned = FusionPass.plan_groups(&captured, &candidates, &SavedTensorSet::new());

    assert_eq!(planned.groups.len(), 1, "refused: {:?}", planned.refused);
    assert_eq!(planned.groups[0].source_node_indices, vec![0, 2]);
    assert_eq!(planned.groups[0].primary_op, OperationKind::Relu);
    assert_eq!(
        planned.refused,
        vec![GroupRefusal::Link {
            producer_idx: 1,
            consumer_idx: 2,
            blocker: FusionBlocker::MultipleProducers,
        }]
    );
}

/// A path whose values do not share one shape is refused as a whole: a single
/// pointwise kernel iterates one geometry.
#[test]
fn plan_groups_refuses_a_path_whose_shapes_disagree() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![2], DTypeId::F32, Some("y".into()));
    let out = graph.add_value(vec![2], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Neg, vec![y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let candidates = FusionPass.find_candidates(&captured);
    assert_eq!(candidates.len(), 1, "{candidates:?}");
    let planned = FusionPass.plan_groups(&captured, &candidates, &SavedTensorSet::new());

    assert!(planned.groups.is_empty());
    assert_eq!(
        planned.refused,
        vec![GroupRefusal::Chain {
            node_indices: vec![0, 1],
            blocker: FusionBlocker::ShapeMismatch,
        }]
    );
}

/// A non-elementwise producer is refused at the link gate.
#[test]
fn check_link_refuses_a_non_elementwise_producer() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::MatMul, vec![x, x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Relu, vec![y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let blocker = FusionPass
        .check_link(&captured, 0, 1)
        .expect_err("MatMul is not in the pointwise fusion class");
    assert_eq!(blocker, FusionBlocker::NotElementwise);
}

/// A producer output that escapes the graph has an external consumer the edge
/// list cannot show, so the link is refused.
#[test]
fn check_link_refuses_when_the_producer_output_escapes_the_graph() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(y);
    graph.mark_output(out);

    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Neg, vec![y], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let blocker = FusionPass
        .check_link(&captured, 0, 1)
        .expect_err("y is read by a caller outside the graph");
    assert_eq!(blocker, FusionBlocker::GraphOutput);
}

/// Two nodes reading the intermediate mean fusing the producer away would
/// strand the second reader.
#[test]
fn check_link_refuses_a_producer_with_two_consumers() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let a = graph.add_value(vec![4], DTypeId::F32, Some("a".into()));
    let b = graph.add_value(vec![4], DTypeId::F32, Some("b".into()));

    graph.mark_input(x);
    graph.mark_output(a);
    graph.mark_output(b);

    graph.add_node(OperationKind::Add, vec![x, x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Relu, vec![y], vec![a], BTreeMap::new());
    graph.add_node(OperationKind::Neg, vec![y], vec![b], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let blocker = FusionPass
        .check_link(&captured, 0, 1)
        .expect_err("y feeds both Relu and Neg");
    assert_eq!(blocker, FusionBlocker::MultipleProducers);
}

/// Nothing consumes the intermediate, so there is no consumer to fuse into.
#[test]
fn check_link_refuses_a_producer_with_no_consumer() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4], DTypeId::F32, Some("y".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::Relu, vec![x], vec![y], BTreeMap::new());
    graph.add_node(OperationKind::Neg, vec![x], vec![out], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    let blocker = FusionPass
        .check_link(&captured, 0, 1)
        .expect_err("y has no consumer to fuse with");
    assert_eq!(blocker, FusionBlocker::NotProvenExclusive);
}

/// A consumer that does not follow its producer in topological order means a
/// malformed graph, not a fusion opportunity.
///
/// The nodes are captured in a valid order, then swapped, so the pair (1, 0)
/// is checked: Relu now sits after the Neg that reads its output.
#[test]
fn check_link_refuses_an_out_of_order_pair() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![4], DTypeId::F32, Some("x".into()));
    let a = graph.add_value(vec![4], DTypeId::F32, Some("a".into()));
    let b = graph.add_value(vec![4], DTypeId::F32, Some("b".into()));
    let out = graph.add_value(vec![4], DTypeId::F32, Some("out".into()));

    graph.mark_input(x);
    graph.mark_output(out);

    graph.add_node(OperationKind::Relu, vec![x], vec![a], BTreeMap::new());
    graph.add_node(OperationKind::Neg, vec![a], vec![b], BTreeMap::new());
    graph.add_node(OperationKind::Abs, vec![b], vec![out], BTreeMap::new());

    let mut captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    captured.nodes.swap(0, 1);

    let blocker = FusionPass
        .check_link(&captured, 1, 0)
        .expect_err("consumer index 0 cannot precede producer index 1");
    assert_eq!(blocker, FusionBlocker::OutOfTopologicalOrder);
}

/// The allow-list is exactly the CMP-005 pointwise set; everything else is
/// opaque and never enters a chain.
#[test]
fn fusion_pattern_kind_classifies_the_pointwise_allow_list() {
    use OperationKind::*;
    for op in [
        Add, Sub, Mul, Div, Relu, Gelu, Sigmoid, Tanh, Swish, Neg, Abs, Exp, Sqrt, Log,
    ] {
        assert_eq!(
            fusion_pattern_kind(op),
            FusionPatternKind::ElemWise,
            "{op:?} must be in the pointwise allow-list"
        );
    }
    for op in [MatMul, Conv2d, Storage, Reduction, Concat, Reshape] {
        assert_eq!(
            fusion_pattern_kind(op),
            FusionPatternKind::Opaque,
            "{op:?} must never be fused"
        );
    }
}
