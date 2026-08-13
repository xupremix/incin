#![cfg(feature = "compiled")]

extern crate incin_core as incin;

use incin_core::exec::OperationIdentity;
use incin_core::exec::{DimExpr, SymbolId};
use incin_core::experimental::compiled::CapturedGraph;
use incin_core::graph::Graph;
use incin_core::prelude::DTypeId;
use incin_core::prelude::OperationKind;
use std::collections::BTreeMap;

incin_core::dim!(Batch);
incin_core::dim!(Sequence);

#[test]
fn test_eager_graph_capture_and_validation() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4, 8], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![2, 8], DTypeId::F32, Some("z".into()));

    graph.mark_input(x);
    graph.mark_input(y);
    graph.mark_output(z);

    graph.add_node(
        OperationKind::MatMulExact,
        vec![x, y],
        vec![z],
        BTreeMap::new(),
    );

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");
    assert_eq!(captured.node_count(), 1);
    assert_eq!(captured.value_count(), 3);
    assert_eq!(captured.inputs, vec![x, y]);
    assert_eq!(captured.outputs, vec![z]);
    assert_eq!(
        captured.nodes[0].operation,
        OperationIdentity::Builtin(OperationKind::MatMulExact)
    );
    assert_eq!(
        captured.nodes[0].execution_site,
        Some(incin_core::exec::ExecutionSite::Kernel)
    );
    assert!(matches!(
        captured.value_metadata[&x].shape_expr.dims[0],
        DimExpr::Symbol(SymbolId(_))
    ));
    assert_eq!(
        captured.value_metadata[&x].dtype.builtin_id(),
        Some(DTypeId::F32)
    );
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
fn typed_input_projection_keeps_static_axes_and_symbols_runtime_axes() {
    let mut graph = Graph::new();
    let input = graph.add_value(vec![7, 768], DTypeId::F32, Some("input".into()));
    graph.mark_input_with_shape::<incin::prelude::s![usize, 768]>(input);

    let expression = &graph.values[&input].shape_expr;
    assert!(matches!(expression.dims[0], DimExpr::Symbol(SymbolId(_))));
    assert_eq!(expression.dims[1], DimExpr::Const(768));
}

#[test]
fn typed_named_projection_preserves_identity_and_shares_named_symbols() {
    let mut graph = Graph::new();
    let first = graph.add_value(vec![7, 768], DTypeId::F32, Some("first".into()));
    let second = graph.add_value(vec![9, 768], DTypeId::F32, Some("second".into()));
    graph.mark_input_with_shape::<incin::prelude::s![Batch, 768]>(first);
    graph.mark_input_with_shape::<incin::prelude::s![Batch, 768]>(second);

    let first_expr = &graph.values[&first].shape_expr.dims[0];
    let second_expr = &graph.values[&second].shape_expr.dims[0];
    assert!(matches!(first_expr, DimExpr::NamedSymbol { name, .. } if name == "Batch"));
    assert_eq!(first_expr, second_expr);
    assert_eq!(graph.values[&first].shape_expr.dims[1], DimExpr::Const(768));
}

#[test]
fn typed_named_projection_does_not_merge_distinct_axis_tags() {
    let mut graph = Graph::new();
    let batch = graph.add_value(vec![7], DTypeId::F32, Some("batch".into()));
    let sequence = graph.add_value(vec![9], DTypeId::F32, Some("sequence".into()));

    graph.mark_input_with_shape::<incin::prelude::s![Batch]>(batch);
    graph.mark_input_with_shape::<incin::prelude::s![Sequence]>(sequence);

    let batch_expr = &graph.values[&batch].shape_expr.dims[0];
    let sequence_expr = &graph.values[&sequence].shape_expr.dims[0];
    assert_ne!(batch_expr, sequence_expr);
    assert!(matches!(
        batch_expr,
        DimExpr::NamedSymbol { name, identity, .. }
            if name == "Batch" && identity.ends_with("::Batch")
    ));
    assert!(matches!(
        sequence_expr,
        DimExpr::NamedSymbol { name, identity, .. }
            if name == "Sequence" && identity.ends_with("::Sequence")
    ));
}

#[test]
fn graph_symbol_allocator_keeps_anonymous_and_named_axes_disjoint() {
    let mut graph = Graph::new();
    let anonymous = graph.add_value(vec![2, 3], DTypeId::F32, Some("anonymous".into()));
    let named = graph.add_value(vec![5], DTypeId::F32, Some("named".into()));
    graph.mark_input(anonymous);
    graph.mark_input_with_shape::<incin::prelude::s![Batch]>(named);

    let anonymous_ids = graph.values[&anonymous]
        .shape_expr
        .dims
        .iter()
        .filter_map(|expr| match expr {
            DimExpr::Symbol(id) | DimExpr::NamedSymbol { id, .. } => Some(*id),
            _ => None,
        })
        .collect::<std::collections::BTreeSet<_>>();
    let named_id = match &graph.values[&named].shape_expr.dims[0] {
        DimExpr::NamedSymbol { id, .. } => *id,
        other => panic!("expected named symbol, got {other:?}"),
    };
    assert!(!anonymous_ids.contains(&named_id));
    assert_eq!(anonymous_ids.len(), 2);
}

#[test]
fn graph_symbol_allocator_survives_serialization_and_continues_fresh_ids() {
    let mut graph = Graph::new();
    let first = graph.add_value(vec![2], DTypeId::F32, Some("first".into()));
    graph.mark_input(first);
    let encoded = serde_json::to_vec(&graph).unwrap();
    let mut restored: Graph = serde_json::from_slice(&encoded).unwrap();
    let second = restored.add_value(vec![3], DTypeId::F32, Some("second".into()));
    restored.mark_input(second);

    let first_id = match restored.values[&first].shape_expr.dims[0] {
        DimExpr::Symbol(id) => id,
        ref other => panic!("expected anonymous symbol, got {other:?}"),
    };
    let second_id = match restored.values[&second].shape_expr.dims[0] {
        DimExpr::Symbol(id) => id,
        ref other => panic!("expected anonymous symbol, got {other:?}"),
    };
    assert_ne!(first_id, second_id);
}
