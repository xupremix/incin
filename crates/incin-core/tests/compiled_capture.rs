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
