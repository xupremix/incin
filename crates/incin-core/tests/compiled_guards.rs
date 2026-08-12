#![cfg(feature = "compiled")]

use incin_core::exec::ShapeExpr;
use incin_core::exec::{Constraint, DimExpr, RankExpr, SymbolId};
use incin_core::experimental::compiled::{CapturedGraph, CompileOptions, CompiledPlan, ShapeGuard};
use incin_core::graph::Graph;
use incin_core::prelude::DTypeId;
use incin_core::prelude::OperationKind;
use std::collections::BTreeMap;

#[test]
fn test_compiled_plan_construction_and_guards() {
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
    let options = CompileOptions::default();
    let plan = CompiledPlan::compile(captured, options);

    assert_eq!(plan.input_guards.len(), 2);
    assert!(plan.verify_input(2, &[1], DTypeId::F32).is_err());
}

#[test]
fn test_shape_guard_verification() {
    let guard = ShapeGuard::new(0, ShapeExpr::concrete(&[2, 4]), DTypeId::F32.into());
    assert!(guard.check(&[2, 4], DTypeId::F32.into()).is_ok());

    // Mismatched shape fails
    assert!(guard.check(&[2, 5], DTypeId::F32.into()).is_err());

    // Mismatched dtype fails
    assert!(guard.check(&[2, 4], DTypeId::F16.into()).is_err());
}

#[test]
fn symbolic_guard_accepts_valid_alternatives_and_rejects_bad_relations() {
    let batch = DimExpr::Symbol(SymbolId(1));
    let shape = ShapeExpr {
        rank: RankExpr::Static(2),
        dims: vec![batch.clone(), DimExpr::Const(768)],
        constraints: vec![
            Constraint::LowerBound {
                value: batch.clone(),
                bound: 1,
            },
            Constraint::Divisible {
                value: batch.clone(),
                divisor: 1,
            },
        ],
    };
    let guard = ShapeGuard::new(0, shape, DTypeId::F16.into());
    assert!(guard.check(&[1, 768], DTypeId::F16.into()).is_ok());
    assert!(guard.check(&[32, 768], DTypeId::F16.into()).is_ok());
    assert!(guard.check(&[32, 767], DTypeId::F16.into()).is_err());
    assert!(guard.check(&[0, 768], DTypeId::F16.into()).is_err());
}
