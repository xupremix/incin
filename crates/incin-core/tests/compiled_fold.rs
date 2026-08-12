#![cfg(feature = "compiled")]

use incin_core::experimental::compiled::{
    CapturedGraph, ConstantFolder, ShapeBucket, WeightPrepacker,
};
use incin_core::graph::{Graph, OpType};
use incin_core::prelude::DTypeId;
use std::collections::BTreeMap;

#[test]
fn incomplete_compiled_passes_fail_closed() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 4], DTypeId::F32, Some("x".into()));
    let y = graph.add_value(vec![4, 8], DTypeId::F32, Some("y".into()));
    let z = graph.add_value(vec![2, 8], DTypeId::F32, Some("z".into()));

    graph.mark_input(x);
    graph.mark_input(y);
    graph.mark_output(z);
    graph.add_node(OpType::MatMul, vec![x, y], vec![z], BTreeMap::new());

    let captured = CapturedGraph::capture(&graph).expect("capture should succeed");

    let folder = ConstantFolder;
    let fold_error = folder
        .fold(&captured)
        .expect_err("folding is not implemented");
    assert_eq!(
        fold_error.to_string(),
        "Operation 'compiled.constant_fold' is not supported by backend 'compiled-prototype'"
    );

    let prepacker = WeightPrepacker;
    let prepack_error = prepacker
        .prepack(&captured)
        .expect_err("prepacking is not implemented");
    assert_eq!(
        prepack_error.to_string(),
        "Operation 'compiled.weight_prepack' is not supported by backend 'compiled-prototype'"
    );
}

#[test]
fn test_shape_bucket_bounds() {
    let bucket = ShapeBucket::new(vec![1, 1], vec![10, 20]).expect("valid bucket");
    assert!(bucket.contains(&[5, 10]));
    assert!(bucket.contains(&[1, 1]));
    assert!(bucket.contains(&[10, 20]));

    assert!(!bucket.contains(&[0, 10]));
    assert!(!bucket.contains(&[5, 25]));
    assert!(!bucket.contains(&[5]));
}

#[test]
fn shape_bucket_rejects_invalid_bounds() {
    assert!(ShapeBucket::new(vec![1], vec![1, 2]).is_err());
    assert!(ShapeBucket::new(vec![3], vec![2]).is_err());
}
