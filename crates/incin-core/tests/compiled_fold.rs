use incin_core::compiled::{CapturedGraph, ConstantFolder, ShapeBucket, WeightPrepacker};
use incin_core::graph::{Graph, OpType};
use incin_core::prelude::DTypeId;
use std::collections::BTreeMap;

#[test]
fn test_constant_folder_and_weight_prepacker_pass() {
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
    let (folded_graph, _folded_ids) = folder.fold(&captured).expect("folding should succeed");
    assert_eq!(folded_graph.node_count(), 1);

    let prepacker = WeightPrepacker;
    let packed_graph = prepacker
        .prepack(&folded_graph)
        .expect("prepack should succeed");
    assert_eq!(packed_graph.node_count(), 1);
}

#[test]
fn test_shape_bucket_bounds() {
    let bucket = ShapeBucket::new(vec![1, 1], vec![10, 20]);
    assert!(bucket.contains(&[5, 10]));
    assert!(bucket.contains(&[1, 1]));
    assert!(bucket.contains(&[10, 20]));

    assert!(!bucket.contains(&[0, 10]));
    assert!(!bucket.contains(&[5, 25]));
    assert!(!bucket.contains(&[5]));
}
