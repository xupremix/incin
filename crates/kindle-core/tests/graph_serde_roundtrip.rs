extern crate kindle_core as kindle;

use kindle_core::prelude::*;
use kindle_core::prelude::{Graph, OpType};

#[test]
/// Graph round trips through serde json.
fn graph_round_trips_through_serde_json() {
    let mut g = Graph::new();
    let v0 = g.add_value(vec![2, 3], DTypeId::F32, Some("x".into()));
    let v1 = g.add_value(vec![2, 3], DTypeId::F32, None);
    g.add_node(OpType::Relu, vec![v0], vec![v1], Default::default());

    let serialized = serde_json::to_string(&g).unwrap();
    let roundtripped: Graph = serde_json::from_str(&serialized).unwrap();

    assert_eq!(g, roundtripped);
}
