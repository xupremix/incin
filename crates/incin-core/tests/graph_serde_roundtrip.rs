extern crate incin_core as incin;

use incin_core::graph::Graph;
use incin_core::prelude::OperationKind;
use incin_core::prelude::*;

use incin_core::exec::catalog::{ScalarAttributes, TraceDescriptor, op};

#[test]
/// Graph round trips through serde json.
fn graph_round_trips_through_serde_json() {
    let mut g = Graph::new();
    let v0 = g.add_value(vec![2, 3], DTypeId::F32, Some("x".into()));
    let v1 = g.add_value(vec![2, 3], DTypeId::F32, None);
    g.add_node(OperationKind::Relu, vec![v0], vec![v1], Default::default());

    let serialized = serde_json::to_string(&g).unwrap();
    let roundtripped: Graph = serde_json::from_str(&serialized).unwrap();

    assert_eq!(g, roundtripped);
}

#[test]
fn captured_descriptor_projection_preserves_typed_attributes() {
    let descriptor = incin_core::exec::catalog::Descriptor::<op::AddScalar>::infer_runtime(
        ScalarAttributes { value: 2.5 },
        vec![incin_core::exec::catalog::LogicalTensorMeta {
            shape: Some(ShapeBuf::from_slice(&[2, 3])),
            dtype: Some(DTypeId::F32.into()),
            device: Some(DeviceId::CPU),
        }],
    )
    .unwrap();
    let attributes = descriptor.descriptor().trace_attributes().unwrap();
    assert_eq!(
        attributes.get("value"),
        Some(&incin_core::graph::AttributeValue::Float(2.5))
    );
}
