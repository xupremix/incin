#![cfg(feature = "compiled")]

use incin_backends::cpu::{CpuBuffer, CpuCompiledInvocation, CpuStorage};
use incin_core::compiled::{CapturedGraph, CompileOptions, CompiledPlan};
use incin_core::exec::OperationIdentity;
use incin_core::exec::catalog::{
    CapturedDescriptor, Descriptor, LogicalTensorMeta, NoAttributes, op,
};
use incin_core::graph::Graph;
use incin_core::prelude::{DTypeId, DeviceId, OperationKind, ShapeBuf};

#[test]
fn compiled_cpu_executes_captured_relu_through_canonical_descriptor() {
    let input_meta = LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(&[2])),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let descriptor = Descriptor::<op::Relu>::infer_runtime(NoAttributes, vec![input_meta]).unwrap();
    let captured = CapturedDescriptor::capture(descriptor.descriptor()).unwrap();

    let mut graph = Graph::new();
    let input = graph.add_value(vec![2], DTypeId::F32, Some("input".into()));
    let output = graph.add_value(vec![2], DTypeId::F32, Some("output".into()));
    graph.mark_input(input);
    graph.mark_output(output);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Relu),
        vec![input],
        vec![output],
        Default::default(),
        Some(incin_core::graph::DescriptorPayload {
            schema: captured.schema(),
            payload: captured.payload().to_vec(),
        }),
    );

    let plan = CompiledPlan::compile(
        CapturedGraph::capture(&graph).unwrap(),
        CompileOptions::new(),
    )
    .unwrap();
    let input = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![-2.0, 3.5]), vec![2]).unwrap();
    let outputs = CpuCompiledInvocation::new(vec![input]).run(&plan).unwrap();

    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].get(&[0]), 0.0);
    assert_eq!(outputs[0].get(&[1]), 3.5);
}
