#![cfg(feature = "compiled")]

use incin_backends::cpu::{CpuBuffer, CpuCompiledInvocation, CpuStorage};
use incin_core::compiled::{CapturedGraph, CompileOptions, CompiledPlan};
use incin_core::exec::OperationIdentity;
use incin_core::exec::catalog::{
    CapturedDescriptor, Descriptor, LogicalTensorMeta, NoAttributes, op,
};
use incin_core::graph::Graph;
use incin_core::prelude::{DTypeId, DeviceId, OperationKind, ShapeBuf};

fn meta(shape: &[usize]) -> LogicalTensorMeta {
    LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(shape)),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    }
}

fn payload<O>(inputs: &[&[usize]]) -> incin_core::graph::DescriptorPayload
where
    O: incin_core::exec::catalog::CanonicalOperation<Attributes = NoAttributes>,
{
    let descriptor = Descriptor::<O>::infer_runtime(
        NoAttributes,
        inputs.iter().map(|shape| meta(shape)).collect(),
    )
    .unwrap();
    let captured = CapturedDescriptor::capture(descriptor.descriptor()).unwrap();
    incin_core::graph::DescriptorPayload {
        schema: captured.schema(),
        payload: captured.payload().to_vec(),
    }
}

#[test]
fn compiled_cpu_executes_captured_relu_through_canonical_descriptor() {
    let descriptor = Descriptor::<op::Relu>::infer_runtime(NoAttributes, vec![meta(&[2])]).unwrap();
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

#[test]
fn compiled_cpu_admission_rejects_a_descriptorless_operation() {
    let mut graph = Graph::new();
    let input = graph.add_value(vec![2], DTypeId::F32, Some("input".into()));
    let output = graph.add_value(vec![2], DTypeId::F32, Some("output".into()));
    graph.mark_input(input);
    graph.mark_output(output);
    graph.add_node(
        OperationKind::Relu,
        vec![input],
        vec![output],
        Default::default(),
    );
    let plan = CompiledPlan::compile(
        CapturedGraph::capture(&graph).unwrap(),
        CompileOptions::new(),
    )
    .unwrap();
    let error = incin_backends::cpu::CpuCompiledPlan::try_new(&plan).unwrap_err();
    assert!(error.to_string().contains("no captured descriptor"));
}

#[test]
fn compiled_cpu_executes_a_chained_matrix_pipeline() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 3], DTypeId::F32, Some("x".into()));
    let w = graph.add_value(vec![3, 4], DTypeId::F32, Some("w".into()));
    let hidden = graph.add_value(vec![2, 4], DTypeId::F32, Some("hidden".into()));
    let activated = graph.add_value(vec![2, 4], DTypeId::F32, Some("activated".into()));
    let v = graph.add_value(vec![4, 2], DTypeId::F32, Some("v".into()));
    let y = graph.add_value(vec![2, 2], DTypeId::F32, Some("y".into()));
    graph.mark_input(x);
    graph.mark_input(w);
    graph.mark_input(v);
    graph.mark_output(y);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::MatMulExact),
        vec![x, w],
        vec![hidden],
        Default::default(),
        Some(payload::<op::MatMulExact>(&[&[2, 3], &[3, 4]])),
    );
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Relu),
        vec![hidden],
        vec![activated],
        Default::default(),
        Some(payload::<op::Relu>(&[&[2, 4]])),
    );
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::MatMulExact),
        vec![activated, v],
        vec![y],
        Default::default(),
        Some(payload::<op::MatMulExact>(&[&[2, 4], &[4, 2]])),
    );

    let plan = CompiledPlan::compile(
        CapturedGraph::capture(&graph).unwrap(),
        CompileOptions::new(),
    )
    .unwrap();
    let x_storage = CpuStorage::try_from_contiguous(
        CpuBuffer::F32(vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0]),
        vec![2, 3],
    )
    .unwrap();
    let w_storage = CpuStorage::try_from_contiguous(
        CpuBuffer::F32(vec![
            1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0,
        ]),
        vec![3, 4],
    )
    .unwrap();
    let v_storage = CpuStorage::try_from_contiguous(
        CpuBuffer::F32(vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0]),
        vec![4, 2],
    )
    .unwrap();
    let outputs = CpuCompiledInvocation::new(vec![x_storage, w_storage, v_storage])
        .run(&plan)
        .unwrap();
    assert_eq!(outputs[0].shape.as_ref(), &[2, 2]);
    assert_eq!(outputs[0].get(&[0, 0]), 6.0);
    assert_eq!(outputs[0].get(&[0, 1]), 9.0);
    assert_eq!(outputs[0].get(&[1, 0]), 1.5);
    assert_eq!(outputs[0].get(&[1, 1]), 3.5);
}

#[test]
fn compiled_cpu_accepts_a_new_symbolic_batch_extent() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 3], DTypeId::F32, Some("x".into()));
    let w = graph.add_value(vec![3, 2], DTypeId::F32, Some("w".into()));
    let y = graph.add_value(vec![2, 2], DTypeId::F32, Some("y".into()));
    graph.mark_input(x);
    graph.mark_input(w);
    graph.mark_output(y);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::MatMulExact),
        vec![x, w],
        vec![y],
        Default::default(),
        Some(payload::<op::MatMulExact>(&[&[2, 3], &[3, 2]])),
    );
    let plan = CompiledPlan::compile(
        CapturedGraph::capture(&graph).unwrap(),
        CompileOptions::new(),
    )
    .unwrap();
    let x_storage = CpuStorage::try_from_contiguous(
        CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]),
        vec![3, 3],
    )
    .unwrap();
    let w_storage = CpuStorage::try_from_contiguous(
        CpuBuffer::F32(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        vec![3, 2],
    )
    .unwrap();
    let outputs = CpuCompiledInvocation::new(vec![x_storage, w_storage])
        .run(&plan)
        .unwrap();
    assert_eq!(outputs[0].shape.as_ref(), &[3, 2]);
    assert_eq!(outputs[0].get(&[2, 0]), 16.0);
    assert_eq!(outputs[0].get(&[2, 1]), 17.0);
}
