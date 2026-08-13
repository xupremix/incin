#![cfg(feature = "compiled")]

use incin_backends::cpu::{
    CpuBuffer, CpuCompiledFunction, CpuCompiledInvocation, CpuStorage, compiled_support,
};
use incin_core::compiled::{
    ArtifactVersion, CapturedGraph, CompileOptions, CompiledArtifact, CompiledPlan,
};
use incin_core::dist::placement::Local;
use incin_core::exec::OperationIdentity;
use incin_core::exec::catalog::{
    AxisAttributes, CapturedDescriptor, Descriptor, LinearAttributes, LogicalTensorMeta,
    NoAttributes, ShapeAttributes, op,
};
use incin_core::exec::{ExecutionContext, TensorHandle, dispatch};
use incin_core::graph::Graph;
use incin_core::prelude::{Cpu, DTypeId, DeviceId, OperationKind, ShapeBuf};

fn meta(shape: &[usize]) -> LogicalTensorMeta {
    LogicalTensorMeta {
        shape: Some(ShapeBuf::from_slice(shape)),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    }
}

#[test]
fn compiled_cpu_support_report_comes_from_canonical_catalog() {
    let report = compiled_support().unwrap();
    assert!(!report.is_empty());
    for supported in report {
        let entry = incin_core::exec::catalog::catalog_entry(supported.operation).unwrap();
        assert_eq!(supported.name, entry.name);
        assert_eq!(supported.descriptor, entry.descriptor);
        assert_eq!(supported.execution_site, entry.site);
        assert!(supported.capture_eligible);
        assert!(entry.site.is_backend_executable());
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

fn payload_with<O>(
    attributes: O::Attributes,
    inputs: &[&[usize]],
) -> incin_core::graph::DescriptorPayload
where
    O: incin_core::exec::catalog::CanonicalOperation,
    O::Attributes: incin_core::exec::catalog::AttributeContract,
{
    let descriptor = Descriptor::<O>::infer_runtime(
        attributes,
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
fn compiled_cpu_function_reuses_admitted_plan() {
    let descriptor = Descriptor::<op::Relu>::infer_runtime(NoAttributes, vec![meta(&[2])]).unwrap();
    let captured = CapturedDescriptor::capture(descriptor.descriptor()).unwrap();
    let mut graph = Graph::new();
    let input_id = graph.add_value(vec![2], DTypeId::F32, Some("input".into()));
    let output_id = graph.add_value(vec![2], DTypeId::F32, Some("output".into()));
    graph.mark_input(input_id);
    graph.mark_output(output_id);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Relu),
        vec![input_id],
        vec![output_id],
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
    let function = CpuCompiledFunction::compile(&plan).unwrap();
    assert_eq!(function.input_count(), 1);
    assert_eq!(function.output_count(), 1);

    let first = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![-1.0, 2.0]), vec![2]).unwrap();
    let second = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![3.0, -4.0]), vec![2]).unwrap();
    assert_eq!(
        function
            .run(CpuCompiledInvocation::new(vec![first]))
            .unwrap()[0]
            .get(&[1]),
        2.0
    );
    assert_eq!(
        function
            .run(CpuCompiledInvocation::new(vec![second]))
            .unwrap()[0]
            .get(&[0]),
        3.0
    );
}

#[test]
fn compiled_cpu_executes_with_captured_initializer() {
    let descriptor = Descriptor::<op::Relu>::infer_runtime(NoAttributes, vec![meta(&[2])]).unwrap();
    let captured = CapturedDescriptor::capture(descriptor.descriptor()).unwrap();

    let mut graph = Graph::new();
    let weights = graph.add_value(vec![2], DTypeId::F32, Some("weights".into()));
    let output = graph.add_value(vec![2], DTypeId::F32, Some("output".into()));
    graph
        .initializers
        .insert(weights, bytemuck::cast_slice(&[-2.0f32, 3.5f32]).to_vec());
    graph.mark_output(output);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Relu),
        vec![weights],
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
    let outputs = CpuCompiledInvocation::new(Vec::new()).run(&plan).unwrap();

    assert_eq!(outputs[0].get(&[0]), 0.0);
    assert_eq!(outputs[0].get(&[1]), 3.5);
}

#[test]
fn compiled_cpu_relu_matches_canonical_eager_dispatch() {
    let input =
        CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![-2.0, 0.5, 3.5, -1.0]), vec![2, 2])
            .unwrap();

    let eager_context = ExecutionContext::new(incin_backends::cpu::CpuBackendImpl::<Cpu>::new());
    let eager_handle =
        TensorHandle::from_storage::<incin_backends::cpu::CpuBackendImpl<Cpu>, f32, Local>(&input);
    let eager =
        dispatch::execute::<op::Relu, _>(&eager_context, NoAttributes, &[eager_handle]).unwrap();

    let descriptor =
        Descriptor::<op::Relu>::infer_runtime(NoAttributes, vec![meta(&[2, 2])]).unwrap();
    let captured = CapturedDescriptor::capture(descriptor.descriptor()).unwrap();
    let mut graph = Graph::new();
    let input_id = graph.add_value(vec![2, 2], DTypeId::F32, Some("input".into()));
    let output_id = graph.add_value(vec![2, 2], DTypeId::F32, Some("output".into()));
    graph.mark_input(input_id);
    graph.mark_output(output_id);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Relu),
        vec![input_id],
        vec![output_id],
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
    let compiled = CpuCompiledInvocation::new(vec![input]).run(&plan).unwrap();

    for index in [[0, 0], [0, 1], [1, 0], [1, 1]] {
        assert_eq!(eager.get(&index), compiled[0].get(&index));
    }
}

#[test]
fn compiled_cpu_executes_captured_reshape_through_canonical_descriptor() {
    let mut graph = Graph::new();
    let input = graph.add_value(vec![2, 2], DTypeId::F32, Some("input".into()));
    let output = graph.add_value(vec![4], DTypeId::F32, Some("output".into()));
    graph.mark_input(input);
    graph.mark_output(output);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::ReshapeExact),
        vec![input],
        vec![output],
        Default::default(),
        Some(payload_with::<op::ReshapeExact>(
            ShapeAttributes { shape: vec![4] },
            &[&[2, 2]],
        )),
    );

    let plan = CompiledPlan::compile(
        CapturedGraph::capture(&graph).unwrap(),
        CompileOptions::new(),
    )
    .unwrap();
    let input =
        CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
            .unwrap();
    let outputs = CpuCompiledInvocation::new(vec![input]).run(&plan).unwrap();
    assert_eq!(outputs[0].shape.as_ref(), &[4]);
    assert_eq!(outputs[0].get(&[3]), 4.0);
}

#[test]
fn compiled_cpu_executes_arithmetic_then_reduction() {
    let mut graph = Graph::new();
    let lhs = graph.add_value(vec![2, 2], DTypeId::F32, Some("lhs".into()));
    let rhs = graph.add_value(vec![2, 2], DTypeId::F32, Some("rhs".into()));
    let added = graph.add_value(vec![2, 2], DTypeId::F32, Some("added".into()));
    let output = graph.add_value(vec![2], DTypeId::F32, Some("output".into()));
    graph.mark_input(lhs);
    graph.mark_input(rhs);
    graph.mark_output(output);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Add),
        vec![lhs, rhs],
        vec![added],
        Default::default(),
        Some(payload::<op::Add>(&[&[2, 2], &[2, 2]])),
    );
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::SumDim),
        vec![added],
        vec![output],
        Default::default(),
        Some(payload_with::<op::SumDim>(
            AxisAttributes { axis: 1 },
            &[&[2, 2]],
        )),
    );

    let plan = CompiledPlan::compile(
        CapturedGraph::capture(&graph).unwrap(),
        CompileOptions::new(),
    )
    .unwrap();
    let lhs = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
        .unwrap();
    let rhs = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![5.0, 6.0, 7.0, 8.0]), vec![2, 2])
        .unwrap();
    let outputs = CpuCompiledInvocation::new(vec![lhs, rhs])
        .run(&plan)
        .unwrap();
    assert_eq!(outputs[0].shape.as_ref(), &[2]);
    assert_eq!(outputs[0].get(&[0]), 14.0);
    assert_eq!(outputs[0].get(&[1]), 22.0);
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
    let error = incin_backends::cpu::CpuCompiledPlan::compile(&plan).unwrap_err();
    assert!(error.to_string().contains("no captured descriptor"));
}

#[test]
fn compiled_cpu_admission_rejects_a_malformed_descriptor() {
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
            schema: 1,
            payload: vec![0xFF, 0x00],
        }),
    );
    let plan = CompiledPlan::compile(
        CapturedGraph::capture(&graph).unwrap(),
        CompileOptions::new(),
    )
    .unwrap();
    let error = incin_backends::cpu::CpuCompiledPlan::compile(&plan).unwrap_err();
    assert!(error.to_string().contains("invalid captured descriptor"));
}

#[test]
fn compiled_cpu_executes_after_artifact_roundtrip() {
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
    let version = ArtifactVersion::new(0, 1, 0);
    let artifact = CompiledArtifact::new(plan, version.clone(), "cpu-roundtrip".into()).unwrap();
    let loaded = CompiledArtifact::load(&artifact.serialize().unwrap(), &version).unwrap();
    let input = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![-2.0, 3.5]), vec![2]).unwrap();
    let outputs = CpuCompiledInvocation::new(vec![input])
        .run(&loaded.plan)
        .unwrap();
    assert_eq!(outputs[0].get(&[0]), 0.0);
    assert_eq!(outputs[0].get(&[1]), 3.5);
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
fn compiled_cpu_reuses_guarded_linear_mlp_for_multiple_batches() {
    let mut graph = Graph::new();
    let x = graph.add_value(vec![2, 3], DTypeId::F32, Some("x".into()));
    let w1 = graph.add_value(vec![4, 3], DTypeId::F32, Some("w1".into()));
    let hidden = graph.add_value(vec![2, 4], DTypeId::F32, Some("hidden".into()));
    let activated = graph.add_value(vec![2, 4], DTypeId::F32, Some("activated".into()));
    let w2 = graph.add_value(vec![2, 4], DTypeId::F32, Some("w2".into()));
    let y = graph.add_value(vec![2, 2], DTypeId::F32, Some("y".into()));
    graph.mark_input(x);
    graph.mark_input(w1);
    graph.mark_input(w2);
    graph.mark_output(y);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Linear),
        vec![x, w1],
        vec![hidden],
        Default::default(),
        Some(payload_with::<op::Linear>(
            LinearAttributes { has_bias: false },
            &[&[2, 3], &[4, 3]],
        )),
    );
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Relu),
        vec![hidden],
        vec![activated],
        Default::default(),
        Some(payload::<op::Relu>(&[&[2, 4]])),
    );
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Linear),
        vec![activated, w2],
        vec![y],
        Default::default(),
        Some(payload_with::<op::Linear>(
            LinearAttributes { has_bias: false },
            &[&[2, 4], &[2, 4]],
        )),
    );
    let plan = CompiledPlan::compile(
        CapturedGraph::capture(&graph).unwrap(),
        CompileOptions::new(),
    )
    .unwrap();

    let w1_storage = CpuStorage::try_from_contiguous(
        CpuBuffer::F32(vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0,
        ]),
        vec![4, 3],
    )
    .unwrap();
    let w2_storage = CpuStorage::try_from_contiguous(
        CpuBuffer::F32(vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        vec![2, 4],
    )
    .unwrap();
    let batch_two = CpuStorage::try_from_contiguous(
        CpuBuffer::F32(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
    )
    .unwrap();
    let output_two =
        CpuCompiledInvocation::new(vec![batch_two, w1_storage.clone(), w2_storage.clone()])
            .run(&plan)
            .unwrap();
    assert_eq!(output_two[0].shape.as_ref(), &[2, 2]);
    assert_eq!(output_two[0].get(&[0, 0]), 1.0);
    assert_eq!(output_two[0].get(&[0, 1]), 6.0);
    assert_eq!(output_two[0].get(&[1, 0]), 4.0);
    assert_eq!(output_two[0].get(&[1, 1]), 15.0);

    let batch_three = CpuStorage::try_from_contiguous(
        CpuBuffer::F32(vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]),
        vec![3, 3],
    )
    .unwrap();
    let output_three = CpuCompiledInvocation::new(vec![batch_three, w1_storage, w2_storage])
        .run(&plan)
        .unwrap();
    assert_eq!(output_three[0].shape.as_ref(), &[3, 2]);
    assert_eq!(output_three[0].get(&[2, 0]), 3.0);
    assert_eq!(output_three[0].get(&[2, 1]), 12.0);
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
