//! Integration coverage for `preview_cpu_relu` on the documented public surface.
use incin::experimental::compiled::{
    CapturedDescriptor, CapturedGraph, CompileOptions, CompiledPlan, CpuBuffer,
    CpuCompiledFunction, CpuCompiledInvocation, CpuStorage, DTypeId, Descriptor, DeviceId, Graph,
    LogicalTensorMeta, NoAttributes, OperationIdentity, OperationKind, op,
};
use std::collections::BTreeMap;

/// Builds and executes a non-empty preview CPU plan through the facade alone.
pub fn preview_cpu_relu() -> Result<(String, f64), incin::Error> {
    let input_meta = LogicalTensorMeta {
        shape: Some((&[2usize][..]).into()),
        dtype: Some(DTypeId::F32.descriptor()),
        device: Some(DeviceId::cpu()),
    };
    let descriptor = Descriptor::<op::Relu>::infer_runtime(NoAttributes, vec![input_meta])
        .map_err(|error| incin::Error::Msg(error.to_string()))?;
    let captured = CapturedDescriptor::capture(descriptor.descriptor())
        .map_err(|error| incin::Error::Msg(error.to_string()))?;

    let mut graph = Graph::new();
    let input = graph.add_value(vec![2], DTypeId::F32, Some("input".into()));
    let output = graph.add_value(vec![2], DTypeId::F32, Some("relu_output".into()));
    graph.mark_input(input);
    graph.mark_output(output);
    graph.add_node_with_descriptor_payload(
        OperationIdentity::Builtin(OperationKind::Relu),
        vec![input],
        vec![output],
        BTreeMap::new(),
        Some(incin::experimental::compiled::DescriptorPayload {
            schema: captured.schema(),
            payload: captured.payload().to_vec(),
        }),
    );

    let plan = CompiledPlan::compile(CapturedGraph::capture(&graph)?, CompileOptions::new())?;
    let function = CpuCompiledFunction::compile(&plan)?;
    let input = CpuStorage::try_from_contiguous(CpuBuffer::F32(vec![-2.0, 3.5]), [2])?;
    let outputs = function.run(CpuCompiledInvocation::new(vec![input]))?;
    let output_name = plan.graph.value_metadata[&output]
        .name
        .clone()
        .expect("fixture output has a name");
    Ok((output_name, outputs[0].get(&[1])))
}

#[cfg(test)]
mod tests {
    use super::preview_cpu_relu;

    #[test]
    fn facade_only_cpu_preview_executes_and_names_its_output() {
        let (name, value) = preview_cpu_relu().expect("preview CPU plan runs");
        assert_eq!(name, "relu_output");
        assert_eq!(value, 3.5);
    }
}
