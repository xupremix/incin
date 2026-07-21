use kindle_core::prelude::dummy::DummyBackend;
use kindle_core::prelude::*;
use kindle_core::prelude::{TRACING_GRAPH, TracingBackend};
extern crate alloc;
use alloc::collections::BTreeMap;
use std::path::Path;

/// Auto-generated documentation for B.
type B = TracingBackend<DummyBackend<f32, Cpu>>;

fn main() -> anyhow::Result<()> {
    // Create a simple model
    let linear = Linear::<Dyn, B>::new_with((10, 5))?;

    // Create a dummy input
    let input = Tensor::<Dyn, B>::zeros([2, 10])?;

    // Mark input in the computation graph
    TRACING_GRAPH.lock().mark_input(input.inner().value_id);

    // Run forward pass
    let output = linear.forward(input)?;
    let output = output.relu()?;

    // Mark output in the computation graph
    TRACING_GRAPH.lock().mark_output(output.inner().value_id);

    // Export to ONNX
    let path = Path::new("model.onnx");
    let mut exporter = OnnxExporter::new(&path);

    // State dict is irrelevant here since the tracing graph already captured everything
    exporter.serialize::<B>(&BTreeMap::new())?;

    println!("ONNX export successful: {:?}", path);

    Ok(())
}
