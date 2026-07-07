use kindle_core::prelude::*;
use kindle_core::nn::{Linear, Module};
use kindle_core::tensor::tracing::{TracingBackend, TRACING_GRAPH};
use kindle_core::tensor::backend::dummy::DummyBackend;
use kindle_core::serialize::Serializer;
use kindle_core::onnx_exporter::OnnxExporter;
use std::collections::HashMap;
use std::path::Path;

type B = TracingBackend<DummyBackend<f32>>;

fn main() -> anyhow::Result<()> {
    // Create a simple model
    let linear = Linear::<Dyn, B>::new_dyn((10, 5))?;

    // Create a dummy input
    let input = Tensor::<Dyn, B>::zeros([2, 10])?;
    
    // Mark input in the computation graph
    TRACING_GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        g.mark_input(input.inner().value_id);
    });

    // Run forward pass
    let output = linear.forward(input)?;
    let output = output.relu()?;

    // Mark output in the computation graph
    TRACING_GRAPH.with(|g| {
        let mut g = g.borrow_mut();
        g.mark_output(output.inner().value_id);
    });

    // Export to ONNX
    let path = Path::new("model.onnx");
    let mut exporter = OnnxExporter::new(&path);
    
    // State dict is irrelevant here since the tracing graph already captured everything
    exporter.serialize::<B>(&HashMap::new())?;

    println!("ONNX export successful: {:?}", path);

    Ok(())
}
