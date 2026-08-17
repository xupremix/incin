use incin_backends::cpu::CpuBackendImpl;
use incin_core::onnx::export_to_onnx;
use incin_core::prelude::*;
use incin_core::tensor::tracing::{
    TracingBackend, extract_graph, tracing_mark_input, tracing_mark_output,
};
extern crate alloc;
use std::path::Path;

/// B.
type B = TracingBackend<CpuBackendImpl>;

fn main() -> anyhow::Result<()> {
    // Create a simple model
    let linear = Linear::<Dyn, B>::build((10, 5))?;

    // Create a dummy input
    let input = Tensor::<Dyn, B>::zeros([2, 10])?;

    // Mark input in the computation graph
    tracing_mark_input(input.inner().value_id);

    // Run forward pass
    let output = linear.forward(input)?;
    let output = output.relu()?;

    // Mark output in the computation graph
    tracing_mark_output(output.inner().value_id);

    // Export to ONNX
    let path = Path::new("model.onnx");
    // State dict is irrelevant here since the tracing graph already captured everything
    let graph = extract_graph();
    export_to_onnx(&graph, path)?;

    println!("ONNX export successful: {:?}", path);

    Ok(())
}
