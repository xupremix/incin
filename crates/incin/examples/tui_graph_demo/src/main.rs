//! Feeds a demo graph into the incin-viz TUI for interactive inspection.
#[macro_use]
extern crate alloc;
use incin::backend_authoring::{SupportsDType, VariableBackend};
use incin::prelude::*;
use incin::{Linear, Module};
use incin_backends::cpu::CpuBackendImpl;
use incin_core::backend_authoring::Execute;
use incin_core::exec::catalog::op;
use incin_core::tensor::tracing::{
    TracingBackend, extract_graph, tracing_mark_input, tracing_mark_output,
};
use incin_telemetry::events::GraphSnapshotEvent;
use incin_telemetry::reporter::Reporter;

/// Nb.
type NB = CpuBackendImpl;
/// Tb.
type TB = TracingBackend<NB>;

#[module]
/// Simple mlp.
pub struct SimpleMlp<B: VariableBackend> {
    /// Fc1.
    pub fc1: Linear<Dyn, B>,
    /// Fc2.
    pub fc2: Linear<Dyn, B>,
    /// Fc3.
    pub fc3: Linear<Dyn, B>,
}

impl<
    B: VariableBackend
        + incin_core::nn::param::ParameterInit<f32>
        + Execute<op::Add>
        + Execute<op::Relu>,
> SimpleMlp<B>
where
    B: SupportsDType<f32>,
    B::Device: ConstDevice,
    B: Execute<op::TransposeExact>,
    <B as Execute<op::TransposeExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Add>>::Output: Into<B::Storage<f32>>,
    B: Execute<op::MatMulExact>,
    <B as Execute<op::MatMulExact>>::Output: Into<B::Storage<f32>>,
    <B as Execute<op::Relu>>::Output: Into<B::Storage<f32>>,
{
    /// New.
    pub fn new() -> Result<Self> {
        Ok(Self {
            fc1: Linear::<Dyn, B>::build((10, 32))?,
            fc2: Linear::<Dyn, B>::build((32, 16))?,
            fc3: Linear::<Dyn, B>::build((16, 2))?,
        })
    }

    /// Forward.
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Dense<Dyn, B, f32, Grad>> {
        let x = self.fc1.forward(x)?.relu()?;
        let x = self.fc2.forward(x)?.relu()?;
        self.fc3.forward(x)
    }
}

fn main() -> anyhow::Result<()> {
    let run_dir = incin_telemetry::run_dir::default_run_dir()?;
    let run_id = incin_telemetry::run_dir::generate_run_id();
    let file_transport = incin_telemetry::transport::file::FileTransport::open(
        &run_dir.join(format!("{}.jsonl", run_id)),
    )?;
    let emitter = incin_telemetry::emitter::Emitter::new(vec![Box::new(file_transport)]);

    println!("run-id: {}", run_id);
    println!("To visualize the model, run in a separate terminal:");
    println!("cargo run -p incin-viz -- --run-id {}", run_id);
    println!();
    println!("Emitting GraphSnapshotEvent...");

    // 1. Initialize the model with TracingBackend
    let model = SimpleMlp::<TB>::new()?;
    let input = Tensor::<Dyn, TB>::zeros([1, 10])?;

    // 2. Mark input
    tracing_mark_input(input.inner().value_id);

    // 3. Forward pass
    let output = model.forward(input)?;

    // 4. Mark output
    tracing_mark_output(output.inner().value_id);

    // 5. Extract graph and emit
    let graph = extract_graph();
    println!("Graph has {} nodes.", graph.nodes.len());
    emitter.log_graph_snapshot(GraphSnapshotEvent {
        schema_version: incin_telemetry::events::CURRENT_SCHEMA_VERSION,
        graph,
    });

    println!("Done! Emitted the graph to telemetry.");
    println!(
        "Keeping process alive for 30 seconds for live attach (though incin-viz reads past history too)..."
    );
    for _ in 0..1 {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    emitter.shutdown();
    Ok(())
}
