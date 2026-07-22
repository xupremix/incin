#[macro_use]
extern crate alloc;
use kindle::prelude::*;
use kindle::{Linear, Module};
use kindle_backends::cpu::CpuBackendImpl;
use kindle_core::prelude::{
    TracingBackend, extract_graph, tracing_mark_input, tracing_mark_output,
};
use kindle_telemetry::events::GraphSnapshotEvent;
use kindle_telemetry::reporter::Reporter;

/// Nb.
type NB = CpuBackendImpl;
/// Tb.
type TB = TracingBackend<NB>;

#[module]
/// Simple mlp.
pub struct SimpleMlp<B: Backend> {
    /// Fc1.
    pub fc1: Linear<Dyn, B>,
    /// Fc2.
    pub fc2: Linear<Dyn, B>,
    /// Fc3.
    pub fc3: Linear<Dyn, B>,
}

impl<B: Backend> SimpleMlp<B>
where
    B: SupportsDType<B::FloatElem>,
    B::FloatElem: ConstDType,
    B::Device: ConstDevice,
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
    pub fn forward(&self, x: Tensor<Dyn, B>) -> Result<Tensor<Dyn, B>> {
        let x = self.fc1.forward(x)?.relu()?;
        let x = self.fc2.forward(x)?.relu()?;
        self.fc3.forward(x)
    }
}

fn main() -> anyhow::Result<()> {
    let run_dir = kindle_telemetry::run_dir::default_run_dir()?;
    let run_id = kindle_telemetry::run_dir::generate_run_id();
    let file_transport = kindle_telemetry::transport::file::FileTransport::open(
        &run_dir.join(format!("{}.jsonl", run_id)),
    )?;
    let emitter = kindle_telemetry::emitter::Emitter::new(vec![Box::new(file_transport)]);

    println!("run-id: {}", run_id);
    println!("To visualize the model, run in a separate terminal:");
    println!("cargo run -p kindle-viz -- --run-id {}", run_id);
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
        schema_version: kindle_telemetry::events::CURRENT_SCHEMA_VERSION,
        graph,
    });

    println!("Done! Emitted the graph to telemetry.");
    println!(
        "Keeping process alive for 30 seconds for live attach (though kindle-viz reads past history too)..."
    );
    for _ in 0..30 {
        std::thread::sleep(std::time::Duration::from_secs(1));
    }

    emitter.shutdown();
    Ok(())
}
