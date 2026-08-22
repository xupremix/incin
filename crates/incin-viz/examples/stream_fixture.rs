//! Writes a synthetic-but-real telemetry stream for verification runs.
//!
//! The file is produced by the actual `Emitter` + `FileTransport` path, so
//! its bytes are the real wire format, not a hand-rolled approximation.
//! Used by `tools/viz-smoke.sh` and recorded as retained evidence for the
//! 0.1.0 functional-verification pass.
//!
//! Usage: `cargo run -p incin-viz --example stream_fixture -- <output.jsonl>`

use incin_telemetry::emitter::Emitter;
use incin_telemetry::events::{
    CURRENT_SCHEMA_VERSION, EpochEvent, GradientNormEvent, MemoryEvent, ScalarEvent,
};
use incin_telemetry::reporter::Reporter;
use incin_telemetry::transport::file::FileTransport;

fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .expect("usage: stream_fixture <output.jsonl>");

    let emitter = Emitter::new(vec![Box::new(FileTransport::open(path.as_ref())?)]);

    for step in 0..50usize {
        let loss = (1.0 / (step as f64 + 1.0)).max(0.01);
        emitter.log_scalar(ScalarEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            name: "loss".into(),
            value: loss,
        });
        emitter.log_scalar(ScalarEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            name: "throughput".into(),
            value: 1000.0 + step as f64,
        });
        emitter.log_scalar(ScalarEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            name: "lr".into(),
            value: 3e-4,
        });
        emitter.log_gradient_norm(GradientNormEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            param_name: "layer0".into(),
            l2_norm: (0.5 + step as f64 * 0.01) as f32,
        });
        emitter.log_memory(MemoryEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            rss_bytes: ((512.0 + step as f64) * 1024.0 * 1024.0) as u64,
        });
        if step % 10 == 9 {
            emitter.log_epoch(EpochEvent {
                schema_version: CURRENT_SCHEMA_VERSION,
                epoch: step / 10,
                metrics: [
                    ("train_loss".into(), loss as f32),
                    ("val_loss".into(), (loss * 1.1) as f32),
                ]
                .into_iter()
                .collect(),
            });
        }
        // The custom metric the verification plugin tracks.
        emitter.log_scalar(incin_telemetry::events::ScalarEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step,
            name: "custom_metric".into(),
            value: step as f64,
        });
    }

    emitter.shutdown();
    eprintln!("wrote synthetic stream to {path}");
    Ok(())
}
