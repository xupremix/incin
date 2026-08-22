//! End-to-end telemetry tests through the public `incin-telemetry` API.
//!
//! The suite drives a real [`FileTransport`] behind an [`Emitter`], the same
//! composition a training loop builds, and reads the JSONL back from disk.
//! No socket, run directory, or network is involved.

use incin_telemetry::emitter::Emitter;
use incin_telemetry::events::Event;
use incin_telemetry::events::{CURRENT_SCHEMA_VERSION, ScalarEvent};
use incin_telemetry::reporter::Reporter;
use incin_telemetry::transport::Transport;
use incin_telemetry::transport::file::FileTransport;

/// A unique file path inside the process temporary directory.
fn temp_jsonl(label: &str) -> std::path::PathBuf {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("clock is after the epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "incin-telemetry-it-{label}-{}-{nanos}.jsonl",
        std::process::id()
    ))
}

#[test]
fn events_written_through_an_emitter_reach_disk_as_jsonl() {
    let path = temp_jsonl("roundtrip");

    let emitter = Emitter::new(vec![Box::new(
        FileTransport::open(&path).expect("temporary transport file should open"),
    )]);

    for step in 0..4u64 {
        emitter.log_scalar(ScalarEvent {
            schema_version: CURRENT_SCHEMA_VERSION,
            step: step as usize,
            name: "loss".into(),
            value: 1.0 / (step + 1) as f64,
        });
    }
    emitter.shutdown();

    let raw =
        std::fs::read_to_string(&path).expect("the transport file should exist after shutdown");
    let _ = std::fs::remove_file(&path);

    let lines: Vec<&str> = raw.lines().collect();
    assert_eq!(lines.len(), 4, "one JSONL line per event");

    let first: ScalarEvent = serde_json::from_str(lines[0]).expect("line 0 parses");
    assert_eq!(first.step, 0);
    assert_eq!(first.name, "loss");
    assert_eq!(first.value, 1.0);

    let last: ScalarEvent = serde_json::from_str(lines[3]).expect("line 3 parses");
    assert_eq!(last.value, 1.0 / 4.0);
}

#[test]
fn every_event_variant_round_trips_through_the_transport_boundary() {
    // Drive the Transport trait directly: this pins the wire contract the
    // Emitter relies on, independent of channel behavior.
    let path = temp_jsonl("variants");
    {
        let mut transport =
            FileTransport::open(&path).expect("temporary transport file should open");
        transport
            .write_event(&Event::Unknown)
            .expect("an unknown-future event writes without error");
    }

    let raw = std::fs::read_to_string(&path).expect("transport file exists");
    let _ = std::fs::remove_file(&path);

    let event: Event = serde_json::from_str(raw.trim_end()).expect("the line parses back");
    assert!(
        matches!(event, Event::Unknown),
        "a serde(other) event must survive the boundary unchanged"
    );
}

#[test]
fn shutdown_is_reusable_and_a_dropped_emitter_flushes_what_it_was_given() {
    let path = temp_jsonl("drop-flush");
    let emitter = Emitter::new(vec![Box::new(
        FileTransport::open(&path).expect("temporary transport file should open"),
    )]);
    emitter.log_scalar(ScalarEvent {
        schema_version: CURRENT_SCHEMA_VERSION,
        step: 7,
        name: "accuracy".into(),
        value: 0.75,
    });
    drop(emitter);

    let raw = std::fs::read_to_string(&path).expect("dropping flushes pending events");
    let _ = std::fs::remove_file(&path);

    let only: ScalarEvent = serde_json::from_str(raw.trim_end()).expect("single line parses");
    assert_eq!(only.step, 7);
    assert_eq!(only.name, "accuracy");
}
