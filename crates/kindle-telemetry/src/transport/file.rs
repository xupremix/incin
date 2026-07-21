//! Crash-durable, append-only JSONL file transport (TELEM-04).
//!
//! **Durability scope (WR-04):** "crash-durable" here means durable against
//! a *process* crash (or `kill`) mid-write -- `write_all`'s stdlib contract
//! guarantees it either writes every byte or returns an `Err` before any
//! partial write is externally observable via a *separate* read of the
//! same file, so a crash can only ever truncate the tail of the last line,
//! never corrupt a prior record. This does **not** cover power-loss / OS
//! crash durability: there is no `fsync`/`sync_data()` call after
//! `write_all`, so data that is only in the OS page cache (not yet flushed
//! to physical storage) can still be lost on a hard power cut, even though
//! ordering and prior-record integrity are preserved either way.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::Path;

use crate::events::Event;
use crate::transport::Transport;

/// Appends one complete, self-delimited JSONL line per [`Event`] written.
/// Opens with `O_APPEND` semantics so a *process* crash mid-write can only
/// ever truncate the tail of the *last* line, never corrupt prior records
/// (T-07-02). See the module-level doc comment (WR-04) for the precise
/// scope of this durability guarantee -- it does not cover power-loss/OS
/// crash durability (no `fsync`/`sync_data()` is performed).
pub struct FileTransport {
    file: File,
}

impl FileTransport {
    /// Opens (creating if absent) the file at `path` for append-only
    /// writes.
    pub fn open(path: &Path) -> std::io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self { file })
    }
}

impl Transport for FileTransport {
    /// Writes a single event as a self-delimited JSONL line to the file.
    fn write_event(&mut self, event: &Event) -> crate::err::Result<()> {
        // Build the complete line in memory first, then issue exactly one
        // write_all call -- never split the JSON body and the trailing
        // newline into separate writes (Pitfall 4).
        let mut line = serde_json::to_string(event)?;
        line.push('\n');
        self.file.write_all(line.as_bytes())?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::ScalarEvent;
    use std::io::{BufRead, BufReader};

    /// Core abstraction for `unique_test_path` within the Kindle framework.
    fn unique_test_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "kindle-telemetry-file-transport-test-{label}-{}.jsonl",
            uuid::Uuid::now_v7()
        ))
    }

    /// Core abstraction for `scalar_event` within the Kindle framework.
    fn scalar_event(step: usize, name: &str, value: f64) -> Event {
        Event::Scalar(ScalarEvent {
            schema_version: crate::events::CURRENT_SCHEMA_VERSION,
            step,
            name: name.to_string(),
            value,
        })
    }

    #[test]
    /// Core abstraction for `open_on_nonexistent_path_creates_file_and_succeeds` within the Kindle framework.
    fn open_on_nonexistent_path_creates_file_and_succeeds() {
        let path = unique_test_path("open-creates");
        assert!(!path.exists());

        let _transport = FileTransport::open(&path).expect("open should succeed");

        assert!(path.exists(), "file should be created on open");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    /// Core abstraction for `write_event_three_times_round_trips_through_readback` within the Kindle framework.
    fn write_event_three_times_round_trips_through_readback() {
        let path = unique_test_path("round-trip");
        let mut transport = FileTransport::open(&path).expect("open should succeed");

        let events = vec![
            scalar_event(0, "loss", 0.5),
            scalar_event(1, "loss", 0.4),
            scalar_event(2, "loss", 0.3),
        ];
        for event in &events {
            transport
                .write_event(event)
                .expect("write_event should succeed");
        }
        drop(transport);

        let file = File::open(&path).expect("file should be readable");
        let lines: Vec<String> = BufReader::new(file)
            .lines()
            .map(|l| l.expect("line should be valid utf8"))
            .collect();

        assert_eq!(lines.len(), 3);
        for (line, expected) in lines.iter().zip(events.iter()) {
            let parsed: Event = serde_json::from_str(line).expect("line should parse as Event");
            match (parsed, expected) {
                (Event::Scalar(p), Event::Scalar(e)) => {
                    assert_eq!(p.step, e.step);
                    assert_eq!(p.name, e.name);
                    assert_eq!(p.value, e.value);
                }
                _ => panic!("expected Event::Scalar for both parsed and expected"),
            }
        }

        std::fs::remove_file(&path).ok();
    }

    #[test]
    /// Core abstraction for `each_line_ends_with_exactly_one_newline_and_no_embedded_newline` within the Kindle framework.
    fn each_line_ends_with_exactly_one_newline_and_no_embedded_newline() {
        let path = unique_test_path("single-newline");
        let mut transport = FileTransport::open(&path).expect("open should succeed");

        transport
            .write_event(&scalar_event(0, "loss", 0.5))
            .expect("write_event should succeed");
        drop(transport);

        let contents = std::fs::read_to_string(&path).expect("file should be readable");
        // Exactly one trailing newline, no embedded newline inside the body.
        assert_eq!(contents.matches('\n').count(), 1);
        assert!(contents.ends_with('\n'));
        assert!(!contents[..contents.len() - 1].contains('\n'));

        std::fs::remove_file(&path).ok();
    }

    #[test]
    /// Core abstraction for `second_transport_at_same_path_appends_rather_than_truncates` within the Kindle framework.
    fn second_transport_at_same_path_appends_rather_than_truncates() {
        let path = unique_test_path("append-not-truncate");

        let mut first = FileTransport::open(&path).expect("first open should succeed");
        first
            .write_event(&scalar_event(0, "loss", 1.0))
            .expect("write_event should succeed");
        drop(first);

        let mut second = FileTransport::open(&path).expect("second open should succeed");
        second
            .write_event(&scalar_event(1, "loss", 2.0))
            .expect("write_event should succeed");
        drop(second);

        let contents = std::fs::read_to_string(&path).expect("file should be readable");
        let lines: Vec<&str> = contents.lines().collect();
        assert_eq!(lines.len(), 2, "second open must append, not truncate");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    /// Core abstraction for `truncated_trailing_line_yields_prior_complete_records_only` within the Kindle framework.
    fn truncated_trailing_line_yields_prior_complete_records_only() {
        let path = unique_test_path("crash-truncation");

        // Write 2 complete lines directly (not through FileTransport) plus
        // a partial third line with no trailing newline, simulating a
        // mid-write crash.
        let line1 = serde_json::to_string(&scalar_event(0, "loss", 0.5)).unwrap();
        let line2 = serde_json::to_string(&scalar_event(1, "loss", 0.4)).unwrap();
        let partial = &serde_json::to_string(&scalar_event(2, "loss", 0.3)).unwrap()[..10];

        let mut raw = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap();
        write!(raw, "{line1}\n{line2}\n{partial}").unwrap();
        drop(raw);

        // Tolerant reader: parse line-by-line, skip a line that fails to
        // parse (the truncated tail) rather than erroring the whole read.
        let file = File::open(&path).unwrap();
        let parsed: Vec<Event> = BufReader::new(file)
            .lines()
            .map_while(Result::ok)
            .filter_map(|line| serde_json::from_str::<Event>(&line).ok())
            .collect();

        assert_eq!(
            parsed.len(),
            2,
            "only the 2 complete prior lines should parse"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    /// Core abstraction for `detach_reattach_sees_all_events_in_order` within the Kindle framework.
    fn detach_reattach_sees_all_events_in_order() {
        let path = unique_test_path("detach-reattach");
        let mut transport = FileTransport::open(&path).expect("open should succeed");

        let first_batch = vec![
            scalar_event(0, "loss", 0.9),
            scalar_event(1, "loss", 0.8),
            scalar_event(2, "loss", 0.7),
        ];
        for event in &first_batch {
            transport
                .write_event(event)
                .expect("write_event should succeed");
        }

        // Independent read handle attaches (models kindle-viz's tailing
        // reader), reads all N lines, then detaches (drop).
        {
            let file = File::open(&path).expect("read handle should open");
            let lines: Vec<String> = BufReader::new(file)
                .lines()
                .map(|l| l.expect("line should be valid utf8"))
                .collect();
            assert_eq!(lines.len(), first_batch.len());
            for (line, expected) in lines.iter().zip(first_batch.iter()) {
                let parsed: Event = serde_json::from_str(line).unwrap();
                match (parsed, expected) {
                    (Event::Scalar(p), Event::Scalar(e)) => assert_eq!(p.step, e.step),
                    _ => panic!("expected Event::Scalar"),
                }
            }
            // read handle dropped here, simulating a viewer detaching
        }

        // Writer, still open, writes M more events without restarting.
        let second_batch = vec![scalar_event(3, "loss", 0.6), scalar_event(4, "loss", 0.5)];
        for event in &second_batch {
            transport
                .write_event(event)
                .expect("write_event should succeed");
        }
        drop(transport);

        // Freshly reopened read handle (reattach), independent of the
        // first, sees all N+M lines in original order.
        let file = File::open(&path).expect("reattach read handle should open");
        let lines: Vec<String> = BufReader::new(file)
            .lines()
            .map(|l| l.expect("line should be valid utf8"))
            .collect();

        assert_eq!(lines.len(), first_batch.len() + second_batch.len());
        let all_expected: Vec<&Event> = first_batch.iter().chain(second_batch.iter()).collect();
        for (line, expected) in lines.iter().zip(all_expected.iter()) {
            let parsed: Event = serde_json::from_str(line).unwrap();
            match (parsed, expected) {
                (Event::Scalar(p), Event::Scalar(e)) => {
                    assert_eq!(p.step, e.step);
                    assert_eq!(p.value, e.value);
                }
                _ => panic!("expected Event::Scalar"),
            }
        }

        std::fs::remove_file(&path).ok();
    }
}
