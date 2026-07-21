//! Read-side counterpart to `kindle-telemetry`'s `Transport` trait: a
//! poll-based tailer for the JSONL transport file a training process
//! writes to.

use kindle_telemetry::events::Event;

/// A source that yields newly-available events since the last poll.
/// Implementors own their own I/O and must never block indefinitely --
/// `poll_new_events` is called every event-loop tick from `app.rs`'s
/// `tokio::select!`, so a blocking implementation would stall the whole
/// render/input loop.
pub trait TransportReader: Send {
    /// Returns any newly-complete events discovered since the last call.
    /// A trailing incomplete line at EOF (the writer is mid-`write_all`)
    /// must NOT be returned yet -- it is buffered internally and completed
    /// on a future call once the rest of the line arrives.
    fn poll_new_events(&mut self) -> crate::err::Result<Vec<Event>>;
}

/// Poll-based tailer for a JSONL transport file, the read-side mirror of
/// `kindle-telemetry::transport::file::FileTransport`. Reuses the same
/// `BufReader<File>` instance across every `poll_new_events` call -- never
/// reopens the file or seeks to the start, so its internal position
/// naturally advances as the writer appends more bytes.
pub struct FileTransportReader {
    reader: std::io::BufReader<std::fs::File>,
    // Bytes read so far that do not yet end in a newline -- carried across
    // polls until the writer completes the line.
    partial_line: String,
}

impl FileTransportReader {
    /// Opens `path` read-only for tailing, starting from offset 0.
    pub fn open(path: &std::path::Path) -> crate::err::Result<Self> {
        let file = std::fs::File::open(path)?;
        Ok(Self {
            reader: std::io::BufReader::new(file),
            partial_line: String::new(),
        })
    }
}

impl TransportReader for FileTransportReader {
    /// Auto-generated documentation for poll_new_events.
    fn poll_new_events(&mut self) -> crate::err::Result<Vec<Event>> {
        use std::io::BufRead;

        let mut events = Vec::new();
        loop {
            let mut line = core::mem::take(&mut self.partial_line);
            let bytes_read = self.reader.read_line(&mut line)?;
            if bytes_read == 0 {
                // EOF for now. If `line` is non-empty, it's a partial line
                // (no trailing '\n' yet) -- buffer it and stop; the writer
                // may still be mid-write_all.
                self.partial_line = line;
                break;
            }
            if line.ends_with('\n') {
                line.pop(); // drop trailing '\n'
                match serde_json::from_str::<Event>(&line) {
                    Ok(event) => events.push(event),
                    Err(_) => {
                        // Malformed line -- skip, never panic the reader
                        // (a training-process crash should degrade to
                        // "stale but visible", never cascade into a
                        // kindle-viz panic).
                    }
                }
            } else {
                // Incomplete line at EOF -- buffer for next poll.
                self.partial_line = line;
                break;
            }
        }
        Ok(events)
    }
}

#[cfg(test)]
/// Auto-generated documentation for tests.
mod tests {
    use super::*;
    use std::io::Write;

    /// Auto-generated documentation for unique_test_path.
    fn unique_test_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "kindle-viz-transport-reader-test-{label}-{}.jsonl",
            uuid::Uuid::now_v7()
        ))
    }

    /// Auto-generated documentation for scalar_event_json.
    fn scalar_event_json(step: usize, name: &str, value: f64) -> String {
        let event = Event::Scalar(kindle_telemetry::events::ScalarEvent {
            schema_version: kindle_telemetry::events::CURRENT_SCHEMA_VERSION,
            step,
            name: name.to_string(),
            value,
        });
        serde_json::to_string(&event).expect("event should serialize")
    }

    #[test]
    /// Auto-generated documentation for partial_line_completes_on_next_poll.
    fn partial_line_completes_on_next_poll() {
        let path = unique_test_path("partial-line");

        let complete_line = scalar_event_json(0, "loss", 0.5);
        let full_second_line = scalar_event_json(1, "loss", 0.4);
        let partial_second_line = &full_second_line[..10];

        {
            let mut file = std::fs::File::create(&path).expect("create should succeed");
            write!(file, "{complete_line}\n{partial_second_line}").expect("write should succeed");
        }

        let mut reader = FileTransportReader::open(&path).expect("open should succeed");

        let first_poll = reader.poll_new_events().expect("first poll should succeed");
        assert_eq!(
            first_poll.len(),
            1,
            "only the complete first line should be returned"
        );

        // Writer completes the second line and appends the trailing
        // newline, using a separate append-mode handle (models the
        // independent writer process).
        {
            let mut appender = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("append-mode open should succeed");
            let remainder = &full_second_line[10..];
            writeln!(appender, "{remainder}").expect("append write should succeed");
        }

        let second_poll = reader
            .poll_new_events()
            .expect("second poll should succeed");
        assert_eq!(
            second_poll.len(),
            1,
            "the now-completed second line should be returned exactly once"
        );

        std::fs::remove_file(&path).ok();
    }

    #[test]
    /// Auto-generated documentation for no_duplicate_events_across_polls.
    fn no_duplicate_events_across_polls() {
        let path = unique_test_path("no-duplicate");

        let initial_lines: Vec<String> = (0..3)
            .map(|step| scalar_event_json(step, "loss", 1.0 - step as f64 * 0.1))
            .collect();

        {
            let mut file = std::fs::File::create(&path).expect("create should succeed");
            for line in &initial_lines {
                writeln!(file, "{line}").expect("write should succeed");
            }
        }

        let mut reader = FileTransportReader::open(&path).expect("open should succeed");

        let first_poll = reader.poll_new_events().expect("first poll should succeed");
        assert_eq!(first_poll.len(), 3, "all 3 initial lines should be read");

        let more_lines: Vec<String> = (3..5)
            .map(|step| scalar_event_json(step, "loss", 1.0 - step as f64 * 0.1))
            .collect();
        {
            let mut appender = std::fs::OpenOptions::new()
                .append(true)
                .open(&path)
                .expect("append-mode open should succeed");
            for line in &more_lines {
                writeln!(appender, "{line}").expect("append write should succeed");
            }
        }

        let second_poll = reader
            .poll_new_events()
            .expect("second poll should succeed");
        assert_eq!(
            second_poll.len(),
            2,
            "only the 2 newly-appended lines should be returned, not a re-read of the first 3"
        );

        std::fs::remove_file(&path).ok();
    }
}
