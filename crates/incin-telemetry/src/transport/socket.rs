//! Optional lower-latency Unix-socket/named-pipe transport (TELEM-05),
//! layered on top of the always-on [`super::file::FileTransport`]. Uses
//! `interprocess` 2.4.2's cross-platform `local_socket` API so the same code
//! path works on Unix domain sockets and Windows named pipes.
//!
//! `interprocess` 2.4.2 API surface confirmed by reading the installed crate
//! source directly (registry checkout, not training-data recall or
//! RESEARCH.md's MEDIUM-confidence sketch), at:
//! `~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/interprocess-2.4.2/src/local_socket/`
//! - `ListenerOptions::new().name(name).create_sync() -> io::Result<Listener>`
//!   (`listener/options.rs`)
//! - `Listener::set_nonblocking(ListenerNonblockingMode)` -- a trait method
//!   (`local_socket::traits::Listener`), not inherent; `ListenerNonblockingMode::Accept`
//!   makes `.accept()` return `WouldBlock` immediately with no pending
//!   connection, while the resulting `Stream` stays blocking
//!   (`listener/trait.rs`)
//! - `Listener::accept(&self) -> io::Result<Stream>` (`listener/enum.rs`)
//! - `"name".to_ns_name::<GenericNamespaced>() -> io::Result<Name<'_>>` via the
//!   `local_socket::traits`/`ToNsName` trait (`name/to_name.rs`, `name/type.rs`)
//! - `Stream: Read + Write` (`stream/enum.rs`)
//!
//! `interprocess::local_socket::prelude::*` brings the `Listener`/`Stream`
//! traits into scope as `_`-renamed imports (no namespace pollution) per
//! `local_socket.rs`'s own `pub mod prelude`.

use std::io::Write;

use interprocess::local_socket::prelude::*;
use interprocess::local_socket::{
    GenericNamespaced, Listener, ListenerNonblockingMode, ListenerOptions, Stream, ToNsName,
};

use crate::events::Event;
use crate::transport::Transport;

/// Maximum accepted length for `run_id` in [`SocketTransport::bind`]
/// (WR-02) -- generous relative to a UUIDv7's 36-character string form, but
/// still bounded so a pathological caller can't construct an
/// arbitrarily-long OS-level namespace string.
const MAX_RUN_ID_LEN: usize = 128;

/// Validates `run_id` before it is formatted into an OS-level socket
/// namespace (WR-02): rejects empty, overly long, or path-like input (path
/// separators or NUL bytes), since every current call site supplies a
/// UUIDv7 from [`crate::run_dir::generate_run_id`] and `bind` is a `pub`
/// API that should not silently accept arbitrary untrusted strings.
fn validate_run_id(run_id: &str) -> std::io::Result<()> {
    if run_id.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "run_id must not be empty",
        ));
    }
    if run_id.len() > MAX_RUN_ID_LEN {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("run_id must not exceed {MAX_RUN_ID_LEN} characters"),
        ));
    }
    if run_id.contains(['/', '\\', '\0']) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "run_id must not contain path separators or NUL bytes",
        ));
    }
    Ok(())
}

/// Broadcasts one JSONL line per [`Event`] to every currently-connected
/// client stream. Byte-identical framing to [`super::file::FileTransport`]
/// (TELEM-05's "same event schema" requirement): exactly one
/// `serde_json::to_string(event) + "\n"` per event, one `write_all` per
/// client.
pub struct SocketTransport {
    listener: Listener,
    clients: Vec<Stream>,
}

impl SocketTransport {
    /// Binds a local socket listener named after `run_id` (e.g.
    /// `incin-viz-{run_id}.sock`), in non-blocking-accept mode. Does not
    /// block: `.accept()` will immediately return `WouldBlock` when no
    /// client is currently trying to connect.
    ///
    /// **`run_id` constraint (WR-02):** every current call site passes a
    /// UUIDv7 from [`crate::run_dir::generate_run_id`], so `run_id` must be
    /// non-empty, reasonably short, and free of path separators (`/`, `\`)
    /// and NUL bytes before being formatted into an OS-level socket
    /// namespace -- this is validated defensively here since `bind` is
    /// `pub` and part of the crate's external API surface, not just an
    /// internal helper restricted to `generate_run_id`'s output.
    pub fn bind(run_id: &str) -> std::io::Result<Self> {
        validate_run_id(run_id)?;
        let name = format!("incin-viz-{run_id}.sock").to_ns_name::<GenericNamespaced>()?;
        let listener = ListenerOptions::new().name(name).create_sync()?;
        // `Accept`: only `.accept()` is nonblocking -- the resulting `Stream`
        // stays blocking, since per-client writes go through a short-lived
        // `write_all` call, not a long-lived read loop on the writer thread.
        listener.set_nonblocking(ListenerNonblockingMode::Accept)?;
        Ok(Self {
            listener,
            clients: Vec::new(),
        })
    }

    /// Drains all currently-pending connections into `clients`. Never
    /// blocks the writer thread: `.accept()` returns `WouldBlock`
    /// immediately once no connection is pending, at which point the loop
    /// stops.
    fn accept_pending(&mut self) {
        loop {
            match self.listener.accept() {
                Ok(stream) => self.clients.push(stream),
                Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => break,
                Err(_) => break,
            }
        }
    }
}

impl Transport for SocketTransport {
    /// Writes a single event as a self-delimited JSONL line to all connected socket clients.
    fn write_event(&mut self, event: &Event) -> crate::err::Result<()> {
        self.accept_pending();

        // Identical framing to FileTransport::write_event: one complete
        // line built in memory, one write_all call per client.
        let mut line = serde_json::to_string(event)?;
        line.push('\n');
        let bytes = line.as_bytes();

        // Broadcast to every connected client, pruning ones whose write
        // fails -- a per-client write failure is not a transport-level
        // failure (per behavior spec's 4th case), so this always returns Ok
        // regardless of how many clients were pruned. Unlike a clean
        // disconnect, the failure is logged (WR-05, matching
        // `write_to_all`'s existing `eprintln!` pattern for `FileTransport`
        // failures) so a client pruned due to a genuine I/O error (e.g. a
        // transient `EPIPE` unrelated to disconnecting) isn't completely
        // invisible relative to file-transport failures.
        self.clients.retain_mut(|c| match c.write_all(bytes) {
            Ok(()) => true,
            Err(e) => {
                eprintln!("incin-telemetry: socket client write failed, pruning client: {e}");
                false
            }
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::events::ScalarEvent;
    use crate::transport::file::FileTransport;
    use std::io::Read;
    use std::time::Duration;

    /// Unique run id.
    fn unique_run_id(label: &str) -> String {
        format!("test-{label}-{}", uuid::Uuid::now_v7())
    }

    /// Scalar event.
    fn scalar_event(step: usize, name: &str, value: f64) -> Event {
        Event::Scalar(ScalarEvent {
            schema_version: crate::events::CURRENT_SCHEMA_VERSION,
            step,
            name: name.to_string(),
            value,
        })
    }

    /// Polls `accept_pending` in a short loop, since the client connect
    /// happens on a separate thread and may not have completed yet.
    fn wait_for_client(transport: &mut SocketTransport) {
        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while transport.clients.is_empty() {
            transport.accept_pending();
            if transport.clients.is_empty() {
                if std::time::Instant::now() >= deadline {
                    panic!("test client did not connect within 5s");
                }
                std::thread::sleep(Duration::from_millis(5));
            }
        }
    }

    #[test]
    /// Bind with fresh run id succeeds without blocking.
    fn bind_with_fresh_run_id_succeeds_without_blocking() {
        let run_id = unique_run_id("bind");
        let start = std::time::Instant::now();
        let _transport = SocketTransport::bind(&run_id).expect("bind should succeed");
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "bind must not block"
        );
    }

    // `SocketTransport` intentionally does not implement `Debug` (its fields —
    // `interprocess`'s `Listener`/`Stream` — don't either), so these tests use an
    // explicit `match` instead of `expect_err`/`unwrap_err`, both of which require
    // `T: Debug` on the `Ok` side even when only the `Err` value is used.
    #[test]
    /// Bind rejects empty run id.
    fn bind_rejects_empty_run_id() {
        let err = match SocketTransport::bind("") {
            Err(e) => e,
            Ok(_) => panic!("empty run_id must be rejected"),
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    /// Bind rejects overly long run id.
    fn bind_rejects_overly_long_run_id() {
        let run_id = "a".repeat(MAX_RUN_ID_LEN + 1);
        let err = match SocketTransport::bind(&run_id) {
            Err(e) => e,
            Ok(_) => panic!("overly long run_id must be rejected"),
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    /// Bind rejects path like run id.
    fn bind_rejects_path_like_run_id() {
        let err = match SocketTransport::bind("../../etc/passwd") {
            Err(e) => e,
            Ok(_) => panic!("path-like run_id must be rejected"),
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    /// Write event with zero clients connected succeeds.
    fn write_event_with_zero_clients_connected_succeeds() {
        let run_id = unique_run_id("no-clients");
        let mut transport = SocketTransport::bind(&run_id).expect("bind should succeed");

        transport
            .write_event(&scalar_event(0, "loss", 0.5))
            .expect("write_event with no clients must still return Ok");
    }

    #[test]
    /// Connected client receives byte identical jsonl line to file transport.
    fn connected_client_receives_byte_identical_jsonl_line_to_file_transport() {
        let run_id = unique_run_id("byte-identical");
        let mut socket_transport = SocketTransport::bind(&run_id).expect("bind should succeed");

        let name = format!("incin-viz-{run_id}.sock")
            .to_ns_name::<GenericNamespaced>()
            .expect("name conversion should succeed");
        let client = std::thread::spawn(move || {
            // Retry connect briefly -- the listener may not have finished
            // setting up between bind() above and this thread starting.
            let deadline = std::time::Instant::now() + Duration::from_secs(5);
            loop {
                match Stream::connect(name.clone()) {
                    Ok(s) => return s,
                    Err(_) if std::time::Instant::now() < deadline => {
                        std::thread::sleep(Duration::from_millis(5));
                    }
                    Err(e) => panic!("client failed to connect: {e}"),
                }
            }
        });

        wait_for_client(&mut socket_transport);
        let mut client = client.join().expect("client thread should not panic");

        let event = scalar_event(0, "loss", 0.5);

        let file_path = std::env::temp_dir().join(format!(
            "incin-telemetry-socket-transport-test-{}.jsonl",
            uuid::Uuid::now_v7()
        ));
        let mut file_transport = FileTransport::open(&file_path).expect("file open should succeed");
        file_transport
            .write_event(&event)
            .expect("file write_event should succeed");
        let expected_line = std::fs::read_to_string(&file_path).expect("file should be readable");
        std::fs::remove_file(&file_path).ok();

        socket_transport
            .write_event(&event)
            .expect("socket write_event should succeed");

        // Read exactly the expected number of bytes from the client stream
        // (blocking read -- the write already happened synchronously above).
        let mut buf = vec![0u8; expected_line.len()];
        client
            .read_exact(&mut buf)
            .expect("client should receive the written bytes");
        let received = String::from_utf8(buf).expect("received bytes should be valid utf8");

        assert_eq!(
            received, expected_line,
            "socket transport's JSONL line must be byte-identical to file transport's"
        );
    }

    #[test]
    /// Disconnected client is pruned without write event returning err.
    fn disconnected_client_is_pruned_without_write_event_returning_err() {
        let run_id = unique_run_id("disconnect");
        let mut transport = SocketTransport::bind(&run_id).expect("bind should succeed");

        let name = format!("incin-viz-{run_id}.sock")
            .to_ns_name::<GenericNamespaced>()
            .expect("name conversion should succeed");
        let client = Stream::connect(name).expect("client should connect");

        wait_for_client(&mut transport);
        assert_eq!(transport.clients.len(), 1);

        // Drop the client, closing its end of the connection.
        drop(client);

        // Give the OS a moment to observe the close.
        std::thread::sleep(Duration::from_millis(50));

        // First write after disconnect: the write to the stale client may or
        // may not fail depending on OS buffering, but the call must not
        // return Err regardless -- and a second write is used to guarantee
        // the stale client has been pruned by then (retain_mut runs on every
        // write_event call).
        transport
            .write_event(&scalar_event(0, "loss", 0.1))
            .expect("write_event must not error on a stale client");
        transport
            .write_event(&scalar_event(1, "loss", 0.2))
            .expect("write_event must not error on a stale client");

        assert!(
            transport.clients.is_empty(),
            "disconnected client must be pruned from the broadcast list"
        );
    }
}
