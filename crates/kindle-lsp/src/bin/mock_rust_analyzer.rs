//! Test-only stand-in for rust-analyzer, used by `tests/proxy_integration.rs`
//! to exercise the real `kindle-lsp` binary end-to-end without depending on
//! an actual rust-analyzer install being present. Not part of the product —
//! see the `test = false, doc = false` bin entry in `Cargo.toml`.
//!
//! Protocol: reads exactly one frame from stdin (the harness's
//! `textDocument/inlayHint` request), then emits three canned frames on
//! stdout — an unrelated notification, a `publishDiagnostics` notification,
//! and the inlayHint response (echoing back the request's own `id`) — each
//! containing raw typenum text for `kindle-lsp` to rewrite, before exiting.

use kindle_lsp::frame::{read_frame, write_frame};
use std::io::{self, BufReader};

fn main() -> io::Result<()> {
    let mut stdin = BufReader::new(io::stdin());
    let request_body = read_frame(&mut stdin)?.expect("harness must send exactly one frame");
    let request: serde_json::Value =
        serde_json::from_slice(&request_body).expect("harness frame must be valid JSON");
    let id = request
        .get("id")
        .cloned()
        .unwrap_or(serde_json::Value::Null);

    let mut stdout = io::stdout();

    // An unrelated notification — must reach the editor byte-for-byte
    // unchanged, since this is a fixed literal, not JSON re-serialized by
    // kindle-lsp.
    write_frame(&mut stdout, br#"{"jsonrpc":"2.0","method":"window/logMessage","params":{"type":3,"message":"mock server started"}}"#)?;

    let diagnostics = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "textDocument/publishDiagnostics",
        "params": {
            "uri": "file:///model.rs",
            "diagnostics": [{
                "range": {"start": {"line": 10, "character": 14}, "end": {"line": 10, "character": 21}},
                "severity": 1,
                "message": "Cannot reshape: source has UInt<UInt<UInt<UTerm, B1>, B1>, B0> elements but the target shape has UInt<UInt<UInt<UInt<UTerm, B1>, B0>, B0>, B0> elements"
            }]
        }
    });
    write_frame(&mut stdout, &serde_json::to_vec(&diagnostics)?)?;

    let inlay_hint_response = serde_json::json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": [{
            "position": {"line": 3, "character": 12},
            "label": "Tensor<(UInt<UInt<UTerm, B1>, B0>, UInt<UInt<UTerm, B1>, B1>), CpuBackendImpl<f32, Cpu>>"
        }]
    });
    write_frame(&mut stdout, &serde_json::to_vec(&inlay_hint_response)?)?;

    Ok(())
}
