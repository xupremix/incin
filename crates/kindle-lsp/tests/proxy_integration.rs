//! End-to-end proof that the real `kindle-lsp` binary — not just its pure
//! `rewrite` functions — correctly pumps and rewrites frames, using
//! `mock-rust-analyzer` as a stand-in server so this test needs no real
//! rust-analyzer install.

use kindle_lsp::frame::{read_frame, write_frame};
use std::io::BufReader;
use std::process::{Command, Stdio};

#[test]
fn proxy_rewrites_diagnostics_and_hints_but_passes_everything_else_through_byte_identical() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_kindle-lsp"))
        .env(
            "KINDLE_LSP_RA_PATH",
            env!("CARGO_BIN_EXE_mock-rust-analyzer"),
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn kindle-lsp");

    let mut proxy_stdin = child.stdin.take().unwrap();
    let request = br#"{"jsonrpc":"2.0","id":42,"method":"textDocument/inlayHint","params":{}}"#;
    write_frame(&mut proxy_stdin, request).unwrap();
    drop(proxy_stdin); // EOF, so the proxy's stdin-forwarding pump can terminate

    let mut proxy_stdout = BufReader::new(child.stdout.take().unwrap());

    // Frame 1: the unrelated notification must be forwarded byte-for-byte —
    // the proxy never re-serializes a message it isn't rewriting.
    let unrelated = read_frame(&mut proxy_stdout).unwrap().unwrap();
    assert_eq!(
        unrelated,
        br#"{"jsonrpc":"2.0","method":"window/logMessage","params":{"type":3,"message":"mock server started"}}"#
    );

    // Frame 2: publishDiagnostics gets its message humanized and hints attached.
    let diagnostics_frame = read_frame(&mut proxy_stdout).unwrap().unwrap();
    let diagnostics: serde_json::Value = serde_json::from_slice(&diagnostics_frame).unwrap();
    assert_eq!(diagnostics["method"], "textDocument/publishDiagnostics");
    let diag = &diagnostics["params"]["diagnostics"][0];
    assert_eq!(
        diag["message"],
        "Cannot reshape: source has 6 elements but the target shape has 8 elements"
    );
    let related = diag["relatedInformation"]
        .as_array()
        .expect("hints attached");
    assert_eq!(related.len(), 2);

    // Frame 3: the inlayHint response (matched by id 42) gets its label shape-humanized.
    let inlay_frame = read_frame(&mut proxy_stdout).unwrap().unwrap();
    let inlay: serde_json::Value = serde_json::from_slice(&inlay_frame).unwrap();
    assert_eq!(inlay["id"], 42);
    assert_eq!(
        inlay["result"][0]["label"],
        "Tensor<[2, 3], CpuBackendImpl<f32, Cpu>>"
    );

    assert!(
        read_frame(&mut proxy_stdout).unwrap().is_none(),
        "no extra frames"
    );

    let status = child.wait().unwrap();
    assert!(
        status.success(),
        "kindle-lsp should exit cleanly once the mock server closes its stdout"
    );
}
