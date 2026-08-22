//! End-to-end proof that the real `incin-lsp` binary - not just its pure
//! `rewrite` functions - correctly pumps and rewrites frames, using
//! a test-only `mock-rust-analyzer` stand-in server so this test needs no
//! real rust-analyzer install.

use incin_lsp::frame::{read_frame, write_frame};
use std::ffi::OsString;
use std::io::BufReader;
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicUsize, Ordering};

static MOCK_BINARY_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

fn build_mock_rust_analyzer() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source = manifest_dir.join("tests/support/mock_rust_analyzer.rs");
    let output_dir = std::env::var_os("CARGO_TARGET_TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let sequence = MOCK_BINARY_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let output = output_dir.join(format!(
        "mock-rust-analyzer-{}-{sequence}{}",
        std::process::id(),
        std::env::consts::EXE_SUFFIX,
    ));
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));

    let result = Command::new(rustc)
        .args([
            "--edition=2024",
            source.to_str().expect("UTF-8 source path"),
            "-o",
        ])
        .arg(&output)
        .status()
        .expect("run rustc for mock-rust-analyzer test support");
    assert!(result.success(), "compile mock-rust-analyzer test support");
    output
}

#[test]
fn proxy_rewrites_diagnostics_and_hints_but_passes_everything_else_through_byte_identical() {
    let mock_rust_analyzer = build_mock_rust_analyzer();
    let mut child = Command::new(env!("CARGO_BIN_EXE_incin-lsp"))
        .env("INCIN_LSP_RA_PATH", &mock_rust_analyzer)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn incin-lsp");

    let mut proxy_stdin = child.stdin.take().unwrap();
    let request = br#"{"jsonrpc":"2.0","id":42,"method":"textDocument/inlayHint","params":{}}"#;
    write_frame(&mut proxy_stdin, request).unwrap();
    drop(proxy_stdin); // EOF, so the proxy's stdin-forwarding pump can terminate

    let mut proxy_stdout = BufReader::new(child.stdout.take().unwrap());

    // Frame 1: the unrelated notification must be forwarded byte-for-byte -
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
        "Tensor<[2, 3], CpuBackendImpl<Cpu>>"
    );

    assert!(
        read_frame(&mut proxy_stdout).unwrap().is_none(),
        "no extra frames"
    );

    let status = child.wait().unwrap();
    assert!(
        status.success(),
        "incin-lsp should exit cleanly once the mock server closes its stdout"
    );

    std::fs::remove_file(mock_rust_analyzer).expect("remove mock-rust-analyzer test support");
}

#[test]
fn proxy_rewrites_inlay_hint_resolve_requests() {
    let mock_rust_analyzer = build_mock_rust_analyzer();
    let mut child = Command::new(env!("CARGO_BIN_EXE_incin-lsp"))
        .env("INCIN_LSP_RA_PATH", &mock_rust_analyzer)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("spawn incin-lsp");

    let mut proxy_stdin = child.stdin.take().unwrap();
    let request = br#"{"jsonrpc":"2.0","id":77,"method":"inlayHint/resolve","params":{}}"#;
    write_frame(&mut proxy_stdin, request).unwrap();
    drop(proxy_stdin);

    let mut proxy_stdout = BufReader::new(child.stdout.take().unwrap());
    let resolve_frame = read_frame(&mut proxy_stdout).unwrap().unwrap();
    let resolve: serde_json::Value = serde_json::from_slice(&resolve_frame).unwrap();
    assert_eq!(resolve["id"], 77);
    assert_eq!(
        resolve["result"]["label"],
        "Tensor<[2, 3], CpuBackendImpl<Cpu>>"
    );

    assert!(
        read_frame(&mut proxy_stdout).unwrap().is_none(),
        "no extra frames"
    );

    let status = child.wait().unwrap();
    assert!(
        status.success(),
        "incin-lsp should exit cleanly once the mock server closes its stdout"
    );

    std::fs::remove_file(mock_rust_analyzer).expect("remove mock-rust-analyzer test support");
}

#[test]
fn proxy_forwards_version_probe_to_rust_analyzer() {
    let mock_rust_analyzer = build_mock_rust_analyzer();
    let output = Command::new(env!("CARGO_BIN_EXE_incin-lsp"))
        .arg("--version")
        .env("INCIN_LSP_RA_PATH", &mock_rust_analyzer)
        .output()
        .expect("run incin-lsp version probe");

    assert!(output.status.success(), "version probe should succeed");
    assert_eq!(output.stdout, b"mock-rust-analyzer 0.1.0\n");
    assert!(
        output.stderr.is_empty(),
        "version probe should not write stderr"
    );

    std::fs::remove_file(mock_rust_analyzer).expect("remove mock-rust-analyzer test support");
}
