//! `incin-lsp` - spawns rust-analyzer as a child process and pumps LSP
//! frames between it and the editor on two threads, rewriting only the
//! frames `incin_lsp::rewrite` recognizes. See `incin_lsp` (the library
//! crate) for the architectural rationale.

use incin_lsp::config::Config;
use incin_lsp::frame::{read_frame, write_frame};
use incin_lsp::rewrite::{PendingRequests, rewrite_incoming_from_server};
use std::io::{self, BufReader};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;

fn main() -> io::Result<()> {
    let config = Config::from_env();
    let arguments: Vec<_> = std::env::args_os().skip(1).collect();

    // Language clients sometimes probe a configured server before starting an
    // LSP session (for example, with `--version`). Those invocations do not
    // speak the framed LSP protocol, so preserve rust-analyzer's regular
    // command-line behavior and its stdio as well as its arguments.
    if arguments.iter().any(|argument| {
        matches!(
            argument.to_str(),
            Some("--version" | "-V" | "--help" | "-h")
        )
    }) {
        let status = Command::new(&config.ra_path)
            .args(&arguments)
            .stdin(Stdio::inherit())
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .status()
            .unwrap_or_else(|e| {
                eprintln!("incin-lsp: failed to spawn '{}': {e}", config.ra_path);
                std::process::exit(1);
            });
        std::process::exit(status.code().unwrap_or(1));
    }

    let mut child = Command::new(&config.ra_path)
        .args(&arguments)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .unwrap_or_else(|e| {
            eprintln!("incin-lsp: failed to spawn '{}': {e}", config.ra_path);
            std::process::exit(1);
        });

    let child_stdin = child.stdin.take().expect("piped child stdin");
    let child_stdout = child.stdout.take().expect("piped child stdout");
    let pending = Arc::new(Mutex::new(PendingRequests::new()));

    // Pump A: editor stdin -> rust-analyzer stdin, forwarded byte-for-byte.
    // Only inspected (never rewritten) to record `textDocument/inlayHint`
    // request ids, so pump B can later recognize their responses.
    let pending_for_pump_a = Arc::clone(&pending);
    let pump_a = thread::spawn(move || -> io::Result<()> {
        let mut editor_stdin = BufReader::new(io::stdin());
        let mut child_stdin = child_stdin;
        while let Some(body) = read_frame(&mut editor_stdin)? {
            if let Ok(msg) = serde_json::from_slice(&body) {
                pending_for_pump_a
                    .lock()
                    .unwrap()
                    .observe_outgoing_to_server(&msg);
            }
            write_frame(&mut child_stdin, &body)?;
        }
        Ok(())
    });

    // Pump B: rust-analyzer stdout -> editor stdout, rewriting the two
    // recognized message kinds and forwarding everything else verbatim.
    let mut server_stdout = BufReader::new(child_stdout);
    let mut editor_stdout = io::stdout();
    while let Some(body) = read_frame(&mut server_stdout)? {
        let rewritten = serde_json::from_slice(&body).ok().and_then(|msg| {
            let mut pending = pending.lock().unwrap();
            rewrite_incoming_from_server(
                &msg,
                &mut pending,
                config.hints_enabled,
                config.shorten_backend,
            )
        });
        match rewritten {
            Some(new_msg) => {
                let new_body =
                    serde_json::to_vec(&new_msg).expect("serialize rewritten LSP message");
                write_frame(&mut editor_stdout, &new_body)?;
            }
            None => write_frame(&mut editor_stdout, &body)?,
        }
    }

    // The editor closed its stdin (or rust-analyzer's stdout closed first);
    // either way, pump A no longer has anything to forward.
    let _ = pump_a.join();
    let status = child.wait()?;
    std::process::exit(status.code().unwrap_or(1));
}
