use kindle_diagnostics::humanize_diagnostic;
use serde_json::Value;
use std::env;
use std::io::{self, BufRead, Read};
use std::process::{Command, Stdio};

/// Renders a message with its translated typenum hints appended underneath.
fn render_translated_diagnostic(original: &str, raw: bool, explain: bool) {
    if raw {
        eprintln!("{}", original);
        return;
    }

    let translated = humanize_diagnostic(original);
    eprintln!("{}", translated.text);

    if !translated.hints.is_empty() {
        eprintln!("  └── 💡 [Typenum Translation Hints]:");
        for (num, typenum_expr) in translated.hints {
            eprintln!("      • {}  <= {}", num, typenum_expr);
        }
    }

    if explain {
        if original.contains("ConcatShape") {
            eprintln!(
                "  └── 📖 [Explain - Concatenation Rule]: All tensor dimensions except the concatenation axis must match exactly."
            );
        } else if original.contains("MatMulShape") || original.contains("matmul") {
            eprintln!(
                "  └── 📖 [Explain - MatMul Rule]: Inner dimensions must match: [M, K] x [K, N] -> [M, N]."
            );
        } else if original.contains("Conv2d") {
            eprintln!(
                "  └── 📖 [Explain - Conv2D Rule]: Input channels must match kernel input channels, and spatial dims must fit stride/padding."
            );
        }
    }
}

fn print_help() {
    println!("cargo-kindle — Seamless Kindle Cargo Interceptor & Model Tool");
    println!();
    println!("USAGE:");
    println!("    cargo kindle <SUBCOMMAND> [FLAGS] [CARGO ARGS...]");
    println!("    cargo-kindle inspect <MODEL_FILE>");
    println!();
    println!("SUBCOMMANDS:");
    println!("    check      Run cargo check with real-time typenum error translation");
    println!("    build      Run cargo build with real-time typenum error translation");
    println!("    test       Run cargo test with real-time typenum error translation");
    println!("    run        Run cargo run with real-time typenum error translation");
    println!("    inspect    Inspect a .safetensors, .gguf, or .onnx model file metadata");
    println!("    translate  Translate raw text containing typenum expressions from stdin or arg");
    println!();
    println!("FLAGS:");
    println!("    --raw      Disable typenum translation (output raw compiler diagnostics)");
    println!("    --explain  Append detailed Kindle shape rule explanations to errors");
    println!("    --help, -h Display this help message");
}

fn main() -> io::Result<()> {
    let mut raw_args: Vec<String> = env::args().collect();

    // Remove `kindle` if invoked via `cargo kindle ...`
    if raw_args.len() > 1 && raw_args[1] == "kindle" {
        raw_args.remove(1);
    }

    if raw_args.iter().any(|a| a == "--help" || a == "-h") || raw_args.len() <= 1 {
        print_help();
        return Ok(());
    }

    let mut raw_mode = false;
    let mut explain_mode = false;
    let mut cargo_args = Vec::new();
    let mut subcommand = String::new();

    for arg in raw_args.into_iter().skip(1) {
        if arg == "--raw" {
            raw_mode = true;
        } else if arg == "--explain" {
            explain_mode = true;
        } else if subcommand.is_empty() {
            subcommand = arg;
        } else {
            cargo_args.push(arg);
        }
    }

    if subcommand == "inspect" {
        if let Some(file_path) = cargo_args.first() {
            println!("🔍 Inspecting model file: {}", file_path);
            match kindle_core::io::inspect_file(file_path) {
                Ok(info) => {
                    println!("  • Format      : {}", info.format);
                    println!("  • File Path   : {}", info.path);
                    println!(
                        "  • Size        : {} bytes ({:.2} MB)",
                        info.file_size_bytes,
                        info.file_size_bytes as f64 / 1_048_576.0
                    );
                    println!("  • Tensor Count: {}", info.tensor_count);
                    if !info.tensors.is_empty() {
                        println!("\n  📦 Tensor Listing:");
                        for t in info.tensors.iter().take(20) {
                            println!(
                                "      • {:<30} {:<15?} {:<6} ({:.2} KB)",
                                t.name,
                                t.shape,
                                t.dtype,
                                t.size_bytes as f64 / 1024.0
                            );
                        }
                        if info.tensors.len() > 20 {
                            println!("      ... and {} more tensors.", info.tensors.len() - 20);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("Error inspecting model file: {}", e);
                }
            }
        } else {
            eprintln!(
                "Error: Please provide a model file path to inspect (e.g. `cargo kindle inspect model.gguf`)"
            );
        }
        return Ok(());
    }

    if subcommand == "translate" {
        if !cargo_args.is_empty() {
            let input = cargo_args.join(" ");
            render_translated_diagnostic(&input, raw_mode, explain_mode);
        } else {
            let mut buffer = String::new();
            io::stdin().read_to_string(&mut buffer)?;
            render_translated_diagnostic(&buffer, raw_mode, explain_mode);
        }
        return Ok(());
    }

    // Command Interception & Forwarding to Cargo
    let mut cmd_args = vec![subcommand];
    if !raw_mode {
        cmd_args.push("--message-format=json".to_string());
    }
    cmd_args.extend(cargo_args);

    let mut child = Command::new("cargo")
        .args(&cmd_args)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()?;

    let stdout = child.stdout.take().expect("Failed to open child stdout");
    let reader = io::BufReader::new(stdout);

    for line in reader.lines() {
        let line = line?;
        if !raw_mode
            && let Ok(json) = serde_json::from_str::<Value>(&line)
            && json.get("reason").and_then(|r| r.as_str()) == Some("compiler-message")
            && let Some(msg_obj) = json.get("message")
            && let Some(rendered) = msg_obj.get("rendered").and_then(|r| r.as_str())
        {
            render_translated_diagnostic(rendered, raw_mode, explain_mode);
            continue;
        }
        println!("{}", line);
    }

    let status = child.wait()?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}
