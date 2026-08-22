use incin_diagnostics::{
    humanize_diagnostic, parse_broadcast_mismatch, parse_concat_mismatch, parse_conv1d_mismatch,
    parse_conv2d_mismatch, parse_flatten_mismatch, parse_matmul_mismatch,
    parse_module_forward_mismatch, parse_pool2d_mismatch, parse_reduce_dim_mismatch,
    parse_reshape_mismatch, parse_slice_mismatch, parse_transpose_mismatch,
};
use serde_json::Value;
use std::env;
use std::io::{self, BufRead, Read};
use std::process::{Command, Stdio};

/// Embedded `cargo incin new` scaffold templates (Task 05.3). Only `mnist`
/// exists today -- `cnn`/`mlp` are noted as future work in
/// `docs/growth/05-observability-and-scaffolding.md` rather than built
/// speculatively ahead of a second real template.
mod templates {
    pub const MNIST_CARGO_TOML: &str = include_str!("templates/mnist/Cargo.toml.template");
    pub const MNIST_MAIN_RS: &str = include_str!("templates/mnist/main.rs.template");
    pub const MNIST_README: &str = include_str!("templates/mnist/README.md.template");
}

/// Embedded Incin AI Agent Skills for model developers and framework contributors.
mod embedded_skills {
    pub const INCIN_EXPERT: &str = include_str!("skills/incin-expert/SKILL.md");
    pub const INCIN_ENGINEERING: &str = include_str!("skills/incin-engineering/SKILL.md");
    pub const INCIN_REPOSITORY: &str = include_str!("skills/incin-repository/SKILL.md");
}

/// Absolute path to this crate's own manifest directory (`crates/incin`),
/// baked in at `cargo-incin`'s own compile time. Scaffolded projects use
/// this to path-depend on the exact incin checkout that built the
/// `cargo-incin` binary generating them -- see the `Cargo.toml.template`
/// comment for why this can't simply be a crates.io version dependency yet.
const INCIN_MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

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
        if translated.text.contains("ConcatShape")
            || translated.text.contains("Cannot concatenate shape")
            || original.contains("ConcatShape")
        {
            eprintln!(
                "  └── 📖 [Explain - Concatenation Rule]: All tensor dimensions except the concatenation axis must match exactly."
            );
            if let Some(mismatch) = parse_concat_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("MatMulShape")
            || translated.text.contains("matrix-multiply")
            || original.contains("MatMulShape")
            || original.contains("matmul")
        {
            eprintln!(
                "  └── 📖 [Explain - MatMul Rule]: matmul requires [M, K] x [K, N] -> [M, N] — the inner dimensions (K) must match."
            );
            if let Some(mismatch) = parse_matmul_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("Conv2D")
            || translated.text.contains("Conv2d")
            || translated.text.contains("incompatible with kernel shape")
            || original.contains("Conv2d")
        {
            eprintln!(
                "  └── 📖 [Explain - Conv2D Rule]: Input channels must match kernel input channels, and spatial dims must fit stride/padding."
            );
            if let Some(mismatch) = parse_conv2d_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("BroadcastShape")
            || translated.text.contains("Cannot broadcast shape")
            || original.contains("Broadcast")
        {
            eprintln!(
                "  └── 📖 [Explain - Broadcast Rule]: Dimensions must either match exactly or one of them must be 1."
            );
            if let Some(mismatch) = parse_broadcast_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("ReshapeShape")
            || translated.text.contains("Cannot reshape from")
            || original.contains("Reshape")
        {
            eprintln!(
                "  └── 📖 [Explain - Reshape Rule]: Total number of elements (product of dimensions) before and after reshape must be identical."
            );
            if let Some(mismatch) = parse_reshape_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("StackShape")
            || translated.text.contains("Cannot stack shape")
            || original.contains("Stack")
        {
            eprintln!(
                "  └── 📖 [Explain - Stack Rule]: All tensors being stacked must have identical shapes."
            );
        } else if translated.text.contains("Slice")
            || translated.text.contains("Cannot slice dimension")
            || original.contains("Slice")
            || original.contains("idx!")
        {
            eprintln!(
                "  └── 📖 [Explain - Slice/Indexing Rule]: Slice ranges must be within tensor dimension bounds."
            );
            if let Some(mismatch) = parse_slice_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("Conv1D")
            || translated.text.contains("Conv1d")
            || translated.text.contains("1D convolution")
        {
            eprintln!(
                "  └── 📖 [Explain - Conv1D Rule]: Conv1D requires a 2D or 3D tensor (C, L) or (B, C, L)."
            );
            if let Some(mismatch) = parse_conv1d_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("Transpose")
            || translated.text.contains("Cannot transpose dimensions")
            || original.contains("Transpose")
        {
            eprintln!(
                "  └── 📖 [Explain - Transpose Rule]: Transpose indices must be < the rank of the tensor."
            );
            if let Some(mismatch) = parse_transpose_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("ReduceDim")
            || translated.text.contains("Cannot reduce dimension")
            || original.contains("ReduceDim")
        {
            eprintln!(
                "  └── 📖 [Explain - Reduction Rule]: Reduction dimension index must be < the rank of the tensor."
            );
            if let Some(mismatch) = parse_reduce_dim_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("Flatten")
            || translated.text.contains("Cannot flatten shape")
            || original.contains("Flatten")
        {
            eprintln!(
                "  └── 📖 [Explain - Flatten Rule]: Flatten range [START, END] requires START <= END and END < rank."
            );
            if let Some(mismatch) = parse_flatten_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("Pool2d") || original.contains("Pool") {
            eprintln!(
                "  └── 📖 [Explain - Pooling Rule]: Spatial input dimensions (H, W) must be larger than or equal to kernel dimensions."
            );
            if let Some(mismatch) = parse_pool2d_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        } else if translated.text.contains("Module")
            || translated.text.contains("forward")
            || original.contains("Module")
        {
            eprintln!(
                "  └── 📖 [Explain - Module Forward Rule]: Layer / Module forward pass input shape must match the layer's expected input shape."
            );
            if let Some(mismatch) = parse_module_forward_mismatch(&translated.text) {
                eprintln!("{}", mismatch.render());
            }
        }
    }
}

fn print_help() {
    println!("cargo-incin — Seamless Incin Cargo Interceptor & Model Tool");
    println!();
    println!("USAGE:");
    println!("    cargo incin <SUBCOMMAND> [FLAGS] [CARGO ARGS...]");
    println!("    cargo-incin inspect <MODEL_FILE>");
    println!();
    println!("SUBCOMMANDS:");
    println!("    check      Run cargo check with real-time typenum error translation");
    println!("    build      Run cargo build with real-time typenum error translation");
    println!("    test       Run cargo test with real-time typenum error translation");
    println!("    run        Run cargo run with real-time typenum error translation");
    println!("    bench, doc, fix, clippy");
    println!("               Same real-time typenum error translation as above");
    println!("    doctor     Report devices, features, caches, and capability probes [--json]");
    println!("    inspect    Inspect a .safetensors, .gguf, or .onnx model file metadata");
    println!("    plan       Generate execution plan report [--json] [--devices DEV] [--epochs N]");
    println!("    tune       Inspect and manage autotune cache [--json] [--clear] [--offline]");
    println!("    translate  Translate raw text containing typenum expressions from stdin or arg");
    println!("    new <template> [path]");
    println!("               Scaffold a ready-to-run training project (templates: mnist)");
    println!(
        "    skills     Manage and install Incin agent skills for AI assistants (list/install)"
    );
    println!(
        "    watch      Launch the incin-viz live telemetry TUI ([--run-id ID] or [--run-dir PATH])"
    );
    println!("    <anything else>");
    println!("               Delegated straight to `cargo <subcommand>` untouched — this");
    println!("               covers every other built-in (fmt, tree, add, update, ...) and");
    println!("               any third-party cargo plugin you have installed, with no need");
    println!("               for cargo-incin to know about it ahead of time.");
    println!();
    println!("FLAGS:");
    println!("    --raw      Disable typenum translation (output raw compiler diagnostics)");
    println!("    --explain  Append detailed Incin shape rule explanations to errors");
    println!("    --help, -h Display this help message");
}

fn main() -> io::Result<()> {
    let mut raw_args: Vec<String> = env::args().collect();

    // Remove `incin` if invoked via `cargo incin ...`
    if raw_args.len() > 1 && raw_args[1] == "incin" {
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

    if subcommand == "doctor" {
        // The report itself lives in the library (`incin::doctor`), not here:
        // `UX-014`'s evidence command is an integration test, and an
        // integration test links the library rather than this binary.
        let (rendered, code) = incin::doctor::run(&cargo_args);
        // A report goes to stdout even when it found something wrong, because
        // a report is what was asked for; only a usage error goes to stderr.
        if code == incin::doctor::EXIT_USAGE {
            eprint!("{rendered}");
        } else {
            print!("{rendered}");
        }
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }

    if subcommand == "plan" {
        #[cfg(feature = "train")]
        {
            let (rendered, code) = incin::experimental::training::plan_report::run(&cargo_args);
            if code == incin::experimental::training::plan_report::EXIT_USAGE {
                eprint!("{rendered}");
            } else {
                print!("{rendered}");
            }
            if code != 0 {
                std::process::exit(code);
            }
            return Ok(());
        }
        #[cfg(not(feature = "train"))]
        {
            eprintln!("Error: cargo incin plan requires the `train` feature.");
            std::process::exit(1);
        }
    }

    if subcommand == "tune" {
        let (rendered, code) = incin::experimental::tuning_report::run(&cargo_args);
        if code == incin::experimental::tuning_report::EXIT_USAGE {
            eprint!("{rendered}");
        } else {
            print!("{rendered}");
        }
        if code != 0 {
            std::process::exit(code);
        }
        return Ok(());
    }

    if subcommand == "inspect" {
        if let Some(file_path) = cargo_args.first() {
            println!("🔍 Inspecting model file: {}", file_path);
            match incin_core::io::inspect_file(file_path) {
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
                "Error: Please provide a model file path to inspect (e.g. `cargo incin inspect model.gguf`)"
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

    if subcommand == "new" {
        let Some(template_name) = cargo_args.first().cloned() else {
            eprintln!("Error: please specify a template (e.g. `cargo incin new mnist`)");
            std::process::exit(1);
        };
        let (cargo_toml, main_rs, readme) = match template_name.as_str() {
            "mnist" => (
                templates::MNIST_CARGO_TOML,
                templates::MNIST_MAIN_RS,
                templates::MNIST_README,
            ),
            other => {
                eprintln!("Error: unknown template `{other}` (available: mnist)");
                std::process::exit(1);
            }
        };
        let target = cargo_args
            .get(1)
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| std::path::PathBuf::from(&template_name));

        if target.exists() {
            eprintln!("Error: `{}` already exists", target.display());
            std::process::exit(1);
        }

        let project_name = target
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&template_name)
            .to_string();

        let incin_path = std::path::Path::new(INCIN_MANIFEST_DIR);
        // `INCIN_MANIFEST_DIR` is `.../crates/incin`; `incin-telemetry`
        // is its workspace sibling under the same `crates/` parent.
        let incin_telemetry_path = incin_path
            .parent()
            .expect("crates/incin always has a parent directory")
            .join("incin-telemetry");

        let substitute = |template: &str| -> String {
            template
                .replace("{{PROJECT_NAME}}", &project_name)
                .replace("{{INCIN_PATH}}", &incin_path.display().to_string())
                .replace(
                    "{{INCIN_TELEMETRY_PATH}}",
                    &incin_telemetry_path.display().to_string(),
                )
        };

        std::fs::create_dir_all(target.join("src"))?;
        std::fs::write(target.join("Cargo.toml"), substitute(cargo_toml))?;
        std::fs::write(target.join("src/main.rs"), substitute(main_rs))?;
        std::fs::write(target.join("README.md"), substitute(readme))?;

        println!(
            "Scaffolded `{template_name}` project at {}",
            target.display()
        );
        println!("    cd {} && cargo run", target.display());
        return Ok(());
    }

    if subcommand == "skills" {
        let action = cargo_args.first().map(|s| s.as_str()).unwrap_or("list");
        match action {
            "list" => {
                println!("Available Incin AI Agent Skills:");
                println!(
                    "  • incin-expert        Expert guide for model developers writing neural networks and training loops."
                );
                println!(
                    "  • incin-engineering   Core framework engineering, custom backend implementation, and invariant contracts."
                );
                println!(
                    "  • incin-repository    Repository navigation, verification gates, docs, and test runners."
                );
                println!();
                println!("Usage:");
                println!(
                    "  cargo incin skills install [--tool <cursor|antigravity|claude|windsurf|all>] [--dir <path>]"
                );
            }
            "install" => {
                let mut tool = "all";
                let mut custom_dir: Option<std::path::PathBuf> = None;
                let mut iter = cargo_args.iter().skip(1);
                while let Some(arg) = iter.next() {
                    match arg.as_str() {
                        "--tool" | "-t" => {
                            if let Some(t) = iter.next() {
                                tool = t.as_str();
                            }
                        }
                        "--dir" | "-d" => {
                            if let Some(d) = iter.next() {
                                custom_dir = Some(std::path::PathBuf::from(d));
                            }
                        }
                        _ => {}
                    }
                }

                let skills = [
                    ("incin-expert", embedded_skills::INCIN_EXPERT),
                    ("incin-engineering", embedded_skills::INCIN_ENGINEERING),
                    ("incin-repository", embedded_skills::INCIN_REPOSITORY),
                ];

                let install_skill =
                    |name: &str, content: &str, dir: &std::path::Path| -> io::Result<()> {
                        let target_dir = dir.join(name);
                        std::fs::create_dir_all(&target_dir)?;
                        let target_file = target_dir.join("SKILL.md");
                        std::fs::write(&target_file, content)?;
                        println!("  ✓ Installed skill {} -> {}", name, target_file.display());
                        Ok(())
                    };

                let install_cursor_rule =
                    |name: &str, content: &str, dir: &std::path::Path| -> io::Result<()> {
                        std::fs::create_dir_all(dir)?;
                        let target_file = dir.join(format!("{}.mdc", name));
                        std::fs::write(&target_file, content)?;
                        println!(
                            "  ✓ Installed Cursor rule {} -> {}",
                            name,
                            target_file.display()
                        );
                        Ok(())
                    };

                if let Some(dir) = custom_dir {
                    println!("Installing Incin agent skills to {}...", dir.display());
                    for (name, content) in &skills {
                        install_skill(name, content, &dir)?;
                    }
                } else {
                    match tool {
                        "antigravity" | "agy" | "gemini" => {
                            println!(
                                "Installing Incin agent skills for Antigravity / Gemini (.agents/skills/)..."
                            );
                            let dir = std::path::Path::new(".agents/skills");
                            for (name, content) in &skills {
                                install_skill(name, content, dir)?;
                            }
                        }
                        "cursor" => {
                            println!(
                                "Installing Incin agent skills for Cursor (.cursor/rules/)..."
                            );
                            let dir = std::path::Path::new(".cursor/rules");
                            for (name, content) in &skills {
                                install_cursor_rule(name, content, dir)?;
                            }
                        }
                        "claude" => {
                            println!(
                                "Installing Incin agent skills for Claude Code (.claude/skills/)..."
                            );
                            let dir = std::path::Path::new(".claude/skills");
                            for (name, content) in &skills {
                                install_skill(name, content, dir)?;
                            }
                        }
                        "windsurf" => {
                            println!(
                                "Installing Incin agent skills for Windsurf (.windsurf/rules/)..."
                            );
                            let dir = std::path::Path::new(".windsurf/rules");
                            for (name, content) in &skills {
                                install_cursor_rule(name, content, dir)?;
                            }
                        }
                        _ => {
                            println!(
                                "Installing Incin agent skills for all environments (.agents/skills/ and .cursor/rules/)..."
                            );
                            let agy_dir = std::path::Path::new(".agents/skills");
                            let cursor_dir = std::path::Path::new(".cursor/rules");
                            for (name, content) in &skills {
                                install_skill(name, content, agy_dir)?;
                                install_cursor_rule(name, content, cursor_dir)?;
                            }
                        }
                    }
                }
                println!("\nSuccessfully installed Incin Agent Skills!");
            }
            other => {
                eprintln!("Error: unknown skills command `{other}` (expected `list` or `install`)");
                std::process::exit(1);
            }
        }
        return Ok(());
    }

    if subcommand == "watch" {
        match Command::new("incin-viz").args(&cargo_args).status() {
            Ok(status) => {
                if !status.success() {
                    std::process::exit(status.code().unwrap_or(1));
                }
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // Not installed as a standalone binary (no `cargo install
                // incin-viz` yet) -- fall back to `cargo run -p
                // incin-viz`, which works from inside this workspace (or
                // any checkout where `incin-viz` is a member) with no
                // separate install step. Note: only the `--file`-transport
                // (default, XDG run-dir) path is supported here --
                // `incin-viz` has no socket *reader* yet (only
                // `incin-telemetry`'s write-side `SocketTransport` exists),
                // so there is no `--socket` flag to pass through.
                let mut fallback_args = vec![
                    "run".to_string(),
                    "--quiet".to_string(),
                    "-p".to_string(),
                    "incin-viz".to_string(),
                    "--".to_string(),
                ];
                fallback_args.extend(cargo_args);
                let status = Command::new("cargo").args(&fallback_args).status()?;
                if !status.success() {
                    std::process::exit(status.code().unwrap_or(1));
                }
            }
            Err(e) => return Err(e),
        }
        return Ok(());
    }

    // Only cargo's own compilation-triggering subcommands understand
    // `--message-format=json` (and thus benefit from diagnostic
    // translation) - everything else, whether a built-in like `fmt`/
    // `tree`/`add`/`update`/`publish` or a third-party plugin the user
    // installed (`cargo-watch`, `cargo-audit`, `cargo-expand`, ...), gets
    // delegated straight through with fully inherited stdio and no
    // interception at all. This is deliberately a small, closed allowlist
    // of cargo's own commands rather than a blocklist of known-bad ones:
    // it means supporting a new third-party subcommand needs zero code
    // here, ever - that's the whole point.
    const JSON_CAPABLE_SUBCOMMANDS: &[&str] = &[
        "build", "b", "check", "c", "test", "t", "bench", "run", "r", "clippy", "doc", "fix",
    ];

    if raw_mode || !JSON_CAPABLE_SUBCOMMANDS.contains(&subcommand.as_str()) {
        let mut cmd_args = vec![subcommand];
        cmd_args.extend(cargo_args);
        let status = Command::new("cargo").args(&cmd_args).status()?;
        if !status.success() {
            std::process::exit(status.code().unwrap_or(1));
        }
        return Ok(());
    }

    // Command Interception & Forwarding to Cargo
    let mut cmd_args = vec![subcommand, "--message-format=json".to_string()];
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
        // `--raw` is handled above by delegating before `--message-format=json`
        // is ever added, so by construction `raw_mode` is always `false` here.
        // `--message-format=json` makes cargo emit *only* JSON on stdout,
        // but its own pretty "Compiling foo v0.1.0 (...)" progress lines
        // always go to *stderr* regardless of `--message-format` - and
        // that stream is inherited untouched (`Stdio::inherit()` above), so
        // that progress already reaches the terminal correctly on its own.
        // Every JSON `reason` on stdout still needs to be handled here or
        // its raw line leaks straight to the terminal; a plain non-JSON
        // line (e.g. a program's own stdout under `cargo incin run`)
        // still just prints.
        let Ok(json) = serde_json::from_str::<Value>(&line) else {
            println!("{}", line);
            continue;
        };
        match json.get("reason").and_then(|r| r.as_str()) {
            Some("compiler-message") => {
                if let Some(rendered) = json
                    .get("message")
                    .and_then(|m| m.get("rendered"))
                    .and_then(|r| r.as_str())
                {
                    render_translated_diagnostic(rendered, raw_mode, explain_mode);
                }
            }
            // `compiler-artifact` (build progress, already shown via
            // stderr above), `build-script-executed`, and `build-finished`
            // are cargo-internal bookkeeping a plain `cargo build` never
            // prints either - suppressed, not dumped raw. Deliberately an
            // explicit list, not a wildcard: `cargo incin run`'s program
            // output could itself be JSON containing an unrelated "reason"
            // key, and a wildcard would silently eat that instead of
            // printing it.
            Some("compiler-artifact") | Some("build-script-executed") | Some("build-finished") => {}
            Some(_) | None => println!("{}", line),
        }
    }

    let status = child.wait()?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }

    Ok(())
}
