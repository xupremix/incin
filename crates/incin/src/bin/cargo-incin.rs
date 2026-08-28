//! The `cargo incin` CLI: a cargo subcommand that wraps ordinary cargo
//! invocations with real-time typenum-error translation, and provides the
//! model inspection (`inspect`), environment diagnosis (`doctor`), and
//! plan/tune tooling documented in the Book. See `--help` for the
//! subcommand surface.
use incin_diagnostics::{
    humanize_diagnostic, parse_broadcast_mismatch, parse_concat_mismatch,
    parse_contraction_mismatch, parse_conv1d_mismatch, parse_conv2d_mismatch,
    parse_flatten_mismatch, parse_matmul_mismatch, parse_module_forward_mismatch,
    parse_pool2d_mismatch, parse_reduce_dim_mismatch, parse_reshape_mismatch, parse_slice_mismatch,
    parse_transpose_mismatch,
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
/// The version a scaffold depends on: the one this tool was built from, so a
/// generated project always matches the `cargo incin new` that wrote it.
const INCIN_VERSION: &str = env!("CARGO_PKG_VERSION");

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

    if let Some(explanation) = explain
        .then(|| explanation(&translated.text, original))
        .flatten()
    {
        eprintln!(
            "  └── 📖 [Explain - {}]: {}",
            explanation.rule.0, explanation.rule.1
        );
        if let Some(detail) = explanation.detail {
            eprintln!("{detail}");
        }
    }
}

/// One explanation: the rule that was broken, and the pointed-out detail when
/// the message carried enough to draw one.
struct Explanation {
    /// Heading and the rule itself.
    rule: (&'static str, &'static str),
    /// The rendered diagram, when a parser recognized the message.
    detail: Option<String>,
}

/// Every fixed `#[diagnostic::on_unimplemented]` message the shape traits emit,
/// paired with the rule it means.
///
/// Matching on these literals rather than on an identifier is the whole point.
/// `original.contains("matmul")` fires on any diagnostic that merely mentions
/// one -- a binding named `matmul_out`, a path `src/matmul.rs`, an unrelated
/// borrow error in a function whose name contains it -- and then explains a
/// rule the reader never broke. These messages, by contrast, exist only when
/// the corresponding bound actually failed.
///
/// The list is exhaustive against `grep -r 'message = "' crates/incin-core`, so
/// a trait that grows a message and is not added here explains nothing rather
/// than explaining the wrong thing.
const RULES: &[(&str, &str, &str)] = &[
    (
        "Cannot matrix-multiply shape `",
        "MatMul Rule",
        "matmul contracts [.., M, K] against [.., K, N] to give [.., M, N]: the two inner dimensions must be equal, and every leading batch dimension must agree.",
    ),
    (
        "Cannot contract dimension `",
        "MatMul Rule",
        "matmul contracts the last axis of the left operand against the second-to-last of the right; those two must be equal, or one of them a runtime `usize`.",
    ),
    (
        "Cannot concatenate shape `",
        "Concatenation Rule",
        "concat joins along one axis: every dimension except that axis must match exactly, and both operands must have the same rank.",
    ),
    (
        "Cannot stack shape `",
        "Stack Rule",
        "stack inserts a new axis, so unlike concat it requires the operands to have identical shapes; use concat if you meant to join along an existing axis.",
    ),
    (
        "Cannot broadcast shape `",
        "Broadcast Rule",
        "shapes are aligned from the right, and at each axis the two extents must be equal or one of them must be 1; a missing leading axis counts as 1.",
    ),
    (
        "Cannot broadcast axis `",
        "Broadcast Rule",
        "this one axis broke the rule: two extents broadcast only when they are equal or one of them is 1. An extent of 1 stretches; any other pair does not.",
    ),
    (
        "Cannot reshape from `",
        "Reshape Rule",
        "reshape preserves the element count: the product of the source dimensions must equal the product of the target dimensions.",
    ),
    (
        "Cannot reshape: source has ",
        "Reshape Rule",
        "the two element counts differ, so no reshape exists between these shapes; reshape re-addresses the same buffer and can neither add nor drop elements.",
    ),
    (
        "Cannot reshape dimension into `",
        "Reshape Rule",
        "a target dimension must be a concrete extent the source can be divided into; an inferred axis needs the remaining elements to divide evenly.",
    ),
    (
        "Cannot slice dimension with `",
        "Slice Rule",
        "a slice range must lie within the axis it addresses: the start must be below the extent and the end must not exceed it.",
    ),
    (
        "Cannot apply a `",
        "Conv Kernel Rule",
        "the kernel's input-channel count must equal the activation's channel extent, and the kernel must fit the spatial extents once stride, padding and dilation are applied.",
    ),
    (
        "Cannot apply 2D convolution to shape `",
        "Conv2D Rule",
        "conv2d takes a [C, H, W] or [N, C, H, W] activation; the channel axis is the third from the end, and the weight is always rank four.",
    ),
    (
        "Cannot apply 1D convolution to shape `",
        "Conv1D Rule",
        "conv1d takes a [C, L] or [N, C, L] activation; the channel axis is the second from the end, and the weight is always rank three.",
    ),
    (
        "Cannot apply 2D pooling to shape `",
        "Pooling Rule",
        "pooling takes a [C, H, W] or [N, C, H, W] activation, and the window must fit the spatial extents once stride, padding and dilation are applied.",
    ),
    (
        "Cannot apply adaptive 2D pooling to shape `",
        "Adaptive Pooling Rule",
        "adaptive pooling takes a [C, H, W] or [N, C, H, W] activation and rewrites only the two trailing axes to the requested output size.",
    ),
    (
        "Cannot use shape `",
        "Layer Shape Rule",
        "this layer pins one axis of its input: the shape given does not carry the channel count or trailing width the layer was built for.",
    ),
    (
        "dimension `",
        "Sharding Rule",
        "a sharded axis must divide evenly by the sharding degree; an axis with a remainder has no even split across ranks.",
    ),
    (
        "placement transition from `",
        "Placement Rule",
        "not every placement can be reached from every other; a transition has to be expressible as a collective this backend supports.",
    ),
    (
        "` is not a valid logical mesh",
        "Mesh Rule",
        "a logical mesh must name a positive extent per axis, and its total size must equal the number of ranks in the process group.",
    ),
    (
        "` is not a valid rank in a two-rank distributed context",
        "Rank Rule",
        "a two-rank context addresses exactly ranks zero and one.",
    ),
    (
        "` is not a collective supported for dtype `",
        "Collective Rule",
        "the collective exists, but not for this dtype on this backend; the supported dtype set per collective is part of the backend's capability table.",
    ),
    (
        "implements `Module<",
        "Module Forward Rule",
        "a module's forward input type is part of its signature: the shape passed in must match the shape the layer was constructed for.",
    ),
];

/// The explanation for a diagnostic, or `None` when nothing in it names a rule
/// this tool knows.
///
/// The match is the message whose prefix appears **earliest**, because a
/// rendered diagnostic leads with its primary message and carries the notes
/// underneath. Taking the first table entry that matches anywhere would let a
/// note decide the explanation for the error above it.
///
/// `original` is searched too, since a long-type substitution can rewrite the
/// span a message sits in, but only for the same fixed literals.
fn explanation(translated: &str, original: &str) -> Option<Explanation> {
    let earliest = |text: &str| {
        RULES
            .iter()
            .filter_map(|entry| text.find(entry.0).map(|at| (at, entry)))
            .min_by_key(|(at, _)| *at)
    };

    let (_, entry) = earliest(translated).or_else(|| earliest(original))?;
    let detail = detail_for(entry.0, translated);
    Some(Explanation {
        rule: (entry.1, entry.2),
        detail,
    })
}

/// The rendered diagram for a message, when a parser recognizes it.
///
/// A message with no parser gets its rule and nothing else, which is the
/// honest answer: several of these carry only the shape that failed and not
/// what it failed against, and there is no diagram to draw from one shape.
fn detail_for(message: &str, text: &str) -> Option<String> {
    let rendered = match message {
        "Cannot matrix-multiply shape `" => parse_matmul_mismatch(text).map(|m| m.render()),
        "Cannot contract dimension `" => parse_contraction_mismatch(text).map(|m| m.render()),
        "Cannot concatenate shape `" => parse_concat_mismatch(text).map(|m| m.render()),
        "Cannot broadcast shape `" => parse_broadcast_mismatch(text).map(|m| m.render()),
        "Cannot reshape from `" => parse_reshape_mismatch(text).map(|m| m.render()),
        "Cannot slice dimension with `" => parse_slice_mismatch(text).map(|m| m.render()),
        "Cannot apply 1D convolution to shape `" => parse_conv1d_mismatch(text).map(|m| m.render()),
        "Cannot apply 2D pooling to shape `" => parse_pool2d_mismatch(text).map(|m| m.render()),
        "Cannot apply a `" => parse_conv2d_mismatch(text).map(|m| m.render()),
        "implements `Module<" => parse_module_forward_mismatch(text).map(|m| m.render()),
        _ => None,
    };
    // The remaining parsers -- transpose, flatten, reduce-dim -- look for
    // messages no trait emits today. They are tried last against the whole
    // text rather than wired to a message that does not exist, so they cost
    // nothing now and start working the moment such a message appears.
    rendered
        .or_else(|| parse_transpose_mismatch(text).map(|m| m.render()))
        .or_else(|| parse_flatten_mismatch(text).map(|m| m.render()))
        .or_else(|| parse_reduce_dim_mismatch(text).map(|m| m.render()))
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
    println!(
        "    doctor     Report devices, features, caches, and capability probes"
    );
    println!(
        "               [--json] [--check-updates: ask crates.io for a newer incin]"
    );
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

        let substitute = |template: &str| -> String {
            template
                .replace("{{PROJECT_NAME}}", &project_name)
                .replace("{{INCIN_VERSION}}", INCIN_VERSION)
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

#[cfg(test)]
mod explain_tests {
    use super::{RULES, explanation};

    /// The regression this table exists for.
    ///
    /// The chain used to select a branch on `original.contains("matmul")`, so
    /// any diagnostic whose source line merely mentioned one -- a binding named
    /// `matmul_out`, a path `src/matmul.rs` -- was explained as a matmul shape
    /// failure. Explaining a rule the reader did not break is worse than
    /// explaining nothing, because they go looking for a mistake that is not
    /// there.
    #[test]
    fn a_diagnostic_that_merely_mentions_an_operation_is_not_explained() {
        let unrelated = "error[E0308]: mismatched types\n \
             --> src/main.rs:6:27\n  |\n6 | let matmul_out: i32 = \"not a number\";\n  \
             |                       ^^^^^^^^^^^^^^ expected `i32`, found `&str`";

        assert!(explanation(unrelated, unrelated).is_none());
    }

    /// Several loose identifiers used to have their own branch: `Stack`,
    /// `Pool`, `Slice`, `Transpose`, `Flatten`, `Module`, `forward`. Each
    /// appears in ordinary Rust diagnostics that have nothing to do with
    /// shapes.
    #[test]
    fn ordinary_rust_identifiers_do_not_select_a_shape_rule() {
        for text in [
            "error: cannot borrow `pool` as mutable",
            "note: required by a bound in `core::slice::Iter`",
            "error[E0599]: no method named `forward` found for struct `Config`",
            "warning: unused import: `std::fmt::Debug`, stack overflow detected",
            "error: `Flatten` is not iterable here",
        ] {
            assert!(
                explanation(text, text).is_none(),
                "explained an unrelated diagnostic: {text}"
            );
        }
    }

    /// The message a shape trait actually emits does select its rule.
    #[test]
    fn the_emitted_message_selects_its_own_rule() {
        let text = "error[E0277]: Cannot contract dimension `3` with `4`";
        let explained = explanation(text, text).expect("the contraction message explains");

        assert_eq!(explained.rule.0, "MatMul Rule");
        assert!(explained.detail.is_some(), "the diagram should render");
    }

    /// A rendered diagnostic leads with its primary message and carries notes
    /// underneath, so the earliest match wins rather than the first table
    /// entry that appears anywhere.
    #[test]
    fn the_primary_message_wins_over_a_note_below_it() {
        let text = "error[E0277]: Cannot broadcast shape `[2, 3]` to `[4, 5]`\n  \
                    = note: required for `Cannot concatenate shape `[1]` with `[2]` along axis `0``";

        let explained = explanation(text, text).expect("the broadcast message explains");
        assert_eq!(explained.rule.0, "Broadcast Rule");
    }

    /// No two table prefixes may shadow one another, or which rule a message
    /// gets would depend on table order rather than on the message.
    #[test]
    fn no_table_prefix_shadows_another() {
        for (outer, ..) in RULES {
            for (inner, ..) in RULES {
                if outer != inner {
                    assert!(!outer.starts_with(inner), "`{inner}` shadows `{outer}`");
                }
            }
        }
    }
}
