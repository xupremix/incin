//! Repository maintenance tasks for the Incin workspace.
//!
//! Run with `cargo xtask <task>`. These are developer tools, not part of the
//! published `incin` API, which is why they live in a `publish = false` crate
//! rather than in `cargo-incin` - a binary target shares its crate's
//! dependencies, so putting a TOML parser in `crates/incin` would put it in
//! every downstream user's dependency graph.

mod budgets;
mod docs;
mod hardware;
mod ledger;
mod onnx;

use std::process::ExitCode;

fn main() -> ExitCode {
    let task = std::env::args().nth(1);
    match task.as_deref() {
        Some("budgets") => budgets::check(),
        Some("ledger") => ledger::check(),
        Some("docs") => docs::run(std::env::args().nth(2).as_deref() == Some("--check")),
        Some("feature-msrv") => docs::run_msrv(),
        Some("hardware-tests") => hardware::check(std::env::args().nth(2).as_deref()),
        Some("onnx") => onnx::run(std::env::args().nth(2).as_deref() == Some("--check")),
        Some("feature-matrix") => {
            let arguments: Vec<_> = std::env::args().skip(2).collect();
            let status = std::process::Command::new("tools/feature-matrix.sh")
                .args(arguments)
                .status()
                .expect("failed to execute tools/feature-matrix.sh");
            if status.success() {
                ExitCode::SUCCESS
            } else {
                ExitCode::FAILURE
            }
        }
        Some(other) => {
            eprintln!("unknown task `{other}`");
            usage();
            ExitCode::FAILURE
        }
        None => {
            usage();
            ExitCode::FAILURE
        }
    }
}

fn usage() {
    eprintln!("Repository maintenance tasks for the Incin workspace.");
    eprintln!();
    eprintln!("USAGE:");
    eprintln!("    cargo xtask <TASK>");
    eprintln!();
    eprintln!("TASKS:");
    eprintln!("    ledger          Validate the PROPOSALS.md execution ledger against");
    eprintln!("                    docs/plan/ledger.toml (GOV-003)");
    eprintln!("    budgets         Validate regression budgets and the Cargo feature");
    eprintln!("                    inventory (GOV-005)");
    eprintln!("    docs            Regenerate README.md's feature tables from the Cargo");
    eprintln!("                    manifests; --check fails instead of writing (UX-013)");
    eprintln!("    onnx            Regenerate incin-core's checked-in ONNX protobuf module");
    eprintln!("                    from proto/onnx.proto; --check fails instead of writing");
    eprintln!("    hardware-tests [LOG]  Print how many ignored CUDA tests the hardware runner");
    eprintln!("                    should execute; with a cargo-test log, fail if fewer ran");
    eprintln!(
        "    feature-matrix [stable]  Check feature contract rows (stable is the MSRV union)"
    );
    eprintln!(
        "    feature-msrv    Derive and check every stable package powerset at the active toolchain"
    );
}
