//! Repository maintenance tasks for the Incin workspace.
//!
//! Run with `cargo xtask <task>`. These are developer tools, not part of the
//! published `incin` API, which is why they live in a `publish = false` crate
//! rather than in `cargo-incin` — a binary target shares its crate's
//! dependencies, so putting a TOML parser in `crates/incin` would put it in
//! every downstream user's dependency graph.

mod ledger;

use std::process::ExitCode;

fn main() -> ExitCode {
    let task = std::env::args().nth(1);
    match task.as_deref() {
        Some("ledger") => ledger::check(),
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
    eprintln!("    ledger    Validate the PROPOSALS.md execution ledger against");
    eprintln!("              docs/plan/ledger.toml (GOV-003)");
}
