//! `kindle-viz` binary entry point: CLI argument parsing and run-path
//! resolution against `kindle-telemetry`'s XDG run-dir convention.

use clap::Parser;

#[derive(Parser)]
#[command(name = "kindle-viz", about = "Terminal UI for observing live Kindle training runs")]
struct Cli {
    /// Run id to attach to, resolved against kindle-telemetry's default
    /// (XDG) run directory.
    #[arg(long)]
    run_id: Option<String>,

    /// Full path to a run's JSONL transport file, bypassing run-id
    /// resolution -- an escape hatch for advanced/manual use.
    #[arg(long)]
    run_dir: Option<std::path::PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    let run_path: Option<std::path::PathBuf> = if let Some(run_dir) = cli.run_dir {
        Some(run_dir)
    } else if let Some(run_id) = cli.run_id {
        Some(kindle_telemetry::run_dir::default_run_dir()?.join(format!("{run_id}.jsonl")))
    } else {
        None
    };

    let Some(run_path) = run_path else {
        println!("kindle-viz — no run specified. Pass --run-id <id> or --run-dir <path>.");
        std::process::exit(0);
    };

    println!("kindle-viz: resolved run path: {}", run_path.display());
    // TODO(08-04): tokio::select! event loop + ratatui terminal setup

    Ok(())
}
