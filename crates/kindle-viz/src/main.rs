//! `kindle-viz` binary entry point: CLI argument parsing, run-path
//! resolution against `kindle-telemetry`'s XDG run-dir convention, and the
//! panic-safe terminal lifecycle around the async event loop.

use clap::Parser;
use kindle_viz::app::{self, App};
use kindle_viz::transport_reader::FileTransportReader;

#[derive(Parser)]
#[command(
    name = "kindle-viz",
    about = "Terminal UI for observing live Kindle training runs"
)]
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

/// Installs a panic hook (before raw mode begins) that best-effort restores
/// the terminal -- disable raw mode, leave the alternate screen -- so a
/// genuine host-level panic (not a caught panel panic; see `dispatch.rs`)
/// never leaves the user's terminal in raw/alternate-screen mode
/// (RESEARCH.md Pitfall 5 / T-08-06).
fn install_panic_hook() {
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |panic_info| {
        previous_hook(panic_info);
        let _ = crossterm::terminal::disable_raw_mode();
        let _ = crossterm::execute!(std::io::stdout(), crossterm::terminal::LeaveAlternateScreen);
    }));
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
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

    let reader = FileTransportReader::open(&run_path)?;
    let mut app = App::new(Box::new(reader), run_path.display().to_string());
    // Loss first so it lands in the left 50% column per UI-SPEC.md's
    // layout; panic-test second/right.
    app.register_panel(Box::new(kindle_viz::panels::loss::LossPanel::new()));
    app.register_panel(Box::new(kindle_viz::panels::panic_test::PanicTestPanel));

    // Panic hook must be installed before ratatui::init() enters raw mode.
    install_panic_hook();
    let terminal = ratatui::init();
    let result = app::run(app, terminal).await;
    // Normal-exit terminal restore; the panic hook above covers the
    // abnormal-exit path.
    ratatui::restore();
    result
}
