//! `incin-viz` binary entry point: CLI argument parsing, run-path
//! resolution against `incin-telemetry`'s XDG run-dir convention, and the
//! panic-safe terminal lifecycle around the async event loop.
#[macro_use]
extern crate alloc;

use clap::Parser;
use incin_viz::app::{self, App};
use incin_viz::transport_reader::FileTransportReader;
use serde::Deserialize;

#[derive(Deserialize, Default)]
/// Config.
struct Config {
    keymap: Option<String>,
}

#[derive(Parser)]
#[command(
    name = "incin-viz",
    about = "Terminal UI for observing live Incin training runs"
)]
/// Cli.
struct Cli {
    /// Run id to attach to, resolved against incin-telemetry's default
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
        let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
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
        Some(incin_telemetry::run_dir::default_run_dir()?.join(format!("{run_id}.jsonl")))
    } else {
        let dir = incin_telemetry::run_dir::default_run_dir()?;
        let mut latest_run = None;
        let mut latest_time = std::time::SystemTime::UNIX_EPOCH;
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.filter_map(Result::ok) {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("jsonl")
                    && let Ok(meta) = entry.metadata()
                    && let Ok(modified) = meta.modified()
                    && modified > latest_time
                {
                    latest_time = modified;
                    latest_run = Some(path);
                }
            }
        }
        latest_run
    };

    let Some(run_path) = run_path else {
        println!(
            "incin-viz — no runs found in default directory. Pass --run-id <id> or --run-dir <path>."
        );
        std::process::exit(0);
    };

    let reader = FileTransportReader::open(&run_path)?;
    let mut app = App::new(Box::new(reader), run_path.display().to_string());
    app.register_panel(Box::new(incin_viz::panels::loss::LossPanel::new()));
    app.register_panel(Box::new(incin_viz::panels::scalar::ScalarPanel::new(
        "throughput",
        "Throughput",
        "throughput",
    )));
    app.register_panel(Box::new(incin_viz::panels::scalar::ScalarPanel::new(
        "lr",
        "Learning Rate",
        "lr",
    )));
    app.register_panel(Box::new(incin_viz::panels::norms::NormsPanel::new(
        incin_viz::panels::norms::NormType::Gradient,
        "Gradient Norms",
        "gradient_norms",
        Some(1e5),
    )));
    app.register_panel(Box::new(incin_viz::panels::norms::NormsPanel::new(
        incin_viz::panels::norms::NormType::Weight,
        "Weight Norms",
        "weight_norms",
        None,
    )));
    app.register_panel(Box::new(incin_viz::panels::system::MemoryPanel::new(
        "Memory (RSS MB)",
        "memory1",
        Some(8000.0),
    )));
    app.register_panel(Box::new(incin_viz::panels::system::MemoryPanel::new(
        "Memory (RSS MB)",
        "memory2",
        Some(8000.0),
    )));
    app.register_panel(Box::new(incin_viz::panels::system::MemoryPanel::new(
        "Memory (RSS MB)",
        "memory3",
        Some(8000.0),
    )));
    app.register_panel(Box::new(
        incin_viz::panels::graph::GraphModuleListPanel::new(),
    ));

    // Load config if exists
    let mut config = Config::default();
    if let Ok(content) = std::fs::read_to_string("incin-viz.toml")
        && let Ok(parsed) = toml::from_str::<Config>(&content)
    {
        config = parsed;
    }

    let keymap: Box<dyn incin_viz_plugin_api::keymap::KeymapProvider> =
        if config.keymap.as_deref() == Some("vim") {
            Box::new(app::VimKeymap)
        } else {
            Box::new(app::DefaultKeymap)
        };

    // Panic hook must be installed before ratatui::init() enters raw mode.
    install_panic_hook();
    let terminal = ratatui::init();
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::EnableMouseCapture);

    let result = app::run(app, terminal, keymap).await;

    // Normal-exit terminal restore; the panic hook above covers the
    // abnormal-exit path.
    let _ = crossterm::execute!(std::io::stdout(), crossterm::event::DisableMouseCapture);
    ratatui::restore();
    result
}
