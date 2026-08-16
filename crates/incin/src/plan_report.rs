//! `cargo incin plan` — plan report generator (`UX-005`).

use crate::train::{HostMachine, Machine, Trainer};
use incin_core::tensor::device::{DevicePreference, DeviceSet};

/// Process exit status for a successful plan report.
pub const EXIT_OK: i32 = 0;
/// Process exit status for invalid plan-report arguments.
pub const EXIT_USAGE: i32 = 2;

/// Runs `cargo incin plan` given CLI arguments.
pub fn run(args: &[String]) -> (String, i32) {
    run_with_machine(args, &HostMachine)
}

/// Runs `cargo incin plan` against a given machine implementation.
pub fn run_with_machine<M: Machine + ?Sized>(args: &[String], machine: &M) -> (String, i32) {
    let mut json_mode = false;
    let mut devices_opt = None;
    let mut epochs_opt = None;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--json" => json_mode = true,
            "--devices" => {
                i += 1;
                if i < args.len() {
                    devices_opt = Some(args[i].as_str());
                } else {
                    return (
                        "Error: missing argument for --devices\n".to_string(),
                        EXIT_USAGE,
                    );
                }
            }
            "--epochs" => {
                i += 1;
                if i < args.len() {
                    if let Ok(n) = args[i].parse::<usize>() {
                        epochs_opt = Some(n);
                    } else {
                        return ("Error: invalid epoch count\n".to_string(), EXIT_USAGE);
                    }
                } else {
                    return (
                        "Error: missing argument for --epochs\n".to_string(),
                        EXIT_USAGE,
                    );
                }
            }
            "--help" | "-h" => {
                return (
                    "cargo incin plan — generate execution plan report\n\nUSAGE:\n    cargo incin plan [--json] [--devices <cpu|cuda|wgpu>] [--epochs <N>]\n".to_string(),
                    EXIT_OK,
                );
            }
            _ => {}
        }
        i += 1;
    }

    let mut builder = Trainer::plan();
    if let Some(epochs) = epochs_opt {
        builder = builder.epochs(epochs);
    }
    if let Some(dev_str) = devices_opt {
        match dev_str {
            "cpu" => builder = builder.device_preference(DevicePreference::Cpu),
            "cuda" => {
                if let Ok(ds) = DeviceSet::cuda(0..1) {
                    builder = builder.devices(ds);
                } else {
                    builder = builder.device_preference(DevicePreference::Fastest);
                }
            }
            "wgpu" => {
                if let Ok(ds) = DeviceSet::wgpu(0..1) {
                    builder = builder.devices(ds);
                } else {
                    builder = builder.device_preference(DevicePreference::Fastest);
                }
            }
            _ => builder = builder.device_preference(DevicePreference::Fastest),
        }
    }

    match builder.build_on(machine) {
        Ok(plan) => {
            let rendered = if json_mode {
                plan.explain_json()
            } else {
                plan.explain()
            };
            (rendered, EXIT_OK)
        }
        Err(err) => (format!("Error building plan: {err}\n"), EXIT_USAGE),
    }
}
