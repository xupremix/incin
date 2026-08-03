//! `UX-005`: `.explain()` and `cargo incin plan` with text and JSON reports.

#![cfg(feature = "train")]

use incin::experimental::training::{Machine, Trainer};
use incin_core::prelude::{DeviceId, DeviceKind, DeviceSet};

struct MockCpuMachine;

impl Machine for MockCpuMachine {
    fn compiled_in(&self, kind: DeviceKind) -> bool {
        kind == DeviceKind::Cpu
    }

    fn has_device(&self, device: DeviceId) -> bool {
        device.kind() == DeviceKind::Cpu
    }
}

#[test]
fn plan_explain_renders_text_report() {
    let plan = Trainer::plan()
        .devices(DeviceSet::cpu())
        .epochs(3)
        .build_on(&MockCpuMachine)
        .expect("building CPU plan on MockCpuMachine must succeed");

    let text = plan.explain();
    assert!(text.contains("Execution Plan:"));
    assert!(text.contains("Devices: 1 device(s) (cpu)"));
    assert!(text.contains("Epochs: 3"));
    assert!(text.contains("Decisions:"));
    assert!(text.contains("devices-requested"));
}

#[test]
fn plan_explain_json_renders_valid_json() {
    let plan = Trainer::plan()
        .devices(DeviceSet::cpu())
        .epochs(5)
        .build_on(&MockCpuMachine)
        .expect("building CPU plan on MockCpuMachine must succeed");

    let json_str = plan.explain_json();
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("explain_json must produce valid JSON");

    assert_eq!(parsed["epochs"], 5);
    assert_eq!(parsed["devices"]["count"], 1);
    assert_eq!(parsed["devices"]["primary"], "cpu");
    assert_eq!(parsed["devices"]["is_multi_device"], false);
    assert!(parsed["decisions"].is_array());
}

#[test]
fn plan_report_runner_executes_with_cli_flags() {
    let (text_out, text_code) = incin::experimental::training::plan_report::run_with_machine(
        &[
            "--devices".to_string(),
            "cpu".to_string(),
            "--epochs".to_string(),
            "10".to_string(),
        ],
        &MockCpuMachine,
    );
    assert_eq!(text_code, 0);
    assert!(text_out.contains("Execution Plan:"));
    assert!(text_out.contains("Epochs: 10"));

    let (json_out, json_code) = incin::experimental::training::plan_report::run_with_machine(
        &[
            "--json".to_string(),
            "--devices".to_string(),
            "cpu".to_string(),
            "--epochs".to_string(),
            "2".to_string(),
        ],
        &MockCpuMachine,
    );
    assert_eq!(json_code, 0);
    let parsed: serde_json::Value =
        serde_json::from_str(&json_out).expect("json report output must be valid JSON");
    assert_eq!(parsed["epochs"], 2);
}
