//! `UX-006`: `cargo incin tune` CLI autotune report and cache round-trip test.

use incin::tune_report;

#[test]
fn tune_cli_renders_text_report() {
    let (text, code) = tune_report::run(&[]);
    assert_eq!(code, 0);
    assert!(text.contains("Autotune Cache Report:"));
    assert!(text.contains("Cache Directory:"));
    assert!(text.contains("State:"));
}

#[test]
fn tune_cli_renders_json_report() {
    let (json_str, code) = tune_report::run(&["--json".to_string()]);
    assert_eq!(code, 0);
    let parsed: serde_json::Value =
        serde_json::from_str(&json_str).expect("tune --json output must be valid JSON");
    assert!(parsed.get("cache_dir").is_some());
    assert!(parsed.get("exists").is_some());
    assert_eq!(parsed["offline"], false);
}

#[test]
fn tune_cli_supports_offline_flag() {
    let (json_str, code) = tune_report::run(&["--json".to_string(), "--offline".to_string()]);
    assert_eq!(code, 0);
    let parsed: serde_json::Value = serde_json::from_str(&json_str).unwrap();
    assert_eq!(parsed["offline"], true);
}

#[test]
fn tune_cli_supports_clear_flag() {
    let (text, code) = tune_report::run(&["--clear".to_string()]);
    assert_eq!(code, 0);
    assert!(text.contains("Autotune cache cleared"));
}
