//! Integration coverage for `check_fixture` on the documented public surface.
use std::path::Path;
use std::process::Command;

fn check_fixture(name: &str, should_pass: bool, expected: &[&str]) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = root
        .join("tests/consumer-fixtures")
        .join(name)
        .join("Cargo.toml");
    let target = root.join("../../target/facade-contract").join(name);
    let output = Command::new(env!("CARGO"))
        .args(["check", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TARGET_DIR", target)
        .env_remove("CARGO_PRIMARY_PACKAGE")
        .output()
        .expect("consumer fixture cargo invocation must start");
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.success() != should_pass {
        eprintln!(
            "=== FIXTURE {name} FAILED (success={}) ===",
            output.status.success()
        );
        eprintln!("{stderr}");
    }

    assert_eq!(
        output.status.success(),
        should_pass,
        "fixture {name} had unexpected status\n{stderr}"
    );
    for reason in expected {
        assert!(
            stderr.contains(reason),
            "fixture {name} did not fail for {reason:?}\n{stderr}"
        );
    }
}

fn test_fixture(name: &str) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest = root
        .join("tests/consumer-fixtures")
        .join(name)
        .join("Cargo.toml");
    let target = root.join("../../target/facade-contract").join(name);
    let output = Command::new(env!("CARGO"))
        .args(["test", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TARGET_DIR", target)
        .env_remove("CARGO_PRIMARY_PACKAGE")
        .output()
        .expect("consumer fixture cargo test invocation must start");
    assert!(
        output.status.success(),
        "fixture {name} test failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn stable_facade_consumer_contracts() {
    check_fixture("default-pass", true, &[]);
    check_fixture("backend-authoring-pass", true, &[]);
    check_fixture("custom-op-cpu-pass", true, &[]);
    check_fixture("experimental-distributed-pass", true, &[]);
    check_fixture("experimental-compiled-pass", true, &[]);
    check_fixture("test-utils-pass", true, &[]);
    check_fixture("no-default-pass", true, &[]);
    check_fixture("internal-absent", false, &["no `Graph` in the root"]);
    check_fixture(
        "backend-authoring-absent",
        false,
        &["could not find `backend_authoring` in `incin`"],
    );
    check_fixture(
        "experimental-absent",
        false,
        &["could not find `compiled` in `experimental`"],
    );
    check_fixture(
        "experimental-compiled-root-absent",
        false,
        &["no `CompiledPlan` in the root"],
    );
    check_fixture(
        "experimental-compiled-prelude-absent",
        false,
        &["no `CompiledPlan` in `prelude`"],
    );
    check_fixture(
        "test-utils-absent",
        false,
        &["could not find `test_utils` in `incin`"],
    );
    // The stand-in backend is gone from the feature that used to carry it,
    // not merely from the default surface.
    check_fixture(
        "dummy-backend-absent",
        false,
        &["no `DummyBackend` in `test_utils`"],
    );
    check_fixture(
        "default-alias-absent",
        false,
        &["no `DefaultBackend` in the root"],
    );
}

#[test]
fn experimental_compiled_facade_executes_a_non_empty_cpu_plan() {
    test_fixture("experimental-compiled-pass");
}

#[test]
fn expert_storage_encoding_has_a_named_path() {
    let _ = incin::types::dtype::StorageEncoding::scalar(4, 4);
}
