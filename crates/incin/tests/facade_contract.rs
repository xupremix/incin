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
        .output()
        .expect("consumer fixture cargo invocation must start");
    let stderr = String::from_utf8_lossy(&output.stderr);

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

#[test]
fn stable_facade_consumer_contracts() {
    check_fixture("default-pass", true, &[]);
    check_fixture("backend-authoring-pass", true, &[]);
    check_fixture("experimental-distributed-pass", true, &[]);
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
        "test-utils-absent",
        false,
        &["could not find `test_utils` in `incin`"],
    );
    check_fixture(
        "default-alias-absent",
        false,
        &["no `DefaultBackend` in the root"],
    );
}
