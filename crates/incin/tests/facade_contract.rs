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
    let mut cmd = Command::new(env!("CARGO"));
    cmd.current_dir("/tmp")
        .args(["check", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TARGET_DIR", target)
        .env_remove("CARGO_PRIMARY_PACKAGE");
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        cmd.env("CARGO_HOME", cargo_home);
    }
    if let Ok(rustup_home) = std::env::var("RUSTUP_HOME") {
        cmd.env("RUSTUP_HOME", rustup_home);
    }
    if let Ok(home) = std::env::var("HOME") {
        cmd.env("HOME", home);
    }
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    let output = cmd
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
    let mut cmd = Command::new(env!("CARGO"));
    cmd.current_dir("/tmp")
        .args(["test", "--quiet", "--manifest-path"])
        .arg(&manifest)
        .env("CARGO_TARGET_DIR", target)
        .env_remove("CARGO_PRIMARY_PACKAGE");
    if let Ok(cargo_home) = std::env::var("CARGO_HOME") {
        cmd.env("CARGO_HOME", cargo_home);
    }
    if let Ok(rustup_home) = std::env::var("RUSTUP_HOME") {
        cmd.env("RUSTUP_HOME", rustup_home);
    }
    if let Ok(home) = std::env::var("HOME") {
        cmd.env("HOME", home);
    }
    if let Ok(path) = std::env::var("PATH") {
        cmd.env("PATH", path);
    }
    let output = cmd
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

/// The published crate, exercised from outside the workspace.
///
/// Ignored because it is the one fixture that resolves `incin` from crates.io
/// rather than from this checkout, so it needs the network and it reports on a
/// release rather than on the working tree. A red result here does not mean the
/// commit under test is broken, which is exactly why it must not sit in the
/// same gate as the fixtures that do mean that. CI runs it on the nightly
/// schedule and on demand; run it by hand with
/// `cargo test -p incin --test facade_contract -- --ignored`.
#[test]
#[ignore = "resolves incin from crates.io; reports on the release, not this checkout"]
fn the_published_release_still_trains() {
    test_fixture("released-consumer");
}

/// The README's install line and the released fixture ask for the same version.
///
/// The fixture exists to check the artifact a reader installs, and the reader
/// installs whatever the front page told them to write. If the two drift, the
/// fixture is still green and still meaningless, because it is testing a
/// release nobody was pointed at. This costs no network, so it runs in the
/// ordinary gate rather than beside the fixture it guards.
#[test]
fn the_released_consumer_tracks_the_readme_quick_start() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let readme = std::fs::read_to_string(root.join("../../README.md"))
        .expect("the repository README must be readable");
    let manifest =
        std::fs::read_to_string(root.join("tests/consumer-fixtures/released-consumer/Cargo.toml"))
            .expect("the released-consumer manifest must be readable");

    let requirement = |text: &str, source: &str| -> String {
        text.lines()
            .find_map(|line| {
                line.trim()
                    .strip_prefix("incin = ")?
                    .trim()
                    .strip_prefix('"')?
                    .strip_suffix('"')
                    .map(str::to_owned)
            })
            .unwrap_or_else(|| panic!("{source} no longer contains an `incin = \"...\"` line"))
    };

    assert_eq!(
        requirement(&readme, "the README quick start"),
        requirement(&manifest, "the released-consumer manifest"),
        "the README tells readers to install one version and the released \
         consumer fixture checks another"
    );
}
