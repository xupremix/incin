//! End-to-end tests for the `cargo incin` binary.
//!
//! The subcommands whose work lives in the library are already covered through
//! that library: `tune` by `tune_cli.rs`, `doctor`'s report by `doctor.rs`.
//! What had no coverage at all is the binary's own layer, which is the part a
//! person actually types at: stripping the `incin` word cargo inserts, deciding
//! which token is the subcommand once global flags are mixed in, routing a
//! usage error to stderr while a report goes to stdout, and the exit codes.
//!
//! These run the real binary rather than calling a function, because argv
//! handling and exit codes are exactly what a function call would skip.
//! `CARGO_BIN_EXE_cargo-incin` is the binary this test run built, so nothing
//! here depends on what is installed on the machine.
//!
//! Nothing in this file reaches the network or invokes cargo, so the suite
//! stays fast and hermetic.
#![cfg(feature = "std")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const BIN: &str = env!("CARGO_BIN_EXE_cargo-incin");

/// A diagnostic carrying typenum's binary encoding of 2 and 6.
///
/// This is the shape of message the tool exists for: `UInt<UInt<UTerm, B1>,
/// B0>` is how the compiler spells `2`, and reading a shape error written that
/// way is the problem `translate` solves.
const TYPENUM_DIAGNOSTIC: &str = "error[E0308]: mismatched types: expected \
     `UInt<UInt<UTerm, B1>, B0>`, found `UInt<UInt<UInt<UTerm, B1>, B1>, B0>`";

/// Runs the binary with `args` and no extra environment.
fn run(args: &[&str]) -> Output {
    Command::new(BIN)
        .args(args)
        .output()
        .expect("the cargo-incin binary this test run built must be executable")
}

/// Runs the binary with one environment variable set.
fn run_with_env(args: &[&str], key: &str, value: &str) -> Output {
    Command::new(BIN)
        .args(args)
        .env(key, value)
        .output()
        .expect("the cargo-incin binary this test run built must be executable")
}

fn stdout(output: &Output) -> String {
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

/// A directory inside this test target's scratch space that does not exist yet.
///
/// `cargo incin new` refuses to write into an existing path, so the point is a
/// name nothing has claimed. The counter keeps parallel tests from colliding.
fn unused_path(label: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join(format!(
        "cargo-incin-{label}-{}-{unique}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&path);
    path
}

// -- argv handling --------------------------------------------------------

/// Invoked as `cargo incin ...`, cargo passes `incin` as the first argument.
/// Stripping it is what makes the same binary work both ways, and nothing
/// else in the suite would notice if it stopped happening.
#[test]
fn the_word_cargo_inserts_is_stripped() {
    let direct = run(&["doctor", "--json"]);
    let through_cargo = run(&["incin", "doctor", "--json"]);

    assert!(direct.status.success(), "doctor --json must succeed");
    assert!(
        through_cargo.status.success(),
        "the same command must succeed when cargo prefixes `incin`"
    );

    let parsed: serde_json::Value =
        serde_json::from_str(&stdout(&through_cargo)).expect("still a JSON report");
    assert_eq!(parsed["schema_version"], incin::doctor::SCHEMA_VERSION);
}

/// With no subcommand there is nothing to do, so the help is the answer
/// rather than an error.
#[test]
fn no_arguments_prints_help_and_succeeds() {
    let output = run(&[]);
    assert!(output.status.success(), "bare invocation must exit 0");
    let text = stdout(&output);
    assert!(text.contains("SUBCOMMANDS:"), "help lists subcommands");
    assert!(text.contains("doctor"), "help names doctor");
}

#[test]
fn both_help_flags_print_the_same_help() {
    let long = run(&["--help"]);
    let short = run(&["-h"]);
    assert!(long.status.success() && short.status.success());
    assert_eq!(stdout(&long), stdout(&short));
    assert!(stdout(&long).contains("USAGE:"));
}

/// `--help` wins wherever it appears, including after a subcommand that would
/// otherwise have run.
#[test]
fn help_anywhere_in_the_arguments_wins() {
    let output = run(&["doctor", "--help"]);
    assert!(output.status.success());
    assert!(
        stdout(&output).contains("SUBCOMMANDS:"),
        "`doctor --help` must print the tool's help, not run the doctor"
    );
}

/// The global flags are position-independent, so the first argument that is
/// not one of them is the subcommand. A flag written first must not be
/// mistaken for the subcommand and delegated to cargo.
#[test]
fn a_global_flag_before_the_subcommand_is_not_the_subcommand() {
    let output = run(&["--raw", "doctor", "--json"]);
    assert!(output.status.success(), "--raw doctor --json must succeed");
    serde_json::from_str::<serde_json::Value>(&stdout(&output))
        .expect("the subcommand still ran and still produced JSON");
}

/// The other half of the same rule: a global flag is consumed by this binary
/// and never forwarded. `doctor` rejects arguments it does not know, so if
/// `--raw` reached it this would exit 2 instead of succeeding.
#[test]
fn a_global_flag_is_consumed_rather_than_forwarded() {
    let output = run(&["doctor", "--raw"]);
    assert!(
        output.status.success(),
        "`--raw` must be swallowed here, not passed through to doctor: {}",
        stderr(&output)
    );
}

// -- doctor routing -------------------------------------------------------

/// A report is what was asked for, so it goes to stdout even when it found
/// problems. Only a usage error goes to stderr, and it names the flags that
/// do exist rather than only the one that does not.
#[test]
fn an_unknown_doctor_flag_is_a_usage_error_on_stderr() {
    let output = run(&["doctor", "--no-such-flag"]);

    assert_eq!(
        output.status.code(),
        Some(incin::doctor::EXIT_USAGE),
        "an unrecognized flag exits with the usage code"
    );
    assert!(
        stdout(&output).is_empty(),
        "a usage error must not print a report to stdout"
    );
    let message = stderr(&output);
    assert!(message.contains("--no-such-flag"), "names the bad flag");
    assert!(message.contains("--json"), "names a flag that does exist");
    assert!(
        message.contains("--check-updates"),
        "names the other flag that does exist"
    );
}

#[test]
fn the_doctor_report_carries_its_schema_and_the_update_key() {
    let output = run(&["doctor", "--json"]);
    let parsed: serde_json::Value =
        serde_json::from_str(&stdout(&output)).expect("doctor --json emits JSON");

    assert_eq!(parsed["schema_version"], incin::doctor::SCHEMA_VERSION);
    assert!(
        parsed.get("update").is_some(),
        "the update key is always present so the key set does not depend on flags"
    );
    assert!(
        parsed["update"].is_null(),
        "and it is null when --check-updates was not asked for"
    );
}

/// The update check must never contact the network behind a person's back.
/// `CARGO_NET_OFFLINE` is the switch that already puts cargo offline, so it
/// covers this too, and the check reports that rather than failing.
#[test]
fn the_update_check_honours_cargo_net_offline() {
    let output = run_with_env(&["doctor", "--check-updates"], "CARGO_NET_OFFLINE", "true");

    assert!(output.status.success() || output.status.code() == Some(1));
    let text = stdout(&output);
    assert!(text.contains("[update]"), "the update section is rendered");
    assert!(
        text.contains("CARGO_NET_OFFLINE") || text.contains("without the `update-check` feature"),
        "offline or not-compiled-in, never a live request: {text}"
    );
}

/// Without the flag there is no update section at all, which is the property
/// that keeps every other command from reaching the network.
#[test]
fn no_update_section_appears_unless_it_is_requested() {
    let output = run(&["doctor"]);
    assert!(
        !stdout(&output).contains("[update]"),
        "the update section must appear only when --check-updates is passed"
    );
}

// -- scaffolding ----------------------------------------------------------

#[test]
fn new_scaffolds_a_project_with_substitutions_applied() {
    let target = unused_path("scaffold");
    let output = run(&["new", "mnist", target.to_str().expect("utf-8 path")]);
    assert!(
        output.status.success(),
        "scaffolding must succeed: {}",
        stderr(&output)
    );

    for relative in ["Cargo.toml", "src/main.rs", "README.md"] {
        assert!(
            target.join(relative).is_file(),
            "the scaffold must write {relative}"
        );
    }

    let manifest = std::fs::read_to_string(target.join("Cargo.toml")).expect("manifest is written");
    assert!(
        !manifest.contains("{{"),
        "every placeholder must be substituted, found one in:\n{manifest}"
    );
    assert!(
        manifest.contains(env!("CARGO_PKG_VERSION")),
        "the scaffold pins the version of the tool that wrote it"
    );

    let main_rs = std::fs::read_to_string(target.join("src/main.rs")).expect("main is written");
    assert!(
        !main_rs.contains("{{"),
        "no placeholder survives in main.rs"
    );

    std::fs::remove_dir_all(&target).expect("cleanup");
}

/// The package name comes from the directory the caller named, not from the
/// template, so two scaffolds in different directories are different crates.
#[test]
fn new_names_the_package_after_the_target_directory() {
    let target = unused_path("named");
    let output = run(&["new", "mnist", target.to_str().expect("utf-8 path")]);
    assert!(output.status.success(), "{}", stderr(&output));

    let expected = target
        .file_name()
        .and_then(|name| name.to_str())
        .expect("the target has a name");
    let manifest = std::fs::read_to_string(target.join("Cargo.toml")).expect("manifest");
    assert!(
        manifest.contains(&format!("name = \"{expected}\"")),
        "expected package name `{expected}` in:\n{manifest}"
    );

    std::fs::remove_dir_all(&target).expect("cleanup");
}

#[test]
fn new_refuses_an_unknown_template() {
    let target = unused_path("unknown-template");
    let output = run(&["new", "not-a-template", target.to_str().expect("utf-8")]);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("unknown template"));
    assert!(stderr(&output).contains("mnist"), "names what is available");
    assert!(
        !target.exists(),
        "a refused scaffold must not leave a directory behind"
    );
}

#[test]
fn new_without_a_template_names_one() {
    let output = run(&["new"]);
    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("mnist"), "suggests a template");
}

/// Overwriting someone's directory is the one mistake this command could make
/// that loses work, so it refuses rather than merging into it.
#[test]
fn new_refuses_to_write_into_an_existing_path() {
    let target = unused_path("existing");
    std::fs::create_dir_all(&target).expect("create the obstacle");
    std::fs::write(target.join("keep-me.txt"), b"untouched").expect("write a file to protect");

    let output = run(&["new", "mnist", target.to_str().expect("utf-8")]);

    assert_eq!(output.status.code(), Some(1));
    assert!(stderr(&output).contains("already exists"));
    assert!(
        target.join("keep-me.txt").is_file(),
        "the existing contents must be untouched"
    );
    assert!(
        !target.join("Cargo.toml").exists(),
        "nothing may be written into an existing directory"
    );

    std::fs::remove_dir_all(&target).expect("cleanup");
}

// -- the remaining subcommands --------------------------------------------

#[test]
fn skills_lists_the_installable_skills() {
    let output = run(&["skills", "list"]);
    assert!(output.status.success(), "{}", stderr(&output));
    let text = stdout(&output);
    for skill in ["incin-expert", "incin-engineering", "incin-repository"] {
        assert!(text.contains(skill), "the listing names {skill}");
    }
}

/// `skills` with no action lists, rather than erroring, because listing is the
/// only harmless thing it could mean.
#[test]
fn skills_defaults_to_listing() {
    let listed = run(&["skills", "list"]);
    let defaulted = run(&["skills"]);
    assert_eq!(stdout(&defaulted), stdout(&listed));
}

/// `translate` is the diagnostic pipeline with the compiler taken out, so it
/// is the one place the typenum table can be exercised without provoking a
/// real compile error first.
///
/// It writes to stderr, not stdout, because it stands in for compiler
/// diagnostics and has to compose with `cargo build 2>&1 | cargo incin
/// translate`. Reading the wrong stream is the easy mistake here, so these
/// assert on stderr deliberately.
#[test]
fn translate_turns_typenum_spam_into_numbers() {
    let output = run(&["translate", TYPENUM_DIAGNOSTIC]);
    assert!(output.status.success(), "{}", stderr(&output));
    let rendered = stderr(&output);

    assert!(
        rendered.contains("expected `2`, found `6`"),
        "the two typenum literals must be rendered as numbers: {rendered}"
    );
    let diagnostic_line = rendered.lines().next().unwrap_or_default();
    assert!(
        !diagnostic_line.contains("UTerm"),
        "no typenum internals survive on the diagnostic line itself: {diagnostic_line}"
    );
    assert!(
        rendered.contains("Typenum Translation Hints"),
        "the original spelling is still offered underneath: {rendered}"
    );
}

/// `--raw` is the escape hatch for when the translation is itself the problem,
/// so it has to hand back exactly what it was given.
#[test]
fn translate_raw_passes_the_diagnostic_through_untouched() {
    let translated = stderr(&run(&["translate", TYPENUM_DIAGNOSTIC]));
    let raw = stderr(&run(&["--raw", "translate", TYPENUM_DIAGNOSTIC]));

    assert_eq!(
        raw.trim_end(),
        TYPENUM_DIAGNOSTIC,
        "raw mode returns the input verbatim"
    );
    assert_ne!(
        raw, translated,
        "and therefore differs from the translation"
    );
}

/// Without `--explain` the shape rule stays out of the way; with it, the rule
/// that was broken is spelled out. The flag is the whole difference.
#[test]
fn explain_appends_the_rule_that_was_broken() {
    const CONTRACTION: &str = "error[E0277]: Cannot contract dimension `3` with `4`";

    let plain = stderr(&run(&["translate", CONTRACTION]));
    let explained = stderr(&run(&["--explain", "translate", CONTRACTION]));

    assert!(
        !plain.contains("MatMul Rule"),
        "no explanation unless asked: {plain}"
    );
    assert!(
        explained.contains("MatMul Rule"),
        "--explain names the rule: {explained}"
    );
    assert!(
        explained.contains('3') && explained.contains('4'),
        "and carries the operands into the explanation: {explained}"
    );
}

/// With no argument the diagnostic comes from stdin, which is how it is
/// actually used: `cargo build 2>&1 | cargo incin translate`.
#[test]
fn translate_reads_a_diagnostic_from_stdin() {
    use std::io::Write as _;
    use std::process::Stdio;

    let mut child = Command::new(BIN)
        .arg("translate")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the binary spawns");

    child
        .stdin
        .take()
        .expect("stdin was piped")
        .write_all(TYPENUM_DIAGNOSTIC.as_bytes())
        .expect("the diagnostic is written");

    let output = child.wait_with_output().expect("the child exits");
    assert!(output.status.success());
    assert!(
        stderr(&output).contains("expected `2`, found `6`"),
        "a piped diagnostic is translated the same as an argument: {}",
        stderr(&output)
    );
}

#[test]
fn inspect_reports_a_missing_file_rather_than_panicking() {
    let missing = unused_path("no-such-model").join("model.safetensors");
    let output = run(&["inspect", missing.to_str().expect("utf-8")]);

    assert!(
        !stderr(&output).contains("panicked"),
        "a missing file is an error, not a panic: {}",
        stderr(&output)
    );
}
