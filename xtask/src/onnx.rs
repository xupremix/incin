//! Regenerate `incin-core`'s checked-in ONNX protobuf module.
//!
//! `crates/incin-core/src/generated/onnx.rs` used to be produced by a build
//! script. That made `protoc` a mandatory system dependency of every crate that
//! depends on `incin-core` - including the overwhelming majority that never
//! call the ONNX exporter - and a first `cargo build` on a machine without it
//! failed with a protobuf error rather than compiling.
//!
//! The generated file is checked in instead, which is only safe if something
//! proves it still equals what the `.proto` compiles to. That is this task:
//! `cargo xtask onnx` rewrites the file, `cargo xtask onnx --check` fails when
//! rewriting it would change anything. The check needs `protoc`, so it runs in
//! CI (which installs it) and not in an ordinary build.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

const PROTO_DIR: &str = "crates/incin-core/proto";
const PROTO_FILE: &str = "crates/incin-core/proto/onnx.proto";

/// Every crate that includes the generated module.
///
/// There are two copies rather than a shared crate on purpose: `incin-macros`
/// is a proc-macro crate that `incin-core` depends on, so it cannot depend on
/// `incin-core`, and adding a third published crate to carry one generated file
/// would put a permanent name on crates.io to solve a problem this task already
/// solves. The copies are written together and compared together, so they
/// cannot drift.
const GENERATED: &[&str] = &[
    "crates/incin-core/src/generated/onnx.rs",
    "crates/incin-macros/src/generated/onnx.rs",
];

pub fn run(check: bool) -> ExitCode {
    let root = match workspace_root() {
        Some(root) => root,
        None => {
            eprintln!("xtask onnx: could not locate the workspace root");
            return ExitCode::FAILURE;
        }
    };

    let generated = match generate(&root) {
        Ok(generated) => generated,
        Err(error) => {
            eprintln!("xtask onnx: {error}");
            eprintln!(
                "xtask onnx: `protoc` is required to regenerate the module. It is not required \
                 to build incin-core, which uses the checked-in output."
            );
            return ExitCode::FAILURE;
        }
    };

    let mut stale = Vec::new();
    for relative in GENERATED {
        let destination = root.join(relative);
        if fs::read_to_string(&destination).unwrap_or_default() != generated {
            stale.push((relative, destination));
        }
    }

    if stale.is_empty() {
        if !check {
            println!("every generated ONNX module is already current");
        }
        return ExitCode::SUCCESS;
    }

    if check {
        for (relative, _) in &stale {
            eprintln!("{relative} does not match what {PROTO_FILE} compiles to");
        }
        eprintln!("Run `cargo xtask onnx` and commit the result.");
        return ExitCode::FAILURE;
    }

    for (relative, destination) in &stale {
        if let Some(parent) = destination.parent()
            && let Err(error) = fs::create_dir_all(parent)
        {
            eprintln!("xtask onnx: {}: {error}", parent.display());
            return ExitCode::FAILURE;
        }
        if let Err(error) = fs::write(destination, &generated) {
            eprintln!("xtask onnx: {}: {error}", destination.display());
            return ExitCode::FAILURE;
        }
        println!("wrote {relative}");
    }
    ExitCode::SUCCESS
}

/// Compile the `.proto` into a string, using a temporary directory so that a
/// failed run cannot leave a half-written file in the source tree.
fn generate(root: &Path) -> Result<String, String> {
    let out = root.join("target").join("xtask-onnx");
    fs::create_dir_all(&out).map_err(|error| format!("{}: {error}", out.display()))?;

    let mut config = prost_build::Config::new();
    // Matches the retired build script: deterministic map ordering, and a map
    // type that does not require `std`.
    config.btree_map(["."]);
    config.out_dir(&out);
    config
        .compile_protos(&[root.join(PROTO_FILE)], &[root.join(PROTO_DIR)])
        .map_err(|error| format!("compiling {PROTO_FILE}: {error}"))?;

    let produced = out.join("onnx.rs");
    fs::read_to_string(&produced).map_err(|error| format!("{}: {error}", produced.display()))
}

/// The directory holding the workspace `Cargo.toml`.
///
/// `CARGO_MANIFEST_DIR` points at `xtask/`, whose parent is the workspace root
/// whether the task was invoked from the root or from a member directory.
fn workspace_root() -> Option<PathBuf> {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let root = manifest.parent()?.to_path_buf();
    root.join("Cargo.toml").is_file().then_some(root)
}
