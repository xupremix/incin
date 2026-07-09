# Coding Conventions

**Analysis Date:** 2026-07-09

## Naming Patterns

**Files:**
- One module/concept per file, `snake_case.rs` matching the primary type (`linear.rs` → `Linear`, `batch_norm.rs` → `BatchNorm2d`, `conv2d.rs` → `Conv2d`).
- Trait-defining files often share a name with their central trait (`module.rs` → `Module`, `Parameters`, `StateDict`).
- Test files under `tests/` are named after the feature under test (`reshape.rs`, `concat_stack.rs`, `builder_permutations.rs`), not `test_*.rs`.
- Compile-fail fixtures live in `crates/kindle-core/tests/compile_fail/*.rs`, one file per negative scenario, named for the exact failure being asserted (`device_mismatch.rs`, `macro_idx_invalid.rs`, `forward_conv2d_static_mismatch.rs`).

**Functions:**
- `snake_case` throughout. Constructors are `new(...)`, fallible constructors return `Result<Self>` (see `crates/kindle-core/src/nn/linear.rs`).
- Fallible operations are prefixed `try_` when a non-fallible/static-shape counterpart also exists, e.g. `reshape` (static, panics/compile-checked) vs `try_reshape` (dynamic, returns `Result`) in `crates/kindle-core/tests/reshape.rs`.
- Conversion functions follow Rust stdlib idiom: `into_dyn`, `to_vec`, `as_slice`.

**Variables:**
- Short, local, `snake_case`. Shape/dimension variables commonly abbreviated (`in_f`, `out_f`) matching their type-level generic counterparts (`InF`, `OutF`) — see `crates/kindle-core/src/nn/linear.rs`.

**Types:**
- `PascalCase` for structs, traits, enums: `Tensor`, `Module`, `Parameters`, `StateDict`, `LinearShape`.
- Type-level shape markers use short PascalCase generics tied to `typenum` (`InF`, `OutF`, `U2`, `U3`) and a special `Dyn` marker type for dynamic (runtime) shapes, contrasted with static shapes built via the `s![...]` macro from `kindle-macros`.
- Backend-parameterized types follow `Thing<Shape, Backend>` ordering consistently, e.g. `Tensor<s![U2, U3], DummyBackend<f32, Cpu>>`.

## Code Style

**Formatting:**
- `cargo fmt --all -- --check` enforced in CI (`.github/workflows/ci.yml`); no custom `rustfmt.toml` present, so default rustfmt style applies uniformly across the workspace.

**Linting:**
- `cargo clippy --workspace --all-targets -- -D warnings` enforced in CI — all clippy warnings are build failures.
- `[workspace.lints.clippy]` table exists in the root `Cargo.toml` (currently empty — no crate-specific clippy allow/deny overrides configured; rely on default clippy lints, but note in-progress work has generated many temporary `fix_*.py`/`refactor_*.py` scripts and `check*.log` files at the repo root from mechanical whitespace/lint fixups — these are not part of the conventions and should not be treated as examples).

## Import Organization

**Order:**
1. `use crate::...` / `use super::...` internal imports first in most files (see `crates/kindle-core/src/nn/module.rs`, `crates/kindle-core/src/nn/linear.rs`).
2. External crate imports (`std`, `alloc`, third-party) interleaved but generally grouped near top of file — no strict rustfmt import grouping config is enforced beyond defaults.
3. Test files import via the crate's `prelude` module: `use kindle::prelude::*;` (integration tests) or `use kindle_core::prelude::*;` (core tests), plus explicit imports for anything not re-exported (e.g. `kindle_macros::s`, `typenum::{U2, U3, U6}`).

**Path Aliases:**
- No `#[path]` aliasing observed. Crates expose a curated `pub mod prelude` (see `crates/kindle-core/src/lib.rs`) that re-exports the commonly used public API surface (`err::*`, `nn::{...}`, tensor types, etc.) so downstream code and tests import a single glob (`use kindle_core::prelude::*;`) instead of deep paths.
- `no_std` support: `kindle-core` is `#![cfg_attr(not(feature = "std"), no_std)]` with `extern crate alloc;` — code must use `alloc::string::String`, `alloc::vec::Vec`, `alloc::format!` instead of `std` equivalents when in shared/core code paths (see `crates/kindle-core/src/err.rs`). The `kindle-macros` proc-macro conditionally emits `alloc::format!`/`std::format!` and `crate`/`kindle` paths depending on an `internal` attribute flag, to support this dual std/no_std, internal/external code generation (`crates/kindle-macros/src/module.rs`).

## Error Handling

**Patterns:**
- Centralized error enum `Error` in `crates/kindle-core/src/err.rs`, built with `thiserror::Error` derive macro; custom `Debug` impl delegates to `Display` (`write!(f, "{self}")`) so `?`-propagated errors print nicely.
- Public `Result<T>` type alias (`pub type Result<T> = core::result::Result<T, Error>;`) used everywhere instead of the raw `core::result::Result`.
- Error variants are structured/data-carrying, not plain strings: `ShapeMismatch { op, expected, got, msg }`, `OutOfMemory { device }`, `UnsupportedBackendOperation { op, backend }`, `DeviceInitializationError { expected, got }`, plus an escape hatch `Msg(String)` and `BackendFailure(#[from] anyhow::Error)` for wrapping backend-library errors via `#[from]`.
- Library code favors `Result`-returning fallible APIs with `?` propagation over `panic!`/`unwrap()` in the public API surface. `panic!`/`unreachable!`/`todo!`/`unimplemented!` usage is rare (15 occurrences total across `crates/*/src`) and should stay that way — reserve them for truly unreachable states or explicit "not yet implemented" markers, not for user-triggerable error conditions.
- `.unwrap()`/`.expect()` are common in test code (integration tests use `Result<()>` + `?` where possible, e.g. `crates/kindle/tests/tensor_ops.rs`) but should be minimized in `src/` — 149 occurrences of `.unwrap()` currently exist under `crates/kindle-core/src`, concentrated in shape/dim conversions where invariants are asserted to hold by construction; new code should prefer propagating `Result` via `?` and only `unwrap()` when the invariant is locally provable.

## Comments

**When to Comment:**
- Doc comments (`///`) are used extensively on public traits, structs, and non-trivial functions, explaining purpose and usage — see `crates/kindle-core/src/nn/module.rs` (`StateDict`, `Parameters`, `ToDevice`) and `crates/kindle-core/src/nn/linear.rs` (`LinearShape`).
- Module-level doc comments (`//!`) at the top of `lib.rs` describe overall crate architecture with a bulleted breakdown of submodules (`crates/kindle-core/src/lib.rs`).
- Inline `//` comments are used sparingly inside test bodies to document test-case rationale/permutations being covered, e.g. `// permutations: positive, negative, zero, very small numbers...` in `crates/kindle/tests/tensor_ops.rs`.

**JSDoc/TSDoc equivalent (rustdoc):**
- Doc comments include runnable/`ignore`-marked code examples using triple backtick ` ```rust,ignore ` blocks for API usage illustration (see `crates/kindle-core/src/lib.rs`, `crates/kindle-core/src/nn/linear.rs`). Use `ignore` when the snippet is illustrative pseudocode rather than a real compiling example.

## Function Design

**Size:** Functions are generally short and single-purpose; larger orchestration logic (e.g. proc-macro codegen in `crates/kindle-macros/src/module.rs`) is broken into local closures/helper `let` bindings that compute one piece of the generated code at a time.

**Parameters:** Generic parameters are heavily used to encode shape/backend/device at the type level (`Tensor<S: Shape, B: Backend>`, `fn to_device<NewD: Device>`); trait bounds are expressed with `where` clauses when they get long (see `StateDict::save_to`/`load_from` in `crates/kindle-core/src/nn/module.rs`).

**Return Values:** Fallible functions return `Result<T>` (the crate-local alias) rather than raw `Result<T, E>`; conversions to a different error type happen explicitly via `.map_err(|e| Error::...)` at call boundaries (e.g. `crates/kindle-core/src/nn/module.rs::load_from`).

## Module Design

**Exports:** Each crate exposes a single curated `pub mod prelude` module aggregating the public API (`crates/kindle-core/src/lib.rs`); internal submodules (`err`, `nn`, `tensor`, `optim`, `serialize`, `shapes`, `graph`, `onnx_exporter`, `onnx_pb`) are also `pub` but consumers are expected to import via `prelude::*`.

**Barrel Files:** The `prelude` module functions as the workspace's barrel-file equivalent; there is no additional re-export layer beyond it.

**Code generation:** Boilerplate trait implementations (`StateDict`, `Parameters`, device-transfer glue) are generated via the `#[kindle::module]` / `#[kindle_macros::module]` proc-macro attribute rather than hand-written per struct — new NN layers should use this macro instead of manually implementing `StateDict`/`Parameters` (see `crates/kindle-macros/src/module.rs` and its usage across `crates/kindle-core/src/nn/*.rs`).

---

*Convention analysis: 2026-07-09*
