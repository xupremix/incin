<!-- GSD:project-start source:PROJECT.md -->

## Project

**Kindle**

Kindle is a Rust deep-learning/tensor library that checks as many invariants
as possible at compile time — shapes, dtypes, devices, and grad-tracking are
all encoded in the `Tensor<S, B, K, D, G>` type, so mismatches like a bad
`matmul` shape or a device mismatch are caught by `rustc`, not at runtime.
Ergonomic macros (`s!`, `idx!`) let users write shapes and slices without
spelling out the full type machinery, so the experience feels close to
PyTorch/NumPy despite the extra compile-time guarantees. Computation is
delegated to a pluggable `Backend` trait — the same `Tensor` code runs on
Candle, ndarray, or (in progress) a native backend, per the user's needs.

**Core Value:** Catch shape/dtype/device mistakes at compile time instead of at runtime —
if it compiles, the tensor math is structurally sound.

### Constraints

- **Tech stack**: Rust, edition 2024. Must integrate with the existing
  `Backend`/`CreationOps`/`NumericOps`/`TensorOps`/`FloatOps`/
  `ReductionOps`/`ModuleOps`/`LossOps` trait surface in
  `crates/kindle-core/src/tensor/backend.rs` — no changes to that trait
  shape are in scope for this milestone, only a new implementor of it

- **Compatibility**: Must not break the existing Candle/ndarray backends or
  their test suites

- **Numerical correctness**: Backward-pass gradients must match Candle's
  within a relative-error tolerance (bit-exact not required — summation
  order will legitimately differ)
<!-- GSD:project-end -->

<!-- GSD:stack-start source:codebase/STACK.md -->

## Technology Stack

## Languages

- Rust (edition 2024) - Entire workspace, all crates under `crates/`
- Python - One-off refactor/fix scripts at repo root (`fix_*.py`, `refactor_*.py`, `rewrite_backend.py`), not part of the build; used as developer tooling to mechanically rewrite Rust source during large refactors
- Protocol Buffers - `crates/kindle-core/proto/onnx.proto` defines the ONNX graph schema compiled via `prost-build`

## Runtime

- Rust toolchain: `stable` (channel pinned in `rust-toolchain.toml`), with `rust-src` component; a commented-out `nightly` channel line exists for future use
- Installed compiler observed: `rustc 1.92.0`
- No `no_std`/embedded target evidenced; `std` is a default cargo feature across crates (`kindle`, `kindle-core`, `kindle-macros`)
- Cargo (workspace with resolver `"2"`), root manifest `Cargo.toml`
- Lockfile: present (`Cargo.lock`, workspace-wide)

## Frameworks

- Not a web/app framework — this is a deep learning / tensor library. Core abstractions: `Backend` trait, statically-typed `Tensor<S, B, T, D, G>` (`crates/kindle-core/src/tensor/`)
- `candle-core` / `candle-nn` `0.9.1` (HuggingFace Candle) - default/primary backend, `crates/kindle-backends/`
- `ndarray` `0.17.2` - optional CPU array backend
- `burn` `0.21.0` (with `autodiff` feature) + `burn-ndarray` `0.21.0` - optional alternative backend
- Feature flags in `crates/kindle/Cargo.toml`: `candle` (default), `ndarray`, `burn`, `cuda`, `metal`
- `crates/kindle-macros` (proc-macro crate) - powers `#[kindle::module]`, `#[kindle::forward]`, `s!`, `idx!`, and `import_model!` macros
- Dependencies: `syn` `2.0.118` (full), `quote` `1.0.46`, `proc-macro2` `1.0.106`
- Built-in `cargo test` / `#[test]` across crates
- `trybuild` `1.0.117` (dev-dependency of `kindle-core`) - compile-fail/UI testing for proc macros and static shape-checking guarantees
- `prost-build` `0.14.4` (build-dependency of `kindle-core`) - compiles `onnx.proto` into Rust structs at build time; requires system `protoc` (Protocol Buffers Compiler) to be installed
- `cargo fmt`, `cargo clippy` (workspace lints defined in root `Cargo.toml` under `[workspace.lints.clippy]`, currently empty/default)

## Key Dependencies

- `candle-core`/`candle-nn` `0.9.1` - default tensor compute backend
- `safetensors` `0.4.1` / `0.4.5` - model weight (de)serialization format, used in `kindle-core` and `kindle-macros`
- `prost` `0.14.4` (kindle-core) / `0.6` (kindle-macros) - Protobuf runtime for ONNX graph representation (note version mismatch between crates)
- `onnx-pb` `0.1.4` - ONNX protobuf message definitions used by `kindle-macros`
- `typenum` `1.20.1` - compile-time type-level numerics, underpins the static shape-verification system
- `serde` `1.0.228`/`1.0` (derive) + `bincode` `1.3.3` + `serde_json` `1.0.150` - serialization (model metadata caching, `.kindle_meta` JSON cache mentioned in README)
- `thiserror` `2.0.18` - error type definitions
- `half` `2.7.1` - half-precision (f16/bf16) float support
- `anyhow` `1.0.103` - error handling/propagation across `kindle`, `kindle-backends`, `kindle-core`, `kindle-data`
- `bytes` `1.12.0` - byte buffer handling in `kindle-core`
- `hf-hub` `0.5.0` (features: `tokio`, `rustls-tls`, `ureq`) - HuggingFace Hub client, used by `kindle-data::hub`
- `ureq` `3.3.0` - blocking HTTP client for direct dataset/file downloads (`kindle-data::downloader`)
- `rayon` `1.12.0` - data-parallelism for `DataLoaderExt`/`into_par_loader()` in `kindle-data`
- `rand` `0.8` - randomness (data shuffling, weight init)
- `flate2` `1.1.9` - gzip decompression for downloaded dataset archives

## Configuration

- `KINDLE_HUB_CACHE_DIR` - overrides HuggingFace Hub cache directory (default `~/.cache/huggingface/hub`)
- `KINDLE_HUB_TOKEN` - HuggingFace auth token for private/gated repos, read in `crates/kindle-data/src/hub.rs`
- `KINDLE_NO_META` - set to `1`/`true` to force `import_model!` macro to bypass its `.kindle_meta` JSON cache and fully re-parse `.safetensors`/`.onnx` graphs during `cargo build`
- No `.env` files present in the repo; configuration is read directly via `std::env::var` at macro-expansion/runtime
- Root `Cargo.toml` - workspace member list and shared `[workspace.dependencies]`/`[workspace.package]`
- `crates/kindle-core/build.rs` - invokes `prost-build` to compile `proto/onnx.proto`; requires system `protoc`
- `rust-toolchain.toml` - pins toolchain to `stable` with `rust-src`
- `.github/workflows/ci.yml` - CI pipeline (see INTEGRATIONS.md)

## Platform Requirements

- Rust stable toolchain (`rustup` recommended) with `rust-src` component
- System-installed Protocol Buffers Compiler (`protoc`) — required to build `kindle-core` due to ONNX proto compilation
- Optional: CUDA toolkit (for `cuda` feature) or Metal (macOS, for `metal` feature) if GPU acceleration is desired
- No deployment target detected — this is a library/framework crate (published to crates.io style workspace, version `0.1.0`), not a deployed service
- Consumers embed `kindle`/`kindle-core`/`kindle-data` as Rust dependencies in their own binaries

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

## Naming Patterns

- One module/concept per file, `snake_case.rs` matching the primary type (`linear.rs` → `Linear`, `batch_norm.rs` → `BatchNorm2d`, `conv2d.rs` → `Conv2d`).
- Trait-defining files often share a name with their central trait (`module.rs` → `Module`, `Parameters`, `StateDict`).
- Test files under `tests/` are named after the feature under test (`reshape.rs`, `concat_stack.rs`, `builder_permutations.rs`), not `test_*.rs`.
- Compile-fail fixtures live in `crates/kindle-core/tests/compile_fail/*.rs`, one file per negative scenario, named for the exact failure being asserted (`device_mismatch.rs`, `macro_idx_invalid.rs`, `forward_conv2d_static_mismatch.rs`).
- `snake_case` throughout. Constructors are `new(...)`, fallible constructors return `Result<Self>` (see `crates/kindle-core/src/nn/linear.rs`).
- Fallible operations are prefixed `try_` when a non-fallible/static-shape counterpart also exists, e.g. `reshape` (static, panics/compile-checked) vs `try_reshape` (dynamic, returns `Result`) in `crates/kindle-core/tests/reshape.rs`.
- Conversion functions follow Rust stdlib idiom: `into_dyn`, `to_vec`, `as_slice`.
- Short, local, `snake_case`. Shape/dimension variables commonly abbreviated (`in_f`, `out_f`) matching their type-level generic counterparts (`InF`, `OutF`) — see `crates/kindle-core/src/nn/linear.rs`.
- `PascalCase` for structs, traits, enums: `Tensor`, `Module`, `Parameters`, `StateDict`, `LinearShape`.
- Type-level shape markers use short PascalCase generics tied to `typenum` (`InF`, `OutF`, `U2`, `U3`) and a special `Dyn` marker type for dynamic (runtime) shapes, contrasted with static shapes built via the `s![...]` macro from `kindle-macros`.
- Backend-parameterized types follow `Thing<Shape, Backend>` ordering consistently, e.g. `Tensor<s![U2, U3], DummyBackend<f32, Cpu>>`.

## Code Style

- `cargo fmt --all -- --check` enforced in CI (`.github/workflows/ci.yml`); no custom `rustfmt.toml` present, so default rustfmt style applies uniformly across the workspace.
- `cargo clippy --workspace --all-targets -- -D warnings` enforced in CI — all clippy warnings are build failures.
- `[workspace.lints.clippy]` table exists in the root `Cargo.toml` (currently empty — no crate-specific clippy allow/deny overrides configured; rely on default clippy lints, but note in-progress work has generated many temporary `fix_*.py`/`refactor_*.py` scripts and `check*.log` files at the repo root from mechanical whitespace/lint fixups — these are not part of the conventions and should not be treated as examples).

## Import Organization

- No `#[path]` aliasing observed. Crates expose a curated `pub mod prelude` (see `crates/kindle-core/src/lib.rs`) that re-exports the commonly used public API surface (`err::*`, `nn::{...}`, tensor types, etc.) so downstream code and tests import a single glob (`use kindle_core::prelude::*;`) instead of deep paths.
- `no_std` support: `kindle-core` is `#![cfg_attr(not(feature = "std"), no_std)]` with `extern crate alloc;` — code must use `alloc::string::String`, `alloc::vec::Vec`, `alloc::format!` instead of `std` equivalents when in shared/core code paths (see `crates/kindle-core/src/err.rs`). The `kindle-macros` proc-macro conditionally emits `alloc::format!`/`std::format!` and `crate`/`kindle` paths depending on an `internal` attribute flag, to support this dual std/no_std, internal/external code generation (`crates/kindle-macros/src/module.rs`).

## Error Handling

- Centralized error enum `Error` in `crates/kindle-core/src/err.rs`, built with `thiserror::Error` derive macro; custom `Debug` impl delegates to `Display` (`write!(f, "{self}")`) so `?`-propagated errors print nicely.
- Public `Result<T>` type alias (`pub type Result<T> = core::result::Result<T, Error>;`) used everywhere instead of the raw `core::result::Result`.
- Error variants are structured/data-carrying, not plain strings: `ShapeMismatch { op, expected, got, msg }`, `OutOfMemory { device }`, `UnsupportedBackendOperation { op, backend }`, `DeviceInitializationError { expected, got }`, plus an escape hatch `Msg(String)` and `BackendFailure(#[from] anyhow::Error)` for wrapping backend-library errors via `#[from]`.
- Library code favors `Result`-returning fallible APIs with `?` propagation over `panic!`/`unwrap()` in the public API surface. `panic!`/`unreachable!`/`todo!`/`unimplemented!` usage is rare (15 occurrences total across `crates/*/src`) and should stay that way — reserve them for truly unreachable states or explicit "not yet implemented" markers, not for user-triggerable error conditions.
- `.unwrap()`/`.expect()` are common in test code (integration tests use `Result<()>` + `?` where possible, e.g. `crates/kindle/tests/tensor_ops.rs`) but should be minimized in `src/` — 149 occurrences of `.unwrap()` currently exist under `crates/kindle-core/src`, concentrated in shape/dim conversions where invariants are asserted to hold by construction; new code should prefer propagating `Result` via `?` and only `unwrap()` when the invariant is locally provable.

## Comments

- Doc comments (`///`) are used extensively on public traits, structs, and non-trivial functions, explaining purpose and usage — see `crates/kindle-core/src/nn/module.rs` (`StateDict`, `Parameters`, `ToDevice`) and `crates/kindle-core/src/nn/linear.rs` (`LinearShape`).
- Module-level doc comments (`//!`) at the top of `lib.rs` describe overall crate architecture with a bulleted breakdown of submodules (`crates/kindle-core/src/lib.rs`).
- Inline `//` comments are used sparingly inside test bodies to document test-case rationale/permutations being covered, e.g. `// permutations: positive, negative, zero, very small numbers...` in `crates/kindle/tests/tensor_ops.rs`.
- Doc comments include runnable/`ignore`-marked code examples using triple backtick ` ```rust,ignore ` blocks for API usage illustration (see `crates/kindle-core/src/lib.rs`, `crates/kindle-core/src/nn/linear.rs`). Use `ignore` when the snippet is illustrative pseudocode rather than a real compiling example.

## Function Design

## Module Design

<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

## System Overview

```text

```

## Component Responsibilities

| Component | Responsibility | File |
|-----------|----------------|------|
| `Shape` trait system | Type-level shape verification (static/dynamic dims) | `crates/kindle-core/src/shapes/shape.rs` |
| `Dim` trait + `typenum` bridge | Type-level integers backing shape dimensions | `crates/kindle-core/src/shapes/dim.rs` |
| `Tensor<S, B, K, D, G>` | Central strongly-typed tensor abstraction | `crates/kindle-core/src/tensor/base.rs` |
| `Backend` trait + op traits | Defines pluggable compute engine contract (`CreationOps`, `NumericOps`, `TensorOps`, `FloatOps`, `ReductionOps`, `ModuleOps`, `LossOps`) | `crates/kindle-core/src/tensor/backend.rs` |
| `Module` / `Parameters` / `StateDict` | Neural network layer contract: forward pass, parameter collection, state (de)serialization | `crates/kindle-core/src/nn/module.rs` |
| Layer implementations | `Linear`, `Conv1d`, `Conv2d`, `BatchNorm2d`, `LayerNorm`, pooling, `Embedding`, `RNN`/`RNNCell`, `LSTM`, activations, losses | `crates/kindle-core/src/nn/*.rs` |
| `Optimizer` / `SGD` / `Gradients` | Gradient-based parameter update abstraction | `crates/kindle-core/src/optim/mod.rs` |
| `Serializer` / `Deserializer` | Weight persistence (SafeTensors) | `crates/kindle-core/src/serialize.rs` |
| `Graph`, `OpType` | Intermediate op-graph representation used for ONNX export | `crates/kindle-core/src/graph.rs` |
| `OnnxExporter` / `OnnxImporter` | Converts internal `Graph` to/from ONNX protobuf | `crates/kindle-core/src/onnx_exporter.rs`, `crates/kindle-core/src/onnx_pb.rs` |
| `CandleBackend<T, D>` | `Backend` impl wrapping Hugging Face Candle (CPU/CUDA/Metal) | `crates/kindle-backends/src/lib.rs` (`candle` module) |
| `NdarrayBackend` | `Backend` impl wrapping the `ndarray` crate (pure-Rust CPU) | `crates/kindle-backends/src/lib.rs` (`ndarray_backend` module) |
| `BurnBackend` (optional feature) | `Backend` impl wrapping the `burn` crate | `crates/kindle-backends/src/lib.rs` (`burn_backend` module) |
| Proc macros | `s![]`, `idx![]`, `#[module]`, `import_model!()`, `impl_arg_into!()`, `generate_shape_ops!()` | `crates/kindle-macros/src/*.rs` |
| `Dataset` / `DataLoader` / `Collate` | Batch loading and iteration abstractions | `crates/kindle-data/src/dataset.rs`, `crates/kindle-data/src/loader.rs` |
| Hub downloader | Fetches models/datasets from Hugging Face Hub | `crates/kindle-data/src/hub.rs`, `crates/kindle-data/src/downloader.rs` |
| `kindle` facade crate | Public API surface; re-exports + convenience type aliases with `DefaultBackend` | `crates/kindle/src/lib.rs` |

## Pattern Overview

- Zero-runtime-cost static shape checking: shape info lives in `PhantomData`-like fields (`_shape: S::Field`) and is stripped at codegen; actual dimensions only exist as `typenum::Unsigned` types for `ConstShape`, or as real runtime `Vec<usize>` for `DynShape`/`Dyn`.
- Backend-agnostic core: `kindle-core` never depends on `candle`, `ndarray`, or `burn` directly — it only defines traits. `kindle-backends` supplies concrete implementations gated by Cargo features (`candle`, `ndarray`, `burn`).
- Macro-driven ergonomics: raw `typenum` bounds and manual `Module`/`Parameters`/`StateDict` trait impls are avoided by users via `s![]`, `idx![]`, and `#[module]`, all implemented in `kindle-macros` as proc-macros that expand into the trait implementations described above.
- Optional `no_std` support: `kindle-core` is `#![cfg_attr(not(feature = "std"), no_std)]`, using `alloc` for `Vec`/`String`/`HashMap` in that mode (see `crates/kindle-core/src/lib.rs:28-35`).
- Facade re-export crate (`kindle`) hides the multi-crate workspace behind a single `kindle::prelude::*` import and injects a `DefaultBackend`/`DefaultDevice` type alias resolved via Cargo features (`cuda` > `metal` > `cpu`), avoiding a hard cyclical dependency between `kindle-core` and `kindle-backends`.

## Layers

- Purpose: Represent and verify tensor shapes at compile time using type-level integers.
- Location: `crates/kindle-core/src/shapes/`
- Contains: `Shape`, `ConstShape`, `DynShape`, `PartialDynShape` traits (`shape.rs`); `Dim` trait and typenum glue (`dim.rs`); reshape validity (`reshape.rs`); broadcast compatibility (`broadcast.rs`); indexing/slicing types (`idx.rs`); convolution/pooling output-shape arithmetic (`spatial.rs`); concat/stack shape rules (`concat.rs`, `stack.rs`); named-dimension helpers (`named.rs`).
- Depends on: `typenum` crate only.
- Used by: `tensor/` (Tensor is generic over `S: Shape`), `nn/` (layer shape parameters), `kindle-macros` (`s![]` expands to these types).
- Purpose: Core `Tensor<S, B, K, D, G>` type and the `Backend` trait contract that any compute engine must satisfy.
- Location: `crates/kindle-core/src/tensor/`
- Contains: `base.rs` (Tensor struct + core methods), `backend.rs` (`Backend`, `CreationOps`, `NumericOps`, `TensorOps`, `FloatOps`, `ReductionOps`, `ModuleOps`, `LossOps` traits), `ops/` (operator implementations dispatching to backend traits: `binary.rs`, `unary.rs`, `reduce.rs`, `manipulation.rs`, `loss.rs`, `index.rs`), `device.rs` (`Device` trait, `Cpu`/`Cuda`/`Metal` markers), `dtype.rs` (`DType` trait, `KindleDType` enum), `grad.rs` (`Grad`/`NoGrad` markers), `matmul.rs`, `conv2d.rs`, `arg.rs`/`arg_into.rs` (constructor argument builders), `tracing.rs` (records ops into `Graph` for ONNX export).
- Depends on: `shapes/` for shape verification, backend traits it defines itself (implemented externally).
- Used by: `nn/` layers wrap `Tensor` and `Param<T, B>`; `optim/` operates on `B::RawVar`; `serialize.rs` (de)serializes `Tensor<Dyn, B>`.
- Purpose: Composable neural network building blocks.
- Location: `crates/kindle-core/src/nn/`
- Contains: `module.rs` (`Module`, `Parameters`, `StateDict`, `ToDevice` traits — the composition contract), `param.rs` (`Param<T, B>` trainable wrapper), layers (`linear.rs`, `conv1d.rs`, `conv2d.rs`, `batch_norm.rs`, `layer_norm.rs`, `embedding.rs`, `flatten.rs`, `max_pool2d.rs`, `avg_pool2d.rs`, `adaptive_avg_pool2d.rs`, `rnn.rs`, `lstm.rs`), `activation.rs` (ReLU/GELU/Sigmoid/Softmax/Swish/Tanh as zero-sized `Module` impls), `loss.rs` (MSE/CrossEntropy/L1/BCEWithLogits), `init.rs` (weight initialization schemes), `save.rs` (state dict save/load helpers), `optional.rs`/`module_optional.rs` (optional-layer composition support).
- Depends on: `tensor/` for `Tensor`/`Backend`, `shapes/` for layer shape parameters (e.g. `LinearShape`).
- Used by: user code composes layers via `Sequential<L1, L2>` or the `#[module]` macro; `kindle` facade re-exports these with `DefaultBackend` defaults.
- Purpose: Training loop support (`optim/`), weight persistence (`serialize.rs`), and model interchange (`graph.rs`, `onnx_exporter.rs`, `onnx_pb.rs`).
- Location: `crates/kindle-core/src/optim/mod.rs`, `crates/kindle-core/src/serialize.rs`, `crates/kindle-core/src/graph.rs`, `crates/kindle-core/src/onnx_exporter.rs`, `crates/kindle-core/src/onnx_pb.rs` (generated from `build.rs` via `prost-build`).
- Depends on: `Backend::RawVar`/`Backend::Grads` (optim), `Tensor<Dyn, B>` (serialize), tensor tracing output (graph/onnx).
- Used by: user training loops (`Optimizer::step`), model save/load (`StateDict::save_to`/`load_from`), `import_model!()` macro (parses ONNX at compile time into typed structs).
- Purpose: Concrete `Backend` trait implementations bridging to real tensor compute libraries.
- Location: `crates/kindle-backends/src/lib.rs` (single large file organized into `candle`, `ndarray_backend`, `burn_backend` sub-modules gated by feature flags).
- Depends on: `kindle-core` (implements its traits), `candle-core`/`candle-nn` (optional), `ndarray` (optional), `burn`/`burn-ndarray` (optional).
- Used by: `kindle` facade sets `DefaultBackend = CandleBackend<f32, DefaultDevice>` when the `candle` feature is enabled (default).

## Data Flow

### Tensor Operation Dispatch (Primary Path)

### Training Loop Flow

### ONNX Export/Import Flow

- No global mutable state in `kindle-core`; state lives in owned `Tensor`/`Param`/module structs.
- Training-time parameter state is held in `B::RawVar` (backend-native variable handles, e.g. `candle_core::Var`), collected into `HashMap<String, B::RawVar>` by `Parameters::parameters()`.
- `#[module]`-derived structs auto-implement `Parameters`/`StateDict`/`ToDevice` by recursively delegating to child fields (see `crates/kindle-macros/src/module.rs`).

## Key Abstractions

- Purpose: Represents tensor dimensionality as a type, enabling compile-time verification.
- Examples: `crates/kindle-core/src/shapes/shape.rs`; concrete static shapes are tuples of `typenum::Unsigned` (e.g. `(U2, U3, U224)`), constructed ergonomically via the `s![]` macro; `Dyn` (`crates/kindle-core/src/tensor/base.rs:8`) represents fully runtime-determined shapes.
- Pattern: Marker-trait + associated-type pattern; `S::Field` holds the actual runtime dimension data (unit type for pure-static shapes, `Vec<usize>` for dynamic ones).
- Purpose: Decouples tensor math from any specific compute library; defines the contract a new backend must satisfy (`CreationOps`, `NumericOps`, `TensorOps`, `FloatOps`, `ReductionOps`, `ModuleOps`, `LossOps`).
- Examples: `crates/kindle-core/src/tensor/backend.rs`; implementations in `crates/kindle-backends/src/lib.rs` (`CandleBackend<T, D>`, ndarray backend, burn backend).
- Pattern: Associated-type-heavy trait (GAT-style `Storage<K: DType>`) so the same backend struct can hold storage for multiple dtypes; each operation category is a separate trait so backends can be partially implemented/tested.
- Purpose: Uniform interface for anything that can be composed into a network, has trainable weights, and can be (de)serialized.
- Examples: `crates/kindle-core/src/nn/module.rs`; concrete layers in `crates/kindle-core/src/nn/*.rs`; auto-derived by `#[module]` in `crates/kindle-macros/src/module.rs`.
- Pattern: Trait objects are avoided — composition uses generic `Sequential<L1, L2>` (a two-slot generic linked-list style container, `crates/kindle-core/src/nn/module.rs`) rather than `Vec<Box<dyn Module>>`, keeping full static typing through the whole network.
- Purpose: Wraps a `Tensor` as a trainable parameter, distinguishing it from ordinary intermediate tensors and buffers.
- Examples: `crates/kindle-core/src/nn/param.rs`.
- Pattern: Newtype wrapper integrated into `Parameters::named_parameters` collection.
- Purpose: An IR capturing recorded tensor operations for ONNX export/import, independent of the live `Tensor` type.
- Examples: `crates/kindle-core/src/graph.rs` (30+ `OpType` variants covering all backend ops).
- Pattern: Flat node/value graph with integer `NodeId`/`ValueId` keys into `HashMap`s, mirroring ONNX's own graph representation.

## Entry Points

- Location: `crates/kindle/src/lib.rs`
- Triggers: Any downstream crate adding `kindle` as a dependency and importing `kindle::prelude::*`.
- Responsibilities: Re-exports `kindle-core`, `kindle-backends`, `kindle-macros`; defines `DefaultBackend`/`DefaultDevice`/`Tensor` type alias resolution based on enabled Cargo features (`candle`, `cuda`, `metal`).
- Location: `crates/kindle-core/src/lib.rs`
- Triggers: Crates that want fine-grained control without the `kindle` facade's default backend wiring (e.g. custom backend authors).
- Responsibilities: Exposes `prelude` with shapes/tensor/nn/optim/serialize traits but no concrete backend.
- Location: `crates/kindle/examples/*.rs` and `crates/kindle/examples/*/src/main.rs` (e.g. `mnist_training.rs`, `resnet_demo.rs`, `native_resnet.rs`, `rnn_sequence_prediction.rs`, `hub_import.rs`, `trace_test.rs`, subdirectory crates `backends/`, `cnn/`, `dataloader/`, `matmul/`, `named_tensors/`, `tensors/`)
- Triggers: `cargo run --example <name>` or `cargo run -p <example-crate>`.
- Responsibilities: End-to-end demonstrations of training loops, ONNX import, dataset loading, custom backend usage; also serve as de facto integration tests for the public API surface.
- Location: `crates/kindle-core/build.rs`
- Triggers: Compilation of `kindle-core`.
- Responsibilities: Uses `prost-build` to generate Rust types from the ONNX protobuf schema into `onnx_pb.rs`.

## Architectural Constraints

- **Threading:** No explicit threading model in `kindle-core`/`kindle-backends`; concurrency is delegated entirely to the chosen backend (e.g. Candle's internal thread pools) and to `kindle-data`'s `rayon`-based parallel data loading (`crates/kindle-data/src/loader.rs`, `Cargo.toml` depends on `rayon`).
- **Global state:** None observed at the `kindle-core`/`kindle-backends` module level; all state is owned by `Tensor`, `Param`, and module structs. The generated ONNX protobuf module (`onnx_pb.rs`) is build-script generated but not a runtime singleton.
- **Circular imports:** Deliberately avoided between `kindle-core` and `kindle-backends` — `kindle-core` has zero dependency on any concrete backend crate; the facade `kindle` crate is the only place that ties `DefaultBackend` to a concrete `kindle-backends` implementation (`crates/kindle/src/lib.rs:99-103`), preventing a cycle.
- **`no_std` compatibility:** `kindle-core` conditionally disables `std` (`#![cfg_attr(not(feature = "std"), no_std)]`, `crates/kindle-core/src/lib.rs:28`) and imports `alloc` explicitly; code contributed to `kindle-core` must use `alloc::vec::Vec`/`alloc::string::String` rather than `std::` equivalents when in shared/generic code paths (see `err.rs` using `alloc::string::String`).
- **Nightly-only path:** `generic_const_exprs` is enabled only under the `nightly` feature (`crates/kindle-core/src/lib.rs:29-33`) and marked `incomplete_features`; code depending on it must be feature-gated.
- **Compile-fail test suite:** `crates/kindle-core/tests/compile_fail/*.rs` + `trybuild` (dev-dependency) assert that specific shape mismatches (matmul, conv, reshape, broadcast, concat, stack, macro misuse) fail to compile — any change to shape-verification traits must keep these compile-fail cases failing for the correct reason.
- **In-progress refactor:** Git status shows nearly every file under `tensor/`, `nn/`, `optim/`, plus `onnx_exporter.rs`, `serialize.rs`, and `kindle-macros/src/module.rs` modified uncommitted, and `tensor/kind.rs` deleted, alongside numerous root-level one-off Python refactor scripts (`fix_*.py`, `refactor_*.py`) and log files (`check*.log`, `check*.txt`) — the workspace is mid-refactor (see git status / recent commit "chore: save state before massive backend device refactor"). Treat current trait signatures as provisional.

## Anti-Patterns

### Root-level throwaway scripts and logs committed to working tree

### Manual `map_err(|e: candle_core::Error| anyhow::anyhow!(e))` repeated on every backend call

## Error Handling

- Backend implementations (`kindle-backends`) convert library-native errors (`candle_core::Error`) into `anyhow::Error` then into `Error::BackendFailure` via `?`/`.into()`.
- Shape-related failures use the structured `Error::ShapeMismatch { op, expected, got, msg }` variant so callers get both the offending operation name and shape details.
- Serialization errors are surfaced through the `Serializer`/`Deserializer` trait's own `Error: Debug + Display` associated type rather than the top-level `Error` enum, then converted to `Error::ShapeMismatch`-shaped generic messages at the `StateDict::load_from` boundary (`crates/kindle-core/src/nn/module.rs:41-49`) — an inconsistency worth normalizing.

## Cross-Cutting Concerns

<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
