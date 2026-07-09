# Codebase Structure

**Analysis Date:** 2026-07-09

## Directory Layout

```
kindle/                                # Cargo workspace root
├── Cargo.toml                         # Workspace manifest, member list, shared deps
├── .cargo/config.toml                 # cargo-hack aliases (hb/ht/hr = feature-powerset build/test/run)
├── crates/
│   ├── kindle-core/                   # Framework core: traits, Tensor type, nn layers, shapes (no_std-capable)
│   │   ├── build.rs                   # prost-build codegen for ONNX protobuf -> onnx_pb.rs
│   │   ├── examples/
│   │   │   └── onnx_export.rs         # Example: exporting a Kindle graph to ONNX
│   │   ├── src/
│   │   │   ├── lib.rs                 # Crate root, feature gates, prelude module
│   │   │   ├── err.rs                 # `Error` enum, `Result<T>` alias
│   │   │   ├── graph.rs               # `Graph`, `OpType`, IR for ONNX export/import
│   │   │   ├── onnx_exporter.rs       # `OnnxExporter`/`OnnxImporter`, Graph <-> ONNX protobuf
│   │   │   ├── onnx_pb.rs             # Generated ONNX protobuf types (via build.rs)
│   │   │   ├── serialize.rs           # `Serializer`/`Deserializer` traits, SafeTensors impl
│   │   │   ├── shapes/                # Type-level shape system (typenum-based)
│   │   │   │   ├── mod.rs, shape.rs, dim.rs, arithmetic.rs, broadcast.rs
│   │   │   │   ├── concat.rs, stack.rs, reshape.rs, idx.rs, named.rs, spatial.rs, shape_ops.rs
│   │   │   ├── tensor/                # Core `Tensor<S,B,K,D,G>` + `Backend` trait contract
│   │   │   │   ├── base.rs            # `Tensor` struct definition and core methods
│   │   │   │   ├── backend.rs         # `Backend`, `CreationOps`, `NumericOps`, `TensorOps`,
│   │   │   │   │                      #   `FloatOps`, `ReductionOps`, `ModuleOps`, `LossOps` traits
│   │   │   │   ├── device.rs          # `Device` trait, `Cpu`/`Cuda`/`Metal` markers
│   │   │   │   ├── dtype.rs           # `DType` trait, `KindleDType` enum
│   │   │   │   ├── grad.rs            # `Grad`/`NoGrad` marker traits
│   │   │   │   ├── matmul.rs          # Matmul shape verification + dispatch
│   │   │   │   ├── conv2d.rs          # Conv2d shape verification + dispatch
│   │   │   │   ├── arg.rs, arg_into.rs # Constructor argument builder traits
│   │   │   │   ├── tracing.rs         # Records tensor ops into `Graph` for ONNX export
│   │   │   │   └── ops/                # Operator implementations (dispatch to Backend)
│   │   │   │       ├── binary.rs, unary.rs, reduce.rs, manipulation.rs, loss.rs, index.rs
│   │   │   ├── nn/                    # Neural network layers and building blocks
│   │   │   │   ├── mod.rs             # Module doc index, re-exports
│   │   │   │   ├── module.rs          # `Module`, `Parameters`, `StateDict`, `ToDevice` traits
│   │   │   │   ├── param.rs           # `Param<T, B>` trainable wrapper
│   │   │   │   ├── linear.rs, conv1d.rs, conv2d.rs, batch_norm.rs, layer_norm.rs
│   │   │   │   ├── embedding.rs, flatten.rs, max_pool2d.rs, avg_pool2d.rs, adaptive_avg_pool2d.rs
│   │   │   │   ├── rnn.rs, lstm.rs    # Recurrent layers
│   │   │   │   ├── activation.rs      # ReLU, GELU, Sigmoid, Softmax, Swish, Tanh
│   │   │   │   ├── loss.rs            # MSE, CrossEntropy, L1, BCEWithLogits
│   │   │   │   ├── init.rs            # Weight initialization schemes
│   │   │   │   ├── save.rs            # State dict save/load helpers
│   │   │   │   └── optional.rs, module_optional.rs  # Optional-layer composition support
│   │   │   └── optim/mod.rs           # `Optimizer`, `SGD`, `Gradients`
│   │   └── tests/                     # Integration + compile-fail tests
│   │       ├── builder_permutations.rs, concat_stack.rs, reshape.rs, compile_tests.rs
│   │       └── compile_fail/          # `trybuild`-driven negative compile tests (18 files)
│   │
│   ├── kindle-backends/               # Concrete `Backend` trait implementations
│   │   ├── src/lib.rs                 # `candle`, `ndarray_backend`, `burn_backend` modules (feature-gated)
│   │   └── tests/
│   │       ├── ndarray.rs             # ndarray backend tests
│   │       └── ops.rs                 # Cross-backend op tests
│   │
│   ├── kindle-data/                   # Dataset loading and Hugging Face Hub integration
│   │   └── src/
│   │       ├── lib.rs                 # Crate root, prelude
│   │       ├── dataset.rs             # `Dataset` trait
│   │       ├── loader.rs              # `DataLoader`, `Collate`
│   │       ├── downloader.rs          # `Downloader` (generic file fetch)
│   │       ├── hub.rs                 # Hugging Face Hub integration (`hf-hub`)
│   │       └── vision/
│   │           ├── mod.rs
│   │           └── mnist.rs           # MNIST dataset loader
│   │
│   ├── kindle-macros/                 # Proc-macro crate (ergonomic frontend)
│   │   └── src/
│   │       ├── lib.rs                 # Macro entry points: `s!`, `idx!`, `#[module]`, `import_model!`, `impl_arg_into!`
│   │       ├── shape.rs               # `s![]` shape macro implementation
│   │       ├── idx.rs                 # `idx![]` slicing macro implementation
│   │       ├── module.rs              # `#[module]` attribute macro (derives Module/Parameters/StateDict/ToDevice)
│   │       ├── onnx.rs                # `import_model!()` — compile-time ONNX -> typed struct codegen
│   │       ├── arg_into.rs            # `impl_arg_into!()` helper macro
│   │       ├── safetensors.rs         # SafeTensors-related codegen helpers
│   │       └── shape_ops.rs           # `generate_shape_ops!()` internal codegen macro
│   │
│   └── kindle/                        # Public facade crate — the crate users depend on
│       ├── src/lib.rs                 # Re-exports core/backends/macros; DefaultBackend/DefaultDevice aliases
│       ├── examples/                  # Runnable examples (standalone files + sub-crates)
│       │   ├── batched_matmul.rs, hub_import.rs, idx_demo.rs, mnist_training.rs
│       │   ├── native_resnet.rs, resnet_demo.rs, rnn_sequence_prediction.rs, trace_test.rs
│       │   ├── backends/src/main.rs   # Sub-crate example: multi-backend usage
│       │   ├── cnn/src/main.rs        # Sub-crate example: CNN training
│       │   ├── dataloader/src/main.rs # Sub-crate example: DataLoader usage
│       │   ├── matmul/src/main.rs     # Sub-crate example: matmul walkthrough
│       │   ├── named_tensors/src/main.rs # Sub-crate example: named dimensions
│       │   └── tensors/src/main.rs    # Sub-crate example: tensor basics
│       └── tests/                     # Crate-level integration tests
│           ├── autograd_tests.rs, broadcast.rs, data_tests.rs, layers.rs
│           ├── macro_tests.rs, nn_tests.rs, onnx_import.rs, optim_tests.rs
│           ├── serde_tests.rs, tensor_ops.rs
│
├── test_models/                       # Sample ONNX model fixtures used by tests/examples
│   ├── advanced.onnx, advanced.onnx.kindle_meta
│   └── if.onnx, if.onnx.kindle_meta
│
└── .planning/                         # GSD planning artifacts (not framework code)
    └── codebase/                      # This directory — generated codebase maps
```

## Directory Purposes

**`crates/kindle-core/src/shapes/`:**
- Purpose: Type-level shape definitions and compile-time verification traits.
- Contains: One file per concern — dimension arithmetic (`arithmetic.rs`), broadcasting rules (`broadcast.rs`), concat/stack rules (`concat.rs`/`stack.rs`), reshape validity (`reshape.rs`), indexing/slicing types (`idx.rs`), named-dimension support (`named.rs`), conv/pool spatial output shapes (`spatial.rs`), the core `Shape`/`ConstShape`/`DynShape`/`PartialDynShape` traits (`shape.rs`).
- Key files: `shape.rs` (trait definitions), `dim.rs` (typenum bridge + `symbolic_dim!` macro).

**`crates/kindle-core/src/tensor/`:**
- Purpose: The `Tensor` type, the `Backend` trait contract, and per-category operator implementations.
- Contains: `base.rs` (struct), `backend.rs` (trait contract — the single most important file for backend authors), `ops/` subdirectory (operator dispatch logic split by category).
- Key files: `backend.rs`, `base.rs`, `ops/mod.rs`.

**`crates/kindle-core/src/nn/`:**
- Purpose: All neural network layer implementations and the `Module` composition contract.
- Contains: One file per layer type; `module.rs` defines the shared trait contract every layer implements (usually via the `#[module]` macro rather than by hand).
- Key files: `module.rs`, `param.rs`.

**`crates/kindle-backends/src/lib.rs`:**
- Purpose: Single file containing all backend implementations, organized into `candle`, `ndarray_backend`, `burn_backend` inline modules, each gated by a Cargo feature.
- Note: This is unusually large (1500+ lines) for a single file; consider splitting into `src/candle.rs`, `src/ndarray_backend.rs`, `src/burn_backend.rs` submodules if extending further (see CONCERNS if generated).

**`crates/kindle-macros/src/`:**
- Purpose: All proc-macro implementations. `lib.rs` only wires up `#[proc_macro]`/`#[proc_macro_attribute]` entry points; actual logic lives in per-macro files.
- Contains: `module.rs` is the largest/most complex (derives `Parameters`/`StateDict`/`ToDevice` for `#[module]`-annotated structs); `onnx.rs` implements compile-time ONNX file parsing for `import_model!()`.

**`crates/kindle-data/src/`:**
- Purpose: Dataset and DataLoader abstractions plus Hugging Face Hub integration, kept separate from `kindle-core` because it depends on `candle-core` directly and pulls in networking (`ureq`, `hf-hub`) and parallelism (`rayon`) dependencies.
- Contains: `dataset.rs`/`loader.rs` (generic abstractions), `hub.rs`/`downloader.rs` (network fetch), `vision/mnist.rs` (concrete dataset).

**`crates/kindle/examples/`:**
- Purpose: Both documentation-by-example and de facto integration smoke tests for the public API.
- Generated: No.
- Committed: Yes — treat as living documentation; update alongside public API changes in `crates/kindle/src/lib.rs`.

**`crates/*/tests/`:**
- Purpose: Crate-level integration tests. `kindle-core/tests/compile_fail/` is special — these files are expected to **fail to compile**, verified via the `trybuild` dev-dependency in `kindle-core/tests/compile_tests.rs`.

**`test_models/`:**
- Purpose: Fixture ONNX files (`.onnx`) plus Kindle-specific sidecar metadata (`.onnx.kindle_meta`) used by ONNX import/export tests and the `import_model!()` macro's compile-time file reads.

## Key File Locations

**Entry Points:**
- `crates/kindle/src/lib.rs`: Public facade — the crate downstream users depend on.
- `crates/kindle-core/src/lib.rs`: Backend-agnostic core entry (advanced/no_std usage).
- `crates/kindle-backends/src/lib.rs`: Backend implementations entry.
- `crates/kindle-macros/src/lib.rs`: Proc-macro entry points.
- `crates/kindle-data/src/lib.rs`: Dataset/loader entry.

**Configuration:**
- `Cargo.toml` (workspace root): Member crates, shared dependency versions, `[workspace.package]` metadata (edition 2024).
- `crates/*/Cargo.toml`: Per-crate feature flags — notably `kindle-core` (`std`, `nightly`, `cuda`, `metal`), `kindle-backends` (`candle`, `ndarray`, `burn`, `cuda`, `metal`), `kindle` (mirrors + adds `default = ["std", "candle"]`).
- `.cargo/config.toml`: `cargo-hack` aliases `hb`/`ht`/`hr` for feature-powerset build/test/run — use these to validate changes across all feature combinations.
- `crates/kindle-core/build.rs`: Generates `onnx_pb.rs` from ONNX's protobuf schema via `prost-build`.

**Core Logic:**
- `crates/kindle-core/src/tensor/backend.rs`: The contract every backend must implement — start here when adding backend operations.
- `crates/kindle-core/src/nn/module.rs`: The contract every layer must implement.
- `crates/kindle-core/src/shapes/shape.rs`: The shape-safety type system foundation.

**Testing:**
- `crates/kindle-core/tests/compile_fail/`: Negative compile-time shape-safety tests (trybuild).
- `crates/kindle/tests/`: End-to-end integration tests exercising the public facade API.
- `crates/kindle-backends/tests/`: Backend-specific operation correctness tests.

## Naming Conventions

**Files:**
- One file per major type/trait-family, named after the primary concept it defines (e.g. `linear.rs` defines `Linear` + `LinearShape`, `batch_norm.rs` defines `BatchNorm2d`).
- Op-category files inside `tensor/ops/` are named after the operator class, not a specific op (`binary.rs`, `unary.rs`, `reduce.rs`, `manipulation.rs`, `loss.rs`, `index.rs`).
- `mod.rs` used for directory-level re-exports and doc comments summarizing the submodule (see `nn/mod.rs`, `shapes/mod.rs`).

**Directories:**
- Crate directories are prefixed `kindle-` except the facade crate itself, simply named `kindle`.
- Feature/domain directories (`shapes/`, `tensor/`, `nn/`, `optim/`) are flat, lowercase, singular-or-plural matching the Rust convention already in use in the codebase (`nn` not `neural_network`, `shapes` plural because it holds multiple shape-related traits).

## Where to Add New Code

**New Tensor Operation:**
- Trait method: add to the relevant trait in `crates/kindle-core/src/tensor/backend.rs` (e.g. add to `TensorOps` for a manipulation op, `FloatOps` for an elementwise float op).
- Public-facing dispatch: add to the matching file in `crates/kindle-core/src/tensor/ops/` (e.g. `manipulation.rs`, `unary.rs`).
- Backend implementations: implement the new trait method for each enabled backend in `crates/kindle-backends/src/lib.rs` (candle/ndarray/burn sections) — leaving `unimplemented!()` stubs for unsupported backends is an existing pattern (see `adaptive_avg_pool2d` in the `candle` module).
- Shape verification (if the op changes output shape): add a trait to `crates/kindle-core/src/shapes/` (mirror `spatial.rs`/`broadcast.rs` patterns).
- Graph/ONNX support: add a variant to `OpType` in `crates/kindle-core/src/graph.rs` if the op should be traceable/exportable.

**New Layer/Module:**
- Implementation: new file in `crates/kindle-core/src/nn/`, following the pattern of `linear.rs`/`conv2d.rs` — define a struct holding `Param<T, B>` fields, annotate with `#[module]` (or hand-implement `Module`/`Parameters`/`StateDict`).
- Register: add `pub mod <layer>;` and `pub use <layer>::*;` to `crates/kindle-core/src/nn/mod.rs`; add to `crates/kindle-core/src/lib.rs` prelude if it should be broadly accessible; add a type alias with `DefaultBackend` default in `crates/kindle/src/lib.rs` for facade ergonomics.
- Tests: `crates/kindle/tests/layers.rs` or `crates/kindle/tests/nn_tests.rs`.

**New Backend:**
- Implementation: new feature-gated module in `crates/kindle-backends/src/lib.rs` (mirror the `candle`/`ndarray_backend` module structure), implementing `Backend` + `CreationOps` + `NumericOps` + `TensorOps` + `FloatOps` + `ReductionOps` + `ModuleOps` + `LossOps` from `kindle-core::tensor::backend`.
- Feature flag: add to `crates/kindle-backends/Cargo.toml` `[features]`.
- Facade wiring: optionally add a `DefaultBackend` resolution branch in `crates/kindle/src/lib.rs` if it should be selectable as default.

**New Example:**
- Standalone: add a `.rs` file directly under `crates/kindle/examples/` (auto-discovered by Cargo as an example binary).
- Multi-file: add a subdirectory under `crates/kindle/examples/<name>/` with its own `Cargo.toml` + `src/main.rs`, and register it as a workspace member glob (already covered by `crates/kindle/examples/*` in the root `Cargo.toml`).

**Utilities:**
- Cross-cutting helpers used by multiple ops/layers belong in `crates/kindle-core/src/tensor/arg.rs`/`arg_into.rs` (constructor argument builders) or as new top-level modules in `crates/kindle-core/src/` if they don't fit `tensor/`/`nn/`/`shapes/`.

## Special Directories

**`crates/kindle-core/tests/compile_fail/`:**
- Purpose: Intentionally-invalid Rust snippets asserting the type system rejects bad shapes/macro usage.
- Generated: No — hand-written.
- Committed: Yes. Run via `trybuild` through `crates/kindle-core/tests/compile_tests.rs`.

**`test_models/`:**
- Purpose: Binary `.onnx` fixture files + `.kindle_meta` sidecar files for ONNX import/export tests.
- Generated: `.onnx` files are external fixtures; `.kindle_meta` files appear to be Kindle-generated metadata sidecars produced by the import/export tooling.
- Committed: Yes.

**`crates/kindle-core/src/onnx_pb.rs`:**
- Purpose: Rust types generated from the ONNX protobuf schema.
- Generated: Yes, by `crates/kindle-core/build.rs` via `prost-build` at build time — do not hand-edit; regenerate by changing the build script/schema source instead.
- Committed: Appears committed as a source file in the listing, but treat as derived — verify against `build.rs` before manual edits.

**Root-level scratch files (untracked, not part of crate structure):**
- Purpose: `fix_*.py`, `refactor_*.py`, `strip_bounds.py`, `restore_tracing.py`, `check*.log`, `check.txt`, `test_proof.rs`, `test_proof.long-type-*.txt`, `expanded.rs` are ad hoc scripts/output from the in-progress backend refactor (per `git status`).
- Generated: Yes (refactor tooling output).
- Committed: No (untracked `??` in git status) — do not treat as part of the library's structure; avoid adding new code here.

---

*Structure analysis: 2026-07-09*
