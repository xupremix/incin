<!-- refreshed: 2026-07-09 -->
# Architecture

**Analysis Date:** 2026-07-09

## System Overview

```text
┌───────────────────────────────────────────────────────────────────────────┐
│                          User Code / Examples                              │
│              `crates/kindle/examples/*`, downstream crates                 │
└───────────────────────────────────┬────────────────────────────────────────┘
                                     │ uses `kindle::prelude::*`
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                        kindle (facade / re-export crate)                   │
│                        `crates/kindle/src/lib.rs`                          │
│  - Re-exports kindle-core + kindle-backends + kindle-macros                │
│  - Defines `DefaultBackend`, `DefaultDevice`, ergonomic type aliases       │
│    (Tensor, Linear, Conv2d, Sequential, Param, RNN, Embedding...)          │
└───────────┬───────────────────────────────────┬─────────────────────────────┘
            │                                    │
            ▼                                    ▼
┌──────────────────────────────┐    ┌──────────────────────────────────────┐
│   kindle-macros (proc-macro) │    │      kindle-data (datasets)          │
│  `crates/kindle-macros/src`  │    │  `crates/kindle-data/src`            │
│  - `s![]` shape macro        │    │  - `Dataset`, `DataLoader`, `Collate` │
│  - `idx![]` slicing macro    │    │  - HuggingFace Hub downloader        │
│  - `#[module]` derive attr   │    │  - MNIST/vision loaders              │
│  - `import_model!()` ONNX    │    └──────────────────────────────────────┘
│    codegen                   │
│  - `impl_arg_into!()`        │
└───────────────┬───────────────┘
                │ generates code depending on
                ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    kindle-backends (backend implementations)               │
│                    `crates/kindle-backends/src/lib.rs`                     │
│  - `candle` module: implements `Backend` trait via candle-core/candle-nn   │
│  - `ndarray_backend` module: implements `Backend` trait via `ndarray`      │
│  - `burn_backend` module: implements `Backend` trait via `burn` (optional) │
│  - `dummy` (in kindle-core tests): mock backend for compile-time tests     │
└───────────────────────────────────┬────────────────────────────────────────┘
                                     │ implements traits defined by
                                     ▼
┌───────────────────────────────────────────────────────────────────────────┐
│                    kindle-core (framework core, no_std-capable)            │
│                    `crates/kindle-core/src/`                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────────┐   │
│  │   shapes/   │  │   tensor/   │  │     nn/     │  │  optim/, graph, │   │
│  │ type-level  │  │  Tensor<S,  │  │  Module,    │  │  serialize,     │   │
│  │ shape       │  │  B,K,D,G>,  │  │  Linear,    │  │  onnx_exporter  │   │
│  │ verification│  │  Backend    │  │  Conv2d,    │  │                 │   │
│  │ (typenum)   │  │  trait, ops │  │  Sequential │  │                 │   │
│  └─────────────┘  └─────────────┘  └─────────────┘  └────────────────┘   │
└───────────────────────────────────────────────────────────────────────────┘
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

**Overall:** Trait-based backend abstraction (a "compile-time strategy pattern") combined with type-level generic programming for shape safety — conceptually similar to `dfdx`/`burn`, distinguishing itself by making `Tensor<Shape, Backend, DType, Device, Grad>` fully generic over five type parameters that are checked at compile time via `typenum` and marker traits.

**Key Characteristics:**
- Zero-runtime-cost static shape checking: shape info lives in `PhantomData`-like fields (`_shape: S::Field`) and is stripped at codegen; actual dimensions only exist as `typenum::Unsigned` types for `ConstShape`, or as real runtime `Vec<usize>` for `DynShape`/`Dyn`.
- Backend-agnostic core: `kindle-core` never depends on `candle`, `ndarray`, or `burn` directly — it only defines traits. `kindle-backends` supplies concrete implementations gated by Cargo features (`candle`, `ndarray`, `burn`).
- Macro-driven ergonomics: raw `typenum` bounds and manual `Module`/`Parameters`/`StateDict` trait impls are avoided by users via `s![]`, `idx![]`, and `#[module]`, all implemented in `kindle-macros` as proc-macros that expand into the trait implementations described above.
- Optional `no_std` support: `kindle-core` is `#![cfg_attr(not(feature = "std"), no_std)]`, using `alloc` for `Vec`/`String`/`HashMap` in that mode (see `crates/kindle-core/src/lib.rs:28-35`).
- Facade re-export crate (`kindle`) hides the multi-crate workspace behind a single `kindle::prelude::*` import and injects a `DefaultBackend`/`DefaultDevice` type alias resolved via Cargo features (`cuda` > `metal` > `cpu`), avoiding a hard cyclical dependency between `kindle-core` and `kindle-backends`.

## Layers

**Shapes layer (`shapes/`):**
- Purpose: Represent and verify tensor shapes at compile time using type-level integers.
- Location: `crates/kindle-core/src/shapes/`
- Contains: `Shape`, `ConstShape`, `DynShape`, `PartialDynShape` traits (`shape.rs`); `Dim` trait and typenum glue (`dim.rs`); reshape validity (`reshape.rs`); broadcast compatibility (`broadcast.rs`); indexing/slicing types (`idx.rs`); convolution/pooling output-shape arithmetic (`spatial.rs`); concat/stack shape rules (`concat.rs`, `stack.rs`); named-dimension helpers (`named.rs`).
- Depends on: `typenum` crate only.
- Used by: `tensor/` (Tensor is generic over `S: Shape`), `nn/` (layer shape parameters), `kindle-macros` (`s![]` expands to these types).

**Tensor layer (`tensor/`):**
- Purpose: Core `Tensor<S, B, K, D, G>` type and the `Backend` trait contract that any compute engine must satisfy.
- Location: `crates/kindle-core/src/tensor/`
- Contains: `base.rs` (Tensor struct + core methods), `backend.rs` (`Backend`, `CreationOps`, `NumericOps`, `TensorOps`, `FloatOps`, `ReductionOps`, `ModuleOps`, `LossOps` traits), `ops/` (operator implementations dispatching to backend traits: `binary.rs`, `unary.rs`, `reduce.rs`, `manipulation.rs`, `loss.rs`, `index.rs`), `device.rs` (`Device` trait, `Cpu`/`Cuda`/`Metal` markers), `dtype.rs` (`DType` trait, `KindleDType` enum), `grad.rs` (`Grad`/`NoGrad` markers), `matmul.rs`, `conv2d.rs`, `arg.rs`/`arg_into.rs` (constructor argument builders), `tracing.rs` (records ops into `Graph` for ONNX export).
- Depends on: `shapes/` for shape verification, backend traits it defines itself (implemented externally).
- Used by: `nn/` layers wrap `Tensor` and `Param<T, B>`; `optim/` operates on `B::RawVar`; `serialize.rs` (de)serializes `Tensor<Dyn, B>`.

**NN layer (`nn/`):**
- Purpose: Composable neural network building blocks.
- Location: `crates/kindle-core/src/nn/`
- Contains: `module.rs` (`Module`, `Parameters`, `StateDict`, `ToDevice` traits — the composition contract), `param.rs` (`Param<T, B>` trainable wrapper), layers (`linear.rs`, `conv1d.rs`, `conv2d.rs`, `batch_norm.rs`, `layer_norm.rs`, `embedding.rs`, `flatten.rs`, `max_pool2d.rs`, `avg_pool2d.rs`, `adaptive_avg_pool2d.rs`, `rnn.rs`, `lstm.rs`), `activation.rs` (ReLU/GELU/Sigmoid/Softmax/Swish/Tanh as zero-sized `Module` impls), `loss.rs` (MSE/CrossEntropy/L1/BCEWithLogits), `init.rs` (weight initialization schemes), `save.rs` (state dict save/load helpers), `optional.rs`/`module_optional.rs` (optional-layer composition support).
- Depends on: `tensor/` for `Tensor`/`Backend`, `shapes/` for layer shape parameters (e.g. `LinearShape`).
- Used by: user code composes layers via `Sequential<L1, L2>` or the `#[module]` macro; `kindle` facade re-exports these with `DefaultBackend` defaults.

**Cross-cutting: optim, serialize, graph/onnx (top-level modules):**
- Purpose: Training loop support (`optim/`), weight persistence (`serialize.rs`), and model interchange (`graph.rs`, `onnx_exporter.rs`, `onnx_pb.rs`).
- Location: `crates/kindle-core/src/optim/mod.rs`, `crates/kindle-core/src/serialize.rs`, `crates/kindle-core/src/graph.rs`, `crates/kindle-core/src/onnx_exporter.rs`, `crates/kindle-core/src/onnx_pb.rs` (generated from `build.rs` via `prost-build`).
- Depends on: `Backend::RawVar`/`Backend::Grads` (optim), `Tensor<Dyn, B>` (serialize), tensor tracing output (graph/onnx).
- Used by: user training loops (`Optimizer::step`), model save/load (`StateDict::save_to`/`load_from`), `import_model!()` macro (parses ONNX at compile time into typed structs).

**Backend implementation layer (`kindle-backends`):**
- Purpose: Concrete `Backend` trait implementations bridging to real tensor compute libraries.
- Location: `crates/kindle-backends/src/lib.rs` (single large file organized into `candle`, `ndarray_backend`, `burn_backend` sub-modules gated by feature flags).
- Depends on: `kindle-core` (implements its traits), `candle-core`/`candle-nn` (optional), `ndarray` (optional), `burn`/`burn-ndarray` (optional).
- Used by: `kindle` facade sets `DefaultBackend = CandleBackend<f32, DefaultDevice>` when the `candle` feature is enabled (default).

## Data Flow

### Tensor Operation Dispatch (Primary Path)

1. User calls a method on `Tensor<S, B, K, D, G>`, e.g. `tensor.matmul(&other)` (`crates/kindle-core/src/tensor/ops/manipulation.rs` or `matmul.rs`).
2. The op module validates/derives the *output* `Shape` type at compile time using traits from `shapes/` (e.g. `MatMulShape`), so an incompatible shape simply fails to compile.
3. At runtime, the op delegates to the corresponding `Backend` associated trait method, e.g. `B::matmul::<K>(lhs.inner(), rhs.inner())` (`crates/kindle-core/src/tensor/backend.rs:125`).
4. The concrete backend (e.g. `CandleBackend`) executes the operation using its underlying library (`crates/kindle-backends/src/lib.rs`, `candle` module, `TensorOps::matmul` impl).
5. The raw result (`B::Storage<K>`) is wrapped back into a new `Tensor` with the statically-computed output shape via `from_parts_unchecked`/`from_parts` (`crates/kindle-core/src/tensor/base.rs:68-120`).

### Training Loop Flow

1. Forward pass: `Module::forward` is called recursively through composed `Sequential<L1, L2>` layers (`crates/kindle-core/src/nn/module.rs`), each layer internally invoking tensor ops as above.
2. Loss computation: a `LossOps` implementation (e.g. `cross_entropy_loss`) reduces the output tensor to a scalar (`crates/kindle-core/src/nn/loss.rs`, dispatched via `Backend::LossOps`).
3. Backward pass: `Backend::backward::<K>(&loss_storage)` triggers backend-native autodiff (e.g. Candle's `.backward()`), returning `Backend::Grads` (`crates/kindle-core/src/tensor/backend.rs:77`).
4. Gradient retrieval + update: `Optimizer::step` (e.g. `SGD::step` in `crates/kindle-core/src/optim/mod.rs`) iterates tracked `B::RawVar` parameters, fetches gradients via `Backend::get_grad`, and applies the update rule via `Backend::assign_var`.

### ONNX Export/Import Flow

1. Tensor ops optionally record themselves into a `Graph` (nodes/values/`OpType`) via tracing hooks (`crates/kindle-core/src/tensor/tracing.rs`).
2. `export_to_onnx(graph, path)` converts the internal `Graph` into an `onnx::GraphProto` (protobuf, generated types in `onnx_pb.rs`) and serializes it with `prost` (`crates/kindle-core/src/onnx_exporter.rs`).
3. Import direction: the `import_model!("model.onnx", Name)` proc-macro (`crates/kindle-macros/src/onnx.rs`) reads an ONNX file **at compile time** and generates a fully-typed Rust struct + `forward` method matching the graph, with weights to be populated later via `StateDict::load_from`.

**State Management:**
- No global mutable state in `kindle-core`; state lives in owned `Tensor`/`Param`/module structs.
- Training-time parameter state is held in `B::RawVar` (backend-native variable handles, e.g. `candle_core::Var`), collected into `HashMap<String, B::RawVar>` by `Parameters::parameters()`.
- `#[module]`-derived structs auto-implement `Parameters`/`StateDict`/`ToDevice` by recursively delegating to child fields (see `crates/kindle-macros/src/module.rs`).

## Key Abstractions

**`Shape` (and `ConstShape`/`DynShape`/`PartialDynShape`):**
- Purpose: Represents tensor dimensionality as a type, enabling compile-time verification.
- Examples: `crates/kindle-core/src/shapes/shape.rs`; concrete static shapes are tuples of `typenum::Unsigned` (e.g. `(U2, U3, U224)`), constructed ergonomically via the `s![]` macro; `Dyn` (`crates/kindle-core/src/tensor/base.rs:8`) represents fully runtime-determined shapes.
- Pattern: Marker-trait + associated-type pattern; `S::Field` holds the actual runtime dimension data (unit type for pure-static shapes, `Vec<usize>` for dynamic ones).

**`Backend` trait family:**
- Purpose: Decouples tensor math from any specific compute library; defines the contract a new backend must satisfy (`CreationOps`, `NumericOps`, `TensorOps`, `FloatOps`, `ReductionOps`, `ModuleOps`, `LossOps`).
- Examples: `crates/kindle-core/src/tensor/backend.rs`; implementations in `crates/kindle-backends/src/lib.rs` (`CandleBackend<T, D>`, ndarray backend, burn backend).
- Pattern: Associated-type-heavy trait (GAT-style `Storage<K: DType>`) so the same backend struct can hold storage for multiple dtypes; each operation category is a separate trait so backends can be partially implemented/tested.

**`Module` / `Parameters` / `StateDict`:**
- Purpose: Uniform interface for anything that can be composed into a network, has trainable weights, and can be (de)serialized.
- Examples: `crates/kindle-core/src/nn/module.rs`; concrete layers in `crates/kindle-core/src/nn/*.rs`; auto-derived by `#[module]` in `crates/kindle-macros/src/module.rs`.
- Pattern: Trait objects are avoided — composition uses generic `Sequential<L1, L2>` (a two-slot generic linked-list style container, `crates/kindle-core/src/nn/module.rs`) rather than `Vec<Box<dyn Module>>`, keeping full static typing through the whole network.

**`Param<T, B>`:**
- Purpose: Wraps a `Tensor` as a trainable parameter, distinguishing it from ordinary intermediate tensors and buffers.
- Examples: `crates/kindle-core/src/nn/param.rs`.
- Pattern: Newtype wrapper integrated into `Parameters::named_parameters` collection.

**`Graph` / `OpType`:**
- Purpose: An IR capturing recorded tensor operations for ONNX export/import, independent of the live `Tensor` type.
- Examples: `crates/kindle-core/src/graph.rs` (30+ `OpType` variants covering all backend ops).
- Pattern: Flat node/value graph with integer `NodeId`/`ValueId` keys into `HashMap`s, mirroring ONNX's own graph representation.

## Entry Points

**Library entry (primary):**
- Location: `crates/kindle/src/lib.rs`
- Triggers: Any downstream crate adding `kindle` as a dependency and importing `kindle::prelude::*`.
- Responsibilities: Re-exports `kindle-core`, `kindle-backends`, `kindle-macros`; defines `DefaultBackend`/`DefaultDevice`/`Tensor` type alias resolution based on enabled Cargo features (`candle`, `cuda`, `metal`).

**Direct core entry (advanced/no_std users):**
- Location: `crates/kindle-core/src/lib.rs`
- Triggers: Crates that want fine-grained control without the `kindle` facade's default backend wiring (e.g. custom backend authors).
- Responsibilities: Exposes `prelude` with shapes/tensor/nn/optim/serialize traits but no concrete backend.

**Examples (developer-facing entry points / smoke tests):**
- Location: `crates/kindle/examples/*.rs` and `crates/kindle/examples/*/src/main.rs` (e.g. `mnist_training.rs`, `resnet_demo.rs`, `native_resnet.rs`, `rnn_sequence_prediction.rs`, `hub_import.rs`, `trace_test.rs`, subdirectory crates `backends/`, `cnn/`, `dataloader/`, `matmul/`, `named_tensors/`, `tensors/`)
- Triggers: `cargo run --example <name>` or `cargo run -p <example-crate>`.
- Responsibilities: End-to-end demonstrations of training loops, ONNX import, dataset loading, custom backend usage; also serve as de facto integration tests for the public API surface.

**Build script (codegen entry):**
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

**What happens:** Numerous Python one-off scripts (`fix_creation_dtype.py`, `fix_float_ops.py`, `fix_misc*.py`, `refactor_*.py`, `strip_bounds.py`, `restore_tracing.py`) and log/output files (`check.log`, `check2-5.log`, `check.txt`, `test_proof.rs`, `test_proof.long-type-*.txt`, `expanded.rs`) sit untracked at the repository root.
**Why it's wrong:** They pollute the workspace root, are not part of the crate structure, and risk being accidentally committed or referenced by future automation; `expanded.rs` and `test_proof.rs` look like `cargo expand`/compiler debug dumps rather than source.
**Do this instead:** Move ad hoc refactor scripts to a `scripts/` or `.scratch/` directory excluded via `.gitignore`, and delete generated debug dumps once the refactor is verified.

### Manual `map_err(|e: candle_core::Error| anyhow::anyhow!(e))` repeated on every backend call

**What happens:** In `crates/kindle-backends/src/lib.rs`, nearly every `candle` operation wraps its `Result` with an identical inline closure converting `candle_core::Error` to `anyhow::Error` (seen 50+ times in the file, e.g. lines 164-425).
**Why it's wrong:** High duplication increases the chance of inconsistent error context and makes the file harder to scan; any future change to error wrapping (e.g. adding op name context) requires touching every call site.
**Do this instead:** Introduce a small helper (e.g. `fn cvt<T>(r: std::result::Result<T, candle_core::Error>) -> Result<T>`) or a `From<candle_core::Error> for Error` conversion plus `?`, centralizing the mapping.

## Error Handling

**Strategy:** Single unified `Error` enum (`crates/kindle-core/src/err.rs`) wrapping backend failures via `#[from] anyhow::Error`, with domain-specific variants (`ShapeMismatch`, `OutOfMemory`, `UnsupportedBackendOperation`, `DeviceInitializationError`, `Msg`). A crate-wide `Result<T> = core::result::Result<T, Error>` alias is used pervasively.

**Patterns:**
- Backend implementations (`kindle-backends`) convert library-native errors (`candle_core::Error`) into `anyhow::Error` then into `Error::BackendFailure` via `?`/`.into()`.
- Shape-related failures use the structured `Error::ShapeMismatch { op, expected, got, msg }` variant so callers get both the offending operation name and shape details.
- Serialization errors are surfaced through the `Serializer`/`Deserializer` trait's own `Error: Debug + Display` associated type rather than the top-level `Error` enum, then converted to `Error::ShapeMismatch`-shaped generic messages at the `StateDict::load_from` boundary (`crates/kindle-core/src/nn/module.rs:41-49`) — an inconsistency worth normalizing.

## Cross-Cutting Concerns

**Logging:** No structured logging framework detected (no `tracing`/`log` dependency in any `Cargo.toml`); errors propagate via `Result`/`anyhow` only.

**Validation:** Primarily compile-time via the `shapes/` type system (matmul, conv, broadcast, reshape, concat, stack all have dedicated shape-verification traits); runtime validation exists as a fallback for `Dyn`/`PartialDynShape` cases (e.g. `from_parts` in `crates/kindle-core/src/tensor/base.rs:111-120` checks expected vs. actual dims and returns `Error::ShapeMismatch` if `S: DynShape`).

**Authentication:** Not applicable to core/backends; `kindle-data`'s Hugging Face Hub downloader (`crates/kindle-data/src/hub.rs`) uses the `hf-hub` crate, which handles HF token-based auth internally for gated/private models.

---

*Architecture analysis: 2026-07-09*
