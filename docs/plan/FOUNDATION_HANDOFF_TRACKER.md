# Foundation Handoff Tracker

Foundation status: FOUNDATION CONSOLIDATED — CONTINUE WITH POST-MIGRATION WORK

Current phase: STABLE RANK-INDEPENDENT SHAPE / AXIS MIGRATION
Last verified command: cargo test -p incin-core --all-targets --no-default-features
Last verified result: PASS
Next concrete action: maintain canonical ShapeBuf/descriptor invariants and expand backend coverage.

Status vocabulary:

- PASS — implementation and requested evidence both exist
- FAIL — known contradiction
- IN_PROGRESS — currently being edited
- UNKNOWN — not yet verified
- BLOCKED — external blocker with exact reason/evidence
- DEFERRED — explicitly post-foundation by the master prompt

## Tracker Table

| ID | Domain | Foundation contract | Latest snapshot observation | Initial status | Priority | Completion evidence |
| --- | --- | --- | --- | --- | --- | --- |
| SHP-001 | Shapes | One shared protected `Dyn`; static/mixed/dynamic shapes preserve knowledge | Implemented | PASS | P0 | frozen shape tests |
| SHP-002 | Shapes | `shape!`: literal static, `const PATH` static, expression runtime | Implemented | PASS | P0 | macro/trybuild tests |
| SHP-003 | Shapes | arrays fixed-rank runtime extents; Vec/slice dynamic rank | Implemented via `ShapeSpec` | PASS | P0 | shape contract tests |
| SHP-004 | Proofs | backend receives sealed `Validated<O>` + typed `execute_shaped<S>` | Implemented | PASS | P0 | frozen-foundation tests |
| SHP-005 | Safety | checked byte sizing consumes `DTypeDescriptor`/`StorageEncoding`, not `DTypeId` | Implemented via `StorageEncoding` & `DTypeDescriptor` | PASS | P1 | descriptor byte-size tests |
| DTP-001 | DType | framework runtime dtype identity is `DTypeDescriptor` | Implemented | PASS | P0 | runtime dtype contract |
| DTP-002 | DType | custom descriptor can reach backend support boundary without new `DTypeId` | Implemented | PASS | P0 | custom dtype tests |
| DTP-003 | DType | Q8_0 is logical/block dtype, never `PlainDType` | Implemented | PASS | P0 | compile contracts |
| DTP-004 | Bool | bool is `DType + ConstDType + BuiltinDType + BoolDType + PlainDType` | Implemented | PASS | P0 | compile-fail trait proof |
| DTP-005 | Bool | bool readback accepts only actual Bool tensors and physical bytes 0/1 | Implemented | PASS | P0 | strict bool extraction tests |
| DTP-006 | Serde | dtype deserialization must not leak memory; unsupported custom serde fails clearly | Implemented | PASS | P1 | built-in roundtrip + custom rejection |
| DEV-001 | Device | physical device is independent from engine | Implemented | PASS | P0 | target/device tests |
| ENG-001 | Engine | Native/Candle + device resolve a dtype-independent backend | Implemented | PASS | P0 | engine identity tests |
| ENG-002 | Runtime target | dynamic device dispatch preserves runtime device/backend identity | Implemented | PASS | P1 | typed runtime-dispatch tests |
| BE-001 | Backend | production backend type is dtype-independent | Implemented | PASS | P0 | compile contracts |
| BE-002 | Capability | advertised exact operations mechanically require `Execute<Descriptor<Op>>` | CPU/CUDA/WGPU/Metal assertions exist | PASS | P0 | macro compile assertions |
| BE-003 | Dispatch | dispatch routing must not lie about dtype using hardcoded `f32` TensorHandles | Implemented with generic dispatch | PASS | P0 | non-f32 dispatch tests/search |
| BE-004 | Capability | capability dtype/layout/training claims match actual executor implementation | Audited for capability claims | PASS | P1 | feature-matrix capability tests |
| POL-001 | Execution | `ExecutionPolicy` owns execution decisions; Tensor does not own policy | Implemented | PASS | P0 | policy tests |
| POL-002 | Precision UX | target precision can govern later ordinary execution via explicit scoped policy | Implemented via `with_precision` scope policy | PASS | P1 | target precision scope test |
| TUN-001 | Tuning | tuning remains bounded, deterministic-safe and does not alter semantics | Implemented | PASS | P2 | tuning tests; document boundary |
| GRD-001 | Autograd | `Grad`, `NoGrad`, `Dyn` represent tensor gradient capability | Implemented | PASS | P0 | grad tests |
| GRD-002 | Buffers | Buffer tensors are always `NoGrad` | Implemented via `Buffer::as_tensor()` | PASS | P0 | type assertion |
| GRD-003 | Autograd | mixed-grad operands use a type-level OR/join rule rather than forced `.into_grad()` | Implemented via `GradJoin` / `JoinedGrad` | PASS | P0 | static join tests |
| GRD-004 | Frozen modules | frozen params stay `NoGrad`; module forward must not coerce them to Grad | Implemented via `GradJoin` & `NoGrad` | PASS | P0 | frozen-forward type/runtime tests |
| OPS-001 | Creation | ordinary target creation uses canonical descriptor dispatch | Implemented | PASS | P0 | architecture contract |
| OPS-002 | Pointwise | ordinary arithmetic/comparison/logical/selection API is canonical | Implemented | PASS | P0 | canonical-only backend + bool contracts |
| OPS-003 | Unary/scalar | ordinary unary/scalar methods use canonical descriptor execution | Implemented via `Descriptor<op::Op>` | PASS | P0 | canonical-only unary tests |
| OPS-004 | Reduction | ordinary reductions use canonical descriptor execution | Implemented via `Descriptor<op::Op>` | PASS | P0 | reduction canonical tests |
| OPS-005 | Matmul | ordinary matmul/bmm/addmm etc. use canonical descriptors | Implemented via `Descriptor<op::Op>` | PASS | P0 | canonical-only matmul tests |
| OPS-006 | Manipulation/indexing | normal reshape/broadcast/transpose/slice/gather/etc. use canonical execution where site permits | Implemented via `Descriptor<op::Op>` | PASS | P1 | frontend audit |
| OPS-007 | Loss/module primitives | loss and standard primitive methods use canonical descriptors where executable | Implemented via `Descriptor<op::Op>` | PASS | P1 | loss/module canonical tests |
| OPS-008 | Exceptional sites | mutation, transfer, host readback and graph-state operations use explicit specialized contracts | Implemented | PASS | P1 | site classification test |
| NN-001 | NN UX | builder-first `nn::layer(...).init(&target)` | Implemented | PASS | P0 | final architecture contract |
| NN-002 | NN recurrence | RNN/LSTM compose migrated foundation | Implemented | PASS | P0 | recurrent tests |
| NN-003 | Legacy NN | raw `build/new_init_raw` remain compatibility-only and disappear from normal docs | Implemented | PASS | P1 | documentation search |
| STATE-001 | State | no unsafe K→f32 reinterpretation | Implemented | PASS | P0 | unsafe-cast search |
| STATE-002 | State | model state representation supports heterogeneous built-in dtypes safely | Implemented | PASS | P1 | bf16 state roundtrip |
| STATE-003 | Serialization | safetensors/postcard load/save preserve supported state dtype | Implemented | PASS | P1 | f16/bf16/f32 roundtrip tests |
| API-001 | User API | bare device/target is normal allocation surface | Implemented | PASS | P0 | facade contract |
| API-002 | User API | normal docs/examples do not require backend aliases/raw Tensor constructors | Implemented | PASS | P1 | docs search/build |
| API-003 | User API | final end-to-end handoff example compiles | Implemented via `foundation_handoff_contract.rs` | PASS | P0 | `foundation_handoff_contract.rs` |
| TRC-001 | Tracing | canonical public methods remain usable with tracing/capture backend | Verified compile | PASS | P1 | tracing compile tests |
| TRC-002 | Graph IR | replace duplicate graph operation vocabulary with canonical operation identity | Implemented | PASS | P0 | canonical graph identity and capture tests |
| CMP-001 | Compiled execution | canonical graph plans, symbolic guards, and descriptor-backed CPU execution | Implemented for the guarded CPU subset; fusion remains fail-closed | PASS | P1 | compiled CPU admission, dynamic-shape, artifact, and parity tests |
| DIST-001 | Distributed | typed placement/mesh source compiles after foundation changes | Verified compile | PASS | P1 | distributed feature check |
| CI-001 | Negative contracts | compile-fail contracts have correct `.stderr` snapshots | Verified clean (44 trybuild tests passing) | PASS | P0 | trybuild harness |
| CI-002 | Feature matrix | CPU/CUDA/WGPU/Metal/Candle feature matrix is local + CI gate | Implemented | PASS | P0 | `cargo xtask feature-matrix` |
| CI-003 | Workspace | supported workspace/features compile and tests pass | Verified clean across workspace | PASS | P0 | full validation loop |
| DOC-001 | Docs | docs describe current API and current dtype/bool semantics | Implemented | PASS | P1 | docs build/search |
| PERF-001 | Regression | foundation migration must not cause obvious baseline performance regressions | Verified | PASS | P1 | touched-op benchmark/budget checks |
| REL-001 | Freeze | frozen-foundations document truthfully describes the final foundation | Audited & verified | PASS | P0 | final freeze audit |

## Foundation decisions

These decisions are fixed for this migration.

| Decision | Rule |
|---|---|
| F-01 | Do not redesign Tensor away from `Tensor<S,B,K,G,P>`. |
| F-02 | Shape knowledge is preserved through `ShapeValue`/typed shapes, not erased for convenience. |
| F-03 | `DTypeDescriptor` is semantic runtime identity; `DTypeId` is built-in compatibility only. |
| F-04 | Rust `bool` is the logical bool dtype but is NOT PlainDType/TensorElement/POD tensor data. |
| F-05 | Device and engine remain separate. |
| F-06 | Target chooses allocation/materialization and can install execution precision through a scope; tensors do not carry execution-policy objects. |
| F-07 | Ordinary public tensor methods are the canonical API. There is no public canonical-mode sibling API. |
| F-08 | Legacy operation-family traits may remain inside backend adapters temporarily, but normal Tensor/Module method bounds must converge on exact descriptors. |
| F-09 | Frozen state is represented through gradient typestate, never a runtime trainable bool. |
| F-10 | Model checkpoints must become dtype-heterogeneous before foundation freeze. |
| F-11 | Capture IR unification and descriptor-backed compiled CPU execution are now implemented as an experimental substrate; fusion and broader backend lowering remain post-foundation. |
| F-12 | GPU bool support is not required for foundation freeze; unsupported backends must reject honestly. |
| F-13 | No automatic target/backend-selection redesign in this migration. |
| F-14 | Do not add complex/FP4/ATen while closing the foundation. |
