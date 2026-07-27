#!/usr/bin/env python3
"""Single source for the PROPOSALS.md ledger table and docs/plan/ledger.toml."""

# (id, tier, theme, status, deps, target, deliverable, evidence)
T = [
# --- governance -------------------------------------------------------------
("GOV-001","core","gov","x",[],"PROPOSALS.md",
 "Architecture RFC exists and is internally consistent","test -f PROPOSALS.md"),
("GOV-002","core","gov","x",["GOV-001"],"PROPOSALS.md :: Appendix C",
 "Decision log locks proof, executor, mesh, and compatibility contracts; one entry per resolved contradiction",
 "cargo xtask ledger"),
("GOV-003","core","gov","x",["GOV-002"],"docs/plan/ledger.toml; xtask/src/ledger.rs",
 "Machine-readable task mirror and validator round-trip every ID, dependency, tier, and evidence field",
 "cargo xtask ledger && cargo test -p xtask"),
("GOV-004","core","gov"," ",["GOV-002"],"crates/incin/benches/; docs/plan/baselines/",
 "CPU and GPU capability, performance, and compile-size baselines with environment metadata",
 "cargo bench -p incin -- --save-baseline main"),
("GOV-005","core","gov"," ",["GOV-004"],".github/workflows/ci.yml",
 "Regression budgets and feature inventory enforced in CI",
 "cargo xtask budgets"),
("GOV-006","core","gov","x",["GOV-002"],"crates/incin-backends/src/external/; crates/incin-core/err.rs; */Cargo.toml",
 "Repo hygiene: track and split external/, delete the orphan crates/incin-core/err.rs, rename candle to external-candle with a deprecated alias",
 "cargo check --workspace --features external-candle"),
("GOV-007","core","gov","x",["GOV-002"],".agents/API_DESIGN.md; docs/API_DESIGN.md",
 "Docs source-of-truth consolidation; .agents/API_DESIGN.md becomes a pointer, not a paraphrase",
 "test $(wc -l < .agents/API_DESIGN.md) -lt 10"),
# --- shape ------------------------------------------------------------------
("SHP-001","core","shape","x",["GOV-002"],"docs/audit/shape-proof-inventory.md; tools/audit-shapes.sh",
 "Audit every shape, dtype, backend, and device rule by proof stage; inventory panic, unwrap, overflow, and static-selector gaps",
 "tools/audit-shapes.sh --check"),
("SHP-002","core","shape","x",["SHP-001"],"crates/incin-core/src/shapes/error.rs; crates/incin-core/src/err.rs",
 "Structured ShapeError plus OperationKind, Axis, RankExpectation, DimensionConstraint, with one rendering test per variant",
 "cargo test -p incin-core --test shape_errors"),
("SHP-003","core","shape","x",["SHP-002"],"crates/incin-core/src/shapes/buf.rs",
 "Checked inline ShapeBuf and StrideBuf with checked numel and byte_len, plus property tests",
 "cargo test -p incin-core --test shape_buf"),
("SHP-004","core","shape","x",["SHP-003"],"crates/incin-core/src/shapes/{broadcast,reshape,shape_ops}.rs; crates/incin-core/src/tensor/ops/",
 "Fallible broadcast, reshape, and flatten with no panic or sentinel output; the from_dyn().unwrap() chain is removed",
 "cargo test -p incin-core --test shape_fallible"),
("SHP-005","core","shape","x",["SHP-003"],"crates/incin-core/src/shapes/spatial.rs",
 "Fallible matmul, conv, and pool geometry as a named checked sequence; rejects stride 0 and stops zeroing spatial dims",
 "cargo test -p incin-core --test spatial_geometry"),
("SHP-006","core","shape","x",["SHP-001"],"crates/incin-macros/src/rank.rs; crates/incin-core/src/shapes/",
 "One rank generator behind a single MAX_RANK; closes the ElementCount rank-4 versus Shape rank-8 gap",
 "cargo test -p incin-core --test rank_matrix"),
("SHP-007","core","shape"," ",["SHP-004","SHP-005","SHP-006"],"crates/incin-core/tests/; crates/incin-core/tests/compile_fail/",
 "Close mixed, named, and rank gaps; compile-pass, compile-fail, and fuzz suites",
 "cargo test -p incin-core"),
("SHP-008","core","shape"," ",["SHP-007"],"crates/incin-core/src/tensor/base.rs",
 "Restrict unchecked construction to a witnessed constructor; audit all ~45 obligations and test Flatten diagnostics",
 "cargo test -p incin-core --test construction_witness"),
# --- executor ---------------------------------------------------------------
("EXE-001","core","exec","x",["SHP-003","GOV-002"],"crates/incin-core/src/exec/spec.rs",
 "Freeze the operation taxonomy and descriptor schema; promote OperationFamily to OperationKind rather than duplicating it",
 "cargo test -p incin-core --test descriptor_schema"),
("EXE-002","core","exec","x",["EXE-001"],"crates/incin-core/src/exec/proof.rs",
 "Sealed Validated<O> and proof provenance, with privacy compile-fail tests and the paranoid-validation feature",
 "cargo test -p incin-core --test compile_tests"),
("EXE-003","core","exec","x",["EXE-002"],"crates/incin-core/src/exec/rule.rs",
 "ShapeRule lowering for broadcast, reduction, reshape, matmul, conv, and pool, each restating its frontend trait Output",
 "cargo test -p incin-core --test lowering_parity"),
("EXE-004","core","exec"," ",["SHP-003"],"crates/incin-core/src/exec/meta.rs; crates/incin-backends/src/{cpu,cuda,wgpu}/storage.rs",
 "Normalize TensorMeta and unify the three LayoutClass enums; view, offset, alignment, and bounds tests",
 "cargo test -p incin-backends --test tensor_meta"),
("EXE-005","core","exec"," ",["EXE-001"],"crates/incin-core/src/exec/capability.rs",
 "Capability registry whose generated matrix matches execution tests",
 "cargo test -p incin-backends --test capability_matrix"),
("EXE-006","core","exec"," ",["EXE-002","EXE-005"],"crates/incin-core/src/tensor/backend.rs",
 "Split storage, Execute<O>, and Capabilities out of the 254-method supertrait; give SupportsDType<K> real per-backend impls",
 "cargo test -p incin-core --test compile_tests"),
("EXE-007","core","exec"," ",["EXE-003","EXE-004","EXE-006"],"crates/incin-backends/src/cpu/",
 "Migrate the CPU vertical slice with parity and overhead evidence",
 "cargo test -p incin-backends --no-default-features --features std,cpu"),
("EXE-008","core","exec"," ",["EXE-007"],"crates/incin-backends/src/{cuda,wgpu,external}/; crates/incin-backends/src/dispatch.rs",
 "Migrate CUDA, WGPU, dispatch, and external adapters; replace the F32-hardcoded byte arithmetic with checked dtype.size_bytes()",
 "cargo test -p incin-backends --no-default-features --features std,cpu,wgpu"),
("EXE-009","core","exec"," ",["EXE-008"],"crates/incin-core/src/tensor/backend.rs; crates/incin-backends/src/dispatch.rs",
 "Remove the monolithic adapter and the default unsupported-operation surface",
 "cargo test --workspace"),
("EXE-010","preview","exec"," ",["EXE-008"],"crates/incin-backends/src/external/",
 "external-candle SDK conformance suite and a backend-authoring template",
 "cargo test -p incin-backends --features external-candle --test conformance"),
# --- tuning -----------------------------------------------------------------
("TUN-000","preview","tune","x",[],"crates/incin-backends/src/tuning.rs",
 "Existing CUDA tuner inventoried: 2 warmups, 7 samples, median selection, 1024-entry cache, single-flight coordination",
 "cargo test -p incin-backends --features autotune"),
("TUN-001","preview","tune"," ",["GOV-004"],"crates/incin-backends/src/tuning/identity.rs",
 "Stable device, compiler, and topology identities replacing ordinal plus compute capability; alias tests",
 "cargo test -p incin-backends --features autotune --test tuning_identity"),
("TUN-002","preview","tune"," ",["TUN-001"],"crates/incin-backends/src/tuning/cache.rs",
 "Atomic bounded persistent cache with corruption, schema, and eviction tests",
 "cargo test -p incin-backends --features autotune --test tuning_cache"),
("TUN-003","preview","tune"," ",["EXE-005","TUN-002"],"crates/incin-backends/src/tuning/service.rs",
 "General disabled, heuristic, coordinated-warmup, and profile-guided tuning service",
 "cargo test -p incin-backends --features autotune --test tuning_service"),
("TUN-004","preview","tune"," ",["EXE-003","TUN-003"],"crates/incin-backends/src/tuning/signature.rs",
 "Shape and layout driven legal-candidate pruning; extends KernelKey rather than adding a parallel KernelSignature",
 "cargo test -p incin-backends --features autotune --test tuning_pruning"),
("TUN-005","preview","tune"," ",["TUN-004"],"crates/incin-backends/src/cuda/ops/",
 "Pointwise, reduction, and normalization CUDA tuning parity",
 "cargo test -p incin-backends --features cuda,autotune  # CUDA hardware"),
("TUN-006","preview","tune"," ",["TUN-005"],"crates/incin-backends/src/cuda/ops/{matmul,conv}.rs",
 "GEMM and convolution library-versus-native tuning with a crossover report",
 "cargo test -p incin-backends --features cuda,autotune  # CUDA hardware"),
("TUN-007","preview","tune"," ",["TUN-003"],"crates/incin-backends/src/tuning/telemetry.rs",
 "Tuning telemetry, provenance, and explain output",
 "cargo test -p incin-backends --features autotune,telemetry"),
("TUN-008","preview","tune"," ",["TUN-006","GOV-005"],".github/workflows/ci.yml",
 "Time, memory, and cache budgets with a no-regression gate",
 "cargo xtask budgets"),
# --- performance ------------------------------------------------------------
("PRF-001","core","perf"," ",["EXE-003","EXE-004"],"crates/incin-backends/src/iteration.rs",
 "Remove repeated hot-path metadata allocation; latency and allocation evidence",
 "cargo bench -p incin -- eager"),
("PRF-002","core","perf"," ",["EXE-007"],"crates/incin-backends/src/cpu/ops/matmul.rs",
 "CPU iteration plans, batched GEMM, optional cpu-blas, and isolated bare-CPU tests",
 "cargo test -p incin-backends --no-default-features --features std,cpu"),
("PRF-003","preview","perf"," ",["EXE-008","TUN-005"],"crates/incin-backends/src/{cuda,wgpu}/",
 "CUDA descriptor launches and WGPU specialization; hardware and sanitizer evidence",
 "cargo test -p incin-backends --features cuda  # CUDA hardware"),
("PRF-004","preview","perf"," ",["TUN-006"],"crates/incin-backends/src/cuda/ops/",
 "Vendor-versus-native selection behind cuda-vendor with numerical and crossover reports",
 "cargo test -p incin-backends --features cuda,cuda-vendor  # CUDA hardware"),
# --- metal ------------------------------------------------------------------
("MTL-001","preview","metal"," ",["EXE-005","EXE-008"],"crates/incin-backends/src/metal/",
 "Native Metal feature, device capabilities, storage modes, and unified-memory guards",
 "cargo test -p incin-backends --features metal  # Apple Silicon"),
("MTL-002","preview","metal"," ",["MTL-001","EXE-003"],"crates/incin-backends/src/metal/shaders/",
 "Generated MSL pointwise and reduction descriptors with parity tests",
 "cargo test -p incin-backends --features metal  # Apple Silicon"),
("MTL-003","preview","metal"," ",["MTL-001","PRF-004"],"crates/incin-backends/src/metal/mps.rs",
 "MPS and MPSGraph structured candidates with explicit native fallback",
 "cargo test -p incin-backends --features metal-mps  # Apple Silicon"),
("MTL-004","preview","metal"," ",["MTL-002","MTL-003","GRD-003"],"crates/incin-backends/src/metal/",
 "Metal forward and backward hardware parity with no hidden readback",
 "cargo test -p incin-backends --features metal  # Apple Silicon"),
("MTL-005","preview","metal"," ",["MTL-003","TUN-003"],"crates/incin-backends/src/metal/tuning.rs",
 "Metal kernel and storage-mode autotuning with a fingerprinted cache",
 "cargo test -p incin-backends --features metal  # Apple Silicon"),
("MTL-006","preview","metal"," ",["MTL-004","MTL-005"],"docs/; README.md",
 "Apple Silicon UX, docs, and laptop plus desktop hardware baselines",
 "cargo bench -p incin --features metal  # Apple Silicon"),
# --- compiled ---------------------------------------------------------------
("CMP-001","preview","compile"," ",["EXE-009"],"crates/incin-core/src/compiled/capture.rs",
 "Capture the eager graph into validated IR with descriptor parity",
 "cargo test -p incin-core --test compiled_capture"),
("CMP-002","preview","compile"," ",["CMP-001"],"crates/incin-core/src/compiled/plan.rs",
 "Immutable compiled plans and dynamic guards",
 "cargo test -p incin-core --test compiled_guards"),
("CMP-003","preview","compile"," ",["CMP-002","PRF-001"],"crates/incin-core/src/compiled/alloc.rs",
 "Liveness and allocation planner with alias and peak-memory tests",
 "cargo test -p incin-core --test compiled_alloc"),
("CMP-004","preview","compile"," ",["CMP-002"],"crates/incin-core/src/compiled/fold.rs",
 "Constant folding, weight prepacking, and bounded shape buckets",
 "cargo test -p incin-core --test compiled_fold"),
("CMP-005","preview","compile"," ",["CMP-003","CMP-004"],"crates/incin-core/src/compiled/fusion.rs",
 "Safe fusion and backward hooks; gradient parity and launch-count reduction",
 "cargo test -p incin-core --test compiled_fusion"),
("CMP-006","preview","compile"," ",["CMP-005"],"crates/incin-core/src/compiled/artifact.rs",
 "Versioned compiled artifacts with compatibility and corruption tests",
 "cargo test -p incin-core --test compiled_artifact"),
# --- autograd ---------------------------------------------------------------
("GRD-001","core","grad"," ",["EXE-006"],"crates/incin-core/src/exec/context.rs",
 "Explicit ExecutionContext with nested and concurrent tests",
 "cargo test -p incin-core --test exec_context"),
("GRD-002","core","grad"," ",["GRD-001"],"crates/incin-core/src/exec/context.rs; crates/incin-core/src/tensor/grad.rs",
 "G to GradMode propagation; NoGrad records zero nodes and saves nothing",
 "cargo test -p incin-core --test nograd_records_nothing"),
("GRD-003","core","grad"," ",["GRD-001"],"crates/incin-core/src/exec/tape.rs",
 "Backend-neutral tape nodes with CPU parity",
 "cargo test -p incin-backends --no-default-features --features std,cpu --test gradient_parity"),
("GRD-004","core","grad"," ",["GRD-003","EXE-008"],"crates/incin-backends/src/{cuda,wgpu}/",
 "CUDA and WGPU gradient recipes with hardware parity",
 "cargo test -p incin-backends --features wgpu --test gradient_parity"),
("GRD-005","core","grad"," ",["GRD-003"],"crates/incin-core/src/exec/tape.rs",
 "Structured backward and NaN failures; no expected-failure panic paths",
 "cargo test -p incin-core --test backward_errors"),
("GRD-006","core","grad"," ",["GRD-004"],"crates/incin-backends/src/{cpu,cuda,wgpu}/tape.rs",
 "Saved-tensor lifetime owned by the graph; delete all three backend-local tapes",
 "cargo test --workspace"),
("GRD-007","preview","grad"," ",["GRD-006","CMP-003"],"crates/incin-core/src/compiled/alloc.rs",
 "Compiled-graph saved-tensor liveness and fusion integration",
 "cargo test -p incin-core --test compiled_alloc"),
# --- distributed ------------------------------------------------------------
("DST-001","preview","dist"," ",["GOV-002","SHP-007"],"crates/incin-core/src/dist/mesh.rs",
 "Typed meshes and ValidMesh; valid and invalid world-size compile tests",
 "cargo test -p incin-core --test mesh_compile"),
("DST-002","preview","dist"," ",["DST-001","EXE-005"],"crates/incin-core/src/dist/mesh.rs",
 "Physical binding, topology fingerprint, and runtime guards",
 "cargo test -p incin-core --test mesh_bind"),
("DST-003","preview","dist"," ",["DST-001","EXE-002"],"crates/incin-core/src/dist/placement.rs; crates/incin-core/src/dist/rule.rs",
 "Placement typestates, PlacementKind, and rules; divisibility and transition compile tests; ValidatedDistributed sealed like Validated",
 "cargo test -p incin-core --test placement_rules"),
("DST-004","preview","dist"," ",["DST-003","EXE-004"],"crates/incin-core/src/tensor/base.rs",
 "Unified Tensor global and local metadata with reshard invariants",
 "cargo test -p incin-core --test placement_tensor"),
("DST-005","preview","dist"," ",["DST-002"],"crates/incin-backends/src/dist/reference.rs",
 "Deterministic CPU reference collectives and their adjoints",
 "cargo test -p incin-backends --features distributed-reference"),
("DST-006","preview","dist"," ",["DST-002","GOV-004"],"crates/incin-backends/src/dist/nccl.rs",
 "Optional NCCL transport; three-GPU order, count, and failure tests",
 "cargo test -p incin-backends --features distributed-nccl  # 3x CUDA"),
("DST-007","preview","dist"," ",["DST-003","DST-005"],"crates/incin-core/src/dist/plan.rs",
 "Collective plans and sequence tokens; divergent-plan preflight test",
 "cargo test -p incin-core --test collective_plan"),
("DST-008","preview","dist"," ",["DST-006","DST-007","GRD-004"],"crates/incin-core/src/dist/",
 "DP=3 training with single-GPU numerical and gradient parity",
 "cargo test -p incin --features distributed-nccl --test dp3  # 3x CUDA"),
("DST-009","preview","dist"," ",["DST-004","DST-006","DST-007"],"crates/incin-core/src/nn/linear.rs",
 "TP=3 column and row linear plus attention parity",
 "cargo test -p incin --features distributed-nccl --test tp3  # 3x CUDA"),
("DST-010","preview","dist"," ",["CMP-002","DST-006","DST-007"],"crates/incin-core/src/dist/pipeline.rs",
 "GPipe then 1F1B PP=3; parity, bubble, and deadlock evidence",
 "cargo test -p incin --features distributed-nccl --test pp3  # 3x CUDA"),
("DST-011","preview","dist"," ",["DST-008","DST-009","DST-010"],"crates/incin-core/src/dist/plan.rs",
 "Hybrid planner and report with feasibility and memory evidence",
 "cargo test -p incin-core --test hybrid_plan"),
("DST-012","preview","dist"," ",["TUN-003","DST-006"],"crates/incin-backends/src/dist/tuning.rs",
 "Coordinated collective tuning; maximum-rank objective and all-rank commit tests",
 "cargo test -p incin-backends --features distributed-nccl  # 3x CUDA"),
("DST-013","preview","dist"," ",["CMP-004","DST-011","DST-012"],"crates/incin-core/src/compiled/tuning.rs",
 "Bounded plan tuning measured against a one-GPU baseline",
 "cargo test -p incin-core --test plan_tuning"),
("DST-014","exploratory","dist"," ",["CMP-003","GRD-007","DST-008"],"crates/incin-core/src/dist/fsdp.rs",
 "FSDP and ZeRO prototype with persistent and transient memory parity",
 "cargo test -p incin --features distributed-nccl --test fsdp  # 3x CUDA"),
("DST-015","preview","dist"," ",["DST-011"],"crates/incin-core/src/dist/context.rs",
 "Multi-process rendezvous and launcher with timeout and shutdown tests",
 "cargo test -p incin --features distributed-nccl --test rendezvous"),
("DST-016","preview","dist"," ",["DST-011","DST-015"],"crates/incin-core/src/nn/save.rs",
 "Global checkpoint manifest and explicit cross-mesh resharded load",
 "cargo test -p incin-core --test checkpoint_reshard"),
# --- UX ---------------------------------------------------------------------
("UX-001","preview","ux"," ",["EXE-005"],"crates/incin/src/train.rs",
 "Automatic Trainer; an unchanged model runs on CPU and on three GPUs",
 "cargo test -p incin --test trainer"),
("UX-002","preview","ux"," ",["DST-001"],"crates/incin-macros/src/mesh.rs",
 "mesh! with expansion, hygiene, span, and compile-fail tests",
 "cargo test -p incin-macros --test mesh_macro"),
("UX-003","preview","ux"," ",["DST-003"],"crates/incin-macros/src/placement.rs",
 "placement! grammar and operation-bound diagnostics",
 "cargo test -p incin-macros --test placement_macro"),
("UX-004","preview","ux"," ",["DST-003"],"crates/incin-macros/src/module.rs",
 "#[parallel] and #[shard] template and conflict tests",
 "cargo test -p incin-macros --test parallel_attrs"),
("UX-005","preview","ux"," ",["DST-011","UX-001"],"crates/incin/src/bin/cargo-incin.rs",
 ".explain() and cargo incin plan with golden text and JSON reports",
 "cargo test -p incin --test plan_report"),
("UX-006","preview","ux"," ",["TUN-007","UX-005"],"crates/incin/src/bin/cargo-incin.rs",
 "cargo incin tune with an offline and stale-cache round trip",
 "cargo test -p incin --test tune_cli"),
("UX-007","preview","ux"," ",["DST-015"],"crates/incin-macros/src/distributed_main.rs",
 "Launcher and #[distributed_main] with shutdown and error tests",
 "cargo test -p incin-macros --test distributed_main"),
("UX-008","preview","ux"," ",["CMP-002","DST-011"],"crates/incin-core/src/compiled/manifest.rs",
 "Reproducibility manifest replay and incompatibility diffs",
 "cargo test -p incin-core --test manifest_replay"),
("UX-009","preview","ux"," ",["SHP-007"],"crates/incin-macros/src/axes.rs",
 "Named axes! with ambiguous and missing-axis diagnostics",
 "cargo test -p incin-macros --test axes_macro"),
("UX-010","exploratory","ux"," ",["EXE-003","DST-003"],"crates/incin-macros/src/einsum.rs",
 "Typed einsum! with parser, shape, placement, and parity tests. Requires a recorded justification before starting",
 "cargo test -p incin-macros --test einsum_macro"),
("UX-011","exploratory","ux"," ",["DST-004"],"crates/incin-macros/src/parallel_block.rs",
 "Evaluate parallel!; implement only with recorded usability evidence",
 "cargo test -p incin-macros --test parallel_block"),
("UX-012","preview","ux"," ",["UX-005","UX-006"],"crates/incin-viz/src/",
 "Visualize placement, memory, timeline, and critical path",
 "cargo test -p incin-viz"),
("UX-013","core","ux"," ",["GOV-005","EXE-005","UX-014"],"docs/; README.md",
 "Feature and capability documentation generated from tested registrations, with compiled examples",
 "cargo test --workspace --doc"),
("UX-014","core","ux"," ",["EXE-005"],"crates/incin/src/bin/cargo-incin.rs",
 "cargo incin doctor with stable text and JSON output and mocked hardware tests",
 "cargo test -p incin --test doctor"),
("UX-015","preview","ux"," ",["EXE-005","GRD-001"],"crates/incin-core/src/exec/precision.rs",
 "PrecisionPolicy and loss scaling extending the existing DTypePolicy; mixed-precision parity tests",
 "cargo test -p incin-core --test precision_policy"),
# --- CI ---------------------------------------------------------------------
("CI-001","core","ci"," ",["GOV-005","GOV-003"],".github/workflows/ci.yml",
 "Feature-powerset CI preserving the bare CPU default; adds cargo doc and drops blanket package exclusions",
 "act -j powerset  # or CI run"),
("CI-002","core","ci"," ",["EXE-008"],".github/workflows/hardware.yml",
 "Scheduled CUDA and WGPU hardware matrix",
 "gh workflow run hardware.yml"),
("CI-003","preview","ci"," ",["DST-008","DST-009","DST-010"],".github/workflows/hardware.yml",
 "Homogeneous three-GPU DP, TP, and PP CI",
 "gh workflow run hardware.yml -f job=dist3"),
("CI-004","preview","ci"," ",["DST-015"],".github/workflows/hardware.yml",
 "Multi-process and multi-node CI with topology metadata",
 "gh workflow run hardware.yml -f job=multinode"),
("CI-005","core","ci"," ",["GOV-005"],"crates/incin-macros/tests/",
 "Macro trybuild, rustfmt, rename, and hygiene suite for the existing s!, idx!, and #[module]",
 "cargo test -p incin-macros"),
("CI-006","preview","ci"," ",["GOV-005","TUN-008","DST-013"],".github/workflows/ci.yml",
 "CPU, GPU, and distributed performance and cache gates",
 "cargo xtask budgets"),
("CI-007","preview","ci"," ",["MTL-004"],".github/workflows/hardware.yml",
 "Scheduled Apple Silicon Metal hardware matrix",
 "gh workflow run hardware.yml -f job=metal"),
("CI-008","preview","ci"," ",["UX-002","UX-003","UX-004"],"crates/incin-macros/tests/",
 "Distributed macro trybuild suite for mesh!, placement!, #[parallel], and #[shard]",
 "cargo test -p incin-macros --features distributed"),
# --- release ----------------------------------------------------------------
("REL-001","core","release"," ",["SHP-008","EXE-009","GRD-006","CI-001","GOV-006","GOV-007"],"CHANGELOG.md; docs/MIGRATION.md",
 "Core stabilization review and migration guide",
 "cargo test --workspace && cargo doc --workspace --no-deps"),
("REL-002","core","release"," ",["REL-001","CI-002","CI-005","UX-013","UX-014","PRF-002","GRD-002","GRD-005"],"CHANGELOG.md",
 "Single-device release-readiness evidence; the deprecated candle alias is removed here",
 "cargo test --workspace --all-features"),
("REL-003","preview","release"," ",["REL-002","CI-003","CI-006","CI-007","CI-008","UX-005","UX-007","UX-008","UX-009","UX-012","UX-015","DST-011","EXE-010","CMP-006","MTL-006","PRF-003","PRF-004","TUN-008","GRD-007"],"CHANGELOG.md",
 "Distributed preview readiness and the fail-stop contract",
 "gh workflow run hardware.yml"),
("REL-004","preview","release"," ",["REL-003","CI-004","DST-016"],"CHANGELOG.md",
 "Multi-node preview scope and recovery limits published",
 "gh workflow run hardware.yml -f job=multinode"),
]

# id -> (date, evidence output). Rule: no task may be "x" without an entry here.
COMPLETED = {
 "GOV-001": ("2026-07-27", "PROPOSALS.md present; internal-consistency script reports 0 undefined deps, 0 cycles, 0 tier violations"),
 "GOV-002": ("2026-07-27", "PROPOSALS.md Appendix C records D-001..D-014; every contradiction in the review maps to exactly one entry"),
 "GOV-003": ("2026-07-27", "cargo xtask ledger -> ok: 100 tasks, 5 complete; 5 unit tests pass; fault injection confirms it fails on unknown dependency, tier violation, table/mirror divergence, and completed-without-evidence"),
 "GOV-006": ("2026-07-27", "external/ tracked and split into candle/{mod,convert,backend,ops/*} (13 files, max 248 lines, was one 1359-line file); orphan crates/incin-core/err.rs removed; candle -> external-candle with alias. cargo check passes for external-candle and the candle alias; fmt/clippy/no_std/cuda-compile clean; 528 tests pass"),
 "GOV-007": ("2026-07-27", ".agents/API_DESIGN.md reduced to a 10-line pointer; its one unique rule merged into docs/API_DESIGN.md as item 3"),
 "TUN-000": ("2026-07-27", "tuning.rs verified: 2 warmups (line 24), 7 samples (line 26), median select (220-222), 1024-entry cache (line 22), single-flight coordinator (284-296)"),
 "SHP-001": ("2026-07-27", "tools/audit-shapes.sh --check -> ok; docs/audit/shape-proof-inventory.md classifies 19 rules by proof stage (T/L/B/N/U) and records the baseline: 28 from_dyn().unwrap(), 2 from_size().unwrap(), 4 Default::default() spatial zeroing, 9 rules short of rank 8 and 1 over, SupportsDType blanket default proving nothing (backend.rs:61). Confirmed defect for SHP-005: Pool2dShape on (U1,U1,usize,usize) with 8x8/k2/s2/p0/d1 returns spatial dims (0,0) instead of (4,4) -- a wrong shape that propagates, not a panic. Fault injection confirms --check fails on both a doctored count in the document and a new unwrap in the source tree"),
 "SHP-004": ("2026-07-27", "cargo test -p incin-core --test shape_fallible -> 13 passed; 0 failed; full workspace 733 passed, 0 failed. The from_dyn().unwrap() chain is at 0, down from the corrected baseline of 39 live sites across 11 files (tools/audit-shapes.sh --check). The shapes surface fell from 13 unwraps to 1 and from 1 panic!-class site to 0. BroadcastShape::output_shape and MatMulShape::output_shape are now fallible; shapes/shape.rs adds field_from_dims() and error.rs adds ShapeError::TargetShapeRejected for the residual generic cases. checked_broadcast_dim converted from assert! to Result per decision D-013, and the named_dims.rs test that asserted the panic now asserts the error and its axis. Three latent wrong-answer bugs found and fixed while converting: (1) broadcast used lhs.max(rhs), so a size-1 axis against a size-0 one yielded 1 instead of 0; (2) Dyn matmul returned vec![] -- the scalar shape -- as a sentinel for every unmatched rank combination, including [m,k] x [k], whose answer is [m] not a scalar; (3) no MatMulShape impl ever checked the contracted dimension, so a disagreeing K produced a confidently wrong output shape (now DimensionMismatch on axis 'k'). The rank-4 x rank-2 'flattened batch' convention is preserved deliberately and pinned by a test. broadcast_dims' unreachable!() arm is gone: a missing axis is now an implicit 1, which is the actual right-alignment rule. no-default-features check clean"),
 "SHP-006": ("2026-07-27", "cargo test -p incin-core --test rank_matrix -> 13 passed; 0 failed; full workspace 746 passed, 0 failed. MAX_RANK now lives only in crates/incin-macros/src/rank.rs and is re-exported as incin_core::shapes::MAX_RANK (a proc-macro crate cannot export a const, and a second copy would reintroduce the drift). rank_sweep! generates every ladder from it, with nine forms covering the argument shapes the existing macro_rules families accept, including two-dimensional rank-pair sweeps for broadcast prepend and rank x axis sweeps for concat and stack; a `max` above MAX_RANK is rejected so no rule can set its own ceiling. All 19 rules now sit at their correct ceiling: 17 rank-preserving at 8, and AppendDim/StackShape at 7 because their Output gains an axis and is bounded by Shape. Nothing sits above the ceiling. Closes the RFC's motivating gap, ElementCount 4->8. Notable finds: ReplaceLastDim generated impls to rank 12, four ranks at which no tuple implements Shape at all, so they could never be selected; HasChannels1D and HasChannels2D each held for exactly ONE rank, so (C, L) and (C, H, W) -- the unbatched forms their own docs name as valid -- did not implement them. 48 hand-written per-axis impls in concat.rs and stack.rs replaced by two macros. Recorded cost: raising a marker trait from one impl to eight loses rustc's 'but trait X is implemented for it' hint, so those diagnostics are less specific; the on_unimplemented note still carries the guidance and the compile failure is unchanged. 11 compile_fail expectations regenerated, every case still errors. The audit tool needed three corrections to measure this honestly, all documented in the inventory. Compile time unchanged (incin-core check ~1.4s). no-default-features clean"),
 "SHP-005": ("2026-07-27", "cargo test -p incin-core --test spatial_geometry -> 17 passed; 0 failed; full workspace 709 passed, 0 failed. spatial.rs gains spatial_out_size(), a named checked sequence for (in + 2p - d*(k-1) - 1)/s + 1: it rejects a 0 stride/kernel/dilation by parameter name, reports a kernel that does not fit its padded input as EmptyOutput instead of underflowing the subtraction, and names each overflowing term individually ('2 * padding', 'input + 2 * padding', 'dilation * (kernel - 1)'). compute_output_shape on Pool2dShape, SpatialConv1d, SpatialConv2d, and AdaptiveAvgPool2dShape now returns Result<Field, ShapeError>; 10 call sites take ?. Confirmed regression closed: pool2d on (U1,U1,usize,usize) with 8x8/k2/s2/p0/d1 returns (4,4), was (0,0). Two further silent defects found and fixed while here: the Dyn conv/pool rules tested len()==4 (or ==3) and returned the input shape unchanged for every other rank, so rank-3 (C,H,W) was never pooled and an unsupported rank was never reported; both accepted ranks are now handled and others report RankMismatch. from_size().unwrap() 2->0 and Default::default() in spatial output 4->0, both verified by tools/audit-shapes.sh --check. The audit's rank scan was corrected twice in the process (variadic macros with fixed trailing tuple elements were undercounted: SpatialConv1d 5->7, SpatialConv2d 4->7; and the repetition-strip pattern missed the $(..)+ form, inflating BroadcastShape 4->6); the SHP-001 exit table also said 9 rules short of rank 8 while its own table listed 14, now corrected. A third measurement fix re-baselined SHP-004: the from_dyn().unwrap() chain was counted with a regex that stops at the first inner ')', so it never matched from_dyn(&broadcast_dims::<Self, (A, B)>(lhs, rhs)).unwrap() -- 14 of the sites were invisible. Counted by balancing parentheses the true figure is 39 live sites across 11 files, not 28"),
 "SHP-003": ("2026-07-27", "cargo test -p incin-core --test shape_buf -> 17 passed; 0 failed. shapes/buf.rs adds InlineOrHeap (inline to INLINE_RANK=8, spills to Vec), ShapeBuf, and StrideBuf. checked_numel/checked_byte_len/contiguous_for/checked_span all use checked_mul and return ShapeError::ArithmeticOverflow naming the failing term; no derived value is cached. Properties are checked against u128 references over 5000 cases each from a fixed-seed generator biased to 0/1/usize::MAX/2^32, with assertions that both the overflow and non-overflow branches were reached. The suite found a real ordering bug in the first implementation: a left-to-right fold made numel([MAX,0,MAX]) = Ok(0) but numel([MAX,MAX,0]) = Err, so the zero case is now short-circuited and order independence is pinned by numel_does_not_depend_on_axis_order (reversal plus rotation). Replaces the panicking cpu::stride::contiguous_strides; EXE-004 migrates the storages. Workspace check, no-default-features check, and cargo test -p incin-core all clean"),
 "EXE-001": ("2026-07-27", "cargo test -p incin-core --test descriptor_schema -> 42 passed; 0 failed; full workspace 0 failures. crates/incin-core/src/exec/{mod,spec}.rs adds DescriptorSchemaVersion (pinned at 1 by test), AxisMask, the sealed OperationSpec trait that binds each descriptor to one OperationKind, and the four descriptors PROPOSALS.md 1.2.1 names: BroadcastSpec, MatMulSpec, ReductionSpec, Conv2dSpec. Every descriptor is #[non_exhaustive] with pub fields -- readable by any backend, constructible only through the checked constructors -- and every field a constructor did not receive is DERIVED from the ones it did: broadcast masks from strides, output shapes from operands, outer/reduced/inner from the input, h_out/w_out via SHP-005's spatial_out_size() rather than a second copy of the formula. Descriptors hold logical geometry only; storage offset, dtype, device, and alignment stay for TensorMeta (EXE-004), which is what lets one descriptor be reused as a cache key. OperationFamily deleted per D-008: incin-backends/src/dtype_policy.rs re-exports incin_core OperationKind and folds through the new OperationKind::family(), which maps the accumulating ops (matmul, conv, pool) onto Reduction and the reindexing ops (every shape manipulation, plus embedding) onto Storage; family() is total and idempotent, asserted over all 23 variants. Design finding the test suite forced: reducing away the only zero axis of [MAX,0,MAX] leaves [MAX,MAX], an output whose element count overflows usize, so all four constructors now reject an unrepresentable output at resolution rather than handing a backend a descriptor it cannot launch. Two SHP loose ends closed while here: INLINE_RANK is now MAX_RANK rather than a literal 8 (the SHP-006 comment promised this), and StrideBuf gained push/pop to match ShapeBuf, needed by the stride-normalizing loops. rustdoc, no-default-features, and tools/audit-shapes.sh --check all clean"),
 "SHP-002": ("2026-07-27", "cargo test -p incin-core --test shape_errors -> 14 passed; 0 failed. shapes/error.rs adds ShapeError (6 variants), OperationKind (23 variants, superset of incin-backends OperationFamily so EXE-001 can delete it per D-008), Axis, RankExpectation, DimensionConstraint. One exact-string rendering test per ShapeError variant plus one per component variant; Error::Shape(#[from] ShapeError) added to err.rs and its ? path tested. Every type is Copy and allocation-free (&'static str + usize only), asserted by test and verified by cargo check -p incin-core --no-default-features. Full workspace check and cargo test -p incin-core clean"),
 "EXE-002": ("2026-07-27", "cargo test -p incin-core --test compile_tests -> 1 passed (55 trybuild cases, 3 new); --test proof_provenance -> 12 passed; --lib exec::proof -> 9 passed, 10 with --features paranoid-validation; full workspace 0 failures. crates/incin-core/src/exec/proof.rs adds ProofLevel (Static/Mixed/Dynamic) and Validated<O>, whose fields are private and whose new() is pub(crate) -- the seal PROPOSALS.md 1.2.1 specifies. Three compile_fail cases prove it from outside the crate and each fails for the right reason, checked by reading the generated .stderr: E0624 on Validated::new, E0451 on a struct literal naming both private fields, E0277 on the Sealed bound when a foreign type tries to join the descriptor taxonomy. The second case matters on its own -- blocking the constructor without private fields would leave a struct literal as an open forgery path. ProofLevel is a meet semilattice topped by Static: meet() takes the weaker operand, and commutativity, idempotence and associativity are asserted over all 27 triples so folding N operands is order-independent. The paranoid-validation feature gates Validated::audit(), which re-derives the constructor obligation EXE-001 established (a representable output), plus a paranoid_audit! macro that expands to nothing when the feature is off so an executor can call it on a hot path. Provenance is derived, never asserted: a caller who could stamp Static on a runtime shape would hand kernels a constant that does not exist"),
 "EXE-003": ("2026-07-27", "cargo test -p incin-core --test lowering_parity -> 23 passed; --test compile_tests -> 1 passed (56 trybuild cases, 1 new); full workspace 74 test binaries, 0 failures; --no-default-features clean; --features paranoid-validation clean; RUSTDOCFLAGS=-D warnings cargo doc clean. crates/incin-core/src/exec/rule.rs adds ShapeRule and the six lowering rules the task names: BroadcastRule, MatMulRule, ReduceRule<D>/ReduceKeepRule<D>, ReshapeRule, Conv2dRule and Pool2dRule. Each restates the Output its frontend trait already computes, which is decision D-007 discharged at the type level; the runtime half is that every rule computes its output twice and compares. Where the frontend trait computes a Field -- BroadcastShape, MatMulShape, SpatialConv2d, Pool2dShape -- its answer is checked against the descriptor axis by axis. Where the trait only names a type -- ReduceDim, ReduceKeepDim -- the descriptor dimensions are rebuilt into that type Field via field_from_dims, which fails on a rank the type does not have and on a statically fixed axis that disagrees. This is the first caller of Validated::new, which EXE-002 left with none. Descriptor is bounded by the sealed OperationSpec, so an outside implementation of ShapeRule can neither invent a descriptor nor wrap one it did not obtain from a rule here; that is why ShapeRule itself is not sealed, and the note says so. A new compile_fail case pins the frontend binding: lowering (U3,U4) against (U3,U5) fails with the BroadcastShape on_unimplemented diagnostic plus a required for BroadcastRule to implement ShapeRule note, verified by reading the generated .stderr")
}

# Where the implementation departed from PROPOSALS.md, and why. Recorded per
# task so `deviations` in the TOML mirror is a real field rather than a
# placeholder -- an unrecorded deviation is how an RFC quietly stops describing
# the code.
DEVIATIONS = {
 "EXE-003": [
   "The RFC signature is lower(inputs: &Inputs, args), but its own worked example passes &(L::Field, R::Field) while writing ShapeRule<(L, R)>. A tuple of shapes is not itself a Shape, so &Inputs cannot name the runtime half; the trait gains an associated type Operands, which each impl sets to the operand fields. No new nameable type is introduced.",
   "Pooling and reshape had no descriptor: 4 of the ledger 6 operations. Added Pool2dSpec and ReshapeSpec to exec/spec.rs, outside the task stated target file, with Appendix A rows and decision D-018. Reusing Conv2dSpec for pooling as a depthwise convolution gives the right geometry and the wrong OperationKind, which every capability query and kernel cache keyed on it would then answer wrongly.",
   "Rules lower dense row-major operands and take no strides. Operand layout is a per-tensor fact owned by TensorMeta (EXE-004), so Args is left as the place it arrives; MatMulSpec::transposed is applied after lowering for the same reason.",
   "BroadcastShape has no same-rank implementation that stretches a U1, so the obvious (U3,U4) against (U1,U4) pair does not typecheck and the tests lower (U3,U4) against (U4,) instead. The gap is SHP-007 scope and is recorded in the test that works around it.",
 ],
 "EXE-002": [
   "PROPOSALS.md 1.2.1 sketches ProofLevel::of::<L, R>() over two operands. Implemented as of::<S>() over one, combined with meet(), because Conv2d lowers three operands (input, weight, bias) and the binary form has nowhere to put the third. The RFC's call is of::<L>().meet(of::<R>()); a test folds the three-operand conv case and asserts it agrees with the pairwise answer.",
   "Deriving the level from the shape types required extending two types outside the task's stated target file: Dim gains STATIC_SIZE (defaulted false) and Shape gains PROOF (defaulted Dynamic). Both defaults are the conservative answer, so a Dim or Shape implemented outside the crate is credited with no proof it has not shown. A lowering rule is generic over its shapes and cannot inspect their axes, so without a per-type const there is no way to compute a proof level at all -- adding a new marker trait instead was rejected as it would need an Appendix A row for a type the design does not otherwise name.",
 ],
}

MATURITY = {"core": 0, "preview": 1, "exploratory": 2}
STATUS_WORD = {" ": "planned", "x": "complete", "~": "active", "!": "blocked", "-": "deferred"}


def validate():
    ids = [t[0] for t in T]
    errs = []
    assert len(ids) == len(set(ids)), "duplicate id"
    idset = set(ids)
    by = {t[0]: t for t in T}
    for t in T:
        for d in t[4]:
            if d not in idset:
                errs.append(f"{t[0]} -> unknown dep {d}")
            elif MATURITY[by[d][1]] > MATURITY[t[1]]:
                errs.append(f"TIER: {t[0]} ({t[1]}) depends on {d} ({by[d][1]})")
    # cycles
    state = {}
    def dfs(n, stack):
        if state.get(n) == 1:
            errs.append(f"CYCLE: {' -> '.join(stack + [n])}")
            return
        if state.get(n) == 2:
            return
        state[n] = 1
        for d in by[n][4]:
            if d in by:
                dfs(d, stack + [n])
        state[n] = 2
    for i in ids:
        dfs(i, [])
    for t in T:
        if t[3] == "x" and t[0] not in COMPLETED:
            errs.append(f"EVIDENCE: {t[0]} is [x] but has no COMPLETED entry")
    referenced = {d for t in T for d in t[4]}
    terminals = [i for i in ids if i not in referenced]
    return errs, terminals


def md():
    out = ["| ID | Tier | Theme | Status | Dependencies | Target crate::module | Deliverable | Evidence |",
           "|---|---|---|---|---|---|---|---|"]
    for tid, tier, theme, st, deps, target, deliv, ev in T:
        d = ",".join(deps) if deps else "—"
        out.append(f"| {tid} | {tier} | {theme} | [{st}] | {d} | `{target}` | {deliv} | `{ev}` |")
    return "\n".join(out)


def tstr(s):
    """Render `s` as a TOML basic string.

    Evidence lines quote real command output, so they contain apostrophes,
    double quotes, and backslashes. Anything less than a real escaper here
    emits a file that `cargo xtask ledger` rejects at parse time -- or worse,
    one it parses into the wrong text.
    """
    esc = {"\\": "\\\\", '"': '\\"', "\b": "\\b", "\t": "\\t",
           "\n": "\\n", "\f": "\\f", "\r": "\\r"}
    body = "".join(esc.get(c, c if c >= " " and c != "\x7f" else f"\\u{ord(c):04X}")
                   for c in s)
    return f'"{body}"'


def toml():
    L = ["# Machine-readable mirror of the PROPOSALS.md execution ledger (GOV-003).",
         "# Regenerate with `python3 tools/gen-ledger.py toml`; validate with `cargo xtask ledger`.",
         "",
         'schema = 1',
         'snapshot = "2026-07-27"',
         "", ]
    for tid, tier, theme, st, deps, target, deliv, ev in T:
        L.append("[[task]]")
        L.append("id = " + tstr(tid))
        L.append("tier = " + tstr(tier))
        L.append("theme = " + tstr(theme))
        L.append("status = " + tstr(STATUS_WORD[st]))
        L.append("deps = [" + ", ".join(tstr(d) for d in deps) + "]")
        L.append("target = [" + ", ".join(tstr(p.strip()) for p in target.split(";")) + "]")
        L.append("deliverable = " + tstr(deliv))
        L.append("evidence = [" + tstr(ev) + "]")
        date, ev = COMPLETED.get(tid, ("", ""))
        L.append("completed_on = " + tstr(date))
        L.append("completed_evidence = " + tstr(ev))
        dev = DEVIATIONS.get(tid, [])
        if dev:
            L.append("deviations = [")
            for d in dev:
                L.append("  " + tstr(d) + ",")
            L.append("]")
        else:
            L.append("deviations = []")
        L.append("")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    errs, terminals = validate()
    if errs:
        print("VALIDATION ERRORS:", file=sys.stderr)
        for e in errs:
            print("  " + e, file=sys.stderr)
        sys.exit(1)
    print(f"# ok: {len(T)} tasks, 0 errors", file=sys.stderr)
    print(f"# terminals ({len(terminals)}): {terminals}", file=sys.stderr)
    if len(sys.argv) > 1 and sys.argv[1] == "toml":
        print(toml())
    else:
        print(md())
