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
("GOV-004","core","gov","x",["GOV-002"],"crates/incin/benches/; docs/plan/baselines/",
 "CPU and GPU capability, performance, and compile-size baselines with environment metadata",
 "cargo bench -p incin -- --save-baseline main"),
("GOV-005","core","gov","x",["GOV-004"],".github/workflows/ci.yml",
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
("SHP-007","core","shape","x",["SHP-004","SHP-005","SHP-006"],"crates/incin-core/tests/; crates/incin-core/tests/compile_fail/",
 "Close mixed, named, and rank gaps; compile-pass, compile-fail, and fuzz suites",
 "cargo test -p incin-core"),
("SHP-008","core","shape","x",["SHP-007"],"crates/incin-core/src/tensor/base.rs",
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
("EXE-004","core","exec","x",["SHP-003"],"crates/incin-core/src/exec/meta.rs; crates/incin-backends/src/{cpu,cuda,wgpu}/storage.rs",
 "Normalize TensorMeta and unify the three LayoutClass enums; view, offset, alignment, and bounds tests",
 "cargo test -p incin-backends --test tensor_meta"),
("EXE-005","core","exec","x",["EXE-001"],"crates/incin-core/src/exec/capability.rs",
 "Capability registry whose generated matrix matches execution tests",
 "cargo test -p incin-backends --test capability_matrix"),
("EXE-006","core","exec","x",["EXE-002","EXE-005"],"crates/incin-core/src/tensor/backend.rs",
 "Split storage, Execute<O>, and Capabilities out of the 254-method supertrait; give SupportsDType<K> real per-backend impls",
 "cargo test -p incin-core --test compile_tests"),
("EXE-007","core","exec","x",["EXE-003","EXE-004","EXE-006"],"crates/incin-backends/src/cpu/",
 "Migrate the CPU vertical slice with parity and overhead evidence",
 "cargo test -p incin-backends --no-default-features --features std,cpu"),
("EXE-008","core","exec","x",["EXE-007"],"crates/incin-backends/src/{cuda,wgpu,external}/; crates/incin-backends/src/dispatch.rs",
 "Migrate CUDA, WGPU, dispatch, and external adapters; replace the F32-hardcoded byte arithmetic with checked dtype.size_bytes()",
 "cargo test -p incin-backends --no-default-features --features std,cpu,wgpu"),
("EXE-009","core","exec","~",["EXE-008"],"crates/incin-core/src/tensor/backend.rs; crates/incin-backends/src/dispatch.rs",
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
("CI-002","core","ci","x",["EXE-008"],".github/workflows/hardware.yml",
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
 "CI-002": ("2026-07-28", "gh workflow run hardware.yml -> run 30359916370 on develop, conclusion success. Resolve Hardware Availability passed in 2s and emitted the two annotations that are the point of the job: 'CUDA hardware job skipped: HARDWARE_CUDA_RUNNER is unset, so no NVIDIA runner is registered' and 'WGPU native-adapter job skipped: HARDWARE_WGPU_RUNNER is unset. The software-adapter job still runs.' WGPU Software Adapter passed in 1m27s with 374 lib tests, 1 ignored, plus 5 wgpu_executor, 21 cpu_executor, 10 dispatch_parity, 12 gradient_parity, 6 ops, 1 safetensors, 6 tensor_meta, and 13 candle/native rows, 0 failed -- the same counts the local run reports, on the pinned 1.92.0 toolchain rather than whatever stable resolves to. .github/workflows/hardware.yml runs weekly and on dispatch. The CUDA job runs the four suites the EXE-004, EXE-005, and EXE-008 hardware obligations name, including both --ignored sets, and writes nvidia-smi and nvcc output into the run summary. Each hardware job asserts tests actually executed: the ignored CUDA suite must report at least 60 results, verified against a real local log that totals 71, because a device-less run reports zero tests and exits zero. The dispatch input rejects dist3, multinode, and metal by name against CI-003, CI-004, and CI-007 rather than matching nothing silently. Full local gate clean: cargo fmt --all --check, cargo test --workspace, cargo xtask ledger, cargo xtask budgets, and tools/audit-shapes.sh --check. Clippy was recorded as unavailable when this row landed, which was true of that machine's toolchain. The CUDA host's Rust 1.92.0 does ship clippy 0.1.92, so the lint gate was run here for the first time and ci.yml's own Clippy Lints step turned out to be failing on develop: DTypeId::size_bytes, added by EXE-008, spells a divisibility test as a remainder comparison, which clippy 1.92 rejects under -D warnings. Nine further findings sat behind it. All are fixed or explicitly allowed with a stated reason, and cargo clippy now exits zero for ci.yml's exact invocation and for the CUDA and WGPU backend feature sets."),
 "EXE-008": ("2026-07-28", "cargo test -p incin-backends --no-default-features --features std,cpu,wgpu -> 367 lib tests passed with 1 ignored, plus 4 wgpu_executor, 6 tensor_meta, 5 capability_matrix, 6 cpu_executor, 12 gradient_parity, 6 ops, and 1 safetensors, 0 failed. cargo test -p incin-backends --no-default-features --features std,cpu,external-candle -> 5 candle_executor tests passed. Bare CPU, CUDA all-target compilation, core no-default, and cargo test --workspace --all-targets all completed with no failures. DTypeId now owns the byte arithmetic: block_elements, block_bytes, and a checked block-aware size_bytes replace every hardcoded width. WGPU's 23 literal '* 4' and size_of::<f32>() allocation sites and CUDA's three duplicate checked_byte_len helpers plus nine raw '* 4' alloc_zeros sites now route through one crate-level bytes::byte_len, and CUDA allocation failures are reported rather than unwrapped. The CUDA shape and concat paths propagated the operand's real dtype while allocating four bytes per element, so an F64 or I64 tensor -- both accepted by the CUDA storage, reshape, and broadcast capability rows -- received half the bytes its own metadata claimed; that is now sized from the dtype. WgpuBuffer::new_zeros_for, reduce_all_to_storage, reduce_dim_to_storage, dispatch_reduce_all, sum_dim_keepdim, sum_dim_squeeze, add_wgpu_storage, and the CUDA conv and pool alloc_zeroed wrappers became fallible, and the WGPU gradient accumulation loops propagate that failure instead of swallowing it in an and_modify closure. WgpuBackendImpl, CudaBackendImpl, DispatchBackend, and CandleBackend all implement StorageBackend and Execute<MatMulSpec> against the same sealed Validated descriptor as the CPU slice; WGPU, CUDA, and Candle also implement Capabilities. CandleStorage pairs a foreign candle_core::Tensor with checked TensorMeta without changing Backend::Storage<K>, so the 1,146-line Candle operation set was not rewritten. TensorMeta::UNALLOCATED, ShapeBuf::SCALAR, StrideBuf::EMPTY, and DeviceId::CPU are new const values so the dispatch storage enum's no-allocation variant is described rather than panicked on, keeping the SHP-001 backend panic count at zero. Formatting, shape audit, diff-check, budgets, and ledger validation clean. Clippy remains unavailable on the installed Rust 1.92 toolchain. The CUDA half of this row was originally compile-verified only and has now run on hardware: cargo test -p incin-backends --no-default-features --features std,cpu,cuda -> 306 passed, 0 failed, 60 ignored, and the same command with --ignored -> 63 passed, 0 failed on a GeForce GTX 1650 (driver 595.71.05, CUDA 12.6, nvcc 12.6.77), covering every CUDA operation test plus the three NVRTC template-compilation families. Two of those tests failed on their first real run and are fixed here. crates/incin-backends/src/cuda/ops/quant.rs sized its Q8_0 output with a literal num_blocks * size_of::<BlockQ8_0>() -- the last hardcoded width this row was meant to remove -- and set CudaBuffer::len to the block count, while every other CUDA buffer sets it to a logical element count and CudaStorage reads it as one. A [2, 32] quantized tensor therefore declared 64 elements over a 2-element allocation, and quantized_matmul_computes_correct_shape and quantized_matmul_rejects_non_multiple_of_32_k both aborted in the EXE-004 bounds check before reaching a kernel. Both sites now go through alloc_zeroed_bytes and DTypeId::size_bytes, which knows a Q8_0 block is 34 bytes for 32 values, and launch_dequantize derives its block count from the tensor shape with an explicit multiple-of-32 check instead of reading the field whose unit was ambiguous. The WGPU evidence command is unchanged and still reports 374 lib tests passed with 1 ignored plus every integration suite green. Clippy was recorded as unavailable when this row landed, which was true of that machine's toolchain. The CUDA host's Rust 1.92.0 does ship clippy 0.1.92, so the lint gate was run here for the first time and ci.yml's own Clippy Lints step turned out to be failing on develop: DTypeId::size_bytes, added by EXE-008, spells a divisibility test as a remainder comparison, which clippy 1.92 rejects under -D warnings. Nine further findings sat behind it. All are fixed or explicitly allowed with a stated reason, and cargo clippy now exits zero for ci.yml's exact invocation and for the CUDA and WGPU backend feature sets."),
 "EXE-007": ("2026-07-28", "cargo test -p incin-backends --no-default-features --features std,cpu -> 269 unit tests passed, 1 ignored, with every integration and doctest target green; cargo test --test cpu_executor -> 6 passed and the internal corrupt-device binder test passed. CpuBackendImpl is now an explicit zero-sized executor with new/Default, StorageBackend<Local, Storage<K> = CpuStorage>, Capabilities delegated to the authoritative CPU registry, and Execute<MatMulSpec, Output = CpuStorage>. The request binder requires exactly two type-erased handles, downcasts them to CpuStorage, rejects non-CPU ordinal, non-F32 capability, unsupported layout/rank, and physical metadata that disagrees with the sealed descriptor before delegating to the established rank-2 or batched kernel. Parity covers exact rank-2 values, batched RHS broadcasting, strided transpose views, and both input gradients; exact rejection coverage includes wrong count, foreign storage, F64, corrupt CUDA metadata, and mismatched descriptor binding. On this 64-bit target CpuBackendImpl is 0 bytes, TensorHandle is 24 bytes, and ExecutionRequest is 32 bytes with no owned buffers. The warmed, order-balanced release probe over 32 runs of 32x32 matmul observed 9.668 us/call through the legacy entry and 10.635 us/call through the descriptor adapter; this approximately 10.0% tiny-matrix overhead is recorded as non-gating evidence, not a portable performance claim. WGPU, CUDA, and external-Candle all-target rows compile; the optional Candle row exposed and fixed three missing imports in its split-module tests. Core no-default, guarded compile tests, cargo test --workspace --all-targets, warning-denied workspace rustdoc, formatting, diff-check, shape audit, budgets, and ledger validation passed. Clippy remains unavailable on the installed Rust 1.92 toolchain."),
 "EXE-006": ("2026-07-28", "cargo test -p incin-core --test compile_tests -> 2 passed, 0 failed, covering 1 compile-pass and 35 guarded compile-fail cases. StorageBackend<P = Local>, Execute<O>, ExecutionRequest, TensorHandle, BackendError, the minimal Local placement foundation, and the backend-owning ExecutionContext now form the descriptor execution contract; requests require sealed Validated<O> and handles obtain TensorMeta only through StorageBackend::metadata. There is deliberately no blanket Execute implementation: a storage-only backend fails trait resolution for BroadcastSpec, while the pass case lowers a real validated BroadcastSpec and executes it through a custom backend. SupportsDType has no default body. CPU and Dummy provide explicit all-storage-dtype resolution; CUDA explicitly proves f32/f64/f16/bf16/i64 plus registry-checked Dyn; WGPU proves f32 plus registry-checked Dyn; dispatch resolves Storage support through the selected registry; tracing delegates an existing proof; Candle explicitly proves its seven non-quantized representations plus conversion-checked Dyn, leaving Q8_0 absent. A real WgpuBackendImpl<f32, Wgpu>: SupportsDType<f64> negative case fails E0277, and a bare BroadcastSpec in ExecutionRequest fails E0308. D-020 records the stable encoding of D-001 because Rust rejects default generic parameters on associated types. The legacy Backend operation families remain callable for staged migration through EXE-007/008 and removal in EXE-009, with no silent Execute fallback. Core no-default, CPU and full WGPU backend tests, CUDA and external-Candle all-target compile rows, cargo test --workspace --all-targets, warning-denied workspace rustdoc, formatting, diff-check, shape audit, budgets, and ledger validation passed. Clippy remains unavailable on the installed Rust 1.92 toolchain."),
 "EXE-005": ("2026-07-28", "cargo test -p incin-backends --test capability_matrix -> 4 passed, 0 failed; with std,cpu,wgpu -> 5 passed, including real WGPU execution for every registered operation; with std,cpu,cuda the matrix and its ignored hardware probe compile without CUDA hardware. The core adds allocation-free CapabilityQuery, SupportLevel, UnsupportedReason, Capabilities, CapabilityRule, and CapabilityRegistry with exact-operation precedence over family rows and deterministic first-unsatisfied-constraint rejection. CPU, CUDA, and WGPU registrations cover dtype, layout, rank, training, MathMode, and native/composed/fallback classification; no fallback is registered. DType policy now queries the same registry, and KernelMathMode is absent after promotion to core MathMode. Generated cases cover every registered dtype-layout-math-mode product at minimum and maximum rank, training-capable rows, exact unsupported reasons, every advertised CPU dtype, every CPU operation/layout row, all WGPU rows on real hardware, and a CUDA hardware-gated execution loop. Conformance exposed and fixed WGPU all-reduction metadata returning [1] instead of scalar []; a later audit also removed false claims for strided CUDA compute, strided Q8 CPU reshape, integer/Q8 training, and rank-zero normalization. Full WGPU backend tests passed, CUDA all-target tests compile, core no-default and cargo test --workspace --all-targets completed with no failures; formatting, shape audit, diff-check, and ledger validation clean. Clippy remains unavailable on the installed Rust 1.92 toolchain. The CUDA hardware-gated execution loop has now run: on a GeForce GTX 1650 (driver 595.71.05, CUDA 12.6) cargo test -p incin-backends --features cuda --test capability_matrix -- --include-ignored -> 5 passed, 0 failed, with every_generated_cuda_row_matches_real_execution_on_hardware executing all thirteen CUDA_CAPABILITIES rows -- Storage, Fill, Random, Pointwise, Reduction, Normalization, two Broadcast rows, two Reshape rows, MatMul, Conv2d, and Pool2d -- at each registered layout and asserting the generated support level, output dtype, and output device against real execution. No row needed removing: the four false claims this row's audit had already deleted by inspection were the only ones, and the surviving set matches the device. No registration or probe changed, so this discharges the deviation without a code diff. Clippy was recorded as unavailable when this row landed, which was true of that machine's toolchain. The CUDA host's Rust 1.92.0 does ship clippy 0.1.92, so the lint gate was run here for the first time and ci.yml's own Clippy Lints step turned out to be failing on develop: DTypeId::size_bytes, added by EXE-008, spells a divisibility test as a remainder comparison, which clippy 1.92 rejects under -D warnings. Nine further findings sat behind it. All are fixed or explicitly allowed with a stated reason, and cargo clippy now exits zero for ci.yml's exact invocation and for the CUDA and WGPU backend feature sets."),
 "EXE-004": ("2026-07-27", "cargo test -p incin-backends --test tensor_meta -> 5 passed, 0 failed; the same suite with WGPU enabled -> 6 passed, including a real materialized transpose on the AMD Radeon 680M, and with CUDA enabled it compiles without hardware. TensorMeta is the sole shape, stride, offset, dtype, device, layout, and alignment source embedded by CPU, CUDA, and WGPU storage; all constructors validate checked span plus offset against allocation capacity. CPU transpose, broadcast, narrow, and contiguous reshape remain metadata-only; CUDA and WGPU materialized outputs report contiguous zero-offset metadata. UnaryLayoutClass, BinaryLayoutClass, and KernelLayout are absent and their consumers use the single core LayoutClass. Tests cover contiguous, transposed, broadcast, nonzero offset, empty, effective alignment weakening, invalid alignment, rank mismatch, arithmetic overflow, and out-of-bounds views. The full WGPU suite passed 361 tests plus 12 parity tests and exposed a packed-Q8 bug: 64 logical values occupy 68 bytes, so Q8 storage now validates its 34-byte block encoding and reports DTypeId::Q8_0 rather than pretending to be dense F32. cargo test --workspace --all-targets completed with no failures; core no-default, formatting, shape audit, diff-check, and ledger validation clean. The CUDA storage half is no longer compile-verified: on a GeForce GTX 1650 (driver 595.71.05, CUDA 12.6) cargo test -p incin-backends --features cuda --test tensor_meta -- --include-ignored -> 6 passed, 0 failed, and the three new ignored storage tests pass inside the 63-test CUDA hardware run. Two claims changed as a result. CUDA TensorMeta reported Alignment::BYTE because CudaSlice<u8> proves only byte alignment; measuring eleven awkward allocation sizes on the device shows every returned pointer is 256-byte aligned, as the CUDA C Programming Guide promises, so the recorded guarantee is now 256 and a view offset still weakens it. CudaStorage also now checks that the allocation covers the element count its metadata claims, comparing CudaSlice byte length against DTypeId::size_bytes; that check is what turns a buffer whose len was recorded in the wrong unit into a reported error rather than a kernel reading past the end of it, and it is pinned by storage_rejects_an_allocation_too_small_for_its_element_count and by packed_q8_storage_reports_logical_elements_over_a_block_sized_allocation, where 64 logical values occupy 68 bytes. Clippy was recorded as unavailable when this row landed, which was true of that machine's toolchain. The CUDA host's Rust 1.92.0 does ship clippy 0.1.92, so the lint gate was run here for the first time and ci.yml's own Clippy Lints step turned out to be failing on develop: DTypeId::size_bytes, added by EXE-008, spells a divisibility test as a remainder comparison, which clippy 1.92 rejects under -D warnings. Nine further findings sat behind it. All are fixed or explicitly allowed with a stated reason, and cargo clippy now exits zero for ci.yml's exact invocation and for the CUDA and WGPU backend feature sets."),
 "GOV-005": ("2026-07-27", "cargo xtask budgets -> budgets ok: 11 runtime, 16 artifacts, 5 feature crates, 25 features. docs/plan/budgets.toml declares complete upper-bound coverage for every GOV-004 runtime confidence-interval high and every compile *_bytes metric; the validator rejects missing or duplicate series, baseline drift, invalid maxima, exceeded budgets, unknown schemas, and unsafe paths. The feature inventory matches exact defaults and forwarding in every feature-bearing workspace member discovered by cargo metadata, so a new uninventoried crate cannot bypass the gate. cargo test -p xtask -> 12 passed, including exceeded-budget, missing-series, duplicate-series, feature-drift, uninventoried-crate, and unsafe-path faults. .github/workflows/ci.yml runs budgets and the shape audit on every ledger job. cargo test --workspace completed with no failures, including 32 semantic trybuild cases and all doctests; formatting, audit-shapes, diff-check, and ledger clean."),
 "GOV-004": ("2026-07-27", "cargo bench -p incin -- --save-baseline main -> 8 stable CPU Criterion series covering f32/u32 creation, f32 add and sum at 1,024 and 65,536 elements, and f32 matmul at 16 and 64; cargo bench -p incin --features wgpu --bench baselines -- '^(capability/wgpu|gpu/)' --save-baseline wgpu-main -> 3 real WGPU series on the AMD Radeon 680M. docs/plan/baselines/main.toml records 95% confidence intervals, exact feature sets, OS, architecture, CPU, memory, Rust/Cargo/LLVM/Criterion versions, revision, device availability, build time, and raw release artifact sizes. CUDA was originally recorded as explicitly unavailable because no NVIDIA adapter or nvidia-smi executable was present; the CUDA series now exist, captured on a second host. cargo bench -p incin --features cuda --bench baselines -- '^(capability/cuda|gpu/cuda)' --save-baseline cuda-main -> 3 real CUDA series on a GeForce GTX 1650 SUPER (Turing TU116, compute capability 7.5, driver 595.71.05, CUDA 12.6): capability/cuda/f32_create at 183.01 ms, gpu/cuda/add_f32/65536 at 15.459 us, and gpu/cuda/matmul_f32/64 at 24.393 us, each with its 95% interval, plus a compile.cuda profile of eight artifact sizes. cargo xtask budgets -> 14 runtime, 24 artifacts, 5 feature crates, 25 features, up from 11 and 16, with accelerator runtime bounds at the 1.5x headroom the WGPU rows use and artifact bounds at 1.2x. The first CUDA capture is itself a finding: capability/cuda/f32_create is five orders of magnitude slower than its WGPU counterpart because cuda_from_bytes calls CudaContext::new on every tensor creation, bypassing the get_cuda_device cache that already exists in cuda/gpu.rs. A direct probe measures CudaContext::new at 114 ms per call even when a context is already live, which accounts for nearly all of the 183 ms. The number is recorded as measured rather than adjusted or omitted, with the cause named in the series note; repairing it is PRF-001/PRF-003 work and will move the series by orders of magnitude. cargo check passes for both default and WGPU benchmark targets; cargo test --workspace completed with no failures, including 32 semantic trybuild cases and all doctests. Formatting, TOML parsing, diff-check, and ledger validation clean. Clippy unavailable: the installed Rust 1.92 toolchain reports cargo-clippy is not applicable."),
 "SHP-008": ("2026-07-27", "cargo test -p incin-core --test construction_witness -> 6 passed, 0 failed; cargo test -p incin-core -> 0 failures, including 32 semantic trybuild cases; cargo test --workspace -> 0 failures. The audit covered all 96 starting references: from_parts_unchecked is absent, the raw constructor is private and requires an unforgeable module-private ConstructionWitness, metadata-only into_dyn/Grad retags use the source Tensor as their witness, and every backend-produced or imported storage validates shape, dtype, and device. Shape::dims is now universal so Tensor<S> can validate every accepted S rather than only call sites carrying an explicit DynShape bound. Runtime Flatten rejects reversed/out-of-rank ranges and product overflow without slicing or arithmetic panics; the static diagnostic is pinned. Enabling validation exposed and fixed DummyBackend matmul returning [M,N] instead of broadcast [B,M,N] when only rhs was batched. fmt, audit-shapes, no-default-features, diff-check, and ledger clean. Clippy unavailable: the installed Rust 1.92 toolchain reports cargo-clippy is not applicable."),
 "SHP-007": ("2026-07-27", "cargo test -p incin-core -> 284 passed, 0 failed, including 30 semantic trybuild cases; cargo test --workspace -> 0 failures. BroadcastShape now resolves same-rank, right-aligned, named, and arbitrary-position runtime axes one axis at a time; a fixed-seed 35,000-case property suite agrees with an independent reference. MatMulShape relates mixed contraction spellings, checks every named/runtime contraction and shared batch value, covers runtime batch axes at every rank, and generates typed batch rules through rank 8. Two rank-4 mixed rules that still skipped contraction checks were fixed. The rank-8 compile-fail case exposed a second ladder outside the SHP-006 inventory: tensor constructor arguments stopped at 7 and NotUnit skipped rank 6; impl_arg_into now takes MAX_RANK directly and rank 6/8 construction is pinned. fmt, audit-shapes, no-default-features, and ledger clean. Clippy unavailable: the installed Rust 1.92 toolchain reports cargo-clippy is not applicable."),
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
 "CI-002": [
  "The row asks for a scheduled CUDA and WGPU hardware matrix, and the CUDA half cannot execute in this repository today. xupremix/incin has zero registered self-hosted runners and no GitHub-hosted runner carries an NVIDIA device, so runs-on with hardware labels would queue the job for roughly a day and then fail for a reason unrelated to the code, on every weekly schedule. The CUDA and native-WGPU jobs are therefore gated behind the HARDWARE_CUDA_RUNNER and HARDWARE_WGPU_RUNNER repository variables, each holding a JSON label array, and the select job records which jobs it skipped and why in the run summary. Registering a runner and setting the variable activates them with no further change; a dispatch that explicitly asks for job=cuda while the variable is unset fails rather than skipping, because an operator who named the job wants an answer. What is proved here is the workflow, the selection logic, and the WGPU software matrix; the CUDA jobs are proved only by the identical commands having been run by hand on a GeForce GTX 1650, which is what EXE-004, EXE-005, and EXE-008 record.",
  "The dispatch input is a free-form string rather than a choice, so the select job can reject dist3, multinode, and metal with a message naming CI-003, CI-004, and CI-007. A choice list containing only the implemented jobs would reject those values with a generic dispatch error instead, and a choice list containing all of them would accept a selector that matches no job and report green.",
 ],
 "EXE-008": [
   "The ledger names the backend directories and dispatch.rs, but the checked replacement for the hardcoded byte arithmetic has to live where every backend can reach it. DTypeId::size_bytes is added in incin-core because the dtype -- not the caller -- is what knows a Q8_0 block is 34 bytes for 32 logical values; crates/incin-backends/src/bytes.rs is the thin wrapper the accelerators share. Leaving the width at each call site is exactly the duplication this row exists to remove.",
   "Candle's storage is candle_core::Tensor, a foreign type that cannot gain a TensorMeta field. Rather than rewrite the 1,146-line Candle operation set, CandleStorage wraps the tensor for StorageBackend::Storage<K> only, leaving the legacy Backend::Storage<K> as the raw tensor. The two are separate associated types on separate traits, so a third-party backend joins the descriptor contract at its boundary; EXE-010 documents this seam as the backend-authoring template.",
   "The WGPU tape's backward closure is Fn(&WgpuStorage) -> Vec<WgpuStorage> and cannot report. scatter_into_zeros therefore still resolves its allocation infallibly, documented at the call site, because its shape was already allocated successfully in the forward pass. Making the backward signature fallible changes 19 closures and belongs with the explicit gradient context in GRD-001, not with a byte-arithmetic migration.",
   "DispatchBackend implements StorageBackend and Execute but deliberately not Capabilities. A runtime-selected backend has no single device, so any answer would either overstate support as the union of every compiled backend or understate it; the concrete executor it routes to queries its own registry, which is the only device that can answer. Dispatch also requires both operands on one backend rather than routing on the first, so a device mismatch is reported as one instead of as a downcast failure.",
   "Each migrated backend implements Execute for MatMulSpec only, matching the operation EXE-007 chose for the CPU slice. The remaining descriptors still have no single agreed execution meaning -- ReductionSpec and Pool2dSpec omit mode -- so giving them implementations here would fabricate semantics ahead of EXE-009.",
   "This row was first recorded with CUDA verified by compilation only, because no device was present. It has since run on a GeForce GTX 1650 (driver 595.71.05, CUDA 12.6) and the deviation is discharged: 63 ignored CUDA tests pass, including every NVRTC template family. The run found one real defect the compiler could not, in exactly the byte-arithmetic area this row owns. launch_quantize sized its allocation with a literal size_of::<BlockQ8_0>() rather than the checked DTypeId::size_bytes this row introduced, and recorded CudaBuffer::len in blocks while every other CUDA buffer records logical elements, so a [2, 32] Q8_0 tensor claimed a two-element allocation and the EXE-004 bounds check rejected it. Both are now the dtype's arithmetic, and launch_dequantize derives its block count from the shape rather than reading the ambiguous field.",
 ],
 "EXE-007": [
   "The roadmap asks for one complete CPU vertical slice but does not name an operation. EXE-007 implements MatMulSpec: it is fully semantic, exercises rank-2/batched/strided/autograd behavior, and PRF-002 depends directly on this row's CPU matmul target. ReductionSpec and Pool2dSpec intentionally omit sum/mean/max mode, while BroadcastSpec describes shared geometry; assigning any of them one execution meaning here would fabricate semantics. They remain unclaimed rather than receiving misleading Execute implementations.",
   "The descriptor executor is a migration adapter over the established CPU rank-2 and batched kernels, preserving their tape behavior and parity. The binder validates resource-to-descriptor agreement, after which the legacy kernel still performs its own defensive shape checks. Removing that duplicate semantic check requires the direct planned-kernel work in PRF-002 and the monolithic-adapter removal in EXE-009; claiming validation-once now would be false.",
   "The optional external-Candle all-target gate failed because tests in the previously split candle/mod.rs no longer imported DTypeId, DeviceId, or the candle_core alias. Adding those three explicit test imports is outside the CPU target but is the minimum repair needed for the required optional-backend compile gate; no Candle runtime behavior changed.",
   "The release timing comparison is deliberately non-gating and machine-local. It alternates execution order after warm-up and reports the observed descriptor cost for a tiny 32x32 matrix, but establishes no cross-host regression budget; GOV-005/TUN-008 remain the owners of enforceable performance budgets.",
 ],
 "EXE-006": [
   "The ledger target names tensor/backend.rs, but an enforceable request contract also requires the owned TensorHandle and ExecutionContext modules, structured BackendError, the minimal Local placement typestate, per-backend SupportsDType implementations, and compile-contract fixtures. These files implement or prove the one interface; concrete backend Execute migrations remain in EXE-007 and EXE-008.",
   "D-001's exact associated-type spelling does not compile on stable Rust because defaults on associated-type generic parameters are forbidden. D-020 moves P: Placement = Local to StorageBackend itself, preserving both dtype and placement selection through <B as StorageBackend<P>>::Storage<K> without making a Core API nightly-only.",
   "ExecutionContext was owned by GRD-001 and the distributed placement vocabulary by DST-003 in Appendix A, but D-002 and D-001 require their types in EXE-006 signatures. EXE-006 adds only backend ownership plus Local/Placement/PlacementKind; GRD-001 and DST-003 extend those same types rather than introducing parallel foundations.",
   "The core trybuild dev dependency enables WGPU so the unsupported static pair is proved against the real WgpuBackendImpl rather than a synthetic test backend. The pass/fail cases compile WGPU code but do not initialize hardware.",
 ],
 "EXE-005": [
   "The ledger names only incin-core exec capability.rs, but authoritative backend registrations and executable conformance probes necessarily live in incin-backends; core policy.rs is also required by D-008 to promote KernelMathMode without creating a parallel vocabulary. The extra files are consumers and evidence for the one core contract, not a second registry.",
   "OperationKind currently names descriptor operations exactly but retains coarse Pointwise, Reduction, Normalization, Fill, Random, and Storage families for the existing broad Backend methods. Registrations are exact where the taxonomy is exact and family-level otherwise; EXE-006 replaces the broad execution surface instead of inventing a second operation vocabulary under EXE-005.",
   "This row was first recorded with the generated CUDA execution probe compiled but never run, because no device or driver was present. It has since run on a GeForce GTX 1650 (driver 595.71.05, CUDA 12.6) and the deviation is discharged with no change to the registrations: all thirteen CUDA rows executed the operation they advertise and returned the dtype and device they claim. That is the outcome worth stating plainly, because it was not the expected one. The WGPU rows exposed a scalar-shape defect the first time they ran, and the audit recorded in this row's evidence had already deleted four false CUDA claims by inspection; the remaining set survives execution. The probe stays ignored by default so a machine without a device still passes cargo test, and CI-002 is what runs it on a schedule.",
 ],
 "EXE-004": [
   "The ledger names the three storage files, but replacing their raw public-in-crate fields necessarily updates every backend consumer plus iteration, kernel, and tuning code that used the three deleted layout enums. Restricting edits to the target paths would leave the workspace uncompilable and preserve duplicate layout vocabularies.",
   "This row originally recorded Alignment::BYTE for CUDA, because CudaSlice<u8> is all the Rust type system knows about a device allocation and one byte is all it proves. That was true and useless: every CUDA tensor would answer 'unaligned' to a kernel choosing between a scalar and a vector load, so the weakest defensible claim would have cost real throughput for the life of the backend. The deviation is discharged by measurement rather than by citation. device_pointers_are_aligned_to_the_documented_boundary allocates eleven awkward sizes -- 1, 3, 17, 34, 68, 127, 256, 257, 1024, 4099, 65537 bytes -- holds them all so the driver cannot reuse one lucky address, and asserts every returned pointer is 256-byte aligned, which is what the CUDA C Programming Guide promises for any allocation routine. CUDA TensorMeta now records that, and a nonzero view offset still weakens it through the existing after_offset_bytes path. The measurement is a hardware test rather than a comment, so a driver that stopped honouring the guarantee would fail CI-002 instead of silently invalidating the claim.",
   "The established infallible backend-produced constructors are retained for compatibility, but each is a thin trusted wrapper over a fallible checked constructor. Imported or explicitly strided metadata uses the fallible path and cannot create an unchecked storage handle.",
 ],
 "GOV-005": [
   "The ledger names only .github/workflows/ci.yml as its target, but an enforceable cargo xtask budgets command requires implementation in xtask and versioned contracts in docs/plan/budgets.toml. CI remains the enforcement point; the additional files are the executable policy it invokes.",
   "GOV-005 validates checked-in runtime and artifact baselines deterministically rather than timing shared CI runners. Live time, memory, cache, and hardware regression execution remains assigned to TUN-008 and CI-006, preventing this governance gate from making noisy performance claims ahead of those tasks.",
 ],
 "GOV-004": [
  "The row asks for CPU and GPU baselines with environment metadata, and the CUDA half was captured on a different machine from the CPU and WGPU half, because the original host had no NVIDIA device. Adding those rows under the unqualified [environment] block would attribute them to a machine that never ran them. The baseline document therefore gains [environment.cuda_host], and every cuda-backed series, the capability.cuda block, and the compile.cuda profile carry a host key naming it. The README forbids diffing a series or an artifact size across two host values; compile sizes are called out explicitly because the two hosts are configured with different linkers, so those bytes are not comparable even for identical code. No xtask change was needed, since the baseline schema does not deny unknown fields and the budget key is already (backend, id).",
  "capability/cuda/f32_create is recorded at 183 ms, five orders of magnitude above the WGPU series it mirrors, because cuda_from_bytes creates a fresh CudaContext for every tensor rather than using the cache in cuda/gpu.rs. GOV-004 records baselines and does not own hot-path repair, so the measurement is entered as taken with its cause named in the series note rather than being adjusted, omitted, or fixed here. Fixing it belongs to PRF-001 or PRF-003 and will require moving both the baseline and its budget in one reviewed commit, which is the process budgets.toml already documents for a deliberate change.",
   "The required evidence command uses the default CPU feature set and therefore cannot truthfully benchmark an optional accelerator. WGPU measurements were captured with a second explicitly feature-enabled command on real hardware; CUDA remains an explicit unavailable capability rather than a synthetic passing series.",
 ],
 "SHP-008": [
   "The ledger estimated roughly 45 unchecked construction obligations, but the current tree contained 96 references across tensor ops, neural-network modules, serialization, and optimizers. All current references were audited; limiting the task to the stale estimate would have left most of the same trust surface intact.",
   "Safe validation needs runtime dimensions for every S accepted by Tensor<S>, while dims lived on the optional DynShape extension and many operation outputs were bounded only by Shape. Shape now owns dims and DynShape retains rank/numel. This is the minimum enforceable boundary: adding DynShape bounds to individual operations would merely move the convention to dozens of callers and still permit future unchecked Shape-only outputs.",
   "Once validation became mandatory it rejected DummyBackend's rhs-only batched matmul: typed metadata said [B,M,N], storage said [M,N]. DummyBackend matmul now performs right-aligned batch broadcasting, outside the row's base.rs target, because weakening the constructor would preserve the invariant bug SHP-008 exists to expose.",
 ],
 "SHP-007": [
   "The ledger target names only test directories, but closing the gaps those tests exposed required changes in broadcast.rs, matmul.rs, the rank generator, and tensor argument conversion. In particular, the rank-8 negative case found that impl_arg_into had an independent literal ceiling of 7 and NotUnit omitted ranks 6 and 8; leaving those untouched would make a supported Shape impossible to construct.",
 ],
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
         'snapshot = "2026-07-28"',
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
