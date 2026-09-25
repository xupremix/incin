# Novel-solutions platform: how incin becomes where researchers try wild ideas

Status: platform analysis, no implementation, no decisions. Branch
`feat/custom-autograd-dtype`. Grounded in the tree as of Sep 2026 plus the
SOTA survey below.

Question from the maintainer: **what makes this framework the place researchers
pick for NEW AND NOVEL solutions?** The answer is not a feature list. Every
framework researchers love has one property: a *mechanism* that turns a wild
idea into a runnable artifact in an afternoon, with a legible path from hack to
supported. This report inventories those mechanisms, audits incin's three
hardest novelty walks honestly, and orders recommendations by
novelty-unlock per effort.

## 0. What incin concretely offers a researcher today (evidence, not pitch)

- **Compile-time shape proofs.** `s![...]` literals
  (`crates/incin-core/src/shapes/`), `ShapeEq` / `BroadcastShape`
  (`crates/incin-core/src/shapes/broadcast.rs`, re-exported in
  `crates/incin/src/lib.rs`), `MatMulShape`
  (`crates/incin-core/src/tensor/matmul.rs`). `BroadcastShape`'s curated
  diagnostic currently never fires (bottom `()` resolves first) — recorded in
  `docs/plan/research/0.2.0/00-index.md` Tier 1 note. That matters below:
  proofs are strong, proof *errors* are the friction.
- **Operation catalog with typed descriptors + generated semantics.**
  `crates/incin-core/src/exec/catalog/` (inference, descriptor, `Validated<O>`
  unforgeable outside the crate); `OPERATION_SEMANTICS` generated text
  (`docs/plan/research/0.2.0/what-to-take-from-sota.md` §4 calls this split
  stronger than ATen's codegen convention because it is a type, not a
  convention). ~180 ops, 164 backend-executable per the index.
- **Capability discovery + `doctor`.** `Capabilities::support` /
  `CapabilityQuery` / `SupportLevel`
  (`crates/incin-core/src/tensor/backend/`), per-backend capability rows,
  `cargo-incin doctor`-style discovery (facade `experimental::` tooling in
  `crates/incin/src/bin/cargo-incin.rs`). Fail-closed: unknown = typed refusal
  (`UnsupportedDType`, arity errors), recently hardened by `validate_cuda_f32_kernel`
  guards and the training-row oracle (see `custom-op-autograd.md` §1, F6).
- **`define_backend_operations!` proc macro (#8, landed).**
  `crates/incin-macros/src/backend_operations.rs` + tests
  (`crates/incin-macros/tests/backend_operations.rs`,
  `compile_pass/backend_operations_grammar.rs`, `compile_fail/`). One
  invocation declares `Execute` shells + companion `Capabilities` routing with
  expansion-time coverage checks and offending-operation spans. Built-in
  backends keep their own private DRY macros; this is the first public
  backend-authoring surface for downstream authors.
- **Custom ops + custom training tests.** `Operation` trait with
  `infer_outputs` contract (`crates/incin/examples/custom_dtype_and_operation.rs`
  is the end-to-end example: contract → kernel → `dispatch::execute` →
  fail-fast proof), `Tensor::apply_op` (single-input shape-preserving entry,
  returns `Dyn` layout deliberately), `execute_shaped_n` for the rest,
  `DifferentiableOp<B>` with associated `Saved` + `Dtype`
  (`crates/incin-core/src/tensor/backend/differentiable.rs`), public
  backend-generic `gradcheck` (`incin_core::exec::gradcheck` +
  `crates/incin-backends/src/cpu/gradcheck_storage.rs` so it ships outside
  `#[cfg(test)]`), uniform `tape_record`/`tape_record_with` seam on all four
  backends, 2286-tuple training oracle that fails rows claiming `training`
  without recording (`custom-op-autograd.md`).
- **`DType` / `ConstDType` + D-110 unseal.** `crates/incin-core/src/tensor/dtype/traits.rs`:
  `DType` (any logical dtype, no `DTypeId` needed), `ConstDType` (compile-time
  identity), `BuiltinDType` (narrow fast-path vocabulary, still required by
  distributed plan digests, collective tuning keys, kernel-table fast paths,
  safetensors export — each documented on the trait), `TensorElement` now OPEN
  via blanket impl over `bytemuck POD + Copy + Debug + Send + Sync`
  (commit `f5a17de0`, D-110/#96), `PlainDType` for real scalar elements,
  semantic markers (`FloatDType`, `IntDType`, `BoolDType`, `QuantDType`).
  Postcard state envelope round-trips custom dtypes; safetensors export refuses
  them with a typed error.
- **`External(DeviceKey)`.** `crates/incin-core/src/tensor/device.rs`:
  `DeviceKind::External(DeviceKey)` third-party device identity alongside
  first-class variants (incl. upcoming ROCm); `Custom(u64)` migrates. Same
  commit `f5a17de0`.
- **Expression-DSL codegen (`dsl` / `ir` / `jit`).**
  `crates/incin-backends/src/codegen/` (`mod.rs`, `catalog.rs`, `dsl.rs`,
  `fragment.rs`, `ir.rs`, `jit.rs`, crate-private `fuser.rs`):
  `dsl::{define_unary_custom_op, define_binary_custom_op}` builds a
  `KernelDefinition` from an `IrExpr` closure with symbolically derived
  backward (`unary_fused_backward` differentiates forward IR — no second
  declaration to drift, `what-to-take-from-sota.md` §5); `CpuJitKernel` is a
  host f64 tree-walking reference (not a compiler), `CudaJitKernel` goes
  through production NVRTC; `catalog` declares the pointwise vocabulary as
  `IrExpr`s; `fragment::lower_scalar` + `kernel::scalar` templates are the body
  seam. IR coverage gap: no `erf`, `asinh`, rounding family.
- **`incin::experimental`.** `crates/incin-core/src/lib.rs` (`pub mod
  experimental`), distributed mesh/placement macros, compiled plan preview
  (CMP-005 group + `compiled` module, fail-closed, exclusivity proof, no
  lowering yet — #112 step 2), facade contract tests
  (`crates/incin/tests/facade_contract.rs`).
- **Book + how-to chapters, `cargo test --doc` compiling book examples.**
  `docs/book/src/` (`custom_operations.md`, `howto_debug_shapes_errors.md`,
  `howto_training_loop.md`, `experimental.md`, `whats_not_finished.md`, …);
  `cargo xtask docs --check`, shape audit, API-example build gates.

## 1. Novelty-mechanism inventory

Each row: the mechanism that actually causes researchers to pick a platform
(verified against sources), who proves it, and incin's analogue with file refs.

| # | Mechanism (the causal bit) | Who proves it | Incin analogue: exists / partial / missing |
|---|---|---|---|
| M1 | **Composable function transforms as primitives** — `jit/grad/vmap/pmap` compose arbitrarily; novelty = writing a plain function and getting batched, differentiated, compiled variants free. Pytrees make nested params first-class; tracing (`jaxpr`) is the shared substrate. | JAX docs: composable transforms, tracing, pytrees (docs.jax.dev key-concepts, 101-transformations). `grad(jit(vmap(f)))` per-example-grads one-liner in jax README. | **Missing.** No transform layer at all. Autograd is tape-over-storage (`BackwardFn` over storage, all four backends walk under `GradMode::Disabled` — `custom-op-autograd.md` §4: rules out higher-order gradients, gradient penalties, meta-learning, HVPs). No vmap/pmap analogue; batching is manual. Closest substrate: `IrExpr::diff` (symbolic differentiation of forward IR, `codegen/catalog.rs`) — a transform on *expressions*, not on *functions*. |
| M2 | **`jax.experimental` staging** — a named, importable place where half-baked ideas live with users' consent; graduation is social, not technical. | `jax.experimental.pallas`, `checkify`, odeint canonically built on `custom_jvp/vjp` (JEP 2026-custom-derivatives). | **Exists (structure), partial (process).** `incin::experimental` exists with facade contract tests, and `codegen::{dsl,jit}` are explicitly "the experimental custom-operation story". What is missing is the *graduation process*: no documented path experimental → catalog → stable, no deprecation/stabilization bar. |
| M3 | **Custom-derivative + custom-primitive escape hatches** — `custom_jvp`/`custom_vjp` for math-level novelty without leaving the system; `core.Primitive` + rules for calling out to solvers/simulators. Design explicitly preserves pdb-debuggability (runtime value inspection was a design goal, JEP). | JEP 2026-custom-derivatives (odeint as canonical user; `custom_jvp_call` acting like `core.call` so vmap passes through). | **Partial, strongest area.** `DifferentiableOp` (`Saved` associated type ≈ `ctx.save_for_backward`, framework-owned — `custom-op-autograd.md` §2 judges this the piece that must not change) + `Tensor::apply_op` (≈ Candle's `apply_op1`, entry point on the tensor) + `define_unary/binary_custom_op` DSL (≈ `custom_jvp` for pointwise math: forward IR, backward derived). Gaps: recipe is per-`Dtype` associated type (author writes `Square<f32>` + `Square<f64>`; none of Candle/PyTorch/Burn ask this — recorded as the outlier in `custom-op-autograd.md` §2); arity-1/shape-preserving fast path vs `execute_shaped_n` cliff for the rest; IR vocabulary missing `erf`/`asinh`/rounding. |
| M4 | **Kernel language mortals can write, staged inside the framework** — Pallas (JAX) / Triton + `triton_op` (PyTorch): kernels written in Python-ish code, debuggable via `interpret=` mode, composable with transforms. The mechanism is *lowering the kernel-authoring cliff without leaving the ecosystem*. | Pallas docs (`interpret=True` runs as `jit` of a scan on CPU — debugging escape hatch); `torch.library.triton_op` docs (traced-into, not opaque, unlike `custom_op`; supports subclasses, AOTInductor, dynamic shapes). | **Partial, CPU-side only.** `codegen::dsl` + `CpuJitKernel` (host reference) + `CudaJitKernel` (NVRTC) is the same shape as Pallas's `interpret` vs device split — but vocabulary is pointwise expressions, not tiled kernels with block specs/scratch memory. No block/grid/scratch vocabulary, no autotune, no `interpret`-mode parity story beyond f64 reference. A researcher wanting FlashAttention-style tiling novelty has no staged path. |
| M5 | **Eager-first with an escape hatch** — default is "just run it, print it, pdb it"; `torch.compile` is opt-in speed. Novelty survives because iteration never requires the compiler's permission. | PyTorch blog FlexAttention (dynamo capturing globals, `mask_mod`/`score_mod` as plain Python callables lowered to fused Triton; backward/vmap reuse). `torch.compile` user-defined-Triton tutorial (2.3→2.6 `triton_op` evolution). | **Exists, inverted.** Incin is eager by default too (CPU backend runs immediately, `doctor` discovers). But the *direction* is inverted: incin's proofs run at compile time, so the "just try it" moment requires satisfying the type system first (see friction walks). PyTorch lets you be sloppy then tighten; incin requires tightness then runs. No `RUSTFLAGS`-style "dynamism escape hatch" (e.g. `Dyn` shapes exist but per-walk they compose poorly with the typed surface). |
| M6 | **`__torch_dispatch__` subclassing: the "try anything" mechanism** — tensor subclasses + modes intercept *every* op; `torch.library.custom_op`/`triton_op` + `register_autograd` define subsystem behavior once. Composability footguns documented (`custom_op` opaque vs `triton_op` traced-into; subclass+compile backward-guard issues #114410/#114389 fixed via `__force_to_same_metadata__`, auto-in-graph ctors). | PyTorch `_python_dispatch.py` (`TorchDispatchMode`, `is_traceable_wrapper_subclass`), `triton.py` source, issues #114410/#114389/#160333. | **Partial.** `Operation` + `Execute<B>` + `Capabilities` is the same interception point (any downstream type can implement `Execute<MyOp> for MyBackend`, and `define_backend_operations!` now generates the shells). Missing: a *tensor-level* subclass story (interpose on existing ops for an existing backend, e.g. "log every matmul" or "quantize every linear") — today you implement a new backend or nothing; there is no mode/subclass that wraps CPU and overrides one op. Also missing: `register_autograd`-style one-line backward registration; backward = hand-written recipe + oracle verdict. |
| M7 | **Radical readability: whole stack fits in head** — one person can read Tensor → LazyBuffer → UOp → renderer → runtime in a sitting; `tinygrad/nn` is a suggestion, not a framework. Novelty mechanism: *the cost of understanding is near zero, so the cost of forking/modifying is near zero*. | tinygrad intro deck (LazyBuffer→LazyOp→UOp→Code pipeline), shapetracker docs (movement ops symbolic, zero-cost views), `extra/models/unet.py` as plain code. | **Partial, trending away.** Incin has the opposite bet: machine-checked proofs instead of readability. The catalog/descriptor split is principled, but the trait count is CuTe-menu-like (74 shape traits per `what-to-take-from-sota.md` §2 — "a menu, not a calculus"). A researcher opening the tree meets `Operation` + `Execute` + `Capabilities` + `SupportsDType` + `StorageBackend` + `HostInterop` + `VariableBackend` + `TensorTarget` before writing one kernel (count the bounds in `custom_dtype_and_operation.rs`). tinygrad's novelty loop is "read 200 lines, change 20"; incin's is "satisfy 8 traits, then run". |
| M8 | **Symbolic shape-tracker: movement ops are free** — reshape/permute/pad tracked symbolically as views; reshape-detects-multiview; indexing lowers to UOp AST. Novelty mechanism: *layout experimentation costs nothing at runtime*, so researchers try wild layouts. | tinygrad shapetracker notes (View/ShapeTracker/`to_indexed_uops`, multiview `%`/`//` rendering, symbolic `Variable` with min/max for loop rendering). | **Exists in weaker form.** `transpose_view` just landed (#113, commit `a094f7af`); `iteration::coalesce_dimensions` merges but never reorders — `what-to-take-from-sota.md` §1 proves the gap with the `[4,3]` example and prescribes TensorIterator-style per-operand output strides. Channels-last: zero hits in tree (§3). So: views exist, but the iteration planner does not exploit them, and there is no symbolic layer (incin's answer is proofs, not symbols). |
| M9 | **Backend in days: narrow, complete seam** — adding UNet/TPU support is days of work because the backend contract is small and the conformance signal is immediate. | tinygrad claims + structure (single UOp IR, per-backend renderer/runtime); evidence is architectural, not a single citation — the mechanism is *seam narrowness*. | **Partial, improving fast.** `define_backend_operations!` (#8) + `External(DeviceKey)` + descriptor-keyed catalog + conformance harness (`crates/incin-backends/src/conformance/`: fixtures, operands, plan, shaped — CPU only, accelerators hand-verified) narrows the seam deliberately. But the seam is still wide: custom backend example implements ~10 traits/handles before one kernel runs; WGPU bool-dtype gap (00-index correction: comparisons type `Bool` but WGPU declares f32-only — "add a dtype, then wire, then register") shows dtype breadth multiplies the seam. |
| M10 | **Systems control + Python interop (Mojo/MAX)** — win where Python's ecosystem meets systems-level hotspots: Mojo→Python stable (`PythonObject`), Python→Mojo preview (25.5/25.6, `pip install mojo`), selective hotspot migration. Novelty mechanism: *meet researchers where they already iterate*. | Modular forum (Python→Mojo preview mid-2025), DeepEngineering Mojo–Python survey Oct 2025 (two-way street maturing, `PythonObject` overhead warnings), MAX 26.1 (Python API out of experimental, eager + `model.compile()`). | **Missing.** No Python bindings at all. Iteration speed in Rust-compile units is the single largest structural disadvantage for novelty (see §3, Walk notes + recommendations). Every wild idea pays `cargo build` latency; no notebook story; no `TinyJit`-style realize-and-time loop. |
| M11 | **Compiler-stack novelty (TVM BYOC / MLIR Transform)** — pattern-table offload (`FuseOpsByPattern` → `MergeCompositeFunctions` → `RunCodegen`), custom datatypes via lowering funcs, scheduling languages (autoTVM/Ansor/MetaSchedule search, PEAK-over-Transform critique of low-level IR-centric APIs). | TVM BYOC tutorial + `external_library_dispatch` arch doc; BYOC custom-datatypes RFC #3060; XTC paper (arXiv 2512.16512) on scheduling-language coupling; PEAK paper (scheduling usability). | **Partial (BYOC shape), missing (scheduling).** `Capabilities` + pattern-fusable `compiled` preview + `fragment` representation is BYOC-shaped (offload fused groups to external codegen; `what-to-take-from-sota.md` §6: "the gap is the pass, not the representation"). Custom datatypes via lowering funcs ≈ incin's `DTypeRegistry` + descriptor-driven dispatch (but incin additionally types them — stronger). Scheduling language: nothing (no TE/schedule split; iteration planner has no reorder — M8). |
| M12 | **Failure-mode mechanisms (what kills novelty)** — Candle: closed `Op`/`DType` enums + per-backend `cpu_fwd/cuda_fwd/metal_fwd` triple-write with backward defaulting to error (`BackwardNotSupported`); novelty dies at arity 4 (only CustomOp1/2/3), at the 4th backend, at the missing `bwd`. Burn: backend-extension is the *only* path — new op = new backend trait + per-backend impl + separate `Backward` state machine + `Autodiff<B,C>` impl; discussion #4535 documents the specialization wall (can't provide default forward + custom backward generically); kernel-authoring cliff = CubeCL/WGSL for a fused pointwise op. TF-graph rigidity: dynamism outside the traceable subset silently traces wrong (JAX's documented `print`-once/`jit`-caches-trace gotcha is the mild form). | Candle `custom_op.rs` + `op.rs` source (triple-fwd, `bwd` default-err, `Op` closed enum); Burn book backend-extension + custom-WGPU/CubeCL kernel chapters + discussion #4535; JAX 101-tracing doc (side-effect/trace-time gotchas). | **Incin's own kill-mechanisms (honest):** (a) `DifferentiableOp::Dtype` associated type = per-dtype recipe multiplication (§1 custom-op-autograd "outlier"); (b) closed consumers still on `BuiltinDType` (dist plan digests, collective tuning keys, kernel-table fast path, safetensors — documented on the trait, but each is a wall a dtype-novelty hits); (c) `BackwardFn` over storage + `GradMode::Disabled` walk = no higher-order grads, ever, until GRD-006 rewrite; (d) 8-trait backend seam for kernel novelty; (e) compile-time proofs front-load the iteration cost (M5 inverted). None is fatal alone; together they define the friction budget in §2. |

## 2. Friction audit: three wild ideas walked through the CURRENT tree

Method: each walk follows the actual files/traits an author touches, names the
concrete error or wall, and estimates honestly. Times assume a strong Rust
researcher who has read the book once.

### Walk A — a new attention variant with custom backward (e.g. ALiBi-tilted online softmax + learned temperature, fused forward, hand-derived backward)

Steps in today's tree:

1. **Express the math.** Options: (i) compose built-ins eagerly (works in
   minutes; `FusedAttention` CPU reference at `crates/incin-backends/src/`
   shows the online-softmax pattern, #104); (ii) `codegen::dsl`
   (`define_unary/binary_custom_op`) — works if the novelty is pointwise
   (bias add, scale, mask). ALiBi tilt = add of a broadcast bias: fits. Learned
   temperature + online softmax normalization does NOT fit the unary/binary
   vocabulary — falls to (iii) full custom `Operation`.
2. **Declare `Operation`.** Implement `Operation` (`KEY` namespace/name/version
   + `Attributes` serde + `infer_outputs` shape/dtype contract). The
   `custom_dtype_and_operation.rs` example is genuinely complete here (~80
   lines). For attention the contract must type the reduction (softmax over
   last axis), the broadcast of the bias, and causal masking — no
   `BroadcastShape`-for-attention helper exists; author hand-proves with
   `ShapeBuf` comparisons, and a mismatch is `DescriptorError::InvalidAttribute`
   (fail-closed, good) with a hand-written reason string (the curated-shape-error
   work in `incin-diagnostics` does not cover custom ops). **~1–2h**, mostly
   rediscovering what `FusedAttention`'s descriptor already knows but does not
   expose as reusable proof pieces.
3. **Write the CPU kernel.** `impl Execute<MyAttn> for Cpu` consuming
   `ExecutionRequest` (verified descriptor + proven outputs). Straightforward;
   tape recording via `tape_record_with`, `Saved` type holds QKVO stats.
   **~2–4h** for forward; the oracle + `gradcheck` verify it.
4. **Write the backward recipe.** `impl DifferentiableOp<Cpu>` — and here the
   per-dtype associated `Dtype` bites: the recipe is written once per element
   type (`MyAttnBwd<f32>`, `<f64>`; f16/bf16 need `PlainDType` plumbing and the
   CUDA side needs `validate_cuda_f32_kernel` clearance). None of the SOTA
   comparators multiply the authoring cost by dtype. Then the oracle's
   `carries_no_gradient` verdict forces an explicit classification (good —
   fail-closed), and central-difference `gradcheck` (public, backend-generic,
   step-size guidance `1e-2` for f32 baked in) proves it. **~1 day** for a
   correct fused backward with replayed statistics; half of it dtype
   multiplication + learning the `Saved`-lifetime rules (GRD-006 pending).
5. **Port to CUDA/WGPU/Metal.** Per-backend `Execute` + `tape_record` seam
   (now uniform, `pub(crate)`-aligned — the good news from
   `custom-op-autograd.md` §3) + capability row + conformance fixture. CUDA:
   write the kernel or NVRTC-lower the IR (pointwise parts only); the fused
   softmax needs a hand kernel + `validate_cuda_f32_kernel` gate per entry
   (nine entry points guarded after the transmute audit — each new entry needs
   its own guard call, easy to forget, caught only by audit not by type).
   WGPU: WGSL + no-bool-dtype wall if the mask materializes as boolean
   (00-index correction: comparisons type `Bool`, WGPU has no bool — author
   must route masks as f32 0.0/1.0 and hand-justify the lie). Metal: same plus
   MPS-vs-custom decision (#7 closed, but the decision record lives in research
   docs, not in a compiler hint). **~2–5 days** for three accelerators, dominated
   by per-backend tape/replay verification, not math.
6. **Stuck points (brutal list):** (a) no `vmap` — per-head/per-example batching
   variants are manual loops or new ops, not a transform; (b) no higher-order
   grads — FlashAttention-style recomputation tricks that need double-backward
   testing are inexpressible (`GradMode::Disabled` walk); (c) IR vocabulary gap
   (`exp` exists, masking/where-cond on WGPU blocked on bool) forces
   full-custom-op path for ideas the DSL *almost* covers; (d) shape-error
   quality for custom ops is hand-rolled strings, while built-ins get
   `incin-diagnostics` curation — the researcher pays the framework's
   proof-tax without receiving its proof-help.

**Plausible total: 1 day (CPU, f32, proven) → 1 week (all backends, dtypes,
training-clean).** PyTorch comparison: FlexAttention variant = an afternoon
(`score_mod` Python + autograd free). The gap is M1+M4+M6, not effort.

### Walk B — a new dtype with a custom kernel on CPU (e.g. `fp6_e3m2` microscaling block format for weights)

Steps in today's tree:

1. **Declare the dtype.** Implement `DType` + `ConstDType` (key, kind,
   encoding descriptors) — genuinely open, no `DTypeId` edit needed. If scalar:
   also `PlainDType` with `Elem: TensorElement` (blanket impl covers any POD —
   D-110 works as advertised). If block format (like Q8_0): `QuantDType`, no
   `Elem`, like the precedent. **~1h. This is the unseal working.**
2. **Construction + host interop.** `from_slice` requires `PlainDType`
   (compile-time gate — good); block formats go through bytes paths. Works.
   **~1h.**
3. **First `matmul` attempt — the wall.** Capability query for
   `(MyDtype, Cpu, matmul)` → typed refusal: kernel table fast path keys on
   `builtin_id()` (`traits.rs` documents it). Author implements
   `Execute<MatMul>`-equivalent for the new descriptor… and discovers the
   closed consumers one by one, each a *different* typed error at a *different*
   layer: `exec/precision.rs Exact<K>` (bound is `ConstDType + FloatDType` —
   fp6 must claim `FloatDType`, which requires `PlainDType`, which block
   formats cannot satisfy — **the semantic-marker lattice contradicts block
   novelty**); distributed plan digest fingerprints by `DTypeId` (collective
   path refuses); safetensors export matches closed builtin table (must use
   postcard envelope + document the gap); `autocast`/`is_host_float` helpers
   key on descriptors but the *rules* (`precision.rs`) assume float =
   f16/bf16/f32/f64. **~1–3 days** of wall-by-wall discovery; every wall is
   fail-closed (no silent wrong numerics — the framework's honor holds), but
   the map of walls exists only in the `BuiltinDType` trait docs, not as a
   single "custom dtype authoring" checklist with one error that points to the
   next step.
4. **Custom kernel via `define_backend_operations!`.** The macro accepts custom
   ops and custom paths (grammar tests prove it) — but the invocation still
   needs an `Execute` shell + capability handler + `SupportsDType` impl, and
   the *coverage check* (expansion-time, good) reports the first missing op,
   not the whole frontier. Author iterates macro-error → row → fixture for each
   op in the training path (matmul, cast, quantize, dequantize…).
   **~1 day per op-family** including oracle + gradcheck (STE gradient per the
   #93 decided contract: PyTorch-style straight-through, per-op admission).
5. **Stuck points:** (a) the marker-trait lattice (`FloatDType: PlainDType`,
   `QuantDType` disjoint) has no home for "block float that participates in
   matmul accumulation" — the researcher's format must either lie about being
   plain or forgo the float rules; (b) five subsystems still demand
   `BuiltinDType` with no adapter shim (no "custom dtype with explicit
   capability declaration" path — it is refuse-only, never delegate); (c) no
   lowering-function registry à la TVM RFC #3060 (register dtype + per-op
   lowering to calls) — incin has the *stronger* primitive (typed descriptors)
   but not the *research* primitive (try-untyped-first lowering); (d) error
   chain is per-wall, not a guided tour: each refusal names its own site, none
   names the remaining four.

**Plausible total: 1 day (declared + CPU-constructible) → 1–2 weeks
(executing in a training graph with casts, STE, serialization story).** TVM
comparison: software-datatype emulation = register + lowering funcs + run, in a
day, because dynamism is the default. Incin's proofs make each step safer and
slower.

### Walk C — a new collective for expert-parallel routing (e.g. capacity-capped sparse dispatch + combine over the reference transport)

Steps in today's tree:

1. **Primitives exist (partially).** `scatter_add`, `log_softmax`,
   `logsumexp_dim`, `one_hot` landed as catalog ops with CPU executors (#103
   first slice); `sort`, `nonzero`, `repeat_interleave`, `bincount`,
   `grouped_matmul` absent. `DuplicateIndexRule` has two variants
   (`LastWriteWins`, and the CPU-refused one) — `Accumulate` semantics needed
   for combine is inexpressible, and adding a variant is a wire-format break
   (`Serialize`/`Deserialize`, not `#[non_exhaustive]` — 00-index correction).
   Researcher starts by reading `103-routing-primitives.md` + `102-moe-typing.md`
   + `102-nameable-spans.md` (offset-array-as-shape target, option B CPU
   interim). **~half day** just to learn what is expressible.
2. **Typing the dispatch.** Target: static outer shapes + E+1 offset array
   (decided). Dynamic spans are "not usefully nameable" (audit) — the
   researcher's capacity-capped routing (data-dependent counts per expert) must
   be expressed as offsets + aux-loss-in-`Module::Output` + E-const/k-const +
   gate-weight-only gradients. If the novelty IS dynamic capacity (the actual
   research question), the type system answers "express it as static + offsets"
   — which pre-answers the research. Workaround: `Dyn` shapes, but then the
   MoE typing proofs do not apply and the author is outside every helper.
   **Stuck point #1: the framework's strongest typing decision (static MoE
   shapes) is also a novelty ceiling for routing research.**
3. **Collective + transport.** Reference transport + data-parallel exists (#97),
   FSDP-ZeRO2/TP/PP in-process runs (#99), `DeviceMesh` + `ValidMesh` typed
   placement. A *new* collective (sparse dispatch) means: new descriptor in
   `dist/plan/` (fingerprinted by `DTypeId` — custom-dtype experts refused,
   see Walk B), tuning key `CollectiveTuningProblem::new_static` (same wall),
   NCCL error-field honesty precondition (D-110 preconditions). No GPU CI
   runner exists (#82 second half outstanding — "everything hardware-gated is
   unfalsifiable"), so multi-GPU routing novelty cannot be *proven* in CI, only
   in-process. **~1 week** for the plan/descriptor/transport slice, untestable
   on real NCCL.
4. **Training through it.** Gate-weight-only gradients (decided default) +
   aux loss in output tuple. Custom backward through dispatch/combine needs
   `DifferentiableOp` per dtype (§Walk A) + the oracle's training-row
   enforcement. Determinism contract + `DuplicateIndexRule::Accumulate`
   warning stand (00-index). **~days**, assuming the wire-format break for the
   third enum variant is approved.
5. **Stuck points:** (a) wire-format-closed `DuplicateIndexRule` (2 variants,
   serialized) makes the core combine semantic a breaking change; (b) static
   MoE typing vs dynamic-capacity research question (ceiling, not friction);
   (c) no NCCL CI → distributed novelty is write-only; (d) plan digests keyed
   on `DTypeId` close the dtype×distributed composition (the exact cross
   researchers want: new dtype ON new routing).

**Plausible total: 1 week (in-process CPU demo with static shapes) → 3+ weeks
+ blocked (real NCCL proof, wire break, dynamic-capacity expressiveness).**
JAX comparison: `pjit`/`shard_map` + explicit sharding lets the same idea run
on real hardware behind a Python script in a day.

## 3. Recommendations, ordered by novelty-unlock per effort

Sized S (<1wk), M (1–3wk), L (month+). Each names what exists to build on and
which wild idea it untryies.

1. **Custom-dtype authoring checklist + guided refusal chain (S).** One book
   chapter + one error: every `BuiltinDType`-gated refusal points at the
   checklist (`docs/book/src/howto_custom_dtype.md`), which enumerates the five
   closed consumers, the postcard-vs-safetensors split, the marker-lattice
   decision tree (scalar→`PlainDType`, block→`QuantDType` + STE per #93), and
   the per-op admission path. Build on: `BuiltinDType` trait docs (already the
   map), `custom_dtype_and_operation.rs` example, `cargo test --doc` gate.
   *Unlocks: Walk B's first week becomes a day; every future dtype issue stops
   re-discovering the walls.* Size S.
2. **Per-dtype recipe removal: dtype-dispatch inside `DifferentiableOp` (M).**
   The documented outlier (`custom-op-autograd.md` §2): move from associated
   `Dtype` (N impls) to one recipe with checked dtype dispatch (match on
   descriptor, fail-closed default) — the Candle `cpu_fwd`-matches-storage
   shape but typed. Precedent: `SupportsDType<K>` already resolves per
   descriptor; `unbroadcast` alignment debate shows where the trait belongs.
   Coordinate with GRD-006 (Saved-lifetime answer lives there too).
   *Unlocks: Walk A backward authoring halves; every custom-op author stops
   paying the dtype tax.* Size M.
3. **`define_backend_operations!` + `define_unary/binary_custom_op` coverage
   expansion: shape-changing + multi-input DSL ops with derived backward (M).**
   Today `apply_op` is arity-1/shape-preserving and the DSL is unary/binary
   pointwise; anything else falls off the `execute_shaped_n` cliff. Add:
   codegen-supported `where/mask/scale-broadcast` trio (unblocks attention
   bias/mask on all backends incl. WGPU-bool workaround as a *typed* mask
   dtype, not f32 lies), then reduction-shape ops (softmax-normalize) with
   symbolic backward through `IrExpr::diff`. Build on: `catalog` IrExpr
   vocabulary, `unary_fused_backward` proof pattern, macro grammar tests.
   *Unlocks: Walk A ideas 1–2 stay in the DSL (afternoon) instead of
   full-custom (week).* Size M.
4. **Interactive shape-debugging: proof-error quality bar (S–M).** Fix the
   known `BroadcastShape` bottom-`()` diagnostic (named uninhabited bottom per
   00-index Tier 1 note), extend `incin-diagnostics` curation to custom-op
   `infer_outputs` failures (a `DescriptorError` renderer that suggests the
   nearest valid shape, à la rustc/checkify error values), and ship
   `cargo-incin explain <error>` replaying the failed proof with values.
   Build on: `crates/incin-diagnostics/` (humanize/mismatch/typenum),
   `howto_debug_shapes_errors.md`, `doctor`.
   *Unlocks: converts the proof-tax from dead time into learning time; every
   walk benefits; directly answers JAX's "debugging transformed code" friction
   with a better offer (errors that teach).* Size S for the bottom fix +
   chapter; M for the renderer.
5. **Kernel-DSL escape hatch: tiled-kernel vocabulary with `interpret` parity
   (L).** Extend `codegen::{dsl,ir,jit}` with block/grid/scratch vocabulary
   (Pallas `BlockSpec`/`GridSpec` shape, Triton-lowering later), `CpuJitKernel`
   as the `interpret=True` reference with NaN-poisoning for OOB (Pallas
   changelog precedent), autotune hook (`tuning::` identity + service already
   exist). Explicitly NOT a Triton dependency on day one: vocabulary +
   reference + NVRTC/WGSL/Metal lowering of the same IR. Build on: `fragment`
   lowering, `compiled_fusion.rs` stepwise-vs-fused proof, tuning service.
   *Unlocks: Walk A fused-attention novelty; the first kernel a mortal writes
   that runs everywhere.* Size L (vocabulary M + lowering per backend).
6. **Tensor-level interposition (mode/subclass analogue) (M).** A wrapper
   backend/mode that delegates to an inner backend and overrides selected ops
   ("log every matmul", "quantize every linear", "STA mask every attention")
   — today's only path is a whole new backend. Design: `VariableBackend`-style
   decorator + `__torch_dispatch__`-like `Mode` with `ignore_compile_internals`
   equivalent for the `compiled` pipeline; must compose with capability
   queries (delegation, not shadowing) and the training oracle (override must
   preserve/re-declare `training`). Build on: `dispatch` + `ExecutionContext`,
   `Capabilities` composition, `polar_cartesian.rs` own-node-list teaching test.
   *Unlocks: the "try anything in 20 lines" loop (Walk A/C instrumentation,
   Walk B quantization shims) without the 8-trait seam.* Size M.
7. **Python bindings for iteration speed (L, staged).** `pyo3`-based
   `incin-py`: construct/run/inspect from Python, `gradcheck` + `doctor` exposed,
   notebook-friendly; later: export traced closures to `KernelDefinition`
   (the reverse-MoJo direction: Python for iteration, Rust for proof). This is
   the Mojo-lesson item (meet researchers where they iterate) and the
   highest-leverage novelty input that is not a compiler feature. Stage 1 needs
   no dynamism compromise: bind the `Dyn` surface only. Build on: `Dyn`
   layouts/shapes, postcard envelope, conformance fixtures as the Python-side
   oracle.
   *Unlocks: afternoon-scale novelty for the Python-native majority; every walk
   shortens by the compile-latency delta.* Size L (S for a minimal REPL
  k, M–L for maintained bindings).
8. **`experimental` graduation process (S).** Document: entry bar (what lands
   in `experimental`), user-consent contract (facade feature + book banner,
   following the `experimental-compiled-pass` fixture pattern), graduation bar
   (oracle + multi-backend + book chapter + generated-docs regen), removal bar
   (what happens to users). Build on: facade contract tests, `feature-matrix`
   + `check-docs.py` gates, `experimental.md` chapter.
   *Unlocks: social permission to land half-baked novelty fast (the actual
   `jax.experimental` mechanism); de-risks items 2/3/5/6 landing
   incrementally.* Size S.
9. **Higher-order autograd design (L, GRD-006 companion).** `BackwardFn` over
   tensors (not storage) + graph-owned tapes so recipes can record and grads
   can flow twice; `needs_input_grad`-style per-input opt-out decided with it
   (the 127-site edit noted in `custom-op-autograd.md` §2 belongs here, not
   before). *Unlocks: gradient penalties, meta-learning, HVP-based research —
   currently inexpressible, not merely slow.* Size L.
10. **Distributed provability (M, mostly #82).** Self-hosted NVIDIA runner +
    NCCL conformance posing accelerators (extend `conformance/` beyond CPU) +
    JSON artifact → verified-on column. Without it Walk C novelty is
    write-only. *Unlocks: expert-parallel/collective research anyone believes.*
    Size M (infra, not design).

Order by unlock/effort: 1 (S, immediate) → 8 (S, enables the rest socially) →
4-bottom-fix (S) → 2 (M, dtype tax) → 3 (M, DSL cliff) → 6 (M, 20-line loop) →
4-renderer (M) → 10 (M infra) → 5 (L kernels) → 7 (L python) → 9 (L H-O grad).

## 4. What NOT to copy from SOTA

- **Dynamism that breaks proofs.** JAX's trace-time vs run-time split
  (`print` runs once, `jit` caches by type, data-dependent branches fail under
  `vmap`/`jit`) and TF-graph rigidity are the same disease: the system accepts
  a program it cannot faithfully run. Incin's compile-time proofs are the
  differentiator (`what-to-take-from-sota.md` "what incin has that these do
  not": reified `ProofLevel`, operation legality at compile time, typed
  distribution). Python bindings (rec. 7) must bind the `Dyn` surface and must
  never smuggle unproven shapes past `Validated<O>`. Eagerness yes; silent
  trace lies no.
- **Stringly-typed registries.** TVM's `register_op` + FFI lookup strings
  (`tvm.custom_datatypes.lower.Add.llvm.bfloat`) buy day-one speed and charge
  permanent debuggability: misspellings fail at lowering, not at authoring.
  Incin's `OperationKey { namespace, name, version }` + `define_backend_operations!`
  expansion-time coverage with spans is already the better answer — extend it
  (rec. 3), don't bypass it with string registries.
- **Magic that defeats fail-closed.** `torch.compile`'s graph-capture
  conveniences (capturing globals without explicit inputs, recompilation
  avoidance heuristics) and dynamo's constructor-tracing workarounds
  (`allow_in_graph`, auto-in-graph ctors) are magic with a correctness tail
  (issues #114410/#114389 backward-guard saga). Incin's oracle (F6: training
  rows must record or fail) and `validate_*` guards are load-bearing honesty
  machinery. Any mode/interposition (rec. 6) or `compiled` lowering (#112) must
  preserve the invariant: *an unverified claim is a test failure, not a silent
  hole*. The double-record finding in `architecture-review.md` (forward built
  of built-ins counted gradients twice, oracle measured depth not exactness)
  is the standing warning: every new convenience needs its oracle verdict.
- **Menu-traits instead of calculus, and closed enums as the default.**
  Don't grow 74 shape traits to 100 (CuTe lesson: composition/complement/
  division/product derive the rest — 0.3.0-scale, gated on nesting), and don't
  repeat Candle's closed-`Op`-enum + arity-capped `CustomOp1/2/3` + per-backend
  triple-write shape. The open `Operation` + `DType` (no `DTypeId` edit) +
  `External(DeviceKey)` direction is right; keep widening bounds
  (`BuiltinDType → ConstDType`, descriptor-keyed subsystems per #96) instead of
  minting new closed vocabularies — with the explicit exception of wire formats
  (`DuplicateIndexRule` break must be deliberate, versioned, documented).
- **Inlining the compiler into the research loop.** TVM/MLIR scheduling
  languages (Ansor templates, Transform dialect SSA-state juggling) optimize
  the last 2× at the cost of the first afternoon (XTC/PEAK critiques agree:
  IR-centric, tightly-coupled search spaces). Incin's order in
  `what-to-take-from-sota.md` stands: TensorIterator reordering (measurable,
  bounded) → ChannelsLast → fusion lowering (#112) → algebra rewrite later.
  Rec. 5's kernel vocabulary must stay mortal-writable (Pallas `interpret`
  parity first, autotune second, scheduling-search never as a novelty
  prerequisite).

## Sources relied on most

JAX: composable-transforms + tracing/pytrees docs (`docs.jax.dev`: key-concepts,
101-transformations); Pallas kernel docs + `pallas_call` API (`interpret` mode);
JEP 2026-custom-derivatives (`custom_jvp`/`custom_vjp`, odeint, pdb-debugging
goal); checkify guide + `jax.debug.print/breakpoint` docs (transformed-code
debugging friction). PyTorch: FlexAttention blog (dynamo capture, `triton_op`
lowering, autograd/vmap reuse); user-defined-Triton tutorial (`triton_op` vs
`custom_op` table); `torch/_library/triton.py` + `_python_dispatch.py` source
(subclass traceability, `return_and_correct_aliasing`); issues #114410, #114389,
#160333 (subclass+compile footguns and fixes). tinygrad: intro deck
(LazyBuffer→UOp pipeline), shapetracker/symbolic notes (views, multiview,
`Variable` loop rendering), `extra/models/unet.py` (nn-as-suggestion). Mojo/MAX:
Modular forum Python→Mojo preview; DeepEngineering Oct-2025 interop survey;
MAX 26.1 release notes (Python API graduation, eager + `model.compile()`).
TVM/MLIR: BYOC tutorial + `external_library_dispatch` arch; custom-datatypes
RFC #3060; XTC scheduling paper (arXiv 2512.16512); PEAK paper (Transform
dialect usability). Failure modes: Candle `custom_op.rs` + `op.rs` source
(closed enums, triple-fwd, `BackwardNotSupported`, arity cap); Burn book
(backend-extension, custom WGPU/CubeCL kernel chapters) + discussion #4535
(specialization wall). Incin tree: files cited inline; prior research
`custom-op-autograd.md`, `custom-op-autograd-decisions.md`,
`architecture-review.md`, `what-to-take-from-sota.md`,
`96-extension-points.md`, `102-moe-typing.md`, `102-nameable-spans.md`,
`103-routing-primitives.md`, `101-attention-modules.md`,
`96-custom-dtypes-devices.md`, `00-index.md` corrections.
