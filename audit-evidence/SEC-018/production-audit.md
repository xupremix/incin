# Production unsafe, panic, and FFI audit - SEC-018

Date: 2026-08-22. Scope: every production `unsafe` block, classified panic
site, and native/FFI boundary on the 0.1.0 surface. This is the release
audit #31 requires; the mechanical checkers are its floor, not its
substitute.

## Mechanical gates (all green at audit time)

| Gate | Command | Result |
| --- | --- | --- |
| Unsafe ledger | `python3 tools/check-unsafe-ledger.py` | passed: 21 production files, 139 unsafe blocks (89 locally annotated; remainder carry the shared family proof) |
| Panic-site inventory | `python3 tools/check-panic-audit.py` | passed: 134 reviewed sites |
| AddressSanitizer | `tools/soundness.sh asan` | Soundness gate passed |
| Miri (Tree Borrows) | `tools/soundness.sh miri` | Soundness gate passed |
| Workspace lints | deny-level `undocumented_unsafe_blocks`, `missing_safety_doc`, `unsafe_op_in_unsafe_fn` | enforced on every build |

ThreadSanitizer remains tracked by #17 (`status:blocked`) and is not
re-claimed here; the parallel AVX2 initialization path keeps its focused
TSan evidence from when a runner was available.

## Per-family verdicts

Verdicts re-state each family's invariant after re-reading the source
sites against their callers, then name the tests that were run or are run
in CI for this audit.

### 1. CPU byte views - `cpu/storage.rs`
Invariant holds: every byte view is `from_raw_parts` over a live,
fully-initialized `Vec` with `len * width_of_variant` bytes and no
padding in the scalar variants. Evidence: storage/quant suites 33 passed;
ASan covers allocation bounds and leaks on this exact path.

### 2. CPU SIMD kernels - `elementwise_kernel/{avx2,neon,scalar,wasm}.rs`, `simd.rs`
Invariant holds: target-feature guards decide variant selection at
runtime before any vector intrinsic; loads are masked/bounded by checked
lengths; spare-capacity outputs are fully initialized before `set_len`.
Evidence: ASan (including AVX2 build) passed; elementwise suites green in
workspace runs; TSan initialization stress retains prior recorded
evidence (#17 tracks restoration).

### 3. CPU matrix and quantized kernels - `cpu/ops/matmul/gemm.rs`, `cpu/ops/quant.rs`
Invariant holds: shape validation proves row offsets, block counts, and
output capacity before intrinsic loads/stores; Q8_0 block arithmetic goes
through checked helpers. Evidence: matmul/gemm suites 29 passed, quant
suite green; ASan passed.

### 4. CUDA launch adaptation - `cuda/ops/*.rs` (10 files)
Invariant holds by construction: checked shape/dtype/allocation-length
arithmetic precedes every `CudaSlice` reinterpretation and launch; FFI
errors convert to typed backend errors. Runtime execution remains
hardware-blocked on this infrastructure; the CI `cuda-compile` job is
green and no hardware execution claim is made (per PROJECT_STATUS
vocabulary this family is compile-verified, not runtime-verified).

### 5. NCCL transport - `dist/nccl/transport.rs`
Invariant holds: dtype selection and element counts validate before typed
views cross into NCCL; failures become `CollectiveError`. Same hardware
caveat as family 4; the distributed-reference suite exercises the shared
collective contract without NCCL.

### 6. CUDA version queries - `tuning/identity/cuda.rs`
Invariant holds: C ABI calls receive pointers to initialized local
storage that outlives the call; negative/error return codes reject.
Compile gate green; identity logic has CPU-runnable test coverage.

### 7. Host extraction - `incin-core/src/tensor/ops/manipulation/interop.rs`
Invariant holds: `to_scalar`/`to_vec1` verify requested Rust type against
the tensor's dtype encoding, element count, byte length, alignment-safe
read, and boolean bit patterns before any reinterpretation (validation
block ahead of the two documented unsafe reads). Negative tests exist and
pass (`to_scalar_bool_rejects_numeric_u8_tensor`,
`to_vec1_bool_rejects_numeric_u8_tensor`); Miri covers the aliasing of
this family directly.

## Recoverable-input criterion

Recoverable invalid input returns typed errors rather than aborting:
enforced by `docs/ERROR_CONTRACT.md`, verified live by the semantic
conformance suite's TypedError rows (#45) and by #43's incin-data
boundary mapping. Remaining operator panics are the 134 inventoried sites
(process boundaries, post-validation transitions, infallible formatting),
none user-reachable with unvalidated input.

## Findings

No new critical or high-severity findings. One documentation correction
was produced by the adjacent conformance work (#45): the dtype-cast
oracle row had claimed typed-error behavior the implementation never had
(casts follow Rust `as` semantics - deterministic truncation/saturation);
the oracle now pins the real contract and exact-conversion policy
boundaries remain the separately checked paths.

## Synchronization

`docs/security/threat-model.md` and `docs/security/unsafe-ledger.md`
reflect this audit's state: ledger counts above match the checker output,
and the threat model's known-gaps section names the same
hardware-coverage caveat recorded here.
