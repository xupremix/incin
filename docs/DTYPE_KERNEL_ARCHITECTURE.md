# DType and Kernel Specialization Architecture

## Decision

Kindle will not maintain one handwritten kernel per operation and dtype. It
will maintain a small set of operation-family templates and specialize them
lazily from a typed execution description:

```text
logical op graph
    -> dtype policy (storage, compute, accumulator, output)
    -> normalized iteration/layout plan
    -> implementation choice (library, fused template, generic template)
    -> backend source/host specialization
    -> device-aware compile and autotune cache
```

This is deliberately more than textual substitution. Dtype, layout,
vectorization, accumulation, hardware features, and numerical mode are all
part of the specialization identity. The first implementation slice lives in
`kindle-backends/src/kernel.rs`: CUDA pointwise, reduction, and normalization
kernels share operation-family templates while rendering F16, BF16, F32, and
F64 storage variants with typed, schema-versioned specialization keys. CUDA
buffers, serialization, and raw launch ABIs carry truthful float-family dtype
and byte-width metadata. The public CUDA capability deliberately remains
F32-only until every reachable operation family is either dtype-safe or
rejected before launch.

The second implementation slice lives in
`kindle-backends/src/dtype_policy.rs`. It is the single capability resolver
for CPU, CUDA, and WGPU and distinguishes storage, fill, random, pointwise,
reduction, and normalization support. It returns explicit storage, compute,
accumulator, and output dtypes. Dispatch, creation, CUDA rendering,
reductions/norms, and WGPU validation now use this resolver, so copying a
dtype's bytes can no longer implicitly advertise every operation.

## Why this matches high-performance systems

- PyTorch's
  [TensorIterator](https://docs.pytorch.org/docs/main/notes/tensor_iterator.html)
  normalizes broadcasting, shape, strides, common dtype, and device before
  kernel dispatch. Its
  [dtype dispatch macros](https://docs.pytorch.org/cppdocs/api/stable/utilities.html#dispatch-macros)
  instantiate one kernel body for selected scalar types.
- [Triton JIT](https://triton-lang.org/main/python-api/generated/triton.jit.html)
  specializes from pointer dtypes and compile-time meta-parameters. Its
  [autotuner](https://triton-lang.org/main/python-api/generated/triton.autotune.html)
  keys candidate selection on workload properties, while its
  [fused softmax example](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)
  shows why eliminating intermediate DRAM traffic matters more than merely
  reducing source duplication.
- [CUTLASS](https://docs.nvidia.com/cutlass/latest/overview.html) decomposes
  kernels into reusable type, layout, data-movement, mainloop, and epilogue
  policies. It explicitly separates input types from accumulator and output
  types and uses specialized tensor-core paths where available.
- [MLIR Linalg](https://mlir.llvm.org/docs/Dialects/Linalg/) represents
  structured iteration once, then applies tiling, fusion, vectorization,
  parallel mapping, and lowering to libraries or hardware intrinsics.
- [XLA:GPU](https://openxla.org/xla/gpu_architecture) treats fusion as its most
  important GPU optimization because intermediate values remain in registers
  or shared memory instead of round-tripping through HBM.
- [cuDNN graphs](https://docs.nvidia.com/deeplearning/cudnn/frontend/latest/developer/overview.html)
  separate a mathematical operation graph from candidate execution plans,
  heuristics, and optional autotuning. They also distinguish tensor I/O dtype
  from compute precision.

The common pattern is: express semantics once, specialize only meaningful
combinations, prefer tuned libraries for structured kernels, fuse
bandwidth-bound graphs, and cache the result.

## Current Kindle gaps

| Area | Current state | Consequence |
| --- | --- | --- |
| CPU elementwise | Arithmetic and unary families dispatch by storage dtype for every layout; a generated AVX2 projection covers F32/F64 contiguous, scalar-broadcast, and vectorizable dense-broadcast inner layouts in serial or Rayon workers | Non-x86 SIMD, packed half types, and arbitrary-stride vectorization remain |
| CUDA elementwise | Float-family source/ABI; normalized strided fallback; metadata-free contiguous/scalar-broadcast paths; scalar ILP and aligned `half2`/`bfloat162`/`float4`/`double2` candidates with masked tails; optional cold-bucket CUDA-event tuning compares access and 128/256/512-thread variants | Real-GPU validation, occupancy pruning, dense-broadcast specialization, and whole-API dtype coverage remain |
| WGPU elementwise | One op-mode shader per family, hardcoded `f32` | Good source reuse, but no native F16 specialization and no dtype/adapter key |
| Reductions/norms | Generated F16/BF16/F32/F64 templates use explicit accumulator policy; last-axis reductions use warp/block cooperation, layer norm uses Welford, and launch-width candidates consume validated cached decisions | Two-pass large reductions, backward normalization, deterministic variants, and real-GPU tuning remain |
| Matmul/conv | Backend-specific handwritten paths | Cannot systematically select tensor cores, libraries, or tuned fallback kernels |
| Caches | Generated CUDA families use a typed canonical key; a bounded device/compute-capability/workload-scoped launch cache stores only synchronized measured winners | Driver/compiler/device UUID identity, concurrency suppression, persistence, and telemetry remain |
| Benchmarks | An ignored release CPU harness covers contiguous, scalar broadcast, dense broadcast, execution-path attribution, seven-sample medians, and same-process Rayon references | CPU model/affinity metadata, tail percentiles, GPU harnesses, and external-framework comparisons remain |

## Core physical model

### 1. Dtype policy

Every dispatch resolves four types, even when they are equal:

```rust
struct DTypePolicy {
    input: SmallVec<DTypeId>,
    storage: DTypeId,
    compute: DTypeId,
    accumulator: DTypeId,
    output: DTypeId,
    math: MathMode,
}
```

Initial policy:

| Operation class | F16/BF16 storage | F32 storage | F64 storage |
| --- | --- | --- | --- |
| Elementwise arithmetic | F32 compute, cast on store; add packed/native vector variants later | F32 | F64 |
| Transcendentals | F32 compute | F32 | F64 where hardware/backend supports it |
| Reductions and norms | F32 accumulator | F32 accumulator | F64 accumulator |
| GEMM/conv | F16/BF16 input, F32 accumulator, policy-selected output | TF32 or F32 accumulator according to math mode | F64 |
| Optimizer state | F32 master state by default | F32 | F64 only when explicitly requested |

CUDA has native half, BF16, float, and double types, with architecture
requirements for some formats; see the
[CUDA floating-point type table](https://docs.nvidia.com/cuda/archive/13.2.0/cuda-programming-guide/05-appendices/mathematical-functions.html).
WGPU F16 is conditional on adapter shader features. BF16 and F64 must not be
advertised as native WGPU arithmetic.

Promotion is a core policy table, not scattered backend matches. Unsupported
combinations fail before allocation or compilation.

### 2. Iteration and layout plan

Introduce a TensorIterator-like internal plan:

```rust
struct IterationPlan {
    shape: SmallVec<usize>,
    operands: SmallVec<OperandLayout>,
    class: LayoutClass,
    vector_width: u8,
    index_width: IndexWidth,
}

enum LayoutClass {
    Contiguous,
    ScalarBroadcast,
    DenseBroadcast,
    Strided,
}
```

The planner validates devices/dtypes, right-aligns broadcast shapes,
coalesces adjacent compatible dimensions, selects 32- versus 64-bit indexing,
and proves alignment. Kernels consume this normalized plan instead of
reimplementing indexing for every operation.

Fast-path order:

1. contiguous and aligned vector loads/stores;
2. contiguous with scalar broadcast;
3. coalesced dense broadcasting;
4. general strided fallback.

The general path preserves coverage. The first three paths are where most
throughput comes from and must avoid uploading shape/stride buffers on every
CUDA launch.

The first implementation slice now provides backend-neutral unary and binary
iteration plans. CPU general views use the plans directly, and CUDA binary
elementwise dispatch consumes the same normalized representation. It converts
right-aligned broadcast axes to zero strides before launch, preserves view
offsets and non-contiguous strides, validates incompatible shapes, and removes
the two input-shape buffers and branches from the generated binary kernel.
The planner now removes unit axes and coalesces adjacent axes only when every
operand preserves the same flattened addressing relation. It also proves four
pointwise classes: contiguous, scalar-left, scalar-right, and strided. CUDA
renders the first three from metadata-free family templates whose launch ABI
contains only pointers, storage offsets, and element count. Only the strided
fallback uploads normalized shape/stride descriptors and reconstructs
coordinates. Layout class is part of the cache family, preventing ABI aliasing.
The CUDA pointwise selector now treats scalar instruction-level unrolling and
packed memory access as distinct strategies. Dense unaligned views may use
scalar unroll widths of two or four without making an alignment claim. Dense
aligned views use `half2`, `bfloat162`, `float4`, or `double2` loads and stores;
the final packet falls back to masked scalar accesses so odd lengths never read
past storage. Strided views remain on the scalar correctness path. The selected
access kind and width are encoded as typed `Scalar` and `Packed` key fields.
With `autotune`, pointwise dispatch consults a bounded cache scoped by device
ordinal, canonical problem identity, and logarithmic workload bucket; it
accepts a cached launch only when it remains a member of the candidates legal
for the current layout and alignment. On a cold bucket, all required source
variants are compiled before timing, each legal candidate receives two warmups
and seven stream-ordered CUDA-event samples, and the median winner is recorded
under the device compute capability and workload bucket. Without `autotune`,
selection remains deterministic. Occupancy pruning, duplicate-tuning
suppression, persistent results, and 32/64-bit index selection remain
subsequent planner stages.

### 3. Kernel specialization key

The target canonical key is:

```text
backend + device architecture/features + operation/fused graph hash
+ input/storage/compute/accumulator/output dtypes
+ layout class + vector width + index width
+ deterministic/fast math mode
+ tile/workgroup policy + source/compiler version
```

Keys are used for:

- in-memory compiled module/pipeline lookup;
- persistent binary cache lookup;
- autotuning results;
- telemetry and benchmark attribution.

Shape values are included only when they affect generated code or tuning
buckets. Otherwise they remain runtime arguments to prevent cache explosion.

The implemented version is a typed, schema-versioned subset containing kernel
family, operation, storage/compute/accumulator/output dtype, normalized layout,
access strategy, index width, and math mode. Its binary `cache_id` includes the
access strategy because scalar-unrolled and packed source differ. Its
`tuning_problem_id` intentionally omits access so legal access and block-width
candidates compete for the same problem. The launch cache adds device ordinal
and logarithmic pointwise or reduction-shape buckets. Block size is a launch
choice, not a source key, because current kernels do not compile it into code.

### 4. Implementation selection

Selection order is performance-driven:

1. vendor or backend library plan for GEMM, convolution, attention, and common
   fused structured graphs;
2. previously autotuned fused kernel;
3. generated family template specialized for dtype/layout/vector width;
4. generic strided correctness fallback;
5. a precise `UnsupportedDType` or `UnsupportedBackendOperation` error.

Do not generate bespoke matmul or convolution source for every dtype when
cuBLASLt/cuDNN/CUTLASS can select tensor-core and data-movement policies more
effectively. Template generation is most valuable for pointwise, reduction,
indexing, and fusion glue.

## Planned repository layout

```text
crates/kindle-backends/src/kernel/
    mod.rs          # internal API and implementation selection
    dtype.rs        # DTypePolicy and promotion/capability tables
    key.rs          # canonical specialization/cache key
    iterator.rs     # normalized broadcast/layout iteration plan
    op.rs           # small scalar/fused expression IR
    cache.rs        # bounded memory cache + versioned persistent cache
    cpu.rs          # typed monomorphization and SIMD lowering
    cuda.rs         # NVRTC templates and launch ABI
    wgsl.rs         # WGSL rendering and adapter feature validation
    tune.rs         # candidates, measurements, cache, telemetry
```

`kernel.rs` is the bootstrap implementation and will be split into this
directory as the model grows.

Backend storage gains authoritative dtype and device metadata. No dispatcher
may infer dtype from a Rust generic when the tensor uses `Dyn`, and no GPU
storage implementation may hardcode F32.

## Phased implementation

### Phase 0 — Measurement baseline

- Add Criterion-style host benchmarks and a backend benchmark binary that
  records latency, effective GB/s, compile latency, cache hit/miss, and
  selected kernel key.
- Cover 1 KiB through 1 GiB, contiguous/broadcast/transpose layouts, and odd
  lengths that exercise masked vector tails.
- Record CPU model/ISA, GPU adapter/architecture, driver, compiler version,
  dtype, and math mode in machine-readable output.
- Establish Candle/PyTorch/library comparisons where dependencies permit.

Exit gate: reproducible baseline artifacts and regression thresholds exist
before changing arithmetic paths.

### Phase 1 — Metadata and dtype policy

- Store `DTypeId` in CUDA and WGPU storage/buffers.
- Make `storage_dtype`, byte lengths, transfers, allocations, variables, and
  serialization use that metadata.
- Add centralized capability/promotion/compute/accumulator tables.
- Enable CUDA F16/BF16 storage and transfer only after byte-level round trips
  pass; keep arithmetic gated independently.

Exit gate: every storage value reports truthful dtype/device metadata and
invalid combinations fail before launch.

Current status: the initial single-input policy and capability matrix is
implemented and exhaustively tested across every current `DTypeId`. Promotion
between differing input dtypes remains intentionally unsupported; when added,
it must extend this table instead of introducing local backend matches.

### Phase 2 — Iterator plus typed CPU kernels

- Build `IterationPlan` once per operation.
- Replace CPU `f64` round-trips with dtype-dispatched, monomorphized kernels.
- Add contiguous vector paths using architecture-specific intrinsics or a
  measured SIMD abstraction, scalar tails, and general strided fallback.
- Use F32 compute for F16/BF16 and F32/F64 native compute otherwise.
- Permit explicit operation-level accuracy exceptions. The initial typed CPU
  GELU keeps F64 polynomial evaluation for F32 storage because native-F32
  evaluation exceeded the established analytical-gradient tolerance.
- Parallelize only above measured grain-size thresholds; small tensors remain
  serial to avoid Rayon overhead.

Current implementation: `Add/Sub/Mul/Div` and the unary activation/
transcendental family use typed kernels for every layout. F16/BF16 compute in
F32, F32/F64 remain native except the documented GELU accuracy policy, and
non-contiguous/dense-broadcast inputs use normalized zero-stride indexing
without allocating per-element index vectors. On x86-64, runtime feature
detection selects explicit AVX2 F32/F64 arithmetic for contiguous and scalar-
broadcast layouts. A small dtype-family macro projects those same writers over
normalized dense-broadcast plans whose inner strides are `(1, 1)`, `(0, 1)`,
or `(1, 0)`; no operation-specific broadcast kernels are generated. These
kernels share allocation-free writers between serial execution and
Rayon-partitioned 128 Ki-element chunks, so large tensors retain explicit SIMD
inside each worker rather than falling back to an opaque scalar closure.
General typed iteration uses a shared serial odometer with a coalesced inner
loop below its separately measured Rayon cutoff, avoiding per-element
division/modulo and duplicate coordinate decoding. Remaining work includes
AArch64/other-ISA SIMD, packed half lanes, and vectorization for arbitrary
non-unit inner strides.

Exit gate: exact dtype preservation, parity tests for every operation family,
and no regression versus the baseline. Contiguous F32 should approach a
substantial fraction of measured memory-copy bandwidth.

### Phase 3 — CUDA typed elementwise and reductions

- The generic raw-pointer launch ABI is now dtype-neutral for F16/BF16/F32/F64,
  and the rendered dtype/element width is checked against storage before
  compilation or allocation. Shape, stride, offset, rank, byte count, and grid
  conversions fail on overflow instead of truncating. Keep public non-F32 CUDA
  support disabled until the remaining operation families have equivalent
  gates and real-GPU byte/numeric tests.
- Use contiguous vector variants (`float4`, `double2`, `half2`, `bfloat162`)
  while keeping the current strided kernel as fallback.
- Move shape/stride metadata into compact launch parameters or cached device
  descriptors instead of allocating/uploading buffers per operation.
- Accumulate reductions/norms in F32 for F16/BF16.
- Include architecture, dtype policy, layout, vector width, and math mode in
  cache keys.

Exit gate: real-GPU correctness, sanitizers, bandwidth benchmarks, warm/cold
JIT measurements, and numerical tolerance tests. Compile-only CI is not enough.

Current status: scalar contiguous and whole-operand scalar-broadcast variants
are selected from the shared iteration plan and already eliminate metadata
uploads. Scalar ILP candidates (`u1`, `u2`, and `u4`) are separate from packed
storage candidates (`half2`, `bfloat162`, `float4`, and `double2`). Packed
selection requires dense addressing, sufficient work, and aligned element
offsets; an in-kernel scalar tail handles incomplete final packets. Generated
source and selection are unit-tested for all four float families, and an
explicit NVRTC compilation test is available when `libnvrtc` is installed.
This host does not currently provide that shared library, so real compiler/GPU
numeric and bandwidth gates remain open. Cached descriptors for genuinely
strided layouts also remain.

Reduction source is now generated from the same storage/compute/accumulator
policy instead of a checked-in F32-only kernel. F16 and BF16 load/store their
native representations while accumulating in F32; F32 and F64 accumulate
natively. Contiguous last-axis reductions assign one output row per block and
use warp shuffles plus one shared value per warp. Arbitrary axes and views use
a dtype-equivalent strided fallback. Both launch ABIs validate axes, bounds,
byte counts, metadata narrowing, grid sizes, and shared-memory sizes before
compilation or launch. Indexed max/min retain the generic path until a measured
segmented arg-reduction design is added.

Normalization source now uses that policy as well. Layer normalization uses a
fused per-row Welford pass, warp-shuffle combination, one `(mean, m2, count)`
record per warp, and a fused affine write. This avoids the old two full shared-
memory reduction trees and improves numerical stability over independent
`sum`/`sum(x*x)` accumulation. Batch-normalization inference is emitted from
the same dtype template with optional affine/running-stat operands. Both paths
validate contiguity, device/dtype agreement, view bounds, byte counts, launch
metadata, and cache identity; unsupported views fail rather than being silently
treated as contiguous.

### Phase 4 — WGPU typed pipelines

- Add dtype-aware storage metadata and pipeline keys.
- Generate F32 and adapter-gated native F16 WGSL from shared family templates.
- Keep BF16/F64 unsupported until a measured emulation path has a compelling
  use case; never silently reinterpret bytes.
- Specialize workgroup size and vector width from adapter limits, then cache
  by adapter identity/features.

Exit gate: software-adapter correctness plus at least one representative
discrete/integrated GPU benchmark lane.

### Phase 5 — Fusion and structured libraries

- Introduce a small side-effect-free scalar expression IR for pointwise chains
  and epilogues.
- Fuse only while register pressure, generated source size, and cache
  cardinality remain bounded.
- Add cuBLASLt/cuDNN/CUTLASS-backed plan selection for GEMM/conv/attention and
  fuse bias/activation epilogues when supported.
- Use cost models to prune candidates before empirical tuning, following the
  same principle as Triton's autotuner.

Exit gate: end-to-end model benchmarks demonstrate lower launch count and HBM
traffic, not just faster isolated microkernels.

### Phase 6 — Quantized and emerging formats

- Model Q8/INT8/FP8/FP4 as storage formats with explicit scale/block metadata,
  not as ordinary scalar dtypes.
- Dispatch dequantize-compute-requantize epilogues or hardware-native block
  scaled kernels.
- Add calibration/rounding/saturation policy and error-budget tests.

Exit gate: accuracy, throughput, and memory reductions are all measured on
representative models.

## Immediate physical execution plan

This is the build order from the current implementation to a production-grade
specializer. Each milestone leaves a complete correctness fallback in place;
an optimization is never the only implementation until it passes its gate.

| Milestone | Physical work | Promotion gate |
| --- | --- | --- |
| A. Capability truth | **In progress:** one operation/dtype/device table now separates storage, compute, accumulator, and output dtype and gates dispatch, creation, CUDA, and WGPU families; mixed-input promotion and hardware byte round trips remain | Exhaustive table tests and byte round trips for every advertised storage dtype |
| B. CUDA elementwise tiers | **In progress:** normalized strided fallback, metadata-free contiguous/scalar-broadcast templates, scalar ILP, packed float-family access, masked tails, and 128/256/512-thread candidates are implemented; optional cold-bucket CUDA-event measurement selects and caches the median winner after JIT | Real-GPU correctness under compute-sanitizer; odd/tail/view tests; material bandwidth win in each declared region |
| C. Reductions and norms | **In progress:** generated float-family reductions use explicit accumulation, contiguous last-axis warp/block cooperation, checked strided fallback, typed Welford layer norm, and 64/128/256/512-thread candidates; value and indexed reductions empirically tune launch width when enabled | Two-pass candidates, normalization backward, adversarial numerical tests, deterministic repeatability, and size/axis benchmarks against CUDA libraries or PyTorch |
| D. Structured operations | Route GEMM/conv/attention to cuBLASLt/cuDNN/CUTLASS plans with explicit epilogue and math policy; retain generated code only as coverage fallback | Shape-bucket benchmark suite, workspace limits, numerical parity, and selected-plan telemetry |
| E. Cache and tuning | **In progress:** typed schema-versioned binary/problem keys, candidate enumeration, pre-timing JIT, CUDA-event warmups/samples, synchronized-median selection, compute-capability identity, deterministic non-autotune fallbacks, and a bounded 1024-entry device/workload cache are implemented; next add occupancy pruning, in-flight suppression, richer device/compiler identity, persistence, and telemetry | No key aliasing; only measured winners are recorded; deterministic invalidation; bounded disk/RAM use; warm launch avoids compilation and retuning |
| F. Fusion | Lower bounded pointwise graphs and structured epilogues to a small scalar IR; estimate register/source pressure before generating candidates | End-to-end launch/HBM reduction and latency win; compile time and cache cardinality stay within budgets |

For elementwise code, the maintained source count should stay approximately
`operation families × backend renderers`, not `operations × dtypes × layouts ×
devices`. Concrete variants are generated on demand from a small scalar
expression IR and a dtype/layout policy. Only performance-significant variants
are materialized: a generic strided fallback, scalar/contiguous forms, and a
small set of packed widths. Shape values remain runtime metadata unless a
tuning result proves specialization worthwhile.

The first hardware benchmark lane should record CUDA architecture, driver and
NVRTC versions, clock/power mode, dtype policy, normalized layout class,
alignment, selected vector width, cache state, and kernel key. Report cold JIT,
warm launch, median, p95, effective bandwidth, and numerical error separately.
Without these fields, an apparent speedup cannot safely drive dispatch policy.

## Benchmark and correctness matrix

Every new dtype/backend family must cover:

- zero, scalar, odd, power-of-two, and very large element counts;
- contiguous, transposed, sliced, scalar-broadcast, and dense-broadcast views;
- all supported storage/compute/accumulator combinations;
- NaN, infinities, signed zero, subnormals, extrema, and integer overflow
  policy;
- same-device and cross-device transfer byte parity;
- forward and backward numerical parity;
- deterministic versus fast-math modes;
- cold compile, warm cache, steady-state latency, and peak memory.

Performance acceptance uses distributions, not one timing: warmups, multiple
samples, median and tail latency, pinned benchmark inputs, synchronized GPU
timing, and noise thresholds. A kernel is promoted only if it wins materially
in its declared region; otherwise the simpler fallback remains selected.

The initial dependency-free CPU baseline can be run with:

    cargo test -p kindle-backends --release --no-default-features \
      --features std,cpu benchmark_cpu_binary_kernels -- --ignored --nocapture

It emits CSV rows for F32 addition from 1 Ki through 4 Mi elements across
contiguous, scalar-broadcast, and dense-broadcast layouts. It reports the
selected execution path, seven-sample median
nanoseconds per element, and median effective GiB/s (two input reads plus one
output write). Local
development-host measurements retain a conservative 256 Ki-element Rayon
crossover for generic/strided loops, while explicit AVX2 dense kernels remain
faster through 1 Mi and switch to Rayon at 2 Mi. Cache-resident measurements
reached roughly 95 GiB/s for two-input contiguous addition and 141 GiB/s for
scalar broadcast; these are effective traffic rates and can exceed physical
memory bandwidth. Projecting the shared AVX2 writers over vectorizable
dense-broadcast plans beat the same-process odometer/iterator reference by
roughly 3.5–22x across the measured sizes. At 2–4 Mi elements,
128 Ki-element parallel AVX2 chunks beat same-process Rayon/autovectorized
references by roughly 10–49%, depending on contiguous versus scalar-broadcast
traffic. This benchmark is intentionally ignored by normal tests. Before
using results for cross-machine promotion decisions, record the CPU model,
core affinity, compiler flags, Rayon thread count, power mode, tail
percentiles, and raw samples.

## Controlling specialization count

The number of source templates stays small because operations share families:
unary pointwise, binary pointwise, reduction, indexing, normalization,
scan/scatter, and structured-library epilogues. The compiled cache remains
bounded because:

- only combinations actually executed are compiled;
- runtime op modes can share a binary when branching cost is negligible;
- shapes are bucketed only for tuning-sensitive kernels;
- invalid hardware/dtype combinations are pruned before compilation;
- persistent entries are versioned and evicted by size/age;
- fused graphs have a maximum node/register/source budget.

This avoids thousands of handwritten kernels without pretending that one
fully dynamic kernel can deliver peak performance everywhere.
