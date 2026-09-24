# Quantization

Quantization is a tensor-level conversion with a per-operation admission
contract (issue #93), not a backend capability you can only reach through
backend-authoring APIs. Two `Tensor` methods are the entire tensor-level
entry surface: `quantize` compresses a float tensor into `Q8_0` blocks,
`dequantize` decodes blocks back into a float tensor. Every other operation
then admits or refuses a quantized dtype on its own — at compile time when
the dtype is statically known, at run time when it is not.

The only quantized representation in this release is `Q8_0`: 32 consecutive
elements packed into one 34-byte block — an `f16` scale and 32 `i8` quants,
`StorageEncoding::block(32, 34, 2)` for 32 logical elements, 34 bytes,
2-byte alignment — laid out flat in row-major order. The authoritative
dtype definitions are in `incin_core::tensor::dtype`.

## The tensor-level API

```rust,no_run
use incin::prelude::*;

// 64 elements on the (only) axis: two whole Q8_0 blocks.
let x = Cpu.zeros(shape![64])?;
let q = x.quantize(-1)?; // same shape, dtype Q8_0
let back = q.dequantize::<f32>()?; // lossy: the f16 block scale rounds
# Ok::<(), incin::Error>(())
```

- `Tensor::quantize(axis)` requires `K: FloatCapable` and returns a tensor
  whose dtype is `Q8_0`. `axis` must resolve to the *last* axis (`-1` or
  `rank - 1`): the kernel indexes the flat row-major buffer, and a last axis
  divisible by 32 is exactly the condition under which 32-element chunks
  never straddle two rows. A different axis, an out-of-bounds axis, or a
  rank-0 tensor is a typed error naming what is wrong — never a panic, and
  never a silently reinterpreted buffer. The operand must be contiguous;
  a strided view is refused rather than copied around. The CPU executor
  narrows the input further to `f32` — its kernel matches the `F32` buffer
  variant rather than converting — so an `f16`/`f64`/`bf16` operand
  compiles and passes descriptor validation but is refused at admission
  with a typed `UnsupportedReason`; CUDA admits every float storage dtype
  through typed kernel entries.
- `Tensor::dequantize::<Kout>()` requires `K: QuantCapable`, where `Kout`
  is any `FloatDType`. The result has the operand's shape. The CPU kernel
  writes `f32` only; a different output dtype is refused with a typed
  `UnsupportedReason` rather than a panic.
- `FloatCapable` and `QuantCapable` are exported from the `incin` root and
  the prelude.

Both methods dispatch the catalog's `quantize`/`dequantize` descriptors,
exactly as the backend-authoring path does, so validation, capability
admission and the gradient node belong to the backend rather than to the
facade.

## Dtype admission: compile time first, run time after

### The bound pair

- `FloatCapable` is implemented blanket-style for every `FloatDType`
  (`f32`, `f64`, `f16`, `bf16`) and for `Dyn`. `Q8_0` and the integer
  dtypes deliberately do not implement it: an elementwise float kernel has
  no block path — there is no per-element `f32` to read out of a 34-byte
  block without decoding it first.
- `QuantCapable` is implemented blanket-style for every `QuantDType`
  (currently `Q8_0`) and for `Dyn`. Float and integer dtypes do not
  implement it: decoding is only meaningful when the input really holds
  blocks.

`quantize` requires `K: FloatCapable`, `dequantize` requires
`K: QuantCapable`, and twenty-five elementwise unary methods — `abs`,
`floor`, `ceil`, `round`, `sign`, `step`, `trunc`, `frac`, `mish`, `elu`,
`powf`, `clamp`, `erf`, `rsqrt`, `log2`, `log10`, `tan`, `asin`, `acos`,
`atan`, `sinh`, `cosh`, `asinh`, `acosh`, `atanh` — plus `norm` require
`K: FloatCapable`. Calling any of them with a statically `Q8_0` operand is
a compile error (`E0277`) whose text names the trait. The compile-fail
fixtures in `crates/incin-core/tests/compile_fail/` pin it:
`quantized_mish_admission`, `quantized_floor_admission`,
`dequantize_rejects_float_input` (a float operand refused by
`QuantCapable`) and `quantize_block_axis_not_divisible` (the const-assert
failure detailed below).

### `Dyn` falls through to the catalog

`Dyn` implements both traits, because its descriptor is only checkable at
run time. The call compiles, and the operation catalog refuses with a
typed descriptor error:

- `quantize` on non-float input metadata: *"quantize requires floating
  input and q8_0 output metadata"*.
- `dequantize` on non-`q8_0` input: *"dequantize requires q8_0 input and
  floating output metadata"*.
- a float-profile operation handed quantized-or-integer input metadata:
  *"operation requires floating-point input metadata"*.

`crates/incin-core/tests/quantized_tensor_ops.rs` (10 tests) pins these
runtime messages for `Dyn` shapes and `Dyn` dtypes, along with the block
rule below.

### Admission is per operation, not a wall

Support is decided per operation (issue #93, Decisions 3 and 8). An operation
with a real path for `Q8_0` accepts it: `quantized_matmul` consumes `q8_0`
operands, and in the CPU capability tables the contiguous `reshape` row and
the `broadcast` rows admit `q8_0` while a strided `reshape` refuses it (a
block has no per-value access for a materializing copy). An
operation with no path refuses it — a compile error when the dtype is
static, a typed descriptor error when it is not. Precision loss and block
constraints are the compromise, not a ban.

The bound covers the unary surface named above. A few unary methods
deliberately keep the unbounded `K: DType` signature — `sin` and `cos` are
reached from in-crate generic callers (rotary-table construction over
`K: DType`, whose trait hierarchy is dtype-open by design), where a float
bound would ripple through — so a quantized operand still compiles there
and is refused at run time by the catalog's float rule instead.

### The catalog slot is under-populated on purpose

`DTypeRule::Quantized` currently marks exactly three of the catalog's 179
operations: `quantize`, `dequantize`, `quantized_matmul`. The other 176
rows carry no quantized-admission statement at all. Populating them is
deferred (issue #93, Decision 7): trait bounds plus the compile-fail
fixtures are this release's enforcement, and the catalog remains the
intended future source of truth for admission tables and generated docs.
This is a documented deferral, not an unnoticed gap.

## Block divisibility

A `Q8_0` block axis must be a whole multiple of 32, proved twice (issue
#93, Decision 4):

- **Static shapes.** A `const { assert! }` reads `Shape::STATIC_EXTENTS`
  and fails compilation at monomorphization (`E0080`) with *"the last axis
  of a Q8_0 block must be a multiple of 32"* — proven by the
  `quantize_block_axis_not_divisible` fixture before any kernel runs.
- **`Dyn` shapes.** The extent is only known at run time, so `quantize`
  returns a typed error instead, e.g. *"quantize: axis 1 has extent 48,
  which is not a multiple of the Q8_0 block size 32; the block axis is the
  last axis, so its extent must be a whole multiple of 32"*.

A violation never reaches a kernel as a panic or a truncated block: the
compile-time assert states the rule, and the `Dyn` error names the axis,
the actual extent and the required multiple.

## The gradient is an estimator, not a derivative

On the CPU backend — the reference implementation, and the one the tests
exercise — `quantize` and `dequantize` record a tape entry (issue #93,
Decision 2): forward computes the true block-quantized value and its exact
decode, and backward passes the cotangent through *unchanged*,
`grad_in = grad_out`, because rounding has no derivative to compute. The
composition `dequantize(quantize(x))` therefore has exactly the identity
backward.

This is the straight-through estimator (STE), PyTorch QAT's `FakeQuantize`
rule, and it is documented as an approximation everywhere it appears: the
catalog carries `GradientRule::StraightThrough`, and the generated
[operation semantics](https://github.com/xupremix/incin/blob/master/docs/OPERATION_SEMANTICS.md)
renders the gradient column as `StraightThrough (approximation: STE)`.
Nothing here computes
the true derivative of quantization. Two honest limits: the CUDA kernels
for these two operations push no tape node yet, so a backward walk on CUDA
stops at that boundary instead of flowing through it (the catalog rule is
the contract they have yet to record), and `quantized_matmul` has no
gradient rule at all — it stays forward-only.

A general STE zeroes the cotangent outside a clip range,
`dx = dy * 1{|x| <= clip}`. `Q8_0` scales every block by its own
`max_abs / 127`, which never saturates, so there is no clip range to mask
against and the pass-through is unconditional. Clip-based masking for
fixed-range formats is not implemented (see the list below).

Two flags are worth keeping apart. The capability rows' `training` flag
gates `ExecutionPolicy`'s training flag, while the STE node above is gated
by gradient recording (the operand's gradient marker under `GradMode`, on
by default) — so the row's flag says nothing about whether an operation is
differentiable. On CPU the `quantize`/`dequantize` rows declare
`training = true`: a training-policy invocation is admitted and records
the STE entry (pinned by
`a_training_context_admits_the_boundary_and_records_both_ste_entries` in
`crates/incin-backends/tests/quantize_ste.rs`). `quantized_matmul` stays
`training = false` on every backend, and all three rows stay `false` on
CUDA, whose kernels record no tape entry yet — there the flag is refused
with *"training is unsupported for {operation}"*.

## Checkpoints, sharding, and the format promise

Issue #93's Decision 1 made the block layout part of the on-disk format and
required sharding to become block-aware:

- **Sharding is block-aware.** `slice_bytes_for_rank` — the resharding
  workhorse behind `load_resharded_checkpoint` — computes byte spans for
  block dtypes through `StorageEncoding::size_bytes` instead of assuming
  scalar bytes, and requires each rank's local extent along the shard axis
  to cover whole blocks. A mid-block boundary is refused: *"Cannot shard
  dtype q8_0 along axis 1: local extent 24 is not a multiple of block size
  32; shard boundary would fall mid-block"*. Nine tests in
  `crates/incin-core/tests/checkpoint_block_quant.rs` cover the slicing,
  the manifest round trip and both refusals.
- **The dtype record is part of the format.** A manifest persists each
  tensor's `CheckpointDType` — a `DTypeKey` (`("incin", "q8_0", 1)`), a
  `DTypeKind` and a `StorageEncoding` — and load refuses rather than
  misreads: an unknown key or version (*"(incin, q8_0, 2) ... registered
  first"*) and an encoding that disagrees with the registered dtype
  (*"Checkpoint dtype metadata does not match registered dtype q8_0"*)
  are both errors. The full commitment, including what a future layout
  change must do (bump the key's version so old and new files are
  distinct keys), lives in
  [`docs/COMPATIBILITY.md`](https://github.com/xupremix/incin/blob/master/docs/COMPATIBILITY.md).
- **State formats draw the byte-level line.** `Format::Postcard`
  round-trips a `Q8_0` tensor; the safetensors writer refuses block dtypes
  explicitly (*"unsupported safetensors dtype q8_0"*) and always has. A
  module whose parameters are `Q8_0` therefore saves through the postcard
  envelope, not through `Format::Safetensors` or the safetensors-backed
  `save_checkpoint` weights file.

## What 0.2.0 does not include

Fail-closed, stated plainly:

- **GPTQ, AWQ, and any post-training quantization search.** `quantize` /
  `dequantize` are the whole boundary into and out of quantized storage;
  [GGUF export](./saving_loading.md) is an export format, not tensor
  operations.
- **Clip-based STE.** No `1{|x| <= clip}` masking — `Q8_0` never
  saturates; a fixed-range format would need it first.
- **Per-operation `DTypeRule::Quantized` catalog population.** Deferred
  (Decision 7): three rows today, as described above.
- **Quantized parameter initialization.** No initializer admits `q8_0`
  (CPU fill rows are `NON_QUANTIZED`, sampling rows are float-only), so
  `Linear<..., K = Q8_0>::build` fails with a typed capability refusal
  naming `q8_0`. The type path compiles end to end (Decision 8); for this
  release the parameters come from `quantize`-ing a float initialization
  or from loading a checkpoint.
- **Q4_0 / NVFP4 / MXFP4 as tensor dtypes.** The block-format audit
  (Decision 6) found that `StorageEncoding::block` + `DTypeKey` already
  fit all three without a contract extension, but none of them is a
  built-in tensor dtype in this release, and NVFP4 still owes a
  serialization-protocol choice before any of those formats freezes.
- **`quantize` / `dequantize` on WGPU and Metal.** Those two backends
  implement no `Execute` for the quantizing operations and their
  capability declarations advertise none (issue #93), so on them the
  methods do not compile — the `B: Execute<op::Quantize>` bound is
  unmet — while CPU and CUDA do.

## For backend authors

The contract does not change: advertise only the quantized operations your
executor really implements, using the same capability rows and `Execute`
checks as every other operation, and keep the dtype and layout claims of
those rows honest. The dtype definitions you advertise against live in
`incin_core::tensor::dtype`.
