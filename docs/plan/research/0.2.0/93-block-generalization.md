#93 block generalization audit (Decision 6) + catalog deferral (Decision 7)
Sibling of `93-quantized-contract.md` (left untouched). Scope: can
`StorageEncoding::block` + `DTypeKey` carry Q8_0 / Q4_0 / NVFP4 / MXFP4 before
#1 / #94 / #95 freeze, and what is the state of `DTypeRule::Quantized`.
Evidence: branch `feat/custom-autograd-dtype`; web sources cited inline for
external format facts; every in-tree claim carries `file:line`.

## Task A — Block generalization audit

### The contract being audited

`StorageEncoding` (`crates/incin-core/src/tensor/dtype/registry.rs:152-156`) has
exactly three fields: `logical_elements_per_block`, `bytes_per_block`,
`alignment`. `StorageEncoding::block(logical, bytes, alignment)`
(`registry.rs:215-231`) asserts all non-zero and alignment a power of two;
`is_power_of_two(1)` holds, so alignment 1 is legal. Storage arithmetic is
centralized:

- `size_bytes` (`registry.rs:282-305`): scalar = `elements × bytes`; block =
  requires `elements % logical == 0` else typed `ShapeError::InvalidParameter`,
  then `(elements / logical) × bytes` checked. Zero elements → `Ok(0)`.
- `scalar_bytes` (`registry.rs:265-271`): `Some(bytes)` only when
  `logical == 1`; `None` for every block encoding — the documented "callers
  must reject block dtypes explicitly" seam.
- `DTypeKey` = `(namespace, name, version)` (`registry.rs:19-23`); builtin
  deserialization is a version-specific match arm (`registry.rs:42-54`),
  unknown versions fall to `DTypeRegistry` and then **refuse** with a typed
  error (`registry.rs:61-69`). `DTypeDescriptor` deserialization cross-checks a
  builtin's key+kind+encoding against the registry (`registry.rs:351-361`).
- The sharded-checkpoint path persists key+kind+encoding as `CheckpointDType`
  (`crates/incin-core/src/nn/save.rs:16-23`) and re-resolves by name arm then
  full key/kind/encoding equality (`nn/save.rs:38-67`): a version or layout
  mismatch is refused ("does not match registered dtype"), never misread.

### Per-format table

| Format | Logical / block | Bytes / block | Align | Scale width in block | Scale dtype | Fits today's three fields? | Gap |
|---|---:|---:|---:|---:|---|---|---|
| Q8_0 (in tree) | 32 | 34 | 2 | 2 B + 32×i8 | f16 | **Yes** — live: `block(32, 34, 2)` (`builtin.rs:198`, `registry.rs:567`) | none |
| Q4_0 (GGUF) | 32 | 18 | 2 | 2 B + 32×4-bit (16 B packed) | f16 | **Yes** — `block(32, 18, 2)` | none |
| NVFP4 | 16 | 9 interleaved / 8 data-only | 1 | 1 B + 16×4-bit (8 B packed) | FP8 E4M3 per block; FP32 per tensor **outside** the block | **Yes** — `block(16, 9, 1)` or `block(16, 8, 1)` + companion scale | serialization-protocol choice (interleaved vs split + global scale), not an encoding-field gap |
| MXFP4 | 32 | 17 | 1 | 1 B + 32×4-bit (16 B packed) | E8M0 per block | **Yes** — `block(32, 17, 1)` | none |
| FP8 e4m3/e5m2 (#94, not block) | 1 (`scalar`) | 1 | 1 | n/a — no in-block scale | separate f32 scale tensor at construction | **Yes** — `scalar(1, 1)` + `DTypeKind::Float` | none; watch width collision (see helpers) |

External corroboration:

- **Q4_0**: GGML `block_q4_0` = `ggml_fp16_t d` + `uint8_t qs[QK4_0/2]` with
  `QK4_0 = 32`, i.e. 18 bytes, GGML type id 2
  ([ggml-quants.h](https://github.com/Tiiny-AI/PowerInfer/blob/main/ggml-quants.h),
  [ggml.c type table](https://github.com/ggml-org/ggml/blob/master/src/ggml.c)).
  In-tree `encode_q4_0` (`crates/incin-core/src/io/gguf.rs:58-99`) writes f16
  scale first then 16 packed bytes (low nibble = indices 0..15, high nibble =
  16..31), capacity `len/128*18`, and `QuantScheme::W4A16_Q4_0` maps to ggml
  type id 2 (`gguf.rs:40`); `inspect.rs:169` names id 2 `Q4_0`. Matches the
  GGML struct byte-for-byte.
- **NVFP4**: block 16, E2M1 data, per-block FP8 **E4M3** scale (the audit
  prompt's "e2m1 scale?" is wrong — the *data* is E2M1, the scale is E4M3), plus
  a per-tensor FP32 `s_global` computed from amax — a two-level scheme whose
  second level is per-tensor, not per-block
  ([Transformer Engine NVFP4](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/features/low_precision_training/nvfp4/nvfp4.html),
  [NVIDIA blog](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference)).
  Effective footprint 9 B / 16 elements (4.5 bpw). NVIDIA tooling commonly keeps
  block scales as a **separate** scale tensor (TE pads scale tensors for
  hardware alignment), so both the interleaved `block(16,9,1)` and the
  data-only `block(16,8,1)` + companion-scale representations exist in the
  ecosystem. 2D weight scaling (16×16 groups) changes only the companion scale
  tensor's shape, not the per-block encoding.
- **MXFP4**: OCP MX v1.0 registers MXFP4 = 32-element blocks, E2M1 elements,
  8-bit E8M0 (power-of-two) scale → 136 bits = 17 bytes per block, 4.25 bpw
  ([OCP MX v1.0 spec](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf),
  [Golden Ruler catalog](https://arxiv.org/html/2605.31035v1)). No per-tensor
  scale in the spec. Block sizes other than 32 are future-spec material; every
  candidate k still fits `block(k, k/2+1, 1)`.

### Per-format findings

**1. Q8_0 — in tree, reference case.** `StorageEncoding::block(32, 34, 2)`
(`builtin.rs:191-200`), `DTypeKind::Quantized`, key `("incin","q8_0",1)`.
Nothing to audit: it is the shape the other three must fit. Postcard snapshot
round-trips it (`serialize.rs:1201-1232` codifies that safetensors **refuses**
it explicitly — `safetensors_dtype` has no block arm, `serialize.rs:17-32`).

**2. Q4_0 (GGUF) — fits today.** Proposed `block(32, 18, 2)` (also the
recommendation already recorded in `3-1-quant-matmul-gguf.md`). Two-level
scale: no (one f16 `d` per block, no min — Q4_1's extra f16 `m` would be
`block(32, 20, 2)`, still fine). Non-uniform block: no; partial final blocks
are refused by `size_bytes` by design, matching GGML's "row length multiple of
block" rule. Scale-dtype-in-bytes: 18 = 2 + 16 split is *not* derivable from
the three fields, but no in-tree code splits block bytes generically — decode
is per-format code keyed by dtype (`QuantScheme`/kernels), so this is invisible
today. `DTypeKey("incin","q4_0",1)`: clean; a layout change (e.g. adopting
Q4_1-style mins) bumps to version 2, the v1 arm disappears, load refuses.

**3. NVFP4 — fits today; one protocol decision owed.** `block(16, 9, 1)`
interleaved, or `block(16, 8, 1)` if scales are a companion tensor. The
**per-tensor FP32 scale is not part of any block** — it is tensor-level data
(same category as #94's required f32 scale tensor) and must be persisted as a
snapshot side tensor / metadata key, not as an `StorageEncoding` field.
Two-level scale: present in the *format*, but its second level lives outside
the encoding, so the three fields do not need to model it. Non-uniform block:
no (every block is 16 elements + 1 scale byte). What #95 must freeze is which
of the two wire layouts its `DTypeKey` names — pinned by name+version, so a
later switch is a version bump, not a contract break.

**4. MXFP4 — fits today.** `block(32, 17, 1)`. One E8M0 scale byte per block,
no global scale, no sub-block structure. Alignment 1 is required (17 is odd →
successive block starts would break 2-byte alignment), and `alignment = 1`
passes every assert and the serde power-of-two check.

**5. FP8 (#94) — not a block dtype; still checked.** Scalar
`scalar(1, 1)` + `DTypeKind::Float` via the `impl_plain_builtin_dtype!`
precedent (f16/bf16 from the `half` crate, `builtin.rs:113-126`). `scalar_bytes`
returns `Some(1)` — correct, but width-collides with `u8`, `bool`, and the
per-element width of `q8_0`. Safe today only because dispatch/ABI checks are
dtype-keyed (`cuda/ops/elementwise.rs:15` compares `kernel.dtype` *and* width);
any future width-only generic path would misread fp8 as u8. The required f32
scale is a construction-time companion tensor (`94-fp8.md`), outside the
encoding.

### What breaks in the helpers if a block format is added

- **`slice_bytes_for_rank` (`nn/save.rs:182-295`)** — **fails today for every
  block encoding, Q8_0 included**: it demands
  `encoding().scalar_bytes().ok_or(UnsupportedDType)` (`nn/save.rs:214-221`)
  then computes byte spans as `elements × elem_bytes`. This is exactly the
  Decision-1 defect. The fix is format-generic and must: (a) compute row/slab
  byte sizes via `size_bytes` instead of element×width; (b) require the shard
  extent on the cut axis to keep blocks whole (row-major: blocks live on the
  innermost axis, so whole-row shards are safe when `row numel % block == 0`,
  matching `conformance/operands.rs:59-75`'s rule that the widened extent goes
  last); (c) keep refusing non-contiguous layouts. Once that lands, Q4_0 /
  NVFP4 / MXFP4 inherit it with zero per-format code — **no format-specific
  field is needed for sharding**.
- **`size_bytes` (`registry.rs:282-305`)** — works unchanged for all four; the
  only new surface is stricter divisibility: `% 32` for Q4_0/MXFP4/Q8_0,
  `% 16` for NVFP4. Violations produce the existing typed
  `InvalidParameter { parameter: "elements" }`, which is Decision 4's runtime
  half. Watch: models whose row dimension is not a block multiple fail at
  allocation/serialization with that error, not silently.
- **`scalar_bytes` (`registry.rs:265-271`)** — `None` for all three new block
  formats. Known callers handle it fail-closed: CUDA kernel ABI check refuses
  with "invalid scalar bytes" (`cuda/ops/elementwise.rs:19-20`), scalar interop
  computes `unwrap_or(0)` and then mismatches into an error
  (`interop.rs:194-201`), kernel tests only call it on known scalar dtypes.
  New block formats get these refusals for free. FP8 is the inverse hazard
  (returns `Some(1)`, collides with u8) — noted above.
- **Snapshot paths beyond the three named helpers** (part of "what breaks"):
  postcard round-trips any descriptor (`serialize.rs:876-879` carries full
  `DTypeDescriptor`), but each new builtin needs a `DTypeKey` serde arm
  (`registry.rs:42-54`) and a `CheckpointDType::descriptor()` name arm
  (`nn/save.rs:39-48`); without them load **refuses** (fail-closed, slightly
  misleading error text for the registry path — see below). safetensors export
  refuses block dtypes outright by design (`serialize.rs:17-32`, test at
  `serialize.rs:1201`).

### DTypeKey versioning verdict

**Clean GO.** A layout change is a `version` bump; loaders compare the full
`(ns, name, version)` plus, on the checkpoint path, kind and encoding bytes
(`nn/save.rs:58-66`), and on the wire path the descriptor cross-check
(`registry.rs:351-361`). Mismatch → typed refusal, never a misread. One
diagnostic nit (not a contract gap): an unknown *version of a builtin name*
falls through to `DTypeRegistry`, whose error suggests
`DTypeRegistry::register(...)` — which then refuses the reserved `"incin"`
namespace (`registry.rs:782-790`). The load still fails correctly; the message
points at an impossible remedy. Worth a wording fix when #93 serialization
work touches that arm.

### GO / GAP verdicts for the freezes

| Issue | Format | Verdict | Condition |
|---|---|---|---|
| **#1** | Q4_0 (GGUF) | **GO** — `block(32, 18, 2)` fits today, no extension | Sharded checkpoints require Decision 1's block-aware `slice_bytes_for_rank` (shared with Q8_0, not a Q4_0 gap); keep #1's existing byte-exact GGML nibble-order fixture requirement |
| **#94** | FP8 e4m3/e5m2 | **GO** — `scalar(1, 1)` + `DTypeKind::Float`, not a block dtype | Encoding trivially fits; scale stays a companion tensor (#94 already says so); note the `scalar_bytes == 1` width collision in the #94 write-up |
| **#95** | NVFP4 + MXFP4 | **GO — no encoding extension needed before freeze** | MXFP4 `block(32,17,1)` is fully determined. NVFP4 must *choose and pin* interleaved `block(16,9,1)` vs data-only `block(16,8,1)` + companion scales, and must specify where the per-tensor FP32 scale is persisted (snapshot side tensor / metadata) — a **serialization-protocol decision under the existing contract**, not a `StorageEncoding` field. Pin both in the `DTypeKey` name+version before freeze. |

**No GAP exists for freezing any of #1 / #94 / #95.** Decision 6's audit
question answers "the three fields cover all four formats".

### Minimal extension proposal (only if a future format actually gaps)

Recorded so a later implementer does not widen the wire spec speculatively —
**do not implement now**:

1. **Preferred**: keep `StorageEncoding`'s three fields frozen; put
   format-side facts (scale width, scale dtype, companion-scale key) in a
   small **`DTypeKey`-keyed side table** consulted by decode/generic materialize
   code. Adding an optional field to `StorageEncoding` changes the serde wire
   of every `DTypeDescriptor`/`CheckpointDType` already written
   (`registry.rs:151`, `nn/save.rs:22`) for facts nothing currently reads.
2. **Reject — encode scale in `bytes` only**: `bytes` is already total
   block size; the scale/data split is not recoverable from it, and no current
   consumer needs the split.
3. **Two-level marker**: only if/when a *per-tensor* scale becomes first-class
   in the type contract (today it is data, like #94's scale tensor — belongs
   beside the tensor, not inside the encoding).
4. Trigger for revisiting: a generic (non-per-kernel) dequant/materialize path
   that must locate the scale inside an arbitrary block, or a checkpoint
   format that interleaves scales per-tensor. Neither exists in the tree.

## Deferred: DTypeRule::Quantized population (Decision 7)

**What exists.** `DTypeRule` (`exec/catalog/classification.rs:63-82`) carries a
`Quantized` variant ("Operates on block-quantized representations",
`classification.rs:78-79`). It reaches catalog rows only through
`profile_semantics(SemanticProfile::Quantized)`
(`exec/catalog/table.rs:249-257`), and exactly **three of the catalog's 179
canonical operations** declare that profile
(`crates/incin-core/src/operation_catalog.rs:195-197`):

| Row | Profile | DTypeRule today |
|---|---|---|
| `quantize` | Quantized | `Quantized` |
| `dequantize` | Quantized | `Quantized` |
| `quantized_matmul` | Quantized | `Quantized` |

(Confirmed in the generated inventory: `docs/OPERATION_SEMANTICS.md:204-206`;
179 canonical operations at line 7.)

**Why that is under-populated.** `DTypeRule::Quantized` currently marks the
three boundary operations; it says nothing about the Decision-8 question —
"which operations accept a quantized input (and via which path)". The other
**176 rows carry no quantized admission statement at all**. Representative
sample of rows a future population pass must decide, with what they say today:

| Sample ops | Current profile / `DTypeRule` | Missing statement |
|---|---|---|
| `matmul`, `bmm`, `addmm`, `dot`, `outer`, `grouped_matmul` | MatMul / `Floating` | whether quantized operands are refused (→ `quantized_matmul`) or get a dequant path; today the float wall in `inference.rs:902-928` refuses them implicitly |
| `linear`, `layer_norm`, `embedding`, `conv1d`… | Module·Composite / `TypedContract` | `embedding` is explicitly *exempt* from the float requirement (`inference.rs:911-914`), so quantized weights reach descriptor validation with no catalog opinion |
| `add`, `mul`, `relu`, `exp`… (pointwise binary) | BinaryBroadcast / `NumericSame`; UnaryFloat / `Floating` | `NumericSame` enforces only same-dtype (`inference.rs:876-901`) — two `q8_0` tensors satisfy `add` at descriptor level; the refusal, if any, arrives only from capability rows |
| `sum*`, `mean*`, `max*`, `norm`, `cumsum`… | Reduction / `Floating` | same implicit float wall, no per-op "no quantized path" record |
| `reshape`, `transpose`, `slice`, `concat`, `narrow`… | Shape / `Preserve` or `TypedContract` | whether q8_0 metadata may pass through; the catalog cannot express the distinction the CPU capability table already draws between contiguous (ALL_DTYPES) and strided (`NON_QUANTIZED`) reshape (`incin-backends/src/capability/tables.rs:60-82`) |
| `to_dtype`, `to_device` | Transfer / `TypedContract`·`ExplicitOutput` | that entering/leaving quantized storage goes through `quantize`/`dequantize`, not `to_dtype` |
| losses, optimizers, norms, attention | Loss·Optimizer·Normalization·Attention / `Floating` | same as reductions |

**Also: `DTypeRule` is documentation-only right now.** The only consumer of
`row.dtype` is `operation_semantics_document` rendering
(`exec/catalog/lookup.rs:122` → `docs/OPERATION_SEMANTICS.md`). No validator
reads it; actual dtype enforcement lives in
`inference.rs::verify_outputs` (profile-keyed `require_float` at 902-928 plus
three Q8_0-hardcoded match arms for `quantize`/`dequantize`/`quantized_matmul`
at 978-1014), in attribute validators (`attributes.rs`), and in the backend
capability tables (`quantized_dtypes = Q8_ONLY` at
`incin-backends/src/capability/tables.rs:58,218,711,959`;
`Q8_ONLY = [q8_0]` at `capability/constants.rs:92`; `NON_QUANTIZED`/`FLOAT_DTYPES`
elsewhere). Populating the slot therefore adds truth to the future source of
truth; it does not currently gate anything.

**The deferral (per Decision 7, settled 2026-09-24).** 0.2.0 enforcement is
**trait bounds + compile-fail fixtures naming the operation** (static case) and
**typed descriptive errors for `Dyn`** (runtime case), not catalog rows. The
catalog slot stays under-populated *by design* for this release and is the
intended future source of truth for "which operations accept a quantized K" —
feeding admission tables and generated docs once bounds exist to check against.
When populated, the pass should: (1) decide each of the 176 remaining rows
(accept-via-quantized-path / accept-after-STE / refuse), (2) special-case the
shape family so contiguous-vs-strided becomes expressible (the capability
table already distinguishes it; `DTypeRule` cannot), (3) consume `row.dtype`
from a validator instead of only from the doc renderer. None of this blocks
#1 / #94 / #95.

## Task C — research index

`00-index.md` does not list memos by file: it narrates issues and links memo
files inline only where relevant (e.g. the #93 entry's
"See `93-quantized-contract.md`"). Per the audit instructions ("if memos
aren't all listed, skip"), **no index line was added**; the index is otherwise
untouched.

## Files written

- `docs/plan/research/0.2.0/93-block-generalization.md` (this file) — created.
- `docs/plan/research/0.2.0/00-index.md` — intentionally unchanged (Task C skip).
- Not edited: Rust code, `93-quantized-contract.md`, book pages, CHANGELOG.

## Gates

- `python3 tools/check-docs.py` — **passed** (38 book chapters, 21 facade
  features, 54 crate feature declarations).
- `python3 tools/check-markdown-links.py` — **passed** (261 files).
- `mdbook build docs/book` — not run: research memos are not part of the
  book; the links check is the applicable gate.
