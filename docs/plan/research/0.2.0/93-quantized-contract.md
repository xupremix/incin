#93 Tensor-level quantized dtype contract — M
Finding: Q8_0 has block StorageEncoding(32,34,2) + DTypeKey versioning; CheckpointDType persists encoding; but slice_bytes_for_rank requires scalar_bytes() -> sharded quantized checkpoints FAIL today. No per-operation dtype admission story, so quantized tensors reach element-wise ops accidentally (or are at risk of a blanket NumericDType wall that would make them useless).

## Decisions (settled by maintainer, 2026-09-24)

These eight answers supersede the draft "Recommendation" line below where they conflict. Implementation must follow this section.

1. **Serialization — standard format, expansion open.**
   Block layout is part of the on-disk format via the existing `CheckpointDType.encoding` + `DTypeKey` (`("incin", "q8_0", version)`). Layout changes bump `version`; load refuses on key/version mismatch rather than misreading bytes. Keep the door open for expansion: new block formats are new keys (or version bumps), and `StorageEncoding::block` stays the shared representation so Q4_0 / NVFP4 / MXFP4 can join without a format fork. Sharding must become block-aware (`slice_bytes_for_rank` cannot demand `scalar_bytes()`).

2. **Gradient — what PyTorch does: straight-through estimator (STE).**
   `quantize` records a tape entry. Forward: `y = dequantize(quantize(x))` semantics at the boundary (or the true quantized value with STE backward). Backward: `dx = dy * 1{|x| <= clip}` — pass gradient through as if quantization were identity within the clip range. This is PyTorch QAT's `FakeQuantize` behavior. Document as an approximation in operation semantics. Do **not** ship silent NoGrad.

3. **Dtype admission — compile error when statically known unsupported; descriptive runtime error when `Dyn`.**
   An operation must not compile when the dtype is statically known to be unsupported by that operation (via trait bounds / compile-fail fixtures that name the operation). When the dtype (or the support fact) is only known at runtime (`Dyn`), fail with a descriptive typed error, not a panic and not silent wrong results. Support is **per-operation**, not a global wall (see 8).

4. **Block divisibility — compile-time for comptime shapes, runtime for `Dyn`.**
   Same pattern as the rest of the shape algebra: static extents prove `extent % block == 0` at compile time (const-generic / `DivisibleBy`-style bound or equivalent proof the codebase already uses for mesh/head divisibility); `Dyn` extents get a typed runtime error naming axis, expected multiple, and actual extent.

5. **Scope — land the tensor-level API now.**
   Breaking changes are acceptable pre-1.0. A user can write a model with quantized weight parameters through the `incin` facade, call `quantize`/`dequantize` on `Tensor`, and round-trip save/load — without writing backend-authoring APIs. Update public-api baselines and `docs/MIGRATION.md` for every affected signature.

6. **Block generalization — verify before freezing #1/#94/#95.**
   Audit that `StorageEncoding::block` + `DTypeKey` actually cover Q8_0, Q4_0 (GGUF), NVFP4, and MXFP4 (different scale widths, block sizes, scale conventions). Write the findings into this memo (or a sibling) **before** any of those three formats is finalized. If a gap exists, extend the encoding contract first; do not special-case one format in the tensor API.

7. **Catalog — bounds only for now; document the gap.**
   Do **not** block 0.2.0 on populating `DTypeRule::Quantized` per operation. Trait bounds + compile-fail fixtures are the enforcement for this release. Explicitly document that the catalog slot (`classification.rs` / `table.rs`) remains under-populated and is the intended future source of truth for "which operations accept a quantized K" (admission tables, generated docs). Leave a pointer in the research note and in whatever book/docs page describes quantized admission.

8. **Quantized tensors stay usable for operations they can meaningfully run — reduced precision, not exile.**
   Do not introduce a blanket `NumericDType` bound that excludes `Q8_0` from every numeric op. Operations that have a defined path for a quantized dtype (true quantized kernel, dequant→op→requant, or STE-fake-quant) must accept that dtype. Operations with **no** path for that dtype are the ones covered by decision 3 (compile error / descriptive Dyn error). Precision loss / scale / block constraints are the compromise, not a ban. `quantize`/`dequantize` remain the explicit total boundary for entering and leaving quantized storage.

## Original draft recommendation (superseded where it conflicts with Decisions 1–8)

Recommendation (draft): (1) block layout recorded in format via CheckpointDType.encoding + block-aware sharding; (2) DTypeKey.version bumps on layout change, load refuses by key mismatch; (3) type-enforced NoGrad at quantize(); add quantize\<Q\>/dequantize\<K\> as total boundary; NumericDType marker on numeric ops.
Example (draft): quantize\<Q\>(axis) -> Tensor\<S,B,Q,NoGrad\>; save writes key("incin","q8_0",1)+block; load refuses v2.
Risk: NumericDType bounds break downstream generics (pre-1.0 OK); STE absence must be documented.

## Acceptance (from issue #93)

- A user can write a model with a quantized weight parameter through the `incin` facade, without calling backend-authoring APIs.
- Applying an operation with no quantized path to a quantized tensor fails to compile, with a diagnostic that names the operation; `Dyn` gets a descriptive runtime error.
- Block divisibility is proven at compile time for static shapes and is a typed runtime error for `Dyn`.
- A quantized parameter survives a save and load round trip through the typed snapshot contract, and a version mismatch in `DTypeKey` is refused rather than misread.
- The gradient rule through the quantize boundary (STE) is documented and enforced in the type system.
- `docs/book/src/quantization.md` describes a tensor-level API rather than deferring one.
- Block generalization findings for Q4_0/NVFP4/MXFP4 are written down before those formats freeze.
- The missing per-op `DTypeRule::Quantized` population is documented as deferred.

## Status

- Decisions: **settled** (see above).
- Implementation: **landed** (2026-09-24) via parallel workstreams: tensor
  surface (`quantize`/`dequantize`, `FloatCapable`/`QuantCapable`,
  compile-fail + block-divisibility admission), block-aware checkpoint
  sharding + `DTypeKey` refusal, identity-STE tape on CPU
  (`GradientRule::StraightThrough`, capability rows training=true on CPU,
  false on CUDA/WGPU/Metal), block generalization audit
  (`93-block-generalization.md`: Q4_0/NVFP4/MXFP4 all fit today's encoding),
  docs (quantization chapter rewrite, how-to cookbook, MIGRATION, CHANGELOG,
  three learning examples).
- Known follow-up (documented in `whats_not_finished.md` Facade gaps):
  `validate_gradient_dtype` still refuses gradient markers on `Q8_0`, so
  facade-level QAT requires `.detach()` and the STE round-trip runs on the
  dispatch surface; no `Tensor::quantized_matmul` method yet; `DTypeRule`
  population deferred per Decision 7.
- Unblocks: #94, #95, mostly #1; #3 unaffected (backend layer).
