#93 Tensor-level quantized dtype contract — M
Finding: Q8_0 has block StorageEncoding(32,34,2) + DTypeKey versioning; CheckpointDType persists encoding; but slice_bytes_for_rank requires scalar_bytes() -> sharded quantized checkpoints FAIL today. No NumericDType marker, so quantized tensors reach element-wise ops accidentally.
Recommendation: (1) block layout recorded in format via CheckpointDType.encoding + block-aware sharding; (2) DTypeKey.version bumps on layout change, load refuses by key mismatch; (3) type-enforced NoGrad at quantize(); add quantize<Q>/dequantize<K> as total boundary; NumericDType marker on numeric ops.
Example: quantize<Q>(axis) -> Tensor<S,B,Q,NoGrad>; save writes key("incin","q8_0",1)+block; load refuses v2.
Risk: NumericDType bounds break downstream generics (pre-1.0 OK); STE absence must be documented.
Unblocks: #94, #95, mostly #1; #3 unaffected (backend layer).
