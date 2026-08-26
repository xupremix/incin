#100 Broadcast-compatible shape pairs — M
Finding: BroadcastShape (shapes/broadcast.rs, SHP-007) already provides option A: Output assoc type, per-axis BroadcastDim/BroadcastExtent, rank promotion, on_unimplemented diagnostics. ShapeEq (ops/index.rs:135) is reflexive-only; where_cond (:852) + masked_fill (:383) bound on it; descriptor inference already derives broadcast outputs (exec/catalog/inference.rs:5-34); CUDA test proves backend broadcasts today.
Recommendation: re-bind where_cond/masked_fill to BroadcastShape, return <S as BroadcastShape<S2>>::Output; leave binary/matmul on ShapeEq; identical shapes keep compiling (verify via public-API baseline).
Risk: "no impl matched" error quality; blanket-impl coherence over named dims.
Example: mask s![8,8].where_cond(&scores s![2,4,8,8]) -> Tensor<s![2,4,8,8]>; compile-fails today.
Unblocks: #101, #102.
