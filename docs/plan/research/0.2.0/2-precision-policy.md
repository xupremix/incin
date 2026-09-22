#2 Precision policy into Trainer — M
Finding: RuntimePrecisionPolicy (exec/precision.rs:241, mixed_bf16()/mixed_f16()) + resolve_precision typed errors exist but NOTHING reads policy at dispatch; TrainerBuilder exposes only LossScaling; fit hard-types forward to f32. CapabilityQuery already carries training+math_mode.
Recommendation: dispatch-scoped policy + capability-derived autocast allowlist (ops whose rows admit BF16/F16 get cast to active_dtype; F32_ONLY ops + accumulators stay f32); f32 master weights as Trainer contract; build_on probes CapabilityQuery{training:true} per op before batch 1.
Risk: CPU rows largely F32_ONLY (fixture needs cast path); byte-width hazards; native_precision misreporting.
Unblocks: mixed precision/AMP (PRF-004), deferred f16/bf16 backend work.

## Findings from external research (2026-09-22)

- Autocast should be op-level policy at dispatch, not module-boundary wrappers: PyTorch's AMP pattern uses an allowlist that downcasts matmul/pointwise ops while reductions/losses stay wide, and applies forward+loss only, never backward. https://docs.pytorch.org/docs/main/amp.html
- GradScaler defaults align with ours (init 65536, growth 2.0, backoff 0.5, interval 2000). BF16 genuinely needs no scaler (shared exponent); F16 does - consistent with the committed `UnsupportedPrecision` rule requiring scaling for f16-active/exact-f32 plans. Sources: torch.amp docs above; oxidelm `amp.rs` https://docs.rs/oxidelm/latest/src/oxidelm/amp.rs.html
- Master weights are an optimizer-level contract: f32 master is the source of truth, working weights re-cast each step, gradients upcast before `step` (oxidelm/candle `MixedAdamW` pattern). https://docs.rs/algocline-nn/latest/src/algocline_nn/train/mixed/index.html
- Status vs on-disk code (as of 2026-09-22): `Trainer::fit`/`fit_scaled` now enter an `ExecutionPolicy` scope carrying `Plan::precision`, so every eagerly built `ExecutionContext` (`from_scope`) reaches `ExecutionRequest::context` with the plan's policy; dispatch admission itself still does not branch on the precision axis, and `fit` hard-types the loss to f32 - the axis is carried and inspectable, not enforced. The autocast allowlist above remains the open slice.
