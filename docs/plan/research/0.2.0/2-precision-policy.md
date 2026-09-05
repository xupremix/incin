#2 Precision policy into Trainer — M
Finding: RuntimePrecisionPolicy (exec/precision.rs:241, mixed_bf16()/mixed_f16()) + resolve_precision typed errors exist but NOTHING reads policy at dispatch; TrainerBuilder exposes only LossScaling; fit hard-types forward to f32. CapabilityQuery already carries training+math_mode.
Recommendation: dispatch-scoped policy + capability-derived autocast allowlist (ops whose rows admit BF16/F16 get cast to active_dtype; F32_ONLY ops + accumulators stay f32); f32 master weights as Trainer contract; build_on probes CapabilityQuery{training:true} per op before batch 1.
Risk: CPU rows largely F32_ONLY (fixture needs cast path); byte-width hazards; native_precision misreporting.
Unblocks: mixed precision/AMP (PRF-004), deferred f16/bf16 backend work.
