#4 CUDA LayerNorm backward — M
Finding: fused Welford forward (ops/norm.rs:155) discards mean/rstd; doc says "no backward written". Tape pattern proven (executor.rs:926 WhereCond). Capability row Training:no.
Recommendation: option 1 — cache mean/rstd [batch] f32 at forward when GradMode records; one backward kernel (dx + atomicAdd dgamma/dbeta pre-zeroed); one TapeEntry capturing input/weight/bias ids + stats. Flip only the training cell after parity.
Risk: atomic nondeterminism vs CPU (tolerance-documented); bf16/f16 need f32 accumulators.
Unblocks: transformer training on GPU.
