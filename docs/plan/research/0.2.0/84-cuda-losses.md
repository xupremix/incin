#84 CUDA losses/dropout/norms — L
Finding: zero CUDA rows for mse/l1/bce/ce/dropout/group_norm/instance_norm; layer_norm+batch_norm Training:no. CPU refs are composition-based; CUDA has composed (RmsNorm/Softmax) and native (pool2d w/ TapeEntry) patterns + compile-enforced advertise-impl proof.
Recommendation: port order mse/l1 (composed) -> bce_with_logits (fused, f32 accum) -> cross_entropy (fused) -> dropout (counter-RNG mask from seed+offset) -> group_norm -> instance_norm(=groups=C). f32 first, widen via #90.
Risk: oracle tolerance drift; i64 targets on device; dropout reproducibility.
Unblocks: full CUDA training step (with #4).
