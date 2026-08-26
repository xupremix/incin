#88 CUDA indexing/layout (11 ops) — L (area:safety)
Finding: missing embedding gather scatter index_select diag pad repeat tril triu unfold pixel_shuffle. kernels/embedding.cu exists but is DEAD code and non-conforming (OOB writes zeros, skips negatives). House rule: invalid input must error, not read garbage.
Recommendation: kernel-side error flags surfaced post-launch as CPU-identical typed errors (no clamp/skip); one dtype-aware strided-copy skeleton deriving pad/repeat/tril/triu/diag/pixel_shuffle as index maps; unfold reuses conv geometry; embedding backward accumulation must match #103; delete or fix the orphan kernel; deliberate-OOB device tests via #83.
Risk: clamp/skip silently corrupts; unsynced error flag races past Ok; i64 indices through f32 pointers.
