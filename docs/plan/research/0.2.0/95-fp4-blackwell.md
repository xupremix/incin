#95 NVFP4/MXFP4 block-scaled FP4 — M (after #85+#93)
Finding: Q8_0's block contract (block(32,34,2), size_bytes divisibility, alignment) already models both FP4 layouts (NVFP4: e2m1+e4m3 scale, blk16, 9B; MXFP4: e8m0 scale, blk32, 17B). cuda-vendor: zero call sites; FP4 needs sm100 + CUDA>=12.8.
Recommendation: both dtypes, encoding at type level; opt-in cuda-blackwell feature pinning cudarc/cuda-12080; runtime compute-capability gate via cargo incin doctor; pure-Rust CPU reference encode/decode per format MANDATORY for #83 oracle; sequence after #94 (de-risks descriptors).
Risk: runner may not be Blackwell (acceptance unverifiable until procured); inherits #85+#93 prerequisites; keep tier opt-in (don't raise global CUDA floor).
