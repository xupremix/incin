#89 Creation + HostReadback rows on accelerators — S
Finding: the 9 CUDA-missing rows are NOT kernels: var_zeros/ones/rand/randn creation primitives ALREADY EXIST (cuda/backend/creation.rs:113-143) and HostReadback/to_bytes exist (contract.rs:31-73) — the gap is registration: declarations.rs sampling/filling groups omit Variable*, readback=[]; no Execute<ToHost*> impls. Same on WGPU/Metal.
Recommendation: extend declaration groups + bind executors reusing present helpers across all three backends; truthful dtype sets (CUDA helpers are f32-only vs CPU's 8); var_* must tape-register exactly like CPU; non-contiguous readbacks -> typed refusal.
Risk: overclaiming dtype sets; autograd parity for var_*.
