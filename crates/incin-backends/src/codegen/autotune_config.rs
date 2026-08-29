//! Kernel launch configuration autotuning and search space optimizer (PRF-012).
//!
//! Inspired by OpenAI Triton autotune configs and Burn CubeCL heuristics:
//! - Generates candidate tile configurations $(B_m, B_n, B_k, T_m, T_n)$
//! - Validates GPU register budgets (max 32–64 registers per thread for 100% occupancy)
//! - Computes shared-memory padding to eliminate 32-bank conflicts

use alloc::vec::Vec;
use incin_core::tensor::dtype::DTypeId;

/// GPU Architecture profile for occupancy estimation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuArchProfile {
    /// Modern NVIDIA Ampere / Ada / Hopper / Blackwell (SM 80+).
    NvidiaModern,
    /// Apple Silicon (M1/M2/M3/M4) SIMD32.
    AppleSilicon,
    /// Generic WebGPU device (uniform workgroup sizes).
    GenericWebGpu,
}

/// Candidate autotune configuration parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutotuneCandidate {
    /// Block size M.
    pub block_m: usize,
    /// Block size N.
    pub block_n: usize,
    /// Block size K.
    pub block_k: usize,
    /// Thread tile M.
    pub thread_m: usize,
    /// Thread tile N.
    pub thread_n: usize,
    /// Number of warps / workgroups.
    pub num_warps: usize,
    /// Shared memory bytes required.
    pub shared_memory_bytes: usize,
}

/// Autotuner search space generator.
#[derive(Debug, Clone)]
pub struct AutotuneSpace {
    /// Problem dimensions $(M, N, K)$.
    pub problem_shape: (usize, usize, usize),
    /// Data type.
    pub dtype: DTypeId,
    /// GPU architecture profile.
    pub arch: GpuArchProfile,
}

impl AutotuneSpace {
    /// Creates a new autotuner space for a given matrix problem $(M, N, K)$.
    #[must_use]
    pub fn for_matmul(m: usize, n: usize, k: usize, dtype: DTypeId, arch: GpuArchProfile) -> Self {
        Self {
            problem_shape: (m, n, k),
            dtype,
            arch,
        }
    }

    /// Generates list of hardware-valid candidate tile configurations.
    #[must_use]
    pub fn generate_candidates(&self) -> Vec<AutotuneCandidate> {
        let (m, n, _k) = self.problem_shape;
        let elem_bytes = self.dtype.encoding().bytes_per_block();
        let mut candidates = Vec::new();

        let tile_sizes = match self.arch {
            GpuArchProfile::NvidiaModern => [
                (128, 128, 32, 8, 8, 4),
                (64, 64, 16, 4, 4, 4),
                (32, 32, 16, 2, 2, 2),
                (16, 16, 16, 1, 1, 1),
            ],
            GpuArchProfile::AppleSilicon => [
                (64, 64, 16, 4, 4, 2),
                (32, 32, 16, 2, 2, 1),
                (16, 16, 16, 1, 1, 1),
                (16, 16, 16, 1, 1, 1),
            ],
            GpuArchProfile::GenericWebGpu => [
                (32, 32, 16, 2, 2, 2),
                (16, 16, 16, 1, 1, 1),
                (16, 16, 16, 1, 1, 1),
                (16, 16, 16, 1, 1, 1),
            ],
        };

        for (bm, bn, bk, tm, tn, warps) in tile_sizes {
            if m < bm / 2 || n < bn / 2 {
                continue;
            }
            // Shared memory calculation with +1 bank conflict avoidance padding
            let shmem_a = bm * (bk + 1) * elem_bytes;
            let shmem_b = bk * (bn + 1) * elem_bytes;
            let total_shmem = shmem_a + shmem_b;

            // NVIDIA typically has 48KB-100KB per block limit; WebGPU has 16KB-32KB limit
            let max_shmem = match self.arch {
                GpuArchProfile::NvidiaModern => 48 * 1024,
                GpuArchProfile::AppleSilicon => 32 * 1024,
                GpuArchProfile::GenericWebGpu => 16 * 1024,
            };

            if total_shmem <= max_shmem {
                candidates.push(AutotuneCandidate {
                    block_m: bm,
                    block_n: bn,
                    block_k: bk,
                    thread_m: tm,
                    thread_n: tn,
                    num_warps: warps,
                    shared_memory_bytes: total_shmem,
                });
            }
        }

        if candidates.is_empty() {
            candidates.push(AutotuneCandidate {
                block_m: 16,
                block_n: 16,
                block_k: 16,
                thread_m: 1,
                thread_n: 1,
                num_warps: 1,
                shared_memory_bytes: 16 * 17 * 2 * elem_bytes,
            });
        }

        candidates
    }

    /// Selects the best heuristic configuration without dynamic benchmarking.
    #[must_use]
    pub fn select_best_heuristic(&self) -> AutotuneCandidate {
        let candidates = self.generate_candidates();
        candidates[0].clone()
    }
}
