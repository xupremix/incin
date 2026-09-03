//! Block-level loop and memory access scheduler inspired by OpenAI Triton and PyTorch Inductor (PRF-011).
//!
//! Provides loop scheduling, block pointer manipulation with boundary masks,
//! and shared-memory staging for cross-backend kernel synthesis:
//! - `BlockTensorPtr`: Triton-style multi-dimensional block pointer with stride offsets and masks
//! - `LoopSchedule`: Inductor-style loop domain classification (`Pointwise1D`, `Tiled2D`, `Reduction`)
//! - Automatic vector coalescing analysis and register double-buffering

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Memory space hierarchy inspired by CubeCL and CUDA memory models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemorySpace {
    /// Global Device VRAM.
    Global,
    /// Block/Workgroup Shared Memory (SRAM).
    Shared,
    /// Thread Local Register.
    Register,
}

/// Triton-style block tensor pointer specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BlockTensorPtr {
    /// Pointer variable name (e.g. "x_ptr").
    pub name: String,
    /// Data type of elements.
    pub dtype: DTypeId,
    /// Memory space.
    pub space: MemorySpace,
    /// Block tile shape $[B_0, B_1, \dots, B_{D-1}]$.
    pub block_shape: Vec<usize>,
    /// Physical strides $[S_0, S_1, \dots, S_{D-1}]$.
    pub strides: Vec<isize>,
    /// Whether boundary masking is required.
    pub requires_mask: bool,
}

impl BlockTensorPtr {
    /// Creates a new global block tensor pointer.
    #[must_use]
    pub fn global(
        name: impl Into<String>,
        dtype: DTypeId,
        block_shape: Vec<usize>,
        strides: Vec<isize>,
    ) -> Self {
        assert_eq!(
            block_shape.len(),
            strides.len(),
            "block shape and strides rank mismatch"
        );
        Self {
            name: name.into(),
            dtype,
            space: MemorySpace::Global,
            block_shape,
            strides,
            requires_mask: true,
        }
    }

    /// Returns whether the innermost dimension is contiguous ($S_{-1} = 1$).
    #[must_use]
    pub fn is_innermost_contiguous(&self) -> bool {
        self.strides.last().copied() == Some(1)
    }

    /// Determines the optimal vector width (1, 2, or 4) for this block pointer.
    #[must_use]
    pub fn optimal_vector_width(&self) -> usize {
        if !self.is_innermost_contiguous() {
            return 1;
        }
        let inner_dim = self.block_shape.last().copied().unwrap_or(1);
        if inner_dim % 4 == 0 {
            4
        } else if inner_dim % 2 == 0 {
            2
        } else {
            1
        }
    }
}

/// Inductor-style loop scheduling strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum LoopScheduleKind {
    /// 1D Pointwise contiguous or strided grid ($N$ total elements).
    Pointwise1D {
        /// Total element count $N$.
        numel: usize,
        /// Elements per thread.
        elements_per_thread: usize,
    },
    /// 2D Tiled block grid with shared-memory staging (e.g. GEMM, 2D Stencils).
    Tiled2D {
        /// Block size along M.
        block_m: usize,
        /// Block size along N.
        block_n: usize,
        /// Block reduction step K.
        block_k: usize,
    },
    /// Split outer batch loop and inner reduction loop ($R$).
    Reduction {
        /// Number of independent reduction rows.
        num_rows: usize,
        /// Size of reduction dimension $R$.
        reduction_dim: usize,
    },
}

/// Comprehensive kernel loop and launch scheduler.
#[derive(Debug, Clone, PartialEq)]
pub struct KernelScheduler {
    /// Kernel function name.
    pub name: String,
    /// Loop schedule kind.
    pub kind: LoopScheduleKind,
    /// Input block tensor pointers.
    pub inputs: Vec<BlockTensorPtr>,
    /// Output block tensor pointers.
    pub outputs: Vec<BlockTensorPtr>,
    /// Number of threads per block / workgroup size.
    pub threads_per_block: usize,
}

impl KernelScheduler {
    /// Creates a new kernel scheduler.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        kind: LoopScheduleKind,
        inputs: Vec<BlockTensorPtr>,
        outputs: Vec<BlockTensorPtr>,
    ) -> Self {
        let threads_per_block = match &kind {
            LoopScheduleKind::Pointwise1D { .. } => 256,
            LoopScheduleKind::Tiled2D {
                block_m, block_n, ..
            } => {
                let threads = (block_m * block_n) / 16;
                threads.clamp(64, 256)
            }
            LoopScheduleKind::Reduction { .. } => 256,
        };

        Self {
            name: name.into(),
            kind,
            inputs,
            outputs,
            threads_per_block,
        }
    }

    /// Computes the recommended grid dimension $(X, Y, Z)$ for CUDA/GPU dispatch.
    #[must_use]
    pub fn recommended_grid_dim(&self) -> (usize, usize, usize) {
        match &self.kind {
            LoopScheduleKind::Pointwise1D {
                numel,
                elements_per_thread,
            } => {
                let total_threads_needed = numel.div_ceil(*elements_per_thread);
                let blocks = total_threads_needed.div_ceil(self.threads_per_block);
                (blocks.max(1), 1, 1)
            }
            LoopScheduleKind::Tiled2D {
                block_m, block_n, ..
            } => {
                let grid_x = 1024_usize.div_ceil(*block_n);
                let grid_y = 1024_usize.div_ceil(*block_m);
                (grid_x.max(1), grid_y.max(1), 1)
            }
            LoopScheduleKind::Reduction { num_rows, .. } => (*num_rows, 1, 1),
        }
    }

    /// Renders the scheduled loop preamble and block index bindings in CUDA C++.
    #[must_use]
    pub fn render_cuda_preamble(&self) -> String {
        let mut out = String::new();
        writeln!(
            out,
            "// Triton/Inductor-inspired Loop Preamble for {}",
            self.name
        )
        .unwrap();

        match &self.kind {
            LoopScheduleKind::Pointwise1D {
                numel,
                elements_per_thread,
            } => {
                writeln!(
                    out,
                    "    const int base_idx = (blockIdx.x * blockDim.x + threadIdx.x) * {};",
                    elements_per_thread
                )
                .unwrap();
                writeln!(out, "    const int total_numel = {numel};").unwrap();
            }
            LoopScheduleKind::Tiled2D {
                block_m, block_n, ..
            } => {
                writeln!(out, "    const int block_row = blockIdx.y * {block_m};").unwrap();
                writeln!(out, "    const int block_col = blockIdx.x * {block_n};").unwrap();
                writeln!(out, "    const int tid = threadIdx.x;").unwrap();
            }
            LoopScheduleKind::Reduction {
                num_rows,
                reduction_dim,
            } => {
                writeln!(out, "    const int row = blockIdx.x;").unwrap();
                writeln!(out, "    const int tid = threadIdx.x;").unwrap();
                writeln!(out, "    const int total_rows = {num_rows};").unwrap();
                writeln!(out, "    const int r_dim = {reduction_dim};").unwrap();
                writeln!(out, "    if (row >= total_rows) return;").unwrap();
            }
        }

        out
    }
}
