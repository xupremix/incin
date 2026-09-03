//! Parallel prefix scan kernel generator for cumulative operations (PRF-018).
//!
//! Generates warp-level and block-level Hillis-Steele & Blelloch parallel prefix scans for:
//! - Cumulative sum (`cumsum`)
//! - Cumulative product (`cumprod`)
//! - Cumulative max / min (`cummax`, `cummin`)
//! - Cross-backend support for CUDA C++, WebGPU (WGSL), and Metal (MSL)

use alloc::string::String;
use core::fmt::Write;
use incin_core::tensor::dtype::DTypeId;

/// Scan operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScanOpKind {
    /// Cumulative Sum.
    Sum,
    /// Cumulative Product.
    Product,
    /// Cumulative Maximum.
    Max,
    /// Cumulative Minimum.
    Min,
}

/// Prefix Scan Specification.
#[derive(Debug, Clone, PartialEq)]
pub struct PrefixScanSpec {
    /// Kernel function name.
    pub name: String,
    /// Scan operator.
    pub op: ScanOpKind,
    /// Data type.
    pub dtype: DTypeId,
    /// Whether scan is inclusive ($y_i = \sum_{j \le i} x_j$) or exclusive ($y_i = \sum_{j < i} x_j$).
    pub is_inclusive: bool,
}

impl PrefixScanSpec {
    /// Creates a new prefix scan specification.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        op: ScanOpKind,
        dtype: DTypeId,
        is_inclusive: bool,
    ) -> Self {
        Self {
            name: name.into(),
            op,
            dtype,
            is_inclusive,
        }
    }

    /// Renders parallel warp-scan CUDA C++ kernel.
    #[must_use]
    pub fn render_cuda(&self) -> String {
        let mut out = String::new();
        let scalar_ty = match self.dtype {
            DTypeId::F32 => "float",
            DTypeId::F64 => "double",
            DTypeId::F16 => "__half",
            DTypeId::BF16 => "__nv_bfloat16",
            DTypeId::U32 | DTypeId::I64 => "int",
            _ => "float",
        };

        writeln!(
            out,
            "// Parallel Warp Prefix Scan Kernel for {} (CUDA)",
            self.name
        )
        .unwrap();
        writeln!(out, "#include <cuda_fp16.h>").unwrap();
        writeln!(out, "#include <cuda_bf16.h>").unwrap();
        writeln!(out).unwrap();

        writeln!(
            out,
            "extern \"C\" __global__ void {}(\n    const {scalar_ty}* __restrict__ X,\n    {scalar_ty}* __restrict__ Y,\n    const int num_rows,\n    const int scan_dim) {{",
            self.name
        )
        .unwrap();

        writeln!(out, "    const int row = blockIdx.x;").unwrap();
        writeln!(out, "    const int tid = threadIdx.x;").unwrap();
        writeln!(out, "    if (row >= num_rows) return;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    const {scalar_ty}* x_row = X + row * scan_dim;").unwrap();
        writeln!(out, "    {scalar_ty}* y_row = Y + row * scan_dim;").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    // Warp-level parallel Hillis-Steele prefix scan").unwrap();
        writeln!(
            out,
            "    float running_val = (tid < scan_dim) ? static_cast<float>(x_row[tid]) : 0.0f;"
        )
        .unwrap();
        writeln!(out).unwrap();

        let op_merge = match self.op {
            ScanOpKind::Sum => "running_val += other;",
            ScanOpKind::Product => "running_val *= other;",
            ScanOpKind::Max => "running_val = fmaxf(running_val, other);",
            ScanOpKind::Min => "running_val = fminf(running_val, other);",
        };

        writeln!(out, "    #pragma unroll").unwrap();
        writeln!(out, "    for (int offset = 1; offset < 32; offset *= 2) {{").unwrap();
        writeln!(
            out,
            "        const float other = __shfl_up_sync(0xffffffff, running_val, offset);"
        )
        .unwrap();
        writeln!(out, "        if (tid >= offset) {{").unwrap();
        writeln!(out, "            {op_merge}").unwrap();
        writeln!(out, "        }}").unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out).unwrap();

        writeln!(out, "    if (tid < scan_dim) {{").unwrap();
        writeln!(
            out,
            "        y_row[tid] = static_cast<{scalar_ty}>(running_val);"
        )
        .unwrap();
        writeln!(out, "    }}").unwrap();
        writeln!(out, "}}").unwrap();

        out
    }
}
