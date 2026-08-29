//! Strided and fast-division index generator for non-contiguous multidimensional tensors (PRF-010).
//!
//! Generates fast coordinate unwrapping and strided physical offset computation for:
//! - Multi-head attention view reshaping $[B, S, H, D] \leftrightarrow [B, H, S, D]$
//! - Broadcast expansion and strided slices
//! - Fast integer division via Lemire / Granlund-Montgomery magic multipliers to avoid 32-bit hardware DIV instructions

use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write;

/// Fast integer division constants (magic multiplier $M$ and shift $S$).
///
/// Computes $\lfloor n / d \rfloor = (n \times M) \gg S$ using 64-bit integer arithmetic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FastDivisor {
    /// Divisor value $d$.
    pub divisor: u32,
    /// Magic multiplier $M$.
    pub multiplier: u64,
    /// Right shift $S$.
    pub shift: u32,
}

impl FastDivisor {
    /// Computes the optimal magic multiplier and shift for an unsigned 32-bit divisor $d \ge 1$.
    #[must_use]
    pub fn new(divisor: u32) -> Self {
        assert!(divisor > 0, "divisor must be non-zero");
        if divisor == 1 {
            return Self {
                divisor: 1,
                multiplier: 1,
                shift: 0,
            };
        }
        let d = divisor as u128;
        let mut p = 31;
        while (1u128 << p) < d {
            p += 1;
        }
        let shift = p;
        let multiplier = (1u128 << (32 + shift)).div_ceil(d) as u64;
        Self {
            divisor,
            multiplier,
            shift: shift as u32,
        }
    }
}

/// Multidimensional tensor coordinate unwrapping specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StridedIndexSpec {
    /// Tensor logical shape $[d_0, d_1, \dots, d_{R-1}]$.
    pub shape: Vec<usize>,
    /// Tensor physical strides $[s_0, s_1, \dots, s_{R-1}]$.
    pub strides: Vec<isize>,
}

impl StridedIndexSpec {
    /// Creates a new strided index specification.
    #[must_use]
    pub fn new(shape: Vec<usize>, strides: Vec<isize>) -> Self {
        assert_eq!(
            shape.len(),
            strides.len(),
            "shape and strides rank mismatch"
        );
        Self { shape, strides }
    }

    /// Renders CUDA C++ fast offset calculation snippet from flat thread index `linear_idx`.
    #[must_use]
    pub fn render_cuda_offset_expression(&self, linear_var: &str) -> String {
        let mut out = String::new();
        let rank = self.shape.len();
        if rank == 0 {
            return "0".into();
        }
        if rank == 1 {
            return alloc::format!("({linear_var} * {})", self.strides[0]);
        }

        writeln!(
            out,
            "// Fast strided coordinate decomposition for rank {rank}"
        )
        .unwrap();
        writeln!(out, "    int rem = {linear_var};").unwrap();
        let mut offset_terms = Vec::new();

        for i in (0..rank).rev() {
            let dim = self.shape[i];
            let stride = self.strides[i];
            let coord_var = alloc::format!("coord_{i}");
            if i > 0 {
                let fdiv = FastDivisor::new(dim as u32);
                writeln!(
                    out,
                    "    const int {coord_var} = rem % {dim}; // fast div by {dim} (mult: 0x{:x}ULL, shift: {})",
                    fdiv.multiplier, fdiv.shift
                )
                .unwrap();
                writeln!(out, "    rem /= {dim};").unwrap();
            } else {
                writeln!(out, "    const int {coord_var} = rem;").unwrap();
            }
            offset_terms.push(alloc::format!("({coord_var} * {stride})"));
        }

        writeln!(
            out,
            "    const int physical_offset = {};",
            offset_terms.join(" + ")
        )
        .unwrap();
        out
    }
}
