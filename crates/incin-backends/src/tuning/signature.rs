//! Shape and layout driven workload signatures and legal candidate pruning.

use alloc::{format, string::String, vec::Vec};
use incin_core::exec::LayoutClass;
use incin_core::prelude::DTypeId;

pub use crate::kernel::{KernelAccess, KernelFamily, KernelKey};
pub use crate::tuning::LaunchCandidate;
pub use incin_core::prelude::OperationKind;

/// Classification of tensor rank for legal-candidate pruning and specialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RankClass {
    Scalar,
    Vector,
    Matrix,
    Volume,
    Tensor4,
    Higher(u8),
}

impl RankClass {
    pub fn from_rank(rank: usize) -> Self {
        match rank {
            0 => Self::Scalar,
            1 => Self::Vector,
            2 => Self::Matrix,
            3 => Self::Volume,
            4 => Self::Tensor4,
            r => Self::Higher(r.min(u8::MAX as usize) as u8),
        }
    }

    pub fn tag(self) -> &'static str {
        match self {
            Self::Scalar => "rank0",
            Self::Vector => "rank1",
            Self::Matrix => "rank2",
            Self::Volume => "rank3",
            Self::Tensor4 => "rank4",
            Self::Higher(_) => "rankN",
        }
    }

    pub fn is_supported_for_vectorization(self) -> bool {
        matches!(
            self,
            Self::Vector | Self::Matrix | Self::Volume | Self::Tensor4
        )
    }
}

/// Shape dimension and size bucket for workload keying and candidate pruning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ShapeBucket {
    pub numel_log2: u8,
    pub primary_dim_log2: u8,
    pub secondary_dim_log2: u8,
}

impl ShapeBucket {
    pub fn from_numel(numel: usize) -> Self {
        Self {
            numel_log2: log2_bucket(numel),
            primary_dim_log2: log2_bucket(numel),
            secondary_dim_log2: 0,
        }
    }

    pub fn from_matrix(rows: usize, cols: usize) -> Self {
        Self {
            numel_log2: log2_bucket(rows.saturating_mul(cols)),
            primary_dim_log2: log2_bucket(rows),
            secondary_dim_log2: log2_bucket(cols),
        }
    }

    pub fn from_gemm(m: usize, n: usize, k: usize) -> Self {
        Self {
            numel_log2: log2_bucket(m.saturating_mul(n)),
            primary_dim_log2: log2_bucket(m),
            secondary_dim_log2: log2_bucket(k),
        }
    }

    pub fn tag(self) -> String {
        format!("p{}_s{}", self.primary_dim_log2, self.secondary_dim_log2)
    }
}

fn log2_bucket(size: usize) -> u8 {
    if size <= 1 {
        0
    } else {
        (usize::BITS - (size - 1).leading_zeros()).min(u8::MAX.into()) as u8
    }
}

/// Classification of memory pointer and stride byte alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlignmentClass {
    Byte,
    Short,
    Word,
    Quad,
    Align256,
}

impl AlignmentClass {
    pub fn from_bytes(bytes: usize) -> Self {
        if bytes == 0 || bytes.is_multiple_of(256) {
            Self::Align256
        } else if bytes.is_multiple_of(16) {
            Self::Quad
        } else if bytes.is_multiple_of(4) {
            Self::Word
        } else if bytes.is_multiple_of(2) {
            Self::Short
        } else {
            Self::Byte
        }
    }

    pub fn bytes(self) -> usize {
        match self {
            Self::Byte => 1,
            Self::Short => 2,
            Self::Word => 4,
            Self::Quad => 16,
            Self::Align256 => 256,
        }
    }

    pub fn is_vector_compatible(self, vector_width: u8, element_size: usize) -> bool {
        let required = (vector_width as usize) * element_size;
        self.bytes() >= required
    }

    pub fn tag(self) -> &'static str {
        match self {
            Self::Byte => "align1",
            Self::Short => "align2",
            Self::Word => "align4",
            Self::Quad => "align16",
            Self::Align256 => "align256",
        }
    }
}

/// Structured identifier for storage, compute, accumulator, and output data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DTypePolicyId {
    pub storage: DTypeId,
    pub compute: DTypeId,
    pub accumulator: DTypeId,
    pub output: DTypeId,
}

impl DTypePolicyId {
    pub fn new(storage: DTypeId, compute: DTypeId, accumulator: DTypeId, output: DTypeId) -> Self {
        Self {
            storage,
            compute,
            accumulator,
            output,
        }
    }

    pub fn tag(self) -> String {
        format!(
            "s{:?}_c{:?}_a{:?}_o{:?}",
            self.storage, self.compute, self.accumulator, self.output
        )
    }
}

impl PartialOrd for DTypePolicyId {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DTypePolicyId {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (
            self.storage as u8,
            self.compute as u8,
            self.accumulator as u8,
            self.output as u8,
        )
            .cmp(&(
                other.storage as u8,
                other.compute as u8,
                other.accumulator as u8,
                other.output as u8,
            ))
    }
}

/// Shape and layout driven workload signature.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct KernelSignature {
    pub policy_id: DTypePolicyId,
    pub rank_class: RankClass,
    pub shape_bucket: ShapeBucket,
    pub alignment: AlignmentClass,
    pub layout: LayoutClass,
    pub op_kind: OperationKind,
}

impl KernelSignature {
    pub fn new(
        policy_id: DTypePolicyId,
        rank_class: RankClass,
        shape_bucket: ShapeBucket,
        alignment: AlignmentClass,
        layout: LayoutClass,
        op_kind: OperationKind,
    ) -> Self {
        Self {
            policy_id,
            rank_class,
            shape_bucket,
            alignment,
            layout,
            op_kind,
        }
    }
}

impl PartialOrd for KernelSignature {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KernelSignature {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (
            &self.policy_id,
            self.rank_class,
            self.shape_bucket,
            self.alignment,
            self.layout,
            format!("{:?}", self.op_kind),
        )
            .cmp(&(
                &other.policy_id,
                other.rank_class,
                other.shape_bucket,
                other.alignment,
                other.layout,
                format!("{:?}", other.op_kind),
            ))
    }
}

/// Candidate GEMM / MatMul tiling parameters for tuning.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatMulTileCandidate {
    pub tile_m: u16,
    pub tile_n: u16,
    pub tile_k: u16,
    pub block_size: u16,
}

impl MatMulTileCandidate {
    pub fn new(tile_m: u16, tile_n: u16, tile_k: u16, block_size: u16) -> Self {
        Self {
            tile_m,
            tile_n,
            tile_k,
            block_size,
        }
    }
}

/// Prune a set of pointwise launch candidates based on shape, layout, alignment, and dtype bounds.
pub fn prune_pointwise_candidates(
    candidates: &[LaunchCandidate],
    signature: &KernelSignature,
    numel: usize,
    element_size: usize,
) -> Vec<LaunchCandidate> {
    let mut pruned = Vec::new();
    for candidate in candidates {
        if is_legal_pointwise_candidate(*candidate, signature, numel, element_size) {
            pruned.push(*candidate);
        }
    }
    if pruned.is_empty() {
        pruned.push(LaunchCandidate {
            block_size: 256,
            access: KernelAccess::Scalar { unroll_width: 1 },
        });
    }
    pruned
}

fn is_legal_pointwise_candidate(
    candidate: LaunchCandidate,
    signature: &KernelSignature,
    numel: usize,
    element_size: usize,
) -> bool {
    if candidate.block_size < 32
        || candidate.block_size > 1024
        || !candidate.block_size.is_power_of_two()
    {
        return false;
    }
    match candidate.access {
        KernelAccess::Scalar { unroll_width } => {
            if unroll_width == 0 || unroll_width > 4 || !unroll_width.is_power_of_two() {
                return false;
            }
            if signature.layout == LayoutClass::Strided && unroll_width > 1 {
                return false;
            }
            true
        }
        KernelAccess::Packed { vector_width } => {
            if signature.layout != LayoutClass::Contiguous {
                return false;
            }
            if !signature
                .alignment
                .is_vector_compatible(vector_width, element_size)
            {
                return false;
            }
            if numel < (vector_width as usize) {
                return false;
            }
            true
        }
        _ => false,
    }
}

/// Prune a set of reduction launch candidates based on shape, layout, and reduction dimensions.
pub fn prune_reduction_candidates(
    candidates: &[LaunchCandidate],
    signature: &KernelSignature,
    _reduction_size: usize,
) -> Vec<LaunchCandidate> {
    let mut pruned = Vec::new();
    for candidate in candidates {
        if is_legal_reduction_candidate(*candidate, signature) {
            pruned.push(*candidate);
        }
    }
    if pruned.is_empty() {
        pruned.push(LaunchCandidate {
            block_size: 256,
            access: KernelAccess::Scalar { unroll_width: 1 },
        });
    }
    pruned
}

fn is_legal_reduction_candidate(candidate: LaunchCandidate, signature: &KernelSignature) -> bool {
    if candidate.block_size < 32
        || candidate.block_size > 1024
        || !candidate.block_size.is_power_of_two()
    {
        return false;
    }
    match candidate.access {
        KernelAccess::WarpReduction | KernelAccess::Welford => {
            if signature.layout == LayoutClass::Strided {
                return false;
            }
            true
        }
        KernelAccess::Scalar { unroll_width: 1 } => true,
        _ => false,
    }
}

/// Prune a set of MatMul tile candidates based on tile size, shared memory capacity, and hardware limits.
pub fn prune_matmul_candidates(
    candidates: &[MatMulTileCandidate],
    _signature: &KernelSignature,
    _m: usize,
    _n: usize,
    _k: usize,
    element_size: usize,
) -> Vec<MatMulTileCandidate> {
    let mut pruned = Vec::new();
    for candidate in candidates {
        let shared_mem_bytes = ((candidate.tile_m as usize) * (candidate.tile_k as usize)
            + (candidate.tile_k as usize) * (candidate.tile_n as usize))
            * element_size;
        if shared_mem_bytes > 49152 {
            continue;
        }
        if candidate.block_size < 32
            || candidate.block_size > 1024
            || !candidate.block_size.is_power_of_two()
        {
            continue;
        }
        pruned.push(*candidate);
    }
    if pruned.is_empty() {
        pruned.push(MatMulTileCandidate {
            tile_m: 16,
            tile_n: 16,
            tile_k: 16,
            block_size: 256,
        });
    }
    pruned
}
