use super::*;

/// Swap the two axes of a 2D `CpuStorage` (thin wrapper over
/// `CpuStorage::transpose(0, 1)`, reused by the backward closure so the
/// gradient composition is built from already-tested primitives rather than
/// a bespoke derivation).
pub(super) fn transpose_2d(t: &CpuStorage) -> CpuStorage {
    t.transpose(0, 1)
        .expect("2D transpose of a 2D matmul operand cannot fail")
}

/// Swap ONLY the last two axes of an N-D (`N >= 2`) `CpuStorage`, leaving
/// every leading batch axis untouched. Generalizes `transpose_2d` to the
/// batched case; both are thin wrappers over the same
/// `CpuStorage::transpose(dim1, dim2)` primitive.
pub(crate) fn transpose_last2(t: &CpuStorage) -> CpuStorage {
    let r = t.shape.len();
    t.transpose(r - 2, r - 1)
        .expect("transpose of the last two axes of a rank>=2 tensor cannot fail")
}
