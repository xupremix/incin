use super::*;

/// A rank-2 view of one operand: where its `[0, 0]` element lives and how far
/// apart its rows and columns are. Constructing one is free, which is the
/// point: the batch loop below builds `batch_total` of them.
#[derive(Clone, Copy)]
pub(super) struct MatrixView<'a> {
    pub(super) buffer: &'a CpuBuffer,
    pub(super) offset: usize,
    pub(super) row_stride: usize,
    pub(super) col_stride: usize,
}

impl<'a> MatrixView<'a> {
    /// View the trailing two axes of a rank `>= 2` operand. Callers check the
    /// rank first, because they can report a better error than this can.
    pub(super) fn trailing(storage: &'a CpuStorage) -> Self {
        let rank = storage.strides.len();
        debug_assert!(rank >= 2, "matrix views need a rank of at least two");
        Self {
            buffer: &storage.buffer,
            offset: storage.offset_elements,
            row_stride: storage.strides[rank - 2],
            col_stride: storage.strides[rank - 1],
        }
    }

    /// The same matrix rebased onto another batch slice.
    pub(super) fn at(self, offset: usize) -> Self {
        Self { offset, ..self }
    }

    #[inline]
    pub(super) fn index(&self, row: usize, col: usize) -> usize {
        self.offset + row * self.row_stride + col * self.col_stride
    }

    #[inline]
    pub(super) fn get(&self, row: usize, col: usize) -> f64 {
        self.buffer.get_f64(self.index(row, col))
    }

    /// The backing slice when this operand is already `f32`, which is what the
    /// SIMD and blocked kernels need in order to read it without converting.
    pub(super) fn f32_data(&self) -> Option<&'a [f32]> {
        match self.buffer {
            CpuBuffer::F32(values) => Some(values),
            _ => None,
        }
    }

    /// True when every index of a `rows` by `cols` read lands inside `len`.
    /// Only the corner has to be checked because strides are unsigned, and it
    /// is checked in full so the `cpu-blas` path can pass raw pointers. Every
    /// other kernel here indexes slices, so this exists only for that path.
    #[cfg(feature = "cpu-blas")]
    pub(super) fn fits_within(&self, rows: usize, cols: usize, len: usize) -> bool {
        if rows == 0 || cols == 0 {
            return true;
        }
        let corner = rows
            .checked_sub(1)
            .and_then(|last| last.checked_mul(self.row_stride))
            .zip(
                cols.checked_sub(1)
                    .and_then(|last| last.checked_mul(self.col_stride)),
            )
            .and_then(|(down, across)| down.checked_add(across))
            .and_then(|extent| extent.checked_add(self.offset));
        corner.is_some_and(|corner| corner < len)
    }
}
