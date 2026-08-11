use alloc::sync::Arc;
use core::ops::Deref;

use incin_core::exec::{Alignment, TensorMeta};
use incin_core::prelude::{DTypeDescriptor, DeviceId, Error, OperationKind, Result};

/// Byte alignment every CUDA device allocation satisfies.
///
/// The CUDA C Programming Guide states that any address returned by a driver or
/// runtime allocation routine is aligned to at least 256 bytes, which is what
/// makes the coalescing and vector-load rules in the same section usable. The
/// Rust type `CudaSlice<u8>` cannot express that, so it was previously recorded
/// as [`Alignment::BYTE`] — a true claim, but far weaker than the allocator
/// actually provides, and one that would force a kernel selecting on alignment
/// to take the scalar path on every tensor. `device_pointers_are_aligned_to_the_
/// documented_boundary` measures real pointers rather than trusting the
/// documentation, so this constant is verified on hardware instead of assumed.
pub(crate) const CUDA_ALLOCATION_ALIGNMENT_BYTES: usize = 256;

/// Re-exported from `incin_core::exec::tape` since `GRD-003`: one identity
/// counter serves the whole workspace.
pub use incin_core::exec::TensorId;

#[derive(Debug)]
pub struct CudaBuffer {
    pub(crate) len: usize,
    pub(crate) dtype: DTypeDescriptor,
    pub(crate) data: Arc<cudarc::driver::CudaSlice<u8>>,
    pub(crate) device: Arc<cudarc::driver::CudaContext>,
    pub(crate) device_id: usize,
}

impl PartialEq for CudaBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.len == other.len
            && self.dtype == other.dtype
            && self.device_id == other.device_id
            && Arc::ptr_eq(&self.data, &other.data)
    }
}

impl Clone for CudaBuffer {
    fn clone(&self) -> Self {
        CudaBuffer {
            len: self.len,
            dtype: self.dtype,
            data: self.data.clone(),
            device: self.device.clone(),
            device_id: self.device_id,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CudaStorage {
    pub(crate) buffer: Arc<CudaBuffer>,
    pub(crate) meta: TensorMeta,
    pub(crate) id: TensorId,
}

impl Deref for CudaStorage {
    type Target = TensorMeta;

    fn deref(&self) -> &Self::Target {
        &self.meta
    }
}

impl CudaStorage {
    pub(crate) fn with_fresh_autograd_identity(mut self) -> Self {
        self.id = TensorId::next();
        self
    }

    /// The alignment guarantee every CUDA allocation carries.
    ///
    /// Infallible despite [`Alignment::new`] being fallible, because
    /// [`CUDA_ALLOCATION_ALIGNMENT_BYTES`] is a power of two by construction and
    /// a `debug_assert` pins that; a bad constant is a source error, not a
    /// runtime condition a caller could handle.
    fn allocation_alignment() -> Alignment {
        debug_assert!(CUDA_ALLOCATION_ALIGNMENT_BYTES.is_power_of_two());
        Alignment::new(CUDA_ALLOCATION_ALIGNMENT_BYTES).unwrap_or(Alignment::BYTE)
    }

    /// Check that the allocation is large enough for the element count it claims.
    ///
    /// `CudaBuffer::len` counts logical elements while `data` counts bytes, and
    /// the two are related by the dtype rather than by a fixed width — a `Q8_0`
    /// block is 34 bytes for 32 values. Comparing them here is what stops a
    /// buffer whose `len` was recorded in the wrong unit from reaching a kernel
    /// that would read past the end of it.
    fn check_allocation_covers_len(buffer: &CudaBuffer) -> Result<()> {
        let required = buffer
            .dtype
            .size_bytes(buffer.len, OperationKind::Storage)
            .map_err(|error| Error::Msg(format!("invalid CUDA allocation length: {error}")))?;
        let allocated = buffer.data.len();
        if allocated < required {
            return Err(Error::Msg(format!(
                "CUDA {:?} allocation holds {allocated} bytes but its metadata claims {} elements, which need {required}",
                buffer.dtype, buffer.len
            )));
        }
        Ok(())
    }

    pub fn try_from_parts(
        buffer: Arc<CudaBuffer>,
        shape: Vec<usize>,
        strides: Vec<usize>,
        offset_elements: usize,
    ) -> Result<Self> {
        Self::check_allocation_covers_len(&buffer)?;
        let meta = TensorMeta::try_new(
            shape.as_slice().into(),
            strides.as_slice().into(),
            offset_elements,
            buffer.dtype,
            DeviceId::cuda(buffer.device_id),
            Self::allocation_alignment(),
            buffer.len,
        )
        .map_err(|error| Error::Msg(format!("invalid CUDA storage metadata: {error}")))?;
        Ok(Self {
            buffer,
            meta,
            id: TensorId::next(),
        })
    }

    pub fn try_new(buffer: Arc<CudaBuffer>, shape: Vec<usize>) -> Result<Self> {
        Self::check_allocation_covers_len(&buffer)?;
        let meta = TensorMeta::contiguous(
            shape.as_slice().into(),
            buffer.dtype,
            DeviceId::cuda(buffer.device_id),
            Self::allocation_alignment(),
            buffer.len,
        )
        .map_err(|error| Error::Msg(format!("invalid CUDA storage metadata: {error}")))?;
        Ok(Self {
            buffer,
            meta,
            id: TensorId::next(),
        })
    }

    pub fn new(buffer: Arc<CudaBuffer>, shape: Vec<usize>) -> Self {
        Self::try_new(buffer, shape)
            .expect("backend-created contiguous CUDA storage must match its allocation")
    }

    pub fn metadata(&self) -> &TensorMeta {
        &self.meta
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::{CudaContext, DevicePtr};

    /// The alignment claim recorded by every CUDA `TensorMeta`, measured.
    ///
    /// Sizes are deliberately awkward — one byte, a prime, a value just past a
    /// block boundary — because a well-behaved allocator that happened to round
    /// every request up to a large power of two would pass a test that only
    /// asked for round sizes. Allocations are held for the duration of the loop
    /// so the driver cannot hand the same suspiciously well-aligned address back
    /// for each one.
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn device_pointers_are_aligned_to_the_documented_boundary() {
        let context = CudaContext::new(0).expect("CUDA device 0");
        let stream = context.default_stream();
        let sizes = [1usize, 3, 17, 34, 68, 127, 256, 257, 1024, 4099, 65_537];
        let mut held = Vec::new();
        for size in sizes {
            let slice = stream.alloc_zeros::<u8>(size).expect("device allocation");
            let address = {
                let (pointer, _sync) = slice.device_ptr(&stream);
                pointer as usize
            };
            assert_eq!(
                address % CUDA_ALLOCATION_ALIGNMENT_BYTES,
                0,
                "a {size}-byte allocation returned {address:#x}, which is not \
                 {CUDA_ALLOCATION_ALIGNMENT_BYTES}-byte aligned"
            );
            held.push(slice);
        }
    }

    /// An allocation smaller than the element count it claims must be rejected.
    ///
    /// This is the shape of the defect the first CUDA hardware run exposed: the
    /// quantize path recorded `len` in blocks while the metadata read it as
    /// logical elements, so the storage claimed thirty-two times the memory it
    /// held. Building the buffer by hand is the only way to reach the check,
    /// since every allocation path now sizes itself from the same dtype.
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn storage_rejects_an_allocation_too_small_for_its_element_count() {
        let context = CudaContext::new(0).expect("CUDA device 0");
        let stream = context.default_stream();
        let buffer = Arc::new(CudaBuffer {
            len: 64,
            dtype: DTypeId::F32.descriptor(),
            data: Arc::new(stream.alloc_zeros::<u8>(16).expect("device allocation")),
            device: context.clone(),
            device_id: 0,
        });
        let error = CudaStorage::try_new(buffer, vec![64]).expect_err("undersized allocation");
        assert!(
            format!("{error}").contains("holds 16 bytes"),
            "unexpected rejection: {error}"
        );
    }

    /// `Q8_0` sizing is block arithmetic, not an element width.
    ///
    /// Sixty-four logical values are two blocks of thirty-four bytes, so the
    /// allocation is 68 bytes while the metadata capacity is 64 elements. A
    /// storage that conflated the two is what failed on hardware.
    #[test]
    #[ignore = "requires CUDA hardware"]
    fn packed_q8_storage_reports_logical_elements_over_a_block_sized_allocation() {
        let context = CudaContext::new(0).expect("CUDA device 0");
        let stream = context.default_stream();
        let bytes = DTypeId::Q8_0
            .size_bytes(64, OperationKind::Storage)
            .expect("64 values are two whole blocks");
        assert_eq!(bytes, 68);
        let buffer = Arc::new(CudaBuffer {
            len: 64,
            dtype: DTypeId::Q8_0.descriptor(),
            data: Arc::new(stream.alloc_zeros::<u8>(bytes).expect("device allocation")),
            device: context.clone(),
            device_id: 0,
        });
        let storage = CudaStorage::try_new(buffer, vec![2, 32]).expect("packed Q8_0 storage");
        assert_eq!(storage.metadata().dtype(), DTypeId::Q8_0.descriptor());
        assert_eq!(storage.metadata().shape().dims(), &[2, 32]);
    }
}
