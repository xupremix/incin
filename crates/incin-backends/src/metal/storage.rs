//! Storage implementation and memory management for the Metal backend.

use alloc::sync::Arc;
use alloc::vec::Vec;
use core::fmt::Debug;

use incin_core::error::{Error, Result};
pub use incin_core::exec::TensorId;
use incin_core::exec::{Alignment, TapeStorage, TensorMeta};
use incin_core::shapes::{OperationKind, ShapeBuf};
use incin_core::tensor::device::DeviceId;
use incin_core::tensor::dtype::DTypeDescriptor;
#[cfg(test)]
use incin_core::tensor::dtype::DTypeId;

/// Storage access mode for Metal buffers on Apple Silicon and macOS.
#[derive(
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    serde::Serialize,
    serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum MetalStorageMode {
    /// Shared memory - CPU and GPU access the same physical memory space.
    /// Default for Apple Silicon unified memory architectures.
    #[default]
    Shared,
    /// Managed memory - explicitly synchronized between CPU and GPU systems.
    Managed,
    /// Private memory - GPU access only, inaccessible directly from CPU host.
    Private,
}

/// Returns whether the host architecture operates under Apple Silicon unified memory rules.
#[must_use]
pub fn is_unified_memory() -> bool {
    #[cfg(target_os = "macos")]
    {
        #[cfg(target_arch = "aarch64")]
        {
            true
        }
        #[cfg(not(target_arch = "aarch64"))]
        {
            false
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// Storage handle backing tensors on Metal devices.
#[derive(Clone)]
pub struct MetalStorage {
    pub(crate) id: TensorId,
    data: Arc<Vec<u8>>,
    metadata: TensorMeta,
    mode: MetalStorageMode,
    device_ordinal: usize,
}

impl Debug for MetalStorage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalStorage")
            .field("id", &self.id)
            .field("metadata", &self.metadata)
            .field("mode", &self.mode)
            .field("device_ordinal", &self.device_ordinal)
            .field("unified_memory", &self.is_unified())
            .finish()
    }
}

impl MetalStorage {
    pub(crate) fn with_fresh_autograd_identity(mut self) -> Self {
        self.id = TensorId::next();
        self
    }

    /// Minimum alignment provided by Metal device buffer allocations (256 bytes).
    pub fn alignment() -> Alignment {
        Alignment::new(256).unwrap_or(Alignment::BYTE)
    }

    /// Creates a new Metal storage buffer from raw bytes, verifying metadata bounds.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if metadata capacity exceeds the backing byte buffer.
    pub fn from_bytes(
        bytes: Vec<u8>,
        metadata: TensorMeta,
        mode: MetalStorageMode,
        device_ordinal: usize,
    ) -> Result<Self> {
        let span_elements = metadata
            .strides
            .checked_span(&metadata.shape, OperationKind::Storage)?;
        let end = metadata.offset_elements.checked_add(span_elements).ok_or(
            incin_core::shapes::ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression: "offset + span",
            },
        )?;
        let required_bytes = metadata.dtype.size_bytes(end, OperationKind::Storage)?;
        if required_bytes > bytes.len() {
            return Err(Error::InvalidByteLength {
                expected: required_bytes,
                got: bytes.len(),
            });
        }
        Ok(Self {
            id: TensorId::next(),
            data: Arc::new(bytes),
            metadata,
            mode,
            device_ordinal,
        })
    }

    /// Allocates a zeroed Metal storage buffer for the given metadata and device.
    ///
    /// # Errors
    ///
    /// Returns [`Error`] if allocation or metadata validation fails.
    pub fn zeros(
        shape: &ShapeBuf,
        dtype: DTypeDescriptor,
        mode: MetalStorageMode,
        device_ordinal: usize,
    ) -> Result<Self> {
        let numel = shape.checked_numel(OperationKind::Storage)?;
        let byte_len = dtype.size_bytes(numel, OperationKind::Storage)?;
        let meta = TensorMeta::contiguous(
            shape.clone(),
            dtype,
            DeviceId::metal(device_ordinal),
            Self::alignment(),
            numel,
        )?;
        let bytes = alloc::vec![0u8; byte_len];
        Self::from_bytes(bytes, meta, mode, device_ordinal)
    }

    /// Metadata describing the shape, stride, layout, and alignment of this storage.
    #[must_use]
    pub fn metadata(&self) -> &TensorMeta {
        &self.metadata
    }

    /// Slice of dimensions of this storage.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.metadata.shape().dims()
    }

    /// The Metal storage mode (`Shared`, `Managed`, or `Private`).
    #[must_use]
    pub fn mode(&self) -> MetalStorageMode {
        self.mode
    }

    /// The device ordinal this storage buffer is assigned to.
    #[must_use]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Returns `true` if this storage buffer resides on a unified memory device.
    #[must_use]
    pub fn is_unified(&self) -> bool {
        is_unified_memory()
    }

    /// Returns a slice of the host-accessible bytes if stored in `Shared` or `Managed` mode.
    ///
    /// # Errors
    ///
    /// Returns [`Error::UnsupportedBackendOperation`] if in `Private` mode without CPU access.
    pub fn as_bytes(&self) -> Result<&[u8]> {
        if self.mode == MetalStorageMode::Private {
            return Err(Error::UnsupportedBackendOperation {
                op: "cpu_readback",
                backend: "Metal (Private Storage)",
            });
        }
        let offset_bytes = self
            .metadata
            .dtype
            .size_bytes(self.metadata.offset_elements, OperationKind::Storage)?;
        let span_elements = self
            .metadata
            .strides
            .checked_span(&self.metadata.shape, OperationKind::Storage)?;
        let end_bytes = self.metadata.dtype.size_bytes(
            self.metadata
                .offset_elements
                .checked_add(span_elements)
                .ok_or(incin_core::shapes::ShapeError::ArithmeticOverflow {
                    operation: OperationKind::Storage,
                    expression: "offset + span",
                })?,
            OperationKind::Storage,
        )?;
        Ok(&self.data[offset_bytes..end_bytes])
    }

    /// Device ID corresponding to this Metal storage handle.
    #[must_use]
    pub fn device(&self) -> DeviceId {
        DeviceId::metal(self.device_ordinal)
    }

    /// Unique identifier for this storage handle.
    #[must_use]
    pub fn id(&self) -> TensorId {
        self.id
    }
}

impl TapeStorage for MetalStorage {
    fn id(&self) -> TensorId {
        self.id
    }

    fn ones_like(&self) -> Result<Self> {
        let numel = self.metadata.shape.checked_numel(OperationKind::Storage)?;
        let data_f32: Vec<f32> = vec![1.0; numel];
        let bytes: Vec<u8> = bytemuck::cast_slice(&data_f32).to_vec();
        let meta = TensorMeta::contiguous(
            self.metadata.shape.clone(),
            self.metadata.dtype,
            self.device(),
            Self::alignment(),
            numel,
        )?;
        Self::from_bytes(bytes, meta, self.mode, self.device_ordinal)
    }

    fn accumulate(&self, contribution: &Self) -> Result<Self> {
        if self.metadata.shape.dims() != contribution.metadata.shape.dims() {
            return Err(Error::ShapeMismatch {
                op: "accumulate",
                expected: self.metadata.shape.dims().to_vec(),
                got: contribution.metadata.shape.dims().to_vec(),
                msg: "shapes must match for gradient accumulation".to_string(),
            });
        }
        let a_bytes = self.as_bytes()?;
        let b_bytes = contribution.as_bytes()?;
        let a_slice: &[f32] = bytemuck::cast_slice(a_bytes);
        let b_slice: &[f32] = bytemuck::cast_slice(b_bytes);
        let out_data: Vec<f32> = a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(x, y)| x + y)
            .collect();
        let out_bytes: Vec<u8> = bytemuck::cast_slice(&out_data).to_vec();
        Self::from_bytes(
            out_bytes,
            self.metadata.clone(),
            self.mode,
            self.device_ordinal,
        )
    }

    fn has_non_finite(&self) -> Result<bool> {
        let bytes = self.as_bytes()?;
        let slice: &[f32] = bytemuck::cast_slice(bytes);
        Ok(slice.iter().any(|x| x.is_nan() || x.is_infinite()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use incin_core::shapes::ShapeBuf;

    #[test]
    fn test_metal_storage_modes_and_defaults() {
        assert_eq!(MetalStorageMode::default(), MetalStorageMode::Shared);
        assert_ne!(MetalStorageMode::Shared, MetalStorageMode::Private);
    }

    #[test]
    fn test_metal_storage_zeros_and_bounds() {
        let shape = ShapeBuf::from_slice(&[2, 3]);
        let storage =
            MetalStorage::zeros(&shape, DTypeId::F32.into(), MetalStorageMode::Shared, 0).unwrap();
        assert_eq!(storage.metadata().dtype(), DTypeId::F32.descriptor());
        assert_eq!(storage.metadata().shape().dims(), &[2, 3]);
        assert_eq!(storage.device(), DeviceId::metal(0));
        assert_eq!(storage.as_bytes().unwrap().len(), 24);
    }

    #[test]
    fn test_metal_private_storage_guard() {
        let shape = ShapeBuf::from_slice(&[4]);
        let storage =
            MetalStorage::zeros(&shape, DTypeId::F32.into(), MetalStorageMode::Private, 0).unwrap();
        assert!(storage.as_bytes().is_err());
    }

    #[test]
    fn test_metal_storage_overflow_check() {
        let shape = ShapeBuf::from_slice(&[100]);
        let meta = TensorMeta::contiguous(
            shape,
            DTypeId::F32.into(),
            DeviceId::metal(0),
            MetalStorage::alignment(),
            100,
        )
        .unwrap();

        // 400 bytes required, pass only 10
        let err =
            MetalStorage::from_bytes(vec![0; 10], meta, MetalStorageMode::Shared, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidByteLength { .. }));
    }
}
