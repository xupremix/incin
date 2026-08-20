//! Backend-family identity and the sealed static-marker vocabulary.

use super::error::IdentityError;
#[cfg(feature = "cuda")]
use incin_core::tensor::device::{Cuda, CudaN};
use incin_core::{tensor::device::Cpu, tensor::device::DeviceKind, typenum::Unsigned};

/// A stable backend-family tag.
///
/// This mirrors the runtime [`DeviceKind`] vocabulary while implementing
/// ordering and hashing for persistent cache keys.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum BackendIdentity {
    /// Host CPU execution.
    Cpu,
    /// NVIDIA CUDA execution.
    Cuda,
    /// WebGPU execution.
    Wgpu,
    /// Native Apple Metal execution.
    Metal,
}

impl BackendIdentity {
    /// The stable lowercase spelling included in identity digests.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
            Self::Wgpu => "wgpu",
            Self::Metal => "metal",
        }
    }

    pub(super) fn from_device_kind(kind: DeviceKind) -> core::result::Result<Self, IdentityError> {
        match kind {
            DeviceKind::Cpu => Ok(Self::Cpu),
            DeviceKind::Cuda => Ok(Self::Cuda),
            DeviceKind::Wgpu => Ok(Self::Wgpu),
            DeviceKind::Metal => Ok(Self::Metal),
            _ => Err(IdentityError::UnsupportedBackend),
        }
    }
}

mod sealed {
    pub trait Sealed {}
}

/// A compile-time device marker with a known backend family.
///
/// This trait is sealed: static identities can only be constructed for Incin
/// device markers whose backend is known. Use `DeviceFingerprint<Dyn>` when
/// the backend itself is selected at runtime.
pub trait StaticBackend: sealed::Sealed + 'static {
    /// Backend encoded by the marker.
    const BACKEND: BackendIdentity;
}

impl sealed::Sealed for Cpu {}
impl StaticBackend for Cpu {
    const BACKEND: BackendIdentity = BackendIdentity::Cpu;
}

#[cfg(feature = "cuda")]
impl sealed::Sealed for Cuda {}
#[cfg(feature = "cuda")]
impl StaticBackend for Cuda {
    const BACKEND: BackendIdentity = BackendIdentity::Cuda;
}

#[cfg(feature = "cuda")]
impl<N: Unsigned + 'static> sealed::Sealed for CudaN<N> {}
#[cfg(feature = "cuda")]
impl<N: Unsigned + 'static> StaticBackend for CudaN<N> {
    const BACKEND: BackendIdentity = BackendIdentity::Cuda;
}

#[cfg(feature = "wgpu")]
impl sealed::Sealed for incin_core::tensor::device::Wgpu {}
#[cfg(feature = "wgpu")]
impl StaticBackend for incin_core::tensor::device::Wgpu {
    const BACKEND: BackendIdentity = BackendIdentity::Wgpu;
}

#[cfg(feature = "wgpu")]
impl<N: Unsigned + 'static> sealed::Sealed for incin_core::tensor::device::WgpuN<N> {}
#[cfg(feature = "wgpu")]
impl<N: Unsigned + 'static> StaticBackend for incin_core::tensor::device::WgpuN<N> {
    const BACKEND: BackendIdentity = BackendIdentity::Wgpu;
}
