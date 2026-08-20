//! CUDA-specific identity queries.
//!
//! This module is gated at the declaration site in `mod.rs`, so its items do
//! not repeat the `cuda` feature attribute individually.

use alloc::{
    format,
    string::{String, ToString},
};

use incin_core::tensor::device::Cuda;

use super::compiler::CompilerFingerprint;
use super::device::DeviceFingerprint;
use super::environment::TuningEnvironmentFingerprint;
use super::error::IdentityError;
use super::primitives::SoftwareVersion;

impl DeviceFingerprint<Cuda> {
    /// Queries UUID, architecture, and driver version from a live CUDA
    /// context. The context ordinal is used only in diagnostics and never
    /// enters the returned identity.
    pub fn from_cuda_context(
        context: &cudarc::driver::CudaContext,
    ) -> core::result::Result<Self, IdentityError> {
        let ordinal = context.ordinal();
        let uuid = context.uuid().map_err(|error| IdentityError::CudaQuery {
            component: "device UUID",
            message: format!("device {ordinal}: {error:?}"),
        })?;
        let (major, minor) =
            context
                .compute_capability()
                .map_err(|error| IdentityError::CudaQuery {
                    component: "compute capability",
                    message: format!("device {ordinal}: {error:?}"),
                })?;
        if major < 0 || minor < 0 {
            return Err(IdentityError::CudaQuery {
                component: "compute capability",
                message: format!("device {ordinal} returned sm_{major}{minor}"),
            });
        }
        Self::new(
            &format_cuda_uuid(uuid.bytes),
            &format!("sm_{major}{minor}"),
            cuda_driver_version()?,
        )
    }
}

impl CompilerFingerprint<Cuda> {
    /// Queries NVRTC and the target architecture used for this CUDA context.
    pub fn from_cuda_context(
        context: &cudarc::driver::CudaContext,
    ) -> core::result::Result<Self, IdentityError> {
        let ordinal = context.ordinal();
        let (major, minor) =
            context
                .compute_capability()
                .map_err(|error| IdentityError::CudaQuery {
                    component: "compiler target",
                    message: format!("device {ordinal}: {error:?}"),
                })?;
        if major < 0 || minor < 0 {
            return Err(IdentityError::CudaQuery {
                component: "compiler target",
                message: format!("device {ordinal} returned sm_{major}{minor}"),
            });
        }
        Self::new(
            "nvrtc",
            nvrtc_version()?,
            &format!("sm_{major}{minor}"),
            &[
                "incin-nvrtc-options-v1",
                "default-math",
                "cuda-include-discovery-v1",
            ],
        )
    }
}

impl TuningEnvironmentFingerprint<Cuda> {
    /// Queries the complete CUDA device/compiler environment.
    pub fn from_cuda_context(
        context: &cudarc::driver::CudaContext,
    ) -> incin_core::error::Result<Self> {
        use incin_core::error::Error;
        Self::new(
            DeviceFingerprint::from_cuda_context(context)
                .map_err(|error| Error::Msg(error.to_string()))?,
            CompilerFingerprint::from_cuda_context(context)
                .map_err(|error| Error::Msg(error.to_string()))?,
        )
        .map_err(|error| Error::Msg(error.to_string()))
    }
}

fn cuda_driver_version() -> core::result::Result<SoftwareVersion, IdentityError> {
    let mut encoded = 0_i32;
    let result = unsafe { cudarc::driver::sys::cuDriverGetVersion(&mut encoded) };
    if result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
        return Err(IdentityError::CudaQuery {
            component: "driver version",
            message: format!("{result:?}"),
        });
    }
    if encoded < 0 {
        return Err(IdentityError::CudaQuery {
            component: "driver version",
            message: format!("negative encoded version {encoded}"),
        });
    }
    let encoded = u32::try_from(encoded).map_err(|_| IdentityError::CudaQuery {
        component: "driver version",
        message: format!("encoded version {encoded} does not fit u32"),
    })?;
    Ok(SoftwareVersion::new(
        encoded / 1000,
        (encoded % 1000) / 10,
        encoded % 10,
    ))
}

fn nvrtc_version() -> core::result::Result<SoftwareVersion, IdentityError> {
    let mut major = 0_i32;
    let mut minor = 0_i32;
    let result = unsafe { cudarc::nvrtc::sys::nvrtcVersion(&mut major, &mut minor) };
    if result != cudarc::nvrtc::sys::nvrtcResult::NVRTC_SUCCESS {
        return Err(IdentityError::CudaQuery {
            component: "NVRTC version",
            message: format!("{result:?}"),
        });
    }
    if major < 0 || minor < 0 {
        return Err(IdentityError::CudaQuery {
            component: "NVRTC version",
            message: format!("negative version {major}.{minor}"),
        });
    }
    let major = u32::try_from(major).map_err(|_| IdentityError::CudaQuery {
        component: "NVRTC version",
        message: format!("major version {major} does not fit u32"),
    })?;
    let minor = u32::try_from(minor).map_err(|_| IdentityError::CudaQuery {
        component: "NVRTC version",
        message: format!("minor version {minor} does not fit u32"),
    })?;
    Ok(SoftwareVersion::new(major, minor, 0))
}

fn format_cuda_uuid(bytes: [core::ffi::c_char; 16]) -> String {
    let mut output = String::from("GPU-");
    for (index, byte) in bytes.into_iter().enumerate() {
        if matches!(index, 4 | 6 | 8 | 10) {
            output.push('-');
        }
        use core::fmt::Write as _;
        write!(&mut output, "{:02x}", byte as u8).expect("writing to String cannot fail");
    }
    output
}
