//! `to_dtype` on Metal: bidirectional casts between the storage dtypes this
//! backend admits for conversion (`F32`/`F64` sources, `{F32, F64, I64}`
//! targets), host-side.
//!
//! Same-dtype is a clone. Float-to-float records a tape entry whose backward
//! casts the gradient back to the source dtype — the identity rule CPU's
//! `canonical_to_dtype` and CUDA's `cuda_to_dtype_storage` both state. An
//! integer target truncates (`value as i64`), so its derivative is zero
//! almost everywhere and nothing is recorded, matching CPU and CUDA. Sources
//! or targets outside the admitted set are refused by name through
//! `BackendError::unsupported(..., UnsupportedReason::DType { .. })`, the
//! same typed refusal CUDA raises for an out-of-set target.
//!
//! The capability row this rides (`broadcast`, `F32_ONLY`) admits the *input*
//! dtype; `admit_invocation` never inspects the target tag, so a wider source
//! set here is belt-and-suspenders for direct `Execute` calls, not a claim
//! the row makes.

use incin_core::error::{Error, Result};
use incin_core::shapes::ShapeBuf;
use incin_core::shapes::error::OperationKind;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

use super::backend::{MetalBackendImpl, storage_from_raw};
use super::storage::MetalStorage;

/// Host-side numeric conversion: reinterpret the shared `f32`/`i64` bytes as
/// values, cast, and repack under `dtype`.
fn cast_values(input: &MetalStorage, dtype: DTypeDescriptor) -> Result<Vec<u8>> {
    let bytes = input.as_bytes()?;
    let shape = input.shape();
    let total = ShapeBuf::from_slice(shape).checked_numel(OperationKind::Storage)?;
    match dtype.builtin_id() {
        Some(DTypeId::F32) => {
            let src: &[f32] = bytemuck::cast_slice(bytes);
            Ok(src[..total]
                .to_vec()
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect())
        }
        Some(DTypeId::F64) => {
            let vals: Vec<f64> = match input.metadata().dtype().builtin_id() {
                Some(DTypeId::F64) => bytemuck::cast_slice::<u8, f64>(bytes)[..total].to_vec(),
                Some(DTypeId::I64) => bytemuck::cast_slice::<u8, i64>(bytes)[..total]
                    .iter()
                    .map(|&v| v as f64)
                    .collect(),
                _ => bytemuck::cast_slice::<u8, f32>(bytes)[..total]
                    .iter()
                    .map(|&v| f64::from(v))
                    .collect(),
            };
            Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        Some(DTypeId::I64) => {
            let vals: Vec<i64> = match input.metadata().dtype().builtin_id() {
                Some(DTypeId::F64) => bytemuck::cast_slice::<u8, f64>(bytes)[..total]
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
                Some(DTypeId::I64) => bytemuck::cast_slice::<u8, i64>(bytes)[..total].to_vec(),
                _ => bytemuck::cast_slice::<u8, f32>(bytes)[..total]
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
            };
            Ok(vals.iter().flat_map(|v| v.to_le_bytes()).collect())
        }
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Metal",
            op: "to_dtype",
        }),
    }
}

impl<D: Device> MetalBackendImpl<D> {
    /// `to_dtype(input, target)`: convert storage dtype, recording a tape
    /// entry only when both sides are floating (CPU's `canonical_to_dtype`).
    pub(crate) fn to_dtype(input: &MetalStorage, target: DTypeDescriptor) -> Result<MetalStorage> {
        let source = input.metadata().dtype();
        let unsupported = |dtype: DTypeDescriptor| {
            Error::Backend(incin_core::error::BackendError::unsupported(
                "Metal",
                incin_core::exec::UnsupportedReason::DType {
                    operation: OperationKind::ToDType,
                    dtype,
                },
            ))
        };
        match source.builtin_id() {
            Some(DTypeId::F32 | DTypeId::F64) => {}
            _ => return Err(unsupported(source)),
        }
        match target.builtin_id() {
            Some(DTypeId::F32 | DTypeId::F64 | DTypeId::I64) => {}
            _ => return Err(unsupported(target)),
        }
        if source == target {
            return Ok(input.clone());
        }
        let shape = input.shape().to_vec();
        let bytes = cast_values(input, target)?;
        let out = storage_from_raw(bytes, &shape, target, input)?;
        let both_float = source.builtin_id().is_some_and(DTypeId::is_float)
            && target.builtin_id().is_some_and(DTypeId::is_float);
        if both_float {
            let (input_id, out_id) = (input.id(), out.id());
            let input_like = input.clone();
            crate::metal::tape::push(crate::metal::tape::TapeEntry {
                output_id: out_id,
                input_ids: vec![input_id],
                backward: Box::new(move |grad_out: &MetalStorage| {
                    // Identity derivative in the input's dtype: cast the
                    // gradient back. `GradMode::Disabled` is already active
                    // during backward, so this cannot re-record.
                    let g_bytes = cast_values(grad_out, source)?;
                    let g = storage_from_raw(g_bytes, &shape, source, &input_like)?;
                    Ok(vec![g])
                }),
            });
        }
        Ok(out)
    }
}
