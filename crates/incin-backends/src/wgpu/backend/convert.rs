//! `to_dtype` on WGPU: bidirectional casts between the storage dtypes this
//! backend can honestly hold (`F32`/`U8`/`U32`/`I64` sources,
//! `{F32, U8, U32, I64}` targets), host-side.
//!
//! Same-dtype is a clone. Every other pair downloads the source buffer at
//! its own physical width, casts value-by-value with Rust `as` (the same
//! truncating conversion CPU's `tensor_to_dtype_storage` performs), and
//! uploads under the target dtype. Sources or targets outside the admitted
//! set are refused by name through `Error::UnsupportedDType`, the same typed
//! refusal CUDA raises for an out-of-set target.
//!
//! No tape entry is ever pushed, and that is a claim rather than an
//! omission: the only float-to-float pair this backend can name is the
//! `F32`-to-`F32` identity (returned as a clone above), and `F32` is the only
//! float storage WGPU has — `device.rs` requests `Features::empty()` (no
//! `SHADER_F16`), every shader reads `array<f32>`, and `validate_wgpu_dtype`
//! refuses `f16`/`bf16`/`f64` outright. So unlike CPU's
//! `canonical_to_dtype` (or CUDA's/Metal's float-to-float backward, which
//! casts the cotangent back to the source dtype), there is no non-trivial
//! float cast whose identity gradient a node could carry. Integer and `bool`
//! casts truncate, so their derivative is zero almost everywhere and
//! recording one would report a sensitivity the forward does not have —
//! CPU records nothing for those either.
//!
//! `Bool` is admittable as a *source* (physical `f32` 0.0/1.0, validated
//! exactly like `HostInterop::to_bytes` validates it) but refused as a
//! *target*: CPU itself has no `bool` conversion in `tensor_to_dtype_storage`
//! and CUDA/Metal refuse `bool` targets the same way, so minting one here
//! would invent cross-backend semantics no other backend honours.
//!
//! The capability row this rides (`broadcast`, `F32_ONLY`) admits the *input*
//! dtype; `admit_invocation` never inspects the target tag, so the wider
//! source set here is belt-and-suspenders for direct `Execute` calls, not a
//! claim the row makes — the same split Metal's `metal/convert.rs` documents.

use incin_core::error::{Error, Result};
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::{DTypeDescriptor, DTypeId};

use super::types::WgpuBackendImpl;
use crate::wgpu::storage::{WgpuBuffer, WgpuStorage};

use super::util::num_elements;

/// Host-side numeric conversion: download `input` at its own physical width,
/// cast value-by-value, and upload under `target`.
fn cast_values(input: &WgpuStorage, target: DTypeDescriptor) -> Result<WgpuStorage> {
    let shape = input.shape().to_vec();
    let total = num_elements(&shape)?;
    let unsupported_target = |dtype: DTypeDescriptor| {
        Error::UnsupportedDType {
            dtype,
            backend: "Wgpu",
            op: "to_dtype",
        }
    };
    match target.builtin_id() {
        Some(DTypeId::F32) => {
            let values: Vec<f32> = read_as_f64(input, total)?
                .into_iter()
                .map(|value| value as f32)
                .collect();
            upload(&values, &shape, target)
        }
        Some(DTypeId::U8) => {
            let values: Vec<u8> = read_as_f64(input, total)?
                .into_iter()
                .map(|value| value as u8)
                .collect();
            upload(&values, &shape, target)
        }
        Some(DTypeId::U32) => {
            let values: Vec<u32> = read_as_f64(input, total)?
                .into_iter()
                .map(|value| value as u32)
                .collect();
            upload(&values, &shape, target)
        }
        Some(DTypeId::I64) => {
            let values: Vec<i64> = read_as_f64(input, total)?
                .into_iter()
                .map(|value| value as i64)
                .collect();
            upload(&values, &shape, target)
        }
        _ => Err(unsupported_target(target)),
    }
}

/// Read every element of `input` as `f64`, through the source dtype's own
/// physical width.
///
/// `Bool` is physical `f32` restricted to exactly 0.0/1.0 — the same
/// strictness `HostInterop::to_bytes` applies — so a corrupted mask is
/// refused rather than cast as an arbitrary float.
fn read_as_f64(input: &WgpuStorage, total: usize) -> Result<Vec<f64>> {
    let source = input.metadata().dtype();
    let values: Vec<f64> = match source.builtin_id() {
        Some(DTypeId::F32) => {
            let data: Vec<f32> = input.buffer.to_vec::<f32>()?;
            checked_len(data.len(), total)?;
            data.into_iter().map(f64::from).collect()
        }
        Some(DTypeId::U8) => {
            let data: Vec<u8> = input.buffer.to_vec::<u8>()?;
            checked_len(data.len(), total)?;
            data.into_iter().map(f64::from).collect()
        }
        Some(DTypeId::U32) => {
            let data: Vec<u32> = input.buffer.to_vec::<u32>()?;
            checked_len(data.len(), total)?;
            data.into_iter().map(f64::from).collect()
        }
        Some(DTypeId::I64) => {
            let data: Vec<i64> = input.buffer.to_vec::<i64>()?;
            checked_len(data.len(), total)?;
            data.into_iter().map(|value| value as f64).collect()
        }
        Some(DTypeId::Bool) => {
            let data: Vec<f32> = input.buffer.to_vec::<f32>()?;
            checked_len(data.len(), total)?;
            let mut out = Vec::with_capacity(data.len());
            for value in data {
                if value == 0.0 {
                    out.push(0.0);
                } else if value == 1.0 {
                    out.push(1.0);
                } else {
                    return Err(Error::Msg(alloc::format!(
                        "WGPU to_dtype refuses a bool source holding {value}, \
                         which is neither 0.0 nor 1.0"
                    )));
                }
            }
            out
        }
        _ => {
            return Err(Error::UnsupportedDType {
                dtype: source,
                backend: "Wgpu",
                op: "to_dtype",
            });
        }
    };
    Ok(values)
}

/// The download must name exactly the shape's element count: a short buffer
/// would otherwise cast garbage padding, and a long one would silently drop
/// a tail.
fn checked_len(got: usize, expected: usize) -> Result<()> {
    if got != expected {
        return Err(Error::InvalidByteLength { expected, got });
    }
    Ok(())
}

/// Upload `values` under `target` for `shape`.
///
/// Uses the fallible `try_from_slice` rather than the panicking `from_slice`:
/// a widening cast (`u8` to `i64` is 8x the bytes) can outgrow the adapter's
/// `max_storage_buffer_binding_size` even when the source fit, and that is a
/// refusal, not a programmer error.
fn upload<T: bytemuck::Pod>(
    values: &[T],
    shape: &[usize],
    target: DTypeDescriptor,
) -> Result<WgpuStorage> {
    let buffer = WgpuBuffer::try_from_slice(values)?;
    WgpuStorage::try_new_with_dtype(buffer, shape.to_vec(), target)
}

impl<D: Device> WgpuBackendImpl<D> {
    /// `to_dtype(input, target)`: convert storage dtype host-side. Records
    /// nothing on the tape (see the module docs for why that is a claim).
    pub(crate) fn to_dtype(
        input: &WgpuStorage,
        target: DTypeDescriptor,
    ) -> Result<WgpuStorage> {
        let source = input.metadata().dtype();
        if source == target {
            return Ok(input.clone());
        }
        cast_values(input, target)
    }
}
