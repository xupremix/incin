use incin_core::backend_authoring::*;
use incin_core::error::{Error, Result};
use incin_core::shapes::Dyn;
use incin_core::tensor::device::{Device, DeviceId};
use incin_core::tensor::dtype::{DType, DTypeDescriptor, DTypeId};

macro_rules! impl_wgpu_supports_dtype {
    ($($t:ty),*) => {
        $(
            impl<D: Device> SupportsDType<$t> for super::backend::WgpuBackendImpl<D> {
                fn resolve_dtype(field: &<$t as DType>::Field, _device: &DeviceId) -> Result<DTypeDescriptor> {
                    let dt = <$t as DType>::descriptor(field);
                    validate_wgpu_dtype(dt, "dtype")?;
                    Ok(dt)
                }
            }
        )*
    };
}

impl_wgpu_supports_dtype!(f32, u32, i64, u8, bool, half::f16, half::bf16);

impl<D: Device> SupportsDType<Dyn> for super::backend::WgpuBackendImpl<D> {
    fn resolve_dtype(field: &DTypeDescriptor, _device: &DeviceId) -> Result<DTypeDescriptor> {
        validate_wgpu_dtype(*field, "dtype")?;
        Ok(*field)
    }
}

/// The dtypes this backend can honestly hold and round-trip: `f32` compute
/// storage, `bool` as a physical `f32` of 0.0/1.0 (WGSL storage buffers
/// cannot hold `bool`), and the integer widths the indexing paths need for
/// index operands (`u8`/`u32`/`i64` at their own physical widths).
///
/// `f16`/`bf16`/`f64`/`q8_0` stay refused, and the refusal is
/// load-bearing beyond this function — the #91 audit that answers
/// "could this backend ever claim half precision?", companion to #90's
/// matmul note in `capability/tables.rs`:
///
/// - The registry is `static WGPU_CAPABILITIES`: rows are compiled
///   constants and cannot vary per adapter, so even an adapter that
///   reports `SHADER_F16` has no plug-in point for a narrower claim —
///   a runtime-gated dtype has nowhere honest to be stated.
/// - `wgpu/device.rs` requests `wgpu::Features::empty()` at
///   `request_device`: the adapter's `SHADER_F16` feature is neither
///   queried nor required, so the device could not compile an `f16`
///   shader even if one existed.
/// - No shader under `wgpu/shaders/` declares `enable f16`; every kernel
///   reads `array<f32>` (or fixed-width integers, or the bool-as-f32
///   encoding above). There is no `f16` kernel variant to advertise.
/// - `bf16` has no WGPU feature bit at all — unlike CUDA/Metal, which
///   carry `bf16` storage — so there is nothing to query or require;
///   serving it would mean a software reinterpretation no kernel here
///   performs.
/// - `native_precision` (below) refuses any non-`f32` storage for the
///   same reason stated positively: compute here is `array<f32>` in
///   every shader.
///
/// Advertising either dtype would need all of: a per-adapter feature
/// query required at `request_device`, `enable f16` shader variants (or
/// widened validators *and* kernels that read the narrow storage), and a
/// registry able to express the conditional claim. None exist; until
/// they do, every row stays `F32_ONLY`/`BOOL_ONLY`/integer and this
/// validator stays the fail-closed chokepoint the creation and
/// dispatch paths route through.
pub(crate) fn validate_wgpu_dtype(dtype: DTypeDescriptor, op: &'static str) -> Result<()> {
    match dtype.builtin_id() {
        Some(DTypeId::F32 | DTypeId::Bool | DTypeId::U8 | DTypeId::U32 | DTypeId::I64) => Ok(()),
        _ => Err(Error::UnsupportedDType {
            dtype,
            backend: "Wgpu",
            op,
        }),
    }
}

/// Precision resolution stays an explicit `F32`-only check, deliberately not
/// routed through [`validate_wgpu_dtype`]: that function now admits the
/// integer and `bool` storage dtypes the indexing rows need, and a widened
/// `native_precision` would then claim `f32` *compute* for an `i64` tensor
/// just because the storage dtype is on the allowed list. Compute here is
/// still `array<f32>` in every shader.
pub(crate) fn native_precision(
    request: &incin_core::exec::PrecisionRequest,
) -> Result<incin_core::exec::ResolvedPrecision> {
    if request.storage != DTypeId::F32.descriptor() {
        return Err(Error::UnsupportedDType {
            dtype: request.storage,
            backend: "Wgpu",
            op: "native_precision",
        });
    }
    Ok(incin_core::exec::ResolvedPrecision::new(
        request.storage,
        DTypeId::F32.descriptor(),
        DTypeId::F32.descriptor(),
        request.output,
        incin_core::exec::LossScaling::None,
    ))
}

impl<D: Device> incin_core::exec::PrecisionCapabilities for super::backend::WgpuBackendImpl<D> {
    fn native_precision(
        &self,
        request: &incin_core::exec::PrecisionRequest,
    ) -> Result<incin_core::exec::ResolvedPrecision> {
        native_precision(request)
    }
}
