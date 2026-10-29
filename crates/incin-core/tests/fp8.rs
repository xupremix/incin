//! FP8 e4m3/e5m2: registry, catalog admission, serialization, and the
//! CPU-rails slice of the scaled-cast surface (issue #94).
//!
//! What runs here is what needs no capability row: descriptors, keys,
//! checkpoint metadata, and the f32→fp8 forward leg (a row admits the f32
//! *source*; the fp8 target is an executor-checked attribute). Anything
//! that reads an fp8 *source* — the return leg, execution, training loops
//! over fp8 tensors — waits on the orchestrator's capability pass and is
//! pinned below as a typed refusal, not as a silent gap.

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;

type B = CpuBackendImpl;

#[test]
fn fp8_descriptors_are_distinct_float_scalars() {
    let e4 = DTypeId::F8E4M3.descriptor();
    let e5 = DTypeId::F8E5M2.descriptor();
    assert_ne!(e4, e5, "e4m3 and e5m2 must never unify");

    for (id, name) in [(DTypeId::F8E4M3, "f8e4m3"), (DTypeId::F8E5M2, "f8e5m2")] {
        let d = id.descriptor();
        assert_eq!(d.key().namespace(), "incin");
        assert_eq!(d.key().name(), name);
        assert_eq!(d.key().version(), 1);
        assert_eq!(d.kind(), DTypeKind::Float);
        assert!(d.is_float());
        assert!(!d.is_quantized() && !d.is_integer() && !d.is_bool());
        assert!(id.is_float());
        let enc = d.encoding();
        assert!(enc.is_scalar());
        assert_eq!(enc.scalar_bytes(), Some(1));
        assert_eq!(id.name(), name);
    }
    // The u8-width collision, at descriptor level: same 1-byte scalar
    // encoding, different logical dtype.
    assert_eq!(
        DTypeId::F8E4M3.descriptor().encoding(),
        DTypeId::U8.descriptor().encoding()
    );
    assert_ne!(
        DTypeId::F8E4M3.descriptor().key(),
        DTypeId::U8.descriptor().key()
    );
}

#[test]
fn fp8_is_a_static_float_dtype() {
    fn assert_float<T: FloatDType>() {}
    assert_float::<F8E4M3>();
    assert_float::<F8E5M2>();
    fn assert_plain<T: PlainDType>() {}
    assert_plain::<F8E4M3>();
    assert_plain::<F8E5M2>();
    assert_eq!(core::mem::size_of::<F8E4M3>(), 1);
    assert_eq!(core::mem::size_of::<F8E5M2>(), 1);
}

#[test]
fn fp8_keys_survive_the_wire() {
    for id in [DTypeId::F8E4M3, DTypeId::F8E5M2] {
        let key = id.descriptor().key();
        let wire = serde_json::to_string(&key).expect("a key always serializes");
        let back: DTypeKey = serde_json::from_str(&wire).expect("builtin keys resolve by arm");
        assert_eq!(back, key);
        // Full descriptor round-trips through the builtin cross-check.
        let desc_wire =
            serde_json::to_string(&id.descriptor()).expect("a descriptor always serializes");
        let desc_back: DTypeDescriptor =
            serde_json::from_str(&desc_wire).expect("builtin descriptors resolve");
        assert_eq!(desc_back, id.descriptor());
    }
}

#[test]
fn fp8_checkpoint_metadata_resolves() {
    use incin_core::nn::CheckpointDType;
    for id in [DTypeId::F8E4M3, DTypeId::F8E5M2] {
        let meta = CheckpointDType::from_descriptor(id.descriptor());
        assert_eq!(meta.descriptor().unwrap(), id.descriptor());
    }
}

#[test]
fn scaled_forward_leg_casts_and_names_its_dtype() {
    let w = Tensor::<s![4], B>::from_slice(&[224.0f32, -112.0, 56.0, -28.0], ()).unwrap();
    let w8 = w.to_dtype_scaled::<F8E4M3>(224.0).unwrap();
    assert_eq!(w8.dtype(), DTypeId::F8E4M3.descriptor());
    assert_eq!(w8.shape_buf().as_ref(), &[4]);
    let g8 = w.to_dtype_scaled::<F8E5M2>(224.0).unwrap();
    assert_eq!(g8.dtype(), DTypeId::F8E5M2.descriptor());
}

/// The return leg: an fp8 source casts back through the widened ToDType
/// row (issue #94 integration). Values are exact here — 224/112/56/28 are
/// all representable in e4m3 — so this pins bit-equality, not tolerance.
#[test]
fn fp8_source_rows_cast_back_exactly() {
    let w = Tensor::<s![4], B>::from_slice(&[224.0f32, -112.0, 56.0, -28.0], ()).unwrap();
    let w8 = w.to_dtype_scaled::<F8E4M3>(224.0).unwrap();
    assert_eq!(w8.dtype(), DTypeId::F8E4M3.descriptor());
    let back = w8.from_dtype_scaled::<f32>(224.0).unwrap();
    assert_eq!(back.dtype(), DTypeId::F32.descriptor());
    assert_eq!(
        back.to_vec1::<f32>().unwrap(),
        [224.0f32, -112.0, 56.0, -28.0],
        "exactly-representable values must round-trip bit-identically"
    );
}

/// Storage-only scaled round-trip (issue #94 slice; no matmul): values that
/// are *not* exactly representable round-trip through
/// `to_dtype_scaled`/`from_dtype_scaled` within the format's documented
/// error bound.
///
/// Documented bounds
/// (`crates/incin-backends/src/conformance/tolerance.rs`, e4m3 row:
/// absolute 0.25 / relative 0.25, e5m2 row: absolute 0.5 / relative 0.5;
/// CPU-only storage admission:
/// `crates/incin-backends/src/capability/constants.rs:27-28`): the fp8 cast
/// rounds the *normalized* value (`self / scale`, in `[-1, 1]` here since
/// `amax == scale`, so nothing saturates) by at most half an ulp, and the
/// return leg re-applies the scale, so the error in original units is at
/// most the normalized-domain tolerance times `scale`.
///
/// Each format runs at three scales, one deliberately above its max normal
/// (e4m3: 448, e5m2: 57344) to prove the scale — not the format range —
/// is what keeps values representable.
#[test]
fn fp8_scaled_roundtrip_within_expected_error() {
    // Normalized pattern: magnitudes in [0.1, 1.0], all in the normal
    // range of both formats (e4m3 min normal 2^-6, e5m2 min normal 2^-14),
    // mostly off-grid so the cast actually rounds.
    let pattern = [1.0f32, 0.333, -0.5, 0.1, -0.75, 0.907, -0.123, 0.456];

    for scale in [4.0f32, 64.0, 1000.0] {
        let values: Vec<f32> = pattern.iter().map(|x| x * scale).collect();
        let w = Tensor::<s![8], B>::from_slice(&values, ()).unwrap();
        let w8 = w.to_dtype_scaled::<F8E4M3>(scale).unwrap();
        let back = w8.from_dtype_scaled::<f32>(scale).unwrap();
        let got = back.to_vec1::<f32>().unwrap();
        let max_err = values
            .iter()
            .zip(got.iter())
            .map(|(want, got)| (want - got).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err <= 0.25 * scale,
            "e4m3 round-trip at scale {scale}: max abs error {max_err} exceeds 0.25 * scale"
        );
    }

    for scale in [4.0f32, 1024.0, 100_000.0] {
        let values: Vec<f32> = pattern.iter().map(|x| x * scale).collect();
        let w = Tensor::<s![8], B>::from_slice(&values, ()).unwrap();
        let w8 = w.to_dtype_scaled::<F8E5M2>(scale).unwrap();
        let back = w8.from_dtype_scaled::<f32>(scale).unwrap();
        let got = back.to_vec1::<f32>().unwrap();
        let max_err = values
            .iter()
            .zip(got.iter())
            .map(|(want, got)| (want - got).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err <= 0.5 * scale,
            "e5m2 round-trip at scale {scale}: max abs error {max_err} exceeds 0.5 * scale"
        );
    }
}
