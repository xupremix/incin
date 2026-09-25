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
