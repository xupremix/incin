//! NVFP4/MXFP4 block-scaled FP4: registry, keys, checkpoint metadata, and
//! version-refusal (issue #95, CPU-real slice).
//!
//! What runs here needs no capability row: descriptors, keys, checkpoint
//! metadata, and the block-arithmetic contract (`size_bytes`, whole-block
//! sharding). Anything that *executes* on FP4 — quantize dispatch, matmul,
//! device rows — waits on the orchestrator's capability pass; the
//! unsupported-hardware audit (`fp4_unsupported_audit.rs` in incin-backends)
//! pins that no accelerator row claims the dtypes meanwhile.
//!
//! mirrors `fp8.rs` (descriptor/key shape) and `checkpoint_block_quant.rs`
//! (version-refusal shape) for the two block dtypes.

extern crate incin_core as incin;

use incin_core::prelude::*;
use incin_core::tensor::dtype::StorageEncoding;
use incin_core::tensor::dtype::{BuiltinDType, ConstDType, MXFP4, NVFP4};

#[test]
fn fp4_descriptors_are_distinct_quantized_blocks() {
    let nv = DTypeId::NVFP4.descriptor();
    let mx = DTypeId::MXFP4.descriptor();
    assert_ne!(nv, mx, "nvfp4 and mxfp4 must never unify");
    assert_ne!(nv, DTypeId::Q8_0.descriptor());

    let cases = [
        (DTypeId::NVFP4, "nvfp4", 16usize, 9usize),
        (DTypeId::MXFP4, "mxfp4", 32usize, 17usize),
    ];
    for (id, name, logical, bytes) in cases {
        let d = id.descriptor();
        assert_eq!(d.key().namespace(), "incin");
        assert_eq!(d.key().name(), name);
        assert_eq!(d.key().version(), 1);
        assert_eq!(d.kind(), DTypeKind::Quantized);
        assert!(d.is_quantized());
        assert!(!d.is_float() && !d.is_integer() && !d.is_bool());
        assert!(id.is_quantized());
        assert_eq!(id.name(), name);
        let enc = d.encoding();
        assert!(enc.is_block());
        assert_eq!(enc.logical_elements_per_block(), logical);
        assert_eq!(enc.bytes_per_block(), bytes);
        assert_eq!(enc.alignment(), 1);
        assert_eq!(enc.scalar_bytes(), None);
        // The audited contract, stated exactly: no encoding extension.
        assert_eq!(enc, StorageEncoding::block(logical, bytes, 1));
    }
}

#[test]
fn fp4_markers_are_static_quant_dtypes() {
    fn assert_quant<T: QuantDType>() {}
    assert_quant::<NVFP4>();
    assert_quant::<MXFP4>();
    fn assert_const<T: ConstDType>() {}
    assert_const::<NVFP4>();
    assert_const::<MXFP4>();
    assert_eq!(NVFP4::DESCRIPTOR, DTypeId::NVFP4.descriptor());
    assert_eq!(MXFP4::DESCRIPTOR, DTypeId::MXFP4.descriptor());
    assert_eq!(NVFP4::DTYPE, DTypeId::NVFP4);
    assert_eq!(MXFP4::DTYPE, DTypeId::MXFP4);
}

#[test]
fn fp4_keys_survive_the_wire() {
    for id in [DTypeId::NVFP4, DTypeId::MXFP4] {
        let key = id.descriptor().key();
        let wire = serde_json::to_string(&key).expect("a key always serializes");
        let back: DTypeKey = serde_json::from_str(&wire).expect("builtin keys resolve by arm");
        assert_eq!(back, key);
        let desc_wire =
            serde_json::to_string(&id.descriptor()).expect("a descriptor always serializes");
        let desc_back: DTypeDescriptor =
            serde_json::from_str(&desc_wire).expect("builtin descriptors resolve");
        assert_eq!(desc_back, id.descriptor());
    }
}

#[test]
fn fp4_checkpoint_metadata_resolves() {
    use incin_core::nn::CheckpointDType;
    for id in [DTypeId::NVFP4, DTypeId::MXFP4] {
        let meta = CheckpointDType::from_descriptor(id.descriptor());
        assert_eq!(meta.descriptor().unwrap(), id.descriptor());
    }
}

#[test]
fn fp4_size_bytes_enforces_whole_blocks() {
    use incin_core::shapes::error::OperationKind;
    // NVFP4: multiples of 16 → 9 bytes each.
    let nv = DTypeId::NVFP4.descriptor();
    assert_eq!(nv.size_bytes(16, OperationKind::Storage).unwrap(), 9);
    assert_eq!(nv.size_bytes(64, OperationKind::Storage).unwrap(), 36);
    assert_eq!(nv.size_bytes(0, OperationKind::Storage).unwrap(), 0);
    assert!(nv.size_bytes(17, OperationKind::Storage).is_err());
    assert!(nv.size_bytes(20, OperationKind::Storage).is_err());
    // MXFP4: multiples of 32 → 17 bytes each.
    let mx = DTypeId::MXFP4.descriptor();
    assert_eq!(mx.size_bytes(32, OperationKind::Storage).unwrap(), 17);
    assert_eq!(mx.size_bytes(64, OperationKind::Storage).unwrap(), 34);
    assert!(mx.size_bytes(16, OperationKind::Storage).is_err());
    assert!(mx.size_bytes(33, OperationKind::Storage).is_err());
}

#[test]
fn fp4_manifest_round_trip_preserves_encoding_and_resolves_descriptor() -> Result<()> {
    use incin_core::nn::save::save_checkpoint_manifest;
    use incin_core::nn::save::{GlobalCheckpointManifest, load_checkpoint_manifest};
    use tempfile::tempdir;
    let dir = tempdir()?;
    let path = dir.path().join("manifest.json");

    let mut manifest = GlobalCheckpointManifest::new(2);
    manifest.add_tensor("block.weight_nv", vec![64], DTypeId::NVFP4, "Sharded:0");
    manifest.add_tensor("block.weight_mx", vec![64], DTypeId::MXFP4, "Sharded:0");
    save_checkpoint_manifest(&manifest, &path)?;

    let wire = std::fs::read_to_string(&path)?;
    assert!(wire.contains("\"nvfp4\""), "{wire}");
    assert!(wire.contains("\"mxfp4\""), "{wire}");

    let loaded = load_checkpoint_manifest(&path)?;
    assert_eq!(manifest, loaded);
    let nv_dtype = &loaded.tensors["block.weight_nv"].dtype;
    assert_eq!(nv_dtype.key, DTypeKey::new("incin", "nvfp4", 1));
    assert_eq!(nv_dtype.kind, DTypeKind::Quantized);
    assert_eq!(nv_dtype.encoding, StorageEncoding::block(16, 9, 1));
    assert_eq!(nv_dtype.descriptor()?, DTypeId::NVFP4.descriptor());
    let mx_dtype = &loaded.tensors["block.weight_mx"].dtype;
    assert_eq!(mx_dtype.key, DTypeKey::new("incin", "mxfp4", 1));
    assert_eq!(mx_dtype.encoding, StorageEncoding::block(32, 17, 1));
    assert_eq!(mx_dtype.descriptor()?, DTypeId::MXFP4.descriptor());
    Ok(())
}

/// DTypeKey versioning (issue #95 AC6, same pattern as #93's checkpoint
/// tests): a known name at an unknown version refuses on load — never
/// coerced to the version this build knows.
#[test]
fn manifest_refuses_fp4_dtype_key_version_2_on_load() -> Result<()> {
    use incin_core::nn::save::save_checkpoint_manifest;
    use incin_core::nn::save::{GlobalCheckpointManifest, load_checkpoint_manifest};
    use tempfile::tempdir;
    for (id, name) in [(DTypeId::NVFP4, "nvfp4"), (DTypeId::MXFP4, "mxfp4")] {
        let dir = tempdir()?;
        let path = dir.path().join("manifest.json");
        let mut manifest = GlobalCheckpointManifest::new(2);
        manifest.add_tensor("block.weight", vec![64], id, "Sharded:0");
        save_checkpoint_manifest(&manifest, &path)?;

        let mut value =
            serde_json::to_value(&manifest).map_err(|e| Error::Msg(format!("to_value: {e}")))?;
        assert_eq!(
            value["tensors"]["block.weight"]["dtype"]["key"][2],
            serde_json::json!(1),
            "serialized DTypeKey should carry version 1 before tampering"
        );
        value["tensors"]["block.weight"]["dtype"]["key"][2] = serde_json::json!(2);
        let json = serde_json::to_string_pretty(&value)
            .map_err(|e| Error::Msg(format!("to_string: {e}")))?;
        std::fs::write(&path, json)?;

        let err = load_checkpoint_manifest(&path).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains(&format!("(incin, {name}, 2)")), "{msg}");
        assert!(msg.contains("registered first"), "{msg}");
    }
    Ok(())
}

#[test]
fn checkpoint_dtype_descriptor_refuses_tampered_fp4_encoding() {
    use incin_core::nn::CheckpointDType;
    // Interleaved-vs-split is pinned by key+version: 8 data-only bytes
    // under the v1 key is a different layout and must refuse.
    let data_only = CheckpointDType {
        key: DTypeKey::new("incin", "nvfp4", 1),
        kind: DTypeKind::Quantized,
        encoding: StorageEncoding::block(16, 8, 1),
    };
    let err = data_only.descriptor().unwrap_err().to_string();
    assert!(
        err.contains("does not match registered dtype nvfp4"),
        "{err}"
    );

    let bad_version = CheckpointDType {
        key: DTypeKey::new("incin", "mxfp4", 2),
        kind: DTypeKind::Quantized,
        encoding: StorageEncoding::block(32, 17, 1),
    };
    let err = bad_version.descriptor().unwrap_err().to_string();
    assert!(
        err.contains("does not match registered dtype mxfp4"),
        "{err}"
    );
}

#[test]
fn fp4_slices_split_whole_blocks_and_refuse_mid_block() -> Result<()> {
    use incin_core::nn::save::slice_bytes_for_rank;
    // NVFP4 [32] = 2 blocks = 18 bytes → world 2 splits cleanly.
    let nv = DTypeId::NVFP4.descriptor();
    let bytes: Vec<u8> = (0..18).map(|i| i as u8).collect();
    let (rank0, shape0) = slice_bytes_for_rank(&bytes, &[32], nv, 0, 0, 2)?;
    assert_eq!((shape0, rank0.len()), (vec![16], 9));
    assert_eq!(rank0.as_slice(), &bytes[0..9]);
    // [32] world 4: local extent 8 < block 16 → mid-block refusal.
    let err = slice_bytes_for_rank(&bytes, &[32], nv, 0, 0, 4)
        .unwrap_err()
        .to_string();
    assert!(err.contains("nvfp4"), "{err}");
    assert!(err.contains("block size 16"), "{err}");
    assert!(err.contains("shard boundary would fall mid-block"), "{err}");
    // MXFP4 [64] = 2 blocks = 34 bytes → world 2 splits cleanly.
    let mx = DTypeId::MXFP4.descriptor();
    let bytes: Vec<u8> = (0..34).map(|i| i as u8).collect();
    let (rank0, shape0) = slice_bytes_for_rank(&bytes, &[64], mx, 0, 0, 2)?;
    assert_eq!((shape0, rank0.len()), (vec![32], 17));
    assert_eq!(rank0.as_slice(), &bytes[0..17]);
    Ok(())
}
