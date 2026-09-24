//! Integration coverage for block-aware checkpoint slicing and Q8_0 dtype-key
//! version refusal on manifest load (issue #93).
#![cfg(feature = "std")]

use incin_core::nn::save::{
    CheckpointDType, GlobalCheckpointManifest, load_checkpoint_manifest, save_checkpoint_manifest,
    slice_bytes_for_rank,
};
use incin_core::prelude::*;
use incin_core::tensor::dtype::StorageEncoding;
use tempfile::tempdir;

/// Q8_0 stores 32 logical elements in one 34-byte physical block
/// (`StorageEncoding::block(32, 34, 2)`), laid out flat over the whole tensor's
/// row-major logical element stream.

#[test]
fn q8_0_slice_1d_world2_splits_into_whole_blocks() -> Result<()> {
    let q8 = DTypeId::Q8_0.descriptor();
    // [64] logical elements = 2 whole blocks = 68 bytes.
    let bytes: Vec<u8> = (0..68).map(|i| i as u8).collect();

    let (rank0, shape0) = slice_bytes_for_rank(&bytes, &[64], q8, 0, 0, 2)?;
    assert_eq!(shape0, vec![32]);
    assert_eq!(rank0.len(), 34);
    assert_eq!(rank0.as_slice(), &bytes[0..34]);

    let (rank1, shape1) = slice_bytes_for_rank(&bytes, &[64], q8, 0, 1, 2)?;
    assert_eq!(shape1, vec![32]);
    assert_eq!(rank1.len(), 34);
    assert_eq!(rank1.as_slice(), &bytes[34..68]);

    // Concatenating the two rank slices reproduces the original storage.
    let mut rebuilt = rank0;
    rebuilt.extend_from_slice(&rank1);
    assert_eq!(rebuilt, bytes);
    Ok(())
}

#[test]
fn q8_0_slice_2d_axis1_world2_preserves_row_blocks() -> Result<()> {
    let q8 = DTypeId::Q8_0.descriptor();
    // [4, 64]: each row is 64 elements = 2 blocks = 68 bytes; total 8 blocks = 272 bytes.
    let bytes: Vec<u8> = (0..272).map(|i| i as u8).collect();

    let (rank0, shape0) = slice_bytes_for_rank(&bytes, &[4, 64], q8, 1, 0, 2)?;
    assert_eq!(shape0, vec![4, 32]);
    assert_eq!(rank0.len(), 4 * 34);

    let (rank1, shape1) = slice_bytes_for_rank(&bytes, &[4, 64], q8, 1, 1, 2)?;
    assert_eq!(shape1, vec![4, 32]);
    assert_eq!(rank1.len(), 4 * 34);

    // Re-interleave per row: each row's first block comes from rank0, its
    // second block from rank1, rebuilding the original flat block stream.
    let mut rebuilt = Vec::with_capacity(272);
    for row in 0..4 {
        rebuilt.extend_from_slice(&rank0[row * 34..(row + 1) * 34]);
        rebuilt.extend_from_slice(&rank1[row * 34..(row + 1) * 34]);
    }
    assert_eq!(rebuilt, bytes);
    Ok(())
}

#[test]
fn q8_0_slice_axis0_flat_layout_splits_whole_blocks() -> Result<()> {
    let q8 = DTypeId::Q8_0.descriptor();
    // [64, 4]: individual rows are only 4 elements (not a block), but the flat
    // row-major stream is 256 elements = 8 whole blocks = 272 bytes. Sharding
    // axis 0 by 2 gives each rank 32*4 = 128 elements = 4 whole blocks.
    let bytes: Vec<u8> = (0..272).map(|i| i as u8).collect();

    let (rank0, shape0) = slice_bytes_for_rank(&bytes, &[64, 4], q8, 0, 0, 2)?;
    assert_eq!(shape0, vec![32, 4]);
    assert_eq!(rank0.len(), 136);
    assert_eq!(rank0.as_slice(), &bytes[0..136]);

    let (rank1, shape1) = slice_bytes_for_rank(&bytes, &[64, 4], q8, 0, 1, 2)?;
    assert_eq!(shape1, vec![32, 4]);
    assert_eq!(rank1.len(), 136);
    assert_eq!(rank1.as_slice(), &bytes[136..272]);

    let mut rebuilt = rank0;
    rebuilt.extend_from_slice(&rank1);
    assert_eq!(rebuilt, bytes);
    Ok(())
}

#[test]
fn q8_0_slice_refuses_mid_block_boundary_1d() {
    let q8 = DTypeId::Q8_0.descriptor();
    let bytes = vec![0u8; 68];
    // [64] world 4: local extent 64/4 = 16, below the 32-element block size.
    let err = slice_bytes_for_rank(&bytes, &[64], q8, 0, 0, 4).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("q8_0"), "{msg}");
    assert!(msg.contains("axis 0"), "{msg}");
    assert!(msg.contains("local extent 16"), "{msg}");
    assert!(msg.contains("block size 32"), "{msg}");
    assert!(msg.contains("shard boundary would fall mid-block"), "{msg}");
}

#[test]
fn q8_0_slice_refuses_mid_block_boundary_2d() {
    let q8 = DTypeId::Q8_0.descriptor();
    let bytes = vec![0u8; 408];
    // [4, 96] world 4 on axis 1: local extent 96/4 = 24, not a multiple of 32.
    let err = slice_bytes_for_rank(&bytes, &[4, 96], q8, 1, 0, 4).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("q8_0"), "{msg}");
    assert!(msg.contains("axis 1"), "{msg}");
    assert!(msg.contains("local extent 24"), "{msg}");
    assert!(msg.contains("block size 32"), "{msg}");
    assert!(msg.contains("shard boundary would fall mid-block"), "{msg}");
}

#[test]
fn q8_0_slice_checks_local_extent_not_the_flattened_span() {
    // Behavior matrix: the contract checks the local extent along the shard
    // axis, not the flattened row-major span. Here the local extent is 48 (not
    // a multiple of 32) while the flattened span 48*64 = 3072 would be a whole
    // number of blocks, so the preferred local-extent rule refuses it.
    let q8 = DTypeId::Q8_0.descriptor();
    let bytes = vec![0u8; 6528]; // [96, 64] = 6144 elements = 192 blocks.
    let err = slice_bytes_for_rank(&bytes, &[96, 64], q8, 0, 0, 2).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("local extent 48"), "{msg}");
    assert!(msg.contains("block size 32"), "{msg}");
    assert!(msg.contains("shard boundary would fall mid-block"), "{msg}");
}

#[test]
fn q8_0_manifest_round_trip_preserves_encoding_and_resolves_descriptor() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("manifest.json");

    let mut manifest = GlobalCheckpointManifest::new(2);
    manifest.add_tensor("quant.weight", vec![64], DTypeId::Q8_0, "Sharded:0");
    save_checkpoint_manifest(&manifest, &path)?;

    let wire = std::fs::read_to_string(&path)?;
    assert!(wire.contains("\"q8_0\""), "{wire}");
    assert!(wire.contains("logical_elements_per_block"), "{wire}");
    assert!(wire.contains("bytes_per_block"), "{wire}");

    let loaded = load_checkpoint_manifest(&path)?;
    assert_eq!(manifest, loaded);

    let dtype = &loaded.tensors["quant.weight"].dtype;
    assert_eq!(dtype.key, DTypeKey::new("incin", "q8_0", 1));
    assert_eq!(dtype.kind, DTypeKind::Quantized);
    assert_eq!(dtype.encoding, StorageEncoding::block(32, 34, 2));

    // The persisted encoding resolves back to the registered descriptor.
    let descriptor = dtype.descriptor()?;
    assert_eq!(descriptor, DTypeId::Q8_0.descriptor());
    Ok(())
}

#[test]
fn manifest_refuses_q8_0_dtype_key_version_2_on_load() -> Result<()> {
    let dir = tempdir()?;
    let path = dir.path().join("manifest.json");

    let mut manifest = GlobalCheckpointManifest::new(2);
    manifest.add_tensor("quant.weight", vec![64], DTypeId::Q8_0, "Sharded:0");

    // Rewrite the serialized key tuple ("incin", "q8_0", 1) to version 2. The
    // refusal is DTypeKey's own: a key this build does not know must not be
    // coerced to the version it does know.
    let mut value =
        serde_json::to_value(&manifest).map_err(|e| Error::Msg(format!("to_value: {e}")))?;
    assert_eq!(
        value["tensors"]["quant.weight"]["dtype"]["key"][2],
        serde_json::json!(1),
        "serialized DTypeKey should carry version 1 before tampering"
    );
    value["tensors"]["quant.weight"]["dtype"]["key"][2] = serde_json::json!(2);
    let json =
        serde_json::to_string_pretty(&value).map_err(|e| Error::Msg(format!("to_string: {e}")))?;
    std::fs::write(&path, json)?;

    let err = load_checkpoint_manifest(&path).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("(incin, q8_0, 2)"), "{msg}");
    assert!(msg.contains("registered first"), "{msg}");
    Ok(())
}

#[test]
fn checkpoint_dtype_descriptor_refuses_tampered_encoding_and_version() {
    let bad_encoding = CheckpointDType {
        key: DTypeKey::new("incin", "q8_0", 1),
        kind: DTypeKind::Quantized,
        encoding: StorageEncoding::block(32, 99, 2),
    };
    let err = bad_encoding.descriptor().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("does not match registered dtype q8_0"),
        "{msg}"
    );

    let bad_version = CheckpointDType {
        key: DTypeKey::new("incin", "q8_0", 2),
        kind: DTypeKind::Quantized,
        encoding: StorageEncoding::block(32, 34, 2),
    };
    let err = bad_version.descriptor().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("does not match registered dtype q8_0"),
        "{msg}"
    );
}
