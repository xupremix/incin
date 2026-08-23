//! Integration coverage for `test_global_checkpoint_manifest_serialization` on the documented public surface.
#![cfg(feature = "std")]

use incin_backends::cpu::CpuBackendImpl;
use incin_core::nn::Linear;
use incin_core::nn::save::{
    GlobalCheckpointManifest, load_checkpoint_manifest, load_resharded_checkpoint, save_checkpoint,
    save_checkpoint_manifest, slice_bytes_for_rank,
};
use incin_core::prelude::*;
use tempfile::tempdir;

type TestBackend = CpuBackendImpl;

#[test]
fn test_global_checkpoint_manifest_serialization() -> Result<()> {
    let dir = tempdir()?;
    let manifest_path = dir.path().join("manifest.json");

    let mut manifest = GlobalCheckpointManifest::new(2);
    manifest.add_tensor("linear.weight", vec![1024, 2048], DTypeId::F32, "Sharded:0");
    manifest.add_tensor("linear.bias", vec![1024], DTypeId::F32, "Replicated");

    save_checkpoint_manifest(&manifest, &manifest_path)?;

    let wire = std::fs::read_to_string(&manifest_path)?;
    assert!(wire.contains("\"key\""));
    assert!(wire.contains("\"kind\""));
    assert!(wire.contains("\"encoding\""));

    let loaded = load_checkpoint_manifest(&manifest_path)?;
    assert_eq!(manifest, loaded);
    Ok(())
}

#[test]
fn test_slice_bytes_for_rank_axis_0() -> Result<()> {
    // 4 rows, 8 cols of F32
    let mut data: Vec<f32> = Vec::new();
    for i in 0..32 {
        data.push(i as f32);
    }
    let bytes: Vec<u8> = data.iter().flat_map(|val| val.to_ne_bytes()).collect();

    let global_shape = vec![4, 8];

    // Slice along axis 0 for rank 0 of 2
    let (rank0_bytes, rank0_shape) =
        slice_bytes_for_rank(&bytes, &global_shape, DTypeId::F32.descriptor(), 0, 0, 2)?;
    assert_eq!(rank0_shape, vec![2, 8]);
    let rank0_f32s: Vec<f32> = rank0_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(rank0_f32s, data[0..16]);

    // Slice along axis 0 for rank 1 of 2
    let (rank1_bytes, rank1_shape) =
        slice_bytes_for_rank(&bytes, &global_shape, DTypeId::F32.descriptor(), 0, 1, 2)?;
    assert_eq!(rank1_shape, vec![2, 8]);
    let rank1_f32s: Vec<f32> = rank1_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();
    assert_eq!(rank1_f32s, data[16..32]);

    Ok(())
}

#[test]
fn test_slice_bytes_for_rank_axis_1() -> Result<()> {
    // 4 rows, 8 cols of F32
    let mut data: Vec<f32> = Vec::new();
    for i in 0..32 {
        data.push(i as f32);
    }
    let bytes: Vec<u8> = data.iter().flat_map(|val| val.to_ne_bytes()).collect();

    let global_shape = vec![4, 8];

    // Slice along axis 1 (cols) for rank 0 of 2 (first 4 cols of each row)
    let (rank0_bytes, rank0_shape) =
        slice_bytes_for_rank(&bytes, &global_shape, DTypeId::F32.descriptor(), 1, 0, 2)?;
    assert_eq!(rank0_shape, vec![4, 4]);
    let rank0_f32s: Vec<f32> = rank0_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();

    let expected_rank0: Vec<f32> = vec![
        0.0, 1.0, 2.0, 3.0, 8.0, 9.0, 10.0, 11.0, 16.0, 17.0, 18.0, 19.0, 24.0, 25.0, 26.0, 27.0,
    ];
    assert_eq!(rank0_f32s, expected_rank0);

    // Slice along axis 1 (cols) for rank 1 of 2 (last 4 cols of each row)
    let (rank1_bytes, rank1_shape) =
        slice_bytes_for_rank(&bytes, &global_shape, DTypeId::F32.descriptor(), 1, 1, 2)?;
    assert_eq!(rank1_shape, vec![4, 4]);
    let rank1_f32s: Vec<f32> = rank1_bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().unwrap()))
        .collect();

    let expected_rank1: Vec<f32> = vec![
        4.0, 5.0, 6.0, 7.0, 12.0, 13.0, 14.0, 15.0, 20.0, 21.0, 22.0, 23.0, 28.0, 29.0, 30.0, 31.0,
    ];
    assert_eq!(rank1_f32s, expected_rank1);

    Ok(())
}

#[test]
fn test_save_and_load_resharded_checkpoint() -> Result<()> {
    let dir = tempdir()?;

    // Create a global Linear module with 8 input features, 16 output features
    let global_linear: Linear<Dyn, TestBackend> = Linear::build((8, 16))?;

    // Save global checkpoint
    save_checkpoint::<TestBackend, _, _>(&global_linear, dir.path(), 2)?;

    // Verify manifest exists and has expected fields
    let manifest = load_checkpoint_manifest(dir.path().join("manifest.json"))?;
    assert_eq!(manifest.world_size, 2);
    assert!(manifest.tensors.contains_key("weight"));
    assert!(manifest.tensors.contains_key("bias"));

    // Create target local module representing rank 0 of a TP=2 mesh (8 output features instead of 16)
    let mut rank0_linear: Linear<Dyn, TestBackend> = Linear::build((8, 8))?;
    load_resharded_checkpoint::<TestBackend, _, _>(&mut rank0_linear, dir.path(), 0, 2)?;

    // Create target local module representing rank 1 of a TP=2 mesh
    let mut rank1_linear: Linear<Dyn, TestBackend> = Linear::build((8, 8))?;
    load_resharded_checkpoint::<TestBackend, _, _>(&mut rank1_linear, dir.path(), 1, 2)?;

    Ok(())
}

#[test]
fn test_reshard_error_rejections() -> Result<()> {
    let dir = tempdir()?;
    let global_linear: Linear<Dyn, TestBackend> = Linear::build((8, 16))?;
    save_checkpoint::<TestBackend, _, _>(&global_linear, dir.path(), 2)?;

    let mut target_linear: Linear<Dyn, TestBackend> = Linear::build((8, 8))?;

    // Out of bounds rank
    let res = load_resharded_checkpoint::<TestBackend, _, _>(&mut target_linear, dir.path(), 2, 2);
    assert!(res.is_err());

    // Incompatible dimension (e.g. trying to load 16 output features into 5 output features)
    let mut target_incompatible: Linear<Dyn, TestBackend> = Linear::build((8, 5))?;
    let res_incompatible =
        load_resharded_checkpoint::<TestBackend, _, _>(&mut target_incompatible, dir.path(), 0, 2);
    assert!(res_incompatible.is_err());

    Ok(())
}
