//! Integration coverage for issue #13: tensor-by-tensor Safetensors
//! placement through the public `ModelExt::load` entry - module atomicity on
//! mid-staging failure, structured deterministic error reporting, and the
//! WGPU binding.
#![cfg(feature = "std")]

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;
use safetensors::tensor::{Dtype, TensorView, serialize_to_file};
use tempfile::tempdir;

type TestBackend = CpuBackendImpl;

type TestModel = SeqTy!(Linear<Dyn, TestBackend>, ReLU, Linear<Dyn, TestBackend>);

/// A two-layer model whose state paths visit `0.weight`, `0.bias`,
/// `2.weight`, `2.bias` in sorted order - so splitting the path list in half
/// puts both `0.*` leaves in the first shard and both `2.*` leaves in the
/// second, and a failure reading the second shard lands after the first
/// shard's leaves have already staged.
fn test_model() -> Result<TestModel> {
    Ok(seq![
        Linear::<Dyn, TestBackend>::build((4, 8))?,
        ReLU,
        Linear::<Dyn, TestBackend>::build((8, 2))?
    ])
}

fn write_shard(path: &Path, entries: &BTreeMap<String, StateValue>) {
    let mut views = BTreeMap::new();
    let mut metadata = HashMap::new();
    for (name, value) in entries {
        metadata.insert(
            format!("incin.state.role.{name}"),
            match value.role() {
                StateRole::Parameter => "parameter".to_string(),
                StateRole::Buffer => "buffer".to_string(),
            },
        );
        let dtype = match value.dtype().builtin_id() {
            Some(DTypeId::F32) => Dtype::F32,
            other => panic!("fixture tensors are f32, found {other:?}"),
        };
        views.insert(
            name.clone(),
            TensorView::new(dtype, value.shape().dims().to_vec(), value.bytes())
                .expect("fixture bytes match their declared shape"),
        );
    }
    serialize_to_file(&views, Some(metadata), path).expect("shard file writes");
}

fn index_json(total: u64, weight_map: &BTreeMap<String, String>) -> String {
    let map = weight_map
        .iter()
        .map(|(name, shard)| format!("\"{name}\":\"{shard}\""))
        .collect::<Vec<_>>()
        .join(",");
    format!("{{\"metadata\":{{\"total_size\":{total}}},\"weight_map\":{{{map}}}}}")
}

fn write_checkpoint(
    dir: &Path,
    first: &BTreeMap<String, StateValue>,
    second: &BTreeMap<String, StateValue>,
    weight_map: &BTreeMap<String, String>,
) -> PathBuf {
    let first_path = dir.join("model-00001-of-00002.safetensors");
    let second_path = dir.join("model-00002-of-00002.safetensors");
    write_shard(&first_path, first);
    write_shard(&second_path, second);
    let total = std::fs::metadata(&first_path)
        .expect("first shard stat")
        .len()
        + std::fs::metadata(&second_path)
            .expect("second shard stat")
            .len();
    let index_path = dir.join("model.safetensors.index.json");
    std::fs::write(&index_path, index_json(total, weight_map)).expect("index is written");
    index_path
}

fn split_snapshot(
    snapshot: &StateSnapshot,
) -> (
    BTreeMap<String, StateValue>,
    BTreeMap<String, StateValue>,
    BTreeMap<String, String>,
) {
    let midpoint = snapshot.len() / 2;
    let mut first = BTreeMap::new();
    let mut second = BTreeMap::new();
    let mut weight_map = BTreeMap::new();
    for (index, (path, value)) in snapshot.iter().enumerate() {
        let name = path.as_str().to_string();
        if index < midpoint {
            first.insert(name.clone(), value.clone());
            weight_map.insert(name, "model-00001-of-00002.safetensors".to_string());
        } else {
            second.insert(name.clone(), value.clone());
            weight_map.insert(name, "model-00002-of-00002.safetensors".to_string());
        }
    }
    assert!(
        !first.is_empty() && !second.is_empty(),
        "both shards carry tensors"
    );
    (first, second, weight_map)
}

/// A sharded checkpoint round-trips through the public entry: `ModelExt::load`
/// streams every tensor and the restored module's state equals the source
/// snapshot exactly (issue #13 acceptance 1, end to end).
#[test]
fn a_sharded_safetensors_checkpoint_round_trips_through_model_ext_load() -> Result<()> {
    let dir = tempdir()?;
    let source = test_model()?;
    let expected = collect_state::<TestBackend, _>(&source)?;
    let (first, second, weight_map) = split_snapshot(&expected);
    let index_path = write_checkpoint(dir.path(), &first, &second, &weight_map);

    let mut restored = test_model()?;
    restored.load(Format::Safetensors, &index_path)?;
    let actual = collect_state::<TestBackend, _>(&restored)?;
    assert_eq!(
        actual, expected,
        "streamed placement reproduces the checkpoint exactly"
    );
    Ok(())
}

/// A wrong-shape tensor in the second shard - same byte length, so the file
/// stays internally consistent and opens - fails placement after the first
/// shard's leaves have already staged. The error names the path, and every
/// staged leaf is cleared so the module is byte-for-byte what it was before
/// the load (issue #13 acceptance 2).
#[test]
fn a_shape_mismatch_mid_staging_rolls_back_and_leaves_the_module_unchanged() -> Result<()> {
    let dir = tempdir()?;
    let source = test_model()?;
    let expected = collect_state::<TestBackend, _>(&source)?;
    let (first, mut second, weight_map) = split_snapshot(&expected);

    let wrong = second
        .get("2.weight")
        .expect("the split puts 2.weight in the second shard")
        .clone();
    let flat: usize = wrong.shape().dims().iter().product();
    let replacement = StateValue::new(
        ShapeBuf::from_slice(&[flat, 1]),
        wrong.dtype(),
        vec![0u8; wrong.bytes().len()],
        wrong.role(),
    )?;
    second.insert("2.weight".to_string(), replacement);
    let index_path = write_checkpoint(dir.path(), &first, &second, &weight_map);

    let mut target = test_model()?;
    let before = collect_state::<TestBackend, _>(&target)?;
    let error = target
        .load(Format::Safetensors, &index_path)
        .expect_err("placement rejects the mismatched shape");
    let message = error.to_string();
    assert!(
        message.contains("shape or dtype mismatch"),
        "the structured error reports the mismatch, got: {message}"
    );
    assert!(
        message.contains("2.weight"),
        "the error names the offending path, got: {message}"
    );
    let after = collect_state::<TestBackend, _>(&target)?;
    assert_eq!(
        after, before,
        "a mid-staging failure leaves the module exactly as it was"
    );
    Ok(())
}

/// An index that maps a tensor to a shard whose header lacks it fails the
/// open with the historical ghost message before any tensor is staged, the
/// module stays untouched, and two consecutive attempts report byte-identical
/// errors (issue #13 acceptance 2 and deterministic reporting).
#[test]
fn an_index_pointing_at_a_missing_tensor_fails_deterministically_without_touching_the_module()
-> Result<()> {
    let dir = tempdir()?;
    let source = test_model()?;
    let expected = collect_state::<TestBackend, _>(&source)?;
    let (first, mut second, weight_map) = split_snapshot(&expected);
    second
        .remove("2.bias")
        .expect("the split puts 2.bias in the second shard");
    let index_path = write_checkpoint(dir.path(), &first, &second, &weight_map);

    let mut target = test_model()?;
    let before = collect_state::<TestBackend, _>(&target)?;
    let mut messages = Vec::new();
    for _ in 0..2 {
        let error = target
            .load(Format::Safetensors, &index_path)
            .expect_err("the ghost tensor fails the load");
        messages.push(error.to_string());
    }
    assert_eq!(
        messages[0], messages[1],
        "repeated failures report byte-identical errors"
    );
    assert!(
        messages[0].contains(
            "the index maps tensor `2.bias` to shard `model-00002-of-00002.safetensors`, \
             which does not contain it"
        ),
        "the historical ghost message is preserved, got: {}",
        messages[0]
    );
    let after = collect_state::<TestBackend, _>(&target)?;
    assert_eq!(
        after, before,
        "a failed open leaves the module exactly as it was"
    );
    Ok(())
}

/// A shard truncated after it was written fails the open with an error that
/// names the shard file, before any tensor is staged, and the module stays
/// untouched (issue #13 acceptance 2, I/O side). The index's `total_size` is
/// re-derived from the filesystem after truncation so the failure lands on
/// the header-versus-file length check inside the shard, not on the earlier
/// index-level total.
#[test]
fn a_truncated_shard_fails_the_open_with_an_error_naming_the_shard() -> Result<()> {
    let dir = tempdir()?;
    let source = test_model()?;
    let expected = collect_state::<TestBackend, _>(&source)?;
    let (first, second, weight_map) = split_snapshot(&expected);
    let index_path = write_checkpoint(dir.path(), &first, &second, &weight_map);

    let shard = dir.path().join("model-00002-of-00002.safetensors");
    let length = std::fs::metadata(&shard).expect("shard stat").len();
    let file = std::fs::OpenOptions::new()
        .write(true)
        .open(&shard)
        .expect("shard opens for truncation");
    file.set_len(length - 8).expect("shard truncates");
    drop(file);
    let total = std::fs::metadata(dir.path().join("model-00001-of-00002.safetensors"))
        .expect("first shard stat")
        .len()
        + std::fs::metadata(&shard).expect("second shard stat").len();
    std::fs::write(&index_path, index_json(total, &weight_map)).expect("index is rewritten");

    let mut target = test_model()?;
    let before = collect_state::<TestBackend, _>(&target)?;
    let error = target
        .load(Format::Safetensors, &index_path)
        .expect_err("the truncated shard fails the load");
    let message = error.to_string();
    assert!(
        message.contains("model-00002-of-00002.safetensors"),
        "the error names the shard it could not read, got: {message}"
    );
    assert!(
        message.contains("declares"),
        "the error reports the declared-versus-actual length, got: {message}"
    );
    let after = collect_state::<TestBackend, _>(&target)?;
    assert_eq!(
        after, before,
        "a failed open leaves the module exactly as it was"
    );
    Ok(())
}

/// A checkpoint whose index and shards both lack a path the module expects
/// fails the exact path-set comparison before any tensor is staged, and the
/// module stays untouched (issue #13 acceptance 2, structural side).
#[test]
fn a_checkpoint_missing_a_path_the_module_expects_fails_the_path_comparison() -> Result<()> {
    let dir = tempdir()?;
    let source = test_model()?;
    let expected = collect_state::<TestBackend, _>(&source)?;
    let (first, mut second, mut weight_map) = split_snapshot(&expected);
    second
        .remove("2.bias")
        .expect("the split puts 2.bias in the second shard");
    weight_map
        .remove("2.bias")
        .expect("the split maps 2.bias in the second shard");
    let index_path = write_checkpoint(dir.path(), &first, &second, &weight_map);

    let mut target = test_model()?;
    let before = collect_state::<TestBackend, _>(&target)?;
    let error = target
        .load(Format::Safetensors, &index_path)
        .expect_err("the missing path fails the load");
    let message = error.to_string();
    assert!(
        message.contains("state paths differ"),
        "the exact path-mismatch report is preserved, got: {message}"
    );
    assert!(
        message.contains("2.bias"),
        "the report names the missing path, got: {message}"
    );
    let after = collect_state::<TestBackend, _>(&target)?;
    assert_eq!(
        after, before,
        "a failed path comparison leaves the module exactly as it was"
    );
    Ok(())
}

/// Compile-level proof (issue #13 acceptance 3): the streaming loader is
/// generic over the variable backend and binds for the WGPU backend through
/// the same `VisitStateMut` traversal and `B::from_bytes` device hook the
/// snapshot loader uses - no CPU-only code exists on the path. Coercing the
/// public `ModelExt::load` entry to a function pointer instantiates
/// `load_state_streaming::<WgpuBackendImpl, ..>` end to end without
/// constructing a device. It lives here rather than in `src/`'s unit tests
/// because unit tests inside the crate cannot link the `incin_backends`
/// dev-dependency - it cycles back into this crate's rlib and rustc rejects
/// the two instances of `incin_core`.
#[test]
fn the_streaming_load_path_binds_for_the_wgpu_backend_through_the_public_entry() {
    use incin_backends::wgpu::WgpuBackendImpl;

    type WgpuModel = SeqTy!(Linear<Dyn, WgpuBackendImpl>, ReLU, Linear<Dyn, WgpuBackendImpl>);

    fn assert_visitors<M>()
    where
        M: VisitState<WgpuBackendImpl> + VisitStateMut<WgpuBackendImpl>,
    {
    }
    assert_visitors::<WgpuModel>();

    let _bind: fn(&mut WgpuModel, Format, &Path) -> Result<()> =
        <WgpuModel as ModelExt<WgpuBackendImpl>>::load;
}
