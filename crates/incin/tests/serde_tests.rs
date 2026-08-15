#![cfg(feature = "cpu")]

use incin::prelude::*;
use incin::state::{collect_state, load_state};
use incin::{StateVisitor, VariableBackend, VisitState};
extern crate alloc;

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = incin_backends::cpu::CpuBackendImpl;

#[module]
struct MixedStateModel {
    fp32: Linear<s![2, 2], CpuBackendImpl>,
    fp16: incin_core::nn::Linear<s![2, 2], CpuBackendImpl, incin_core::nn::optional::True, f16>,
}

#[module]
struct ExplicitStateNames {
    #[state(name = "q_proj")]
    internal_query_projection: Linear<s![2, 2], CpuBackendImpl>,
}

#[test]
/// Test state dict extraction.
fn test_state_dict_extraction() -> Result<()> {
    let layer = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    let map = collect_state::<CpuBackendImpl, _>(&layer)?;

    // Linear has weight and bias (bias is optional, but new() creates it by default)
    assert_eq!(map.len(), 2);
    assert!(map.iter().any(|(path, _)| path.as_str() == "weight"));
    assert!(map.iter().any(|(path, _)| path.as_str() == "bias"));

    // Load state
    let mut new_layer = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    load_state::<CpuBackendImpl, _>(&mut new_layer, &map)?;

    // Test optimizer-owned parameter collection
    let params = ParameterGroup::<CpuBackendImpl, f32>::from_module(&layer)?;
    assert_eq!(params.len(), 2);

    Ok(())
}

#[test]
fn test_owned_heterogeneous_snapshot_extraction() -> Result<()> {
    let layer = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    let snapshot = collect_state::<CpuBackendImpl, _>(&layer)?;
    assert_eq!(snapshot.len(), 2);
    assert!(snapshot.iter().all(|(path, value)| {
        (path.as_str().ends_with("weight") || path.as_str().ends_with("bias"))
            && !value.bytes().is_empty()
    }));
    let mut restored = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    load_state::<CpuBackendImpl, _>(&mut restored, &snapshot)?;
    Ok(())
}

#[test]
fn test_postcard_snapshot_round_trip() -> Result<()> {
    let layer = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    let path = std::env::temp_dir().join(format!("incin-state-{}.postcard", std::process::id()));
    layer.save(Format::Postcard, &path)?;
    let mut restored = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    restored.load(Format::Postcard, &path, &DeviceId::cpu())?;
    std::fs::remove_file(path).ok();
    Ok(())
}

#[test]
fn test_mixed_dtype_snapshot_round_trip() -> Result<()> {
    let model = MixedStateModel {
        fp32: Linear::build(())?,
        fp16: incin_core::nn::Linear::<
            s![2, 2],
            CpuBackendImpl,
            incin_core::nn::optional::True,
            f16,
        >::build(())?,
    };
    let snapshot = collect_state::<CpuBackendImpl, _>(&model)?;
    assert_eq!(snapshot.len(), 4);
    assert!(
        snapshot
            .iter()
            .any(|(path, value)| path.as_str() == "fp32.weight"
                && value.dtype() == DTypeId::F32.descriptor())
    );
    assert!(
        snapshot
            .iter()
            .any(|(path, value)| path.as_str() == "fp16.weight"
                && value.dtype() == DTypeId::F16.descriptor())
    );

    let path = std::env::temp_dir().join(format!("incin-mixed-{}.safetensors", std::process::id()));
    model.save(Format::Safetensors, &path)?;
    let loaded = incin_core::prelude::load_safetensors_snapshot(&path)?;
    assert_eq!(loaded, snapshot);
    std::fs::remove_file(path).ok();
    Ok(())
}

#[test]
fn test_mixed_dtype_checkpoint_round_trip() -> Result<()> {
    let model = MixedStateModel {
        fp32: Linear::build(())?,
        fp16: incin_core::nn::Linear::<
            s![2, 2],
            CpuBackendImpl,
            incin_core::nn::optional::True,
            f16,
        >::build(())?,
    };
    let expected = collect_state::<CpuBackendImpl, _>(&model)?;
    let dir = std::env::temp_dir().join(format!("incin-mixed-checkpoint-{}", std::process::id()));
    std::fs::create_dir_all(&dir)?;
    incin_core::nn::save::save_checkpoint::<CpuBackendImpl, _, _>(&model, &dir, 1)?;

    let manifest = incin_core::nn::save::load_checkpoint_manifest(dir.join("manifest.json"))?;
    assert_eq!(manifest.tensors["fp16.weight"].dtype.key.name(), "f16");
    assert_eq!(manifest.tensors["fp32.weight"].dtype.key.name(), "f32");

    let mut restored = MixedStateModel {
        fp32: Linear::build(())?,
        fp16: incin_core::nn::Linear::<
            s![2, 2],
            CpuBackendImpl,
            incin_core::nn::optional::True,
            f16,
        >::build(())?,
    };
    incin_core::nn::save::load_resharded_checkpoint::<CpuBackendImpl, _, _>(
        &mut restored,
        &dir,
        0,
        1,
    )?;
    assert_eq!(collect_state::<CpuBackendImpl, _>(&restored)?, expected);
    std::fs::remove_dir_all(dir).ok();
    Ok(())
}

#[test]
fn test_explicit_state_name_is_schema_stable() -> Result<()> {
    let model = ExplicitStateNames {
        internal_query_projection: Linear::build(())?,
    };
    let snapshot = collect_state::<CpuBackendImpl, _>(&model)?;
    assert!(
        snapshot
            .iter()
            .any(|(path, _)| path.as_str() == "q_proj.weight")
    );
    assert!(
        snapshot
            .iter()
            .any(|(path, _)| path.as_str() == "q_proj.bias")
    );
    Ok(())
}

#[test]
fn test_strict_load_rejects_bad_snapshots_without_mutation() -> Result<()> {
    let mut model = Linear::<s![2, 2], CpuBackendImpl>::build(())?;
    let original = collect_state::<CpuBackendImpl, _>(&model)?;

    let mut missing = StateSnapshot::new();
    for (path, value) in original.iter().filter(|(path, _)| path.as_str() != "bias") {
        missing.insert(path.clone(), value.clone())?;
    }
    assert!(load_state::<CpuBackendImpl, _>(&mut model, &missing).is_err());
    assert_eq!(collect_state::<CpuBackendImpl, _>(&model)?, original);

    let mut unexpected = original.clone();
    unexpected.insert(
        StatePath::new("unexpected")?,
        StateValue::new(
            incin_core::prelude::ShapeBuf::from_slice(&[1]),
            DTypeId::U8.descriptor(),
            vec![0],
            StateRole::Buffer,
        )?,
    )?;
    assert!(load_state::<CpuBackendImpl, _>(&mut model, &unexpected).is_err());
    assert_eq!(collect_state::<CpuBackendImpl, _>(&model)?, original);

    let weight = original.get(&StatePath::new("weight")?).unwrap();
    let mut wrong_shape = StateSnapshot::new();
    for (path, value) in original.iter() {
        wrong_shape.insert(
            path.clone(),
            if path.as_str() == "weight" {
                StateValue::new(
                    incin_core::prelude::ShapeBuf::from_slice(&[1, 4]),
                    weight.dtype(),
                    weight.bytes().to_vec(),
                    weight.role(),
                )?
            } else {
                value.clone()
            },
        )?;
    }
    assert!(load_state::<CpuBackendImpl, _>(&mut model, &wrong_shape).is_err());
    assert_eq!(collect_state::<CpuBackendImpl, _>(&model)?, original);

    let mut wrong_dtype = StateSnapshot::new();
    for (path, value) in original.iter() {
        wrong_dtype.insert(
            path.clone(),
            if path.as_str() == "weight" {
                StateValue::new(
                    weight.shape().clone(),
                    DTypeId::F16.descriptor(),
                    vec![0; weight.shape().numel().unwrap() * 2],
                    weight.role(),
                )?
            } else {
                value.clone()
            },
        )?;
    }
    assert!(load_state::<CpuBackendImpl, _>(&mut model, &wrong_dtype).is_err());
    assert_eq!(collect_state::<CpuBackendImpl, _>(&model)?, original);
    Ok(())
}

#[test]
fn test_late_state_failure_does_not_commit_earlier_leaves() -> Result<()> {
    let mut model = MixedStateModel {
        fp32: Linear::build(())?,
        fp16: incin_core::nn::Linear::<
            s![2, 2],
            CpuBackendImpl,
            incin_core::nn::optional::True,
            f16,
        >::build(())?,
    };
    let original = collect_state::<CpuBackendImpl, _>(&model)?;
    let fp16_weight = original
        .get(&StatePath::new("fp16.weight")?)
        .expect("mixed model has an fp16 weight");
    let mut invalid = StateSnapshot::new();
    for (path, value) in original.iter() {
        invalid.insert(
            path.clone(),
            if path.as_str() == "fp16.weight" {
                StateValue::new(
                    fp16_weight.shape().clone(),
                    DTypeId::F32.descriptor(),
                    vec![0; 16],
                    fp16_weight.role(),
                )?
            } else {
                value.clone()
            },
        )?;
    }

    assert!(load_state::<CpuBackendImpl, _>(&mut model, &invalid).is_err());
    assert_eq!(collect_state::<CpuBackendImpl, _>(&model)?, original);
    Ok(())
}

struct FailingState;

impl<B: VariableBackend> VisitState<B> for FailingState {
    fn visit_state<V: StateVisitor<B>>(&self, _: &StatePath, _: &mut V) -> Result<()> {
        Err(Error::Msg("intentional state readback failure".into()))
    }
}

#[test]
fn test_state_extraction_failure_propagates() {
    assert!(collect_state::<CpuBackendImpl, _>(&FailingState).is_err());
}

#[test]
/// Sequential state paths use flat positional numbering independent of the
/// right-nested Rust representation used by `seq!`.
fn test_sequential_state_dict_flat_keys_and_round_trip() -> Result<()> {
    let seq = seq!(
        Linear::<s![10, 5], CpuBackendImpl>::build(())?,
        ReLU,
        Linear::<s![5, 2], CpuBackendImpl>::build(())?
    );
    let map = collect_state::<CpuBackendImpl, _>(&seq)?;

    let mut keys: Vec<&str> = map.iter().map(|(path, _)| path.as_str()).collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        vec!["0.bias", "0.weight", "2.bias", "2.weight"],
        "expected stable sequential state paths"
    );

    // Round trip: a fresh Sequential of the same shape loads the saved
    // state without error.
    let mut new_seq = seq!(
        Linear::<s![10, 5], CpuBackendImpl>::build(())?,
        ReLU,
        Linear::<s![5, 2], CpuBackendImpl>::build(())?
    );
    load_state::<CpuBackendImpl, _>(&mut new_seq, &map)?;

    Ok(())
}
