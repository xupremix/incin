use kindle::StateDict;
use kindle::prelude::*;
extern crate alloc;
use alloc::collections::BTreeMap;

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = kindle_backends::cpu::CpuBackendImpl;

#[test]
/// Test state dict extraction.
fn test_state_dict_extraction() -> Result<()> {
    let layer = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    let mut map = BTreeMap::new();

    // Extract state
    layer.state_dict("linear.", &mut map);

    println!("Map keys: {:?}", map.keys().collect::<Vec<_>>());

    // Linear has weight and bias (bias is optional, but new() creates it by default)
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("linear.weight."));
    assert!(map.contains_key("linear.bias."));

    // Load state
    let mut new_layer = Linear::<s![10, 5], CpuBackendImpl>::build(())?;
    new_layer.load_state_dict("linear.", &map)?;

    // Test parameters
    let params = layer.parameters();
    assert_eq!(params.len(), 2);

    Ok(())
}

#[test]
/// `state_dict`'s prefix convention differs from `named_parameters`'s (the
/// caller must already include a trailing `.`, unlike `named_parameters`
/// where the `#[module]`-generated body appends it), so `Sequential`'s flat
/// numbering needs its own, separately-verified test even though it shares
/// the same `flat_width` mechanism.
fn test_sequential_state_dict_flat_keys_and_round_trip() -> Result<()> {
    let seq = seq!(
        Linear::<s![10, 5], CpuBackendImpl>::build(())?,
        ReLU,
        Linear::<s![5, 2], CpuBackendImpl>::build(())?
    );
    let mut map = BTreeMap::new();
    seq.state_dict("", &mut map);

    let mut keys: Vec<&String> = map.keys().collect();
    keys.sort();
    assert_eq!(
        keys,
        vec!["0.bias.", "0.weight.", "2.bias.", "2.weight."],
        "expected flat PyTorch-style numbering, got nested keys instead"
    );

    // Round trip: a fresh Sequential of the same shape loads the saved
    // state without error.
    let mut new_seq = seq!(
        Linear::<s![10, 5], CpuBackendImpl>::build(())?,
        ReLU,
        Linear::<s![5, 2], CpuBackendImpl>::build(())?
    );
    new_seq.load_state_dict("", &map)?;

    Ok(())
}
