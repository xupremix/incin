use kindle::StateDict;
use kindle::prelude::*;
extern crate alloc;
use alloc::collections::BTreeMap;

/// Implementation of `CpuBackend` for the respective backend.
type CpuBackend = kindle_backends::cpu::CpuBackend<f32, Cpu>;

#[test]
/// Test state dict extraction.
fn test_state_dict_extraction() -> Result<()> {
    let layer = Linear::<s![10, 5], CpuBackend>::new()?;
    let mut map = BTreeMap::new();

    // Extract state
    layer.state_dict("linear.", &mut map);

    println!("Map keys: {:?}", map.keys().collect::<Vec<_>>());

    // Linear has weight and bias (bias is optional, but new() creates it by default)
    assert_eq!(map.len(), 2);
    assert!(map.contains_key("linear.weight."));
    assert!(map.contains_key("linear.bias."));

    // Load state
    let mut new_layer = Linear::<s![10, 5], CpuBackend>::new()?;
    new_layer.load_state_dict("linear.", &map)?;

    // Test parameters
    let params = layer.parameters();
    assert_eq!(params.len(), 2);

    Ok(())
}
