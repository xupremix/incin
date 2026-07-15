use kindle::prelude::*;
use kindle::{ConstShape, DynShape, Shape};

type CpuBackend = DefaultBackend;

#[test]
fn test_s_macro() {
    // Static dimensions
    type StaticShape = s![10, 20];
    assert_eq!(<StaticShape as ConstShape>::DIMS, [10, 20]);

    // Dynamic dimensions
    type DynShapeType = s![dyn, dyn];
    let arg = (10, 20); // Arg depends on the dynamic fields
    let field = <DynShapeType as Shape>::init(arg);
    assert_eq!(<DynShapeType as DynShape>::dims(&field), [10, 20]);

    // Mixed dimensions
    type MixedShape = s![2, dyn, 5, dyn];
    let mixed_arg = ((), 3, (), 7);
    let mixed_field = <MixedShape as Shape>::init(mixed_arg);
    assert_eq!(<MixedShape as DynShape>::dims(&mixed_field), [2, 3, 5, 7]);
}

#[test]
fn test_idx_macro() {
    // Basic indexing
    type Ranges = idx![0..5, 2..10, 0, ..];
    // Compiling is enough to verify type parsing
}

#[module]
pub struct MyCustomLayer<B: Backend> {
    pub linear: Linear<s![10, 5], B>,
    pub ln: LayerNorm<s![5], B>,
}

#[test]
fn test_module_macro() -> Result<()> {
    // Verify that #[module] derived Parameters and StateDict automatically
    let layer = MyCustomLayer::<CpuBackend> {
        linear: Linear::new()?,
        ln: LayerNorm::new(1e-5)?,
    };

    // Since #[module] implements Parameters, this should compile:
    let params = layer.parameters();
    assert_eq!(params.len(), 4); // linear.weight, linear.bias, ln.weight, ln.bias

    Ok(())
}
