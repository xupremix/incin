//! Integration coverage for `ConsumerModule` on the documented public surface.
use incin::prelude::*;

#[module]
pub struct ConsumerModule {
    layer: Linear<s![3, 4]>,
}
pub fn tensor_and_module_surface() -> Result<()> {
    let tensor = Tensor::<Dyn>::zeros(vec![2, 3])?;
    let _module = ConsumerModule {
        layer: Linear::build(())?,
    };
    assert_eq!(tensor.dims(), vec![2, 3]);
    Ok(())
}
