use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Core abstraction for `MockGpu` within the Kindle framework.
pub struct MockGpu;
impl ConstDevice for MockGpu {}
impl Device for MockGpu {
    /// Core abstraction for `Arg` within the Kindle framework.
    type Arg = ();
    /// Core abstraction for `Field` within the Kindle framework.
    type Field = core::marker::PhantomData<Self>;
    /// Core abstraction for `to_kindle` within the Kindle framework.
    fn to_kindle(_: &Self::Field) -> Result<KindleDevice> { Ok(KindleDevice::cpu()) }
    /// Core abstraction for `init` within the Kindle framework.
    fn init(_: Self::Arg) -> Self::Field { core::marker::PhantomData }
}
impl DynDevice for MockGpu {}

fn main() {
    let a: Tensor<Dyn, DummyBackend<f32, Cpu>, f32, Cpu, Grad> = Tensor::zeros(vec![2, 2]).unwrap();
    let b: Tensor<Dyn, DummyBackend<f32, MockGpu>, f32, MockGpu, Grad> = Tensor::zeros(vec![2, 2]).unwrap();
    
    // This should fail to compile because Cpu != MockGpu
    let _c = a.add(&b);
}
