use kindle_core::prelude::*;
use kindle_core::prelude::dummy::DummyBackend;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Auto-generated documentation for MockGpu.
pub struct MockGpu;
impl ConstDevice for MockGpu {}
impl Device for MockGpu {
    /// Auto-generated documentation for Arg.
    type Arg = ();
    /// Auto-generated documentation for Field.
    type Field = core::marker::PhantomData<Self>;
    /// Auto-generated documentation for to_kindle.
    fn to_kindle(_: &Self::Field) -> Result<KindleDevice> { Ok(KindleDevice::cpu()) }
    /// Auto-generated documentation for init.
    fn init(_: Self::Arg) -> Self::Field { core::marker::PhantomData }
}
impl DynDevice for MockGpu {}

fn main() {
    let a: Tensor<Dyn, DummyBackend<f32, Cpu>, f32, Cpu, Grad> = Tensor::zeros(vec![2, 2]).unwrap();
    let b: Tensor<Dyn, DummyBackend<f32, MockGpu>, f32, MockGpu, Grad> = Tensor::zeros(vec![2, 2]).unwrap();
    
    // This should fail to compile because Cpu != MockGpu
    let _c = a.add(&b);
}
