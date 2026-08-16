use incin_core::prelude::*;
use incin_core::test_utils::DummyBackend;

#[derive(Debug, Default, Clone, PartialEq, Eq)]
/// Mock gpu.
pub struct MockGpu;
impl ConstDevice for MockGpu {}
impl Device for MockGpu {
    /// Arg.
    type Arg = ();
    /// Field.
    type Field = core::marker::PhantomData<Self>;
    /// To incin.
    fn to_incin(_: &Self::Field) -> Result<DeviceId> { Ok(DeviceId::cpu()) }
    /// Init.
    fn init(_: Self::Arg) -> Self::Field { core::marker::PhantomData }
}

fn main() {
    let a: Tensor<Dyn, DummyBackend<Cpu>, f32, Grad> = Tensor::zeros(vec![2, 2]).unwrap();
    let b: Tensor<Dyn, DummyBackend<MockGpu>, f32, Grad> = Tensor::zeros(vec![2, 2]).unwrap();
    
    // This should fail to compile because Cpu != MockGpu
    let _c = a.add_exact(&b);
}
