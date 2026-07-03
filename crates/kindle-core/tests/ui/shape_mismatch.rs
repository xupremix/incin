use kindle_core::prelude::*;

pub struct DummyBackend;
impl<S: Shape> Backend<S> for DummyBackend {
    type RawTensor = ();
    fn zeros(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
    fn ones(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
    fn rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
    fn randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
    fn relu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn abs(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn add(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn sub(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn mul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn div(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn matmul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn reshape(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> { Ok(()) }
}

fn main() {
    let t1: Tensor<[usize; 2], DummyBackend> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<[usize; 3], DummyBackend> = Tensor::zeros([2, 3, 4]).unwrap();

    // Mismatched shapes should not compile
    t1.add(&t2).unwrap();
}
