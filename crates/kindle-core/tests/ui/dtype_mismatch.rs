use kindle_core::prelude::*;

pub struct DummyBackend;
impl Backend for DummyBackend {
    type Device = Cpu;
    type DType = f32;
    type BackendWithDType<NewT: kindle_core::prelude::DType> = DummyBackend;
    type BackendWithDevice<NewD: kindle_core::prelude::Device> = DummyBackend;

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
    fn max_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn min_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn sum_dim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn sum_keepdim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn mean_dim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn mean_keepdim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn max_dim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn max_keepdim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn min_dim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn min_keepdim(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn to_dtype(_t: &Self::RawTensor, _dtype: KindleDType) -> Result<Self::RawTensor> { Ok(()) }
    fn broadcast_as(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> { Ok(()) }
    fn broadcast_left(_t: &Self::RawTensor, _shape: &[usize]) -> Result<Self::RawTensor> { Ok(()) }
    fn transpose(_t: &Self::RawTensor, _dim1: usize, _dim2: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn flatten(_t: &Self::RawTensor, _start: usize, _end: usize) -> Result<Self::RawTensor> { Ok(()) }
}

fn main() {
    let t1: Tensor<Dyn, DummyBackend, f32> = Tensor::zeros(alloc::vec![2]).unwrap();
    let t2: Tensor<Dyn, DummyBackend, f64> = Tensor::zeros(alloc::vec![2]).unwrap();

    // Mismatched dtypes should not compile
    t1.add(&t2).unwrap();
}
