use kindle_core::prelude::*;

pub struct DummyBackend;
impl Backend for DummyBackend {
    type Device = Cpu;
    type DType = f32;
    type BackendWithDType<NewT: kindle_core::prelude::DType> = DummyBackend;
    type BackendWithDevice<NewD: kindle_core::prelude::Device> = DummyBackend;

    type RawTensor = ();
    type RawVar = ();
    type Grads = ();
    
    fn var_as_tensor(_var: &Self::RawVar) -> Result<Self::RawTensor> { Ok(()) }
    fn var_zeros(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }
    fn var_ones(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }
    fn var_rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }
    
    fn tensor_to_device(_t: &Self::RawTensor, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
    fn var_to_device(_var: &Self::RawVar, _device: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }

    fn var_randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawVar> { Ok(()) }

    fn zeros(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
    fn ones(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
    fn rand(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
    fn randn(_shape: &[usize], _dtype: KindleDType, _device: &KindleDevice) -> Result<Self::RawTensor> { Ok(()) }
    
    fn neg(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn sqrt(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn exp(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn log(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn tanh(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn sigmoid(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    
    fn relu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn gelu(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn abs(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn add(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn sub(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn mul(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn div(_lhs: &Self::RawTensor, _rhs: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn mul_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> { Ok(()) }
    fn add_scalar(_t: &Self::RawTensor, _scalar: f64) -> Result<Self::RawTensor> { Ok(()) }
    fn sum_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn mean_all(_t: &Self::RawTensor) -> Result<Self::RawTensor> { Ok(()) }
    fn narrow(_t: &Self::RawTensor, _dim: usize, _start: usize, _len: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn squeeze(_t: &Self::RawTensor, _dim: usize) -> Result<Self::RawTensor> { Ok(()) }
    fn conv2d(_t: &Self::RawTensor, _w: &Self::RawTensor, _b: Option<&Self::RawTensor>, _s: usize, _p: usize, _d: usize) -> Result<Self::RawTensor> { Ok(()) }
    
    fn backward(_loss: &Self::RawTensor) -> Result<Self::Grads> { Ok(()) }
    fn step_sgd(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> { Ok(()) }
    fn step_adamw(_params: &mut [Self::RawVar], _grads: &Self::Grads, _lr: f64) -> Result<()> { Ok(()) }
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
    let t1: Tensor<[usize; 2], DummyBackend> = Tensor::zeros([2, 3]).unwrap();
    let t2: Tensor<[usize; 3], DummyBackend> = Tensor::zeros([2, 3, 4]).unwrap();

    // Mismatched shapes should not compile
    t1.add(&t2).unwrap();
}
