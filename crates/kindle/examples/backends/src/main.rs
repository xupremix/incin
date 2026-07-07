use kindle::{Tensor, DefaultDevice};
use kindle::prelude::*;

fn main() -> Result<()> {
    // 1. Using the default Candle Backend
    let candle_tensor: Tensor<s![3, 3], CandleBackend<f32, DefaultDevice>> = Tensor::zeros(())?;
    let res = candle_tensor.relu()?;
    println!("Candle Backend Shape: {:?}", res.dims());

    // 2. Using the Ndarray Backend
    let ndarray_tensor: Tensor<s![3, 3], NdarrayBackend<f32, DefaultDevice>> = Tensor::zeros(())?;
    let res2 = ndarray_tensor.relu()?;
    println!("Ndarray Backend Shape: {:?}", res2.dims());

    Ok(())
}
