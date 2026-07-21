extern crate alloc;
use kindle::prelude::*;
use kindle::{DefaultDevice, Tensor};

fn main() -> Result<()> {
    // 1. Using the default Candle Backend
    let candle_tensor: Tensor<s![3, 3], CandleBackend<f32, DefaultDevice>> = Tensor::zeros(())?;
    let res = candle_tensor.relu()?;
    println!("Candle Backend Shape: {:?}", res.dims());

    // 2. Using the Wgpu Backend
    let wgpu_tensor: Tensor<s![3, 3], WgpuBackend> = Tensor::zeros(())?;
    let res2 = wgpu_tensor.relu()?;
    println!("Wgpu Backend Shape: {:?}", res2.dims());

    Ok(())
}
