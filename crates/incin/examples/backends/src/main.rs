extern crate alloc;
use incin::Tensor;
use incin::prelude::*;

fn main() -> Result<()> {
    type CpuB = IncinBackend<f32, Cpu>;
    let cpu_tensor: Tensor<s![3, 3], CpuB> = Tensor::zeros(())?;
    let res = cpu_tensor.relu()?;
    println!("CPU backend shape: {:?}", res.dims());

    type WgpuB = IncinBackend<f32, Wgpu>;
    let wgpu_tensor: Tensor<s![3, 3], WgpuB> = Tensor::zeros(())?;
    let res2 = wgpu_tensor.relu()?;
    println!("WGPU backend shape: {:?}", res2.dims());

    Ok(())
}
