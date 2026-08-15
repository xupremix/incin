extern crate alloc;
use incin::Tensor;
use incin::prelude::*;

#[allow(clippy::type_complexity)]
fn main() -> Result<()> {
    // Tier 3: Fully compile-time CPU (only one CPU, always Tier 3)
    type CpuB = IncinBackend<Cpu>;
    let cpu_tensor: Tensor<s![3, 3], CpuB> = Tensor::zeros(())?;
    let res = cpu_tensor.relu()?;
    println!("CPU backend shape: {:?}", res.dims());

    // Tier 3: Fully compile-time WGPU — both backend family and ordinal (U0 = adapter 0)
    // known at compile time, no runtime argument needed.
    type WgpuB = IncinBackend<WgpuN<typenum::U0>>;
    let wgpu_tensor: Tensor<s![3, 3], WgpuB> = Tensor::zeros(())?;
    let res2 = wgpu_tensor.relu()?;
    println!("WGPU backend shape: {:?}", res2.dims());

    Ok(())
}
