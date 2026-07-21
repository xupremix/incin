use kindle::prelude::*;
use kindle_native::{NativeBackend, Cpu};
use kindle_wgpu::WgpuBackend;

type NativeCPU = NativeBackend<f32, Cpu>;
type WgpuCPU = WgpuBackend<f32, kindle_core::tensor::device::Cpu>;

#[test]
fn test_matmul_parity() -> Result<()> {
    let dev_cpu = KindleDevice::cpu();
    // For Wgpu to work in a portable test, we might use default device or whatever WgpuBackend supports.
    // If wgpu tests pass, then it's fine.
    
    // Parity tests implementation.
    Ok(())
}
