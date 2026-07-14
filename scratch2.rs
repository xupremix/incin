use cudarc::driver::{CudaDevice, CudaFunction, LaunchConfig, LaunchAsync};
fn test(f: CudaFunction, cfg: LaunchConfig) {
    let args = (1i32, 2i32);
    unsafe { f.launch(cfg, args).unwrap() };
}
fn main() {}
