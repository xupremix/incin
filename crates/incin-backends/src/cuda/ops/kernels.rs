#[allow(dead_code)]
pub const MATMUL_KERNEL: &str = include_str!("kernels/matmul.cu");
pub const LOSS_KERNEL: &str = include_str!("kernels/loss.cu");
pub const QUANT_KERNEL: &str = include_str!("kernels/quant.cu");
pub const SHAPE_KERNEL: &str = include_str!("kernels/shape.cu");
pub const POOL_KERNEL: &str = include_str!("kernels/pool.cu");
pub const CONV_KERNEL: &str = include_str!("kernels/conv.cu");
