#[allow(dead_code)]
pub const MATMUL_SWIGLU_KERNEL: &str = include_str!("kernels/matmul_swiglu.cu");
#[allow(dead_code)]
pub const FLASH_ATTENTION_LITE_KERNEL: &str = include_str!("kernels/flash_attention_lite.cu");
#[allow(dead_code)]
pub const FUSED_ADAMW_KERNEL: &str = include_str!("kernels/fused_adamw.cu");
#[allow(dead_code)]
pub const MATMUL_KERNEL: &str = include_str!("kernels/matmul.cu");
pub const LOSS_KERNEL: &str = include_str!("kernels/loss.cu");
pub const QUANT_KERNEL: &str = include_str!("kernels/quant.cu");
pub const SHAPE_KERNEL: &str = include_str!("kernels/shape.cu");
