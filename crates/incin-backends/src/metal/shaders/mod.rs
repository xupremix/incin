//! Metal Shading Language (MSL) shader definitions and descriptor generator.

/// Pointwise MSL shader code.
pub const POINTWISE_MSL: &str = include_str!("pointwise.metal");

/// Reduction MSL shader code.
pub const REDUCTION_MSL: &str = include_str!("reduction.metal");

/// Descriptor for a Metal pointwise compute kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalPointwiseDescriptor {
    /// MTL shader function name for unary kernels.
    pub kernel_name: &'static str,
    /// Numeric op tag passed to the shader.
    pub op_type: u32,
    /// Number of buffers the kernel expects.
    pub num_buffers: usize,
}

impl MetalPointwiseDescriptor {
    /// Creates a unary operation descriptor.
    #[must_use]
    pub const fn unary(op_type: u32) -> Self {
        Self {
            kernel_name: "pointwise_unary_f32",
            op_type,
            num_buffers: 3,
        }
    }

    /// Creates a binary operation descriptor.
    #[must_use]
    pub const fn binary(op_type: u32) -> Self {
        Self {
            kernel_name: "pointwise_binary_f32",
            op_type,
            num_buffers: 4,
        }
    }
}

/// Descriptor for a Metal reduction compute kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalReductionDescriptor {
    /// MTL shader function name for binary kernels.
    pub kernel_name: &'static str,
    /// Numeric op tag passed to the shader.
    pub op_type: u32,
    /// Threads per threadgroup for the launch.
    pub threads_per_group: u32,
}

impl MetalReductionDescriptor {
    /// Creates a reduction descriptor (0=Sum, 1=Max, 2=Min).
    #[must_use]
    pub const fn new(op_type: u32, threads_per_group: u32) -> Self {
        Self {
            kernel_name: "reduce_f32",
            op_type,
            threads_per_group,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pointwise_descriptor_parity() {
        let relu = MetalPointwiseDescriptor::unary(0);
        assert_eq!(relu.kernel_name, "pointwise_unary_f32");
        assert_eq!(relu.op_type, 0);

        let add = MetalPointwiseDescriptor::binary(0);
        assert_eq!(add.kernel_name, "pointwise_binary_f32");
        assert_eq!(add.op_type, 0);

        assert!(POINTWISE_MSL.contains("kernel void pointwise_unary_f32"));
        assert!(POINTWISE_MSL.contains("kernel void pointwise_binary_f32"));
    }

    #[test]
    fn test_reduction_descriptor_parity() {
        let sum_reduce = MetalReductionDescriptor::new(0, 256);
        assert_eq!(sum_reduce.kernel_name, "reduce_f32");
        assert_eq!(sum_reduce.op_type, 0);
        assert_eq!(sum_reduce.threads_per_group, 256);

        assert!(REDUCTION_MSL.contains("kernel void reduce_f32"));
    }
}
