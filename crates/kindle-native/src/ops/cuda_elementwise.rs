pub const ELEMENTWISE_UNARY_TEMPLATE: &str = r#"
extern "C" __global__ void unary_op(
    const float* input,
    float* output,
    const int* shape,
    const int* strides,
    int offset,
    int numel,
    int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        // Resolve logical index to flat index
        int flat_idx = offset;
        int temp = idx;
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_idx = temp % shape[i];
            temp /= shape[i];
            flat_idx += dim_idx * strides[i];
        }
        float x = input[flat_idx];
        float out_val = {OP};
        output[idx] = out_val;
    }
}
"#;

pub const ELEMENTWISE_BINARY_TEMPLATE: &str = r#"
extern "C" __global__ void binary_op(
    const float* lhs,
    const float* rhs,
    float* output,
    const int* out_shape,
    const int* lhs_shape,
    const int* rhs_shape,
    const int* lhs_strides,
    const int* rhs_strides,
    int lhs_offset,
    int rhs_offset,
    int numel,
    int ndim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numel) {
        int temp = idx;
        int lhs_flat = lhs_offset;
        int rhs_flat = rhs_offset;
        
        for (int i = ndim - 1; i >= 0; i--) {
            int dim_idx = temp % out_shape[i];
            temp /= out_shape[i];
            
            // Broadcast logic for LHS
            int lhs_dim = lhs_shape[i];
            int lhs_dim_idx = (lhs_dim == 1) ? 0 : dim_idx;
            lhs_flat += lhs_dim_idx * lhs_strides[i];
            
            // Broadcast logic for RHS
            int rhs_dim = rhs_shape[i];
            int rhs_dim_idx = (rhs_dim == 1) ? 0 : dim_idx;
            rhs_flat += rhs_dim_idx * rhs_strides[i];
        }
        
        float a = lhs[lhs_flat];
        float b = rhs[rhs_flat];
        float out_val = {OP};
        output[idx] = out_val;
    }
}
"#;

pub fn generate_unary(op: &str) -> String {
    ELEMENTWISE_UNARY_TEMPLATE.replace("{OP}", op)
}

pub fn generate_binary(op: &str) -> String {
    ELEMENTWISE_BINARY_TEMPLATE.replace("{OP}", op)
}
