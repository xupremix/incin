#include <metal_stdlib>
using namespace metal;

// MSL Pointwise Unary Operations Kernel
kernel void pointwise_unary_f32(
    device const float* in [[buffer(0)]],
    device float* out [[buffer(1)]],
    constant uint& op_type [[buffer(2)]],
    uint id [[thread_position_in_grid]]
) {
    float x = in[id];
    float res = x;
    switch (op_type) {
        case 0: res = max(x, 0.0f); break;            // ReLU
        case 1: res = 1.0f / (1.0f + exp(-x)); break; // Sigmoid
        case 2: res = tanh(x); break;                 // Tanh
        case 3: res = abs(x); break;                  // Abs
        case 4: res = -x; break;                      // Neg
        case 5: res = sqrt(x); break;                 // Sqrt
        case 6: res = exp(x); break;                  // Exp
        case 7: res = log(x); break;                  // Log
        default: break;
    }
    out[id] = res;
}

// MSL Pointwise Binary Operations Kernel
kernel void pointwise_binary_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* out [[buffer(2)]],
    constant uint& op_type [[buffer(3)]],
    uint id [[thread_position_in_grid]]
) {
    float a = lhs[id];
    float b = rhs[id];
    float res = a;
    switch (op_type) {
        case 0: res = a + b; break; // Add
        case 1: res = a - b; break; // Sub
        case 2: res = a * b; break; // Mul
        case 3: res = a / b; break; // Div
        default: break;
    }
    out[id] = res;
}
