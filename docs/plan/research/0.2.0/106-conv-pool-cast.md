#106 Last 6 unowned CUDA rows — M
Finding: conv1d, conv_transpose2d, adaptive_avg_pool2d, to_dtype, quantize, dequantize. Kernels partially EXIST unwired: im2col_1d/col2im_1d in conv.cu; adaptive_avg_pool2d forward/backward in pool.cu. cudnn enabled but unused (nondeterminism, f16/f32-only, no adaptive pooling -> poor fit).
Recommendation: implicit-GEMM family sharing one checked out_size() geometry (conv_transpose2d = im2col + GEMM + col2im scatter-add with output_padding folded); wire adaptive pool; new cast kernels with per-pair rounding documented; Q8_0 block quantize/dequantize mirroring CPU natives. Closing zeroes the UNOWNED CUDA gap.
Risk: output_padding off-by-one (explicit oracle rows); f32->bf16 rounding drift; Q8_0 block boundaries.
