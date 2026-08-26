#90 Dtype-parametric matmul — M
Finding: core Tensor::matmul already generic over K: DType (matmul.rs:317); gate is purely the CUDA executor: kernels/matmul.cu is const float* with unconditional .transmute::<f32>() — bf16 today = silent byte misread. bf16 pointwise rendering + cublas feature already exist; backward calls launch_matmul 3x (all must convert together).
Recommendation: generic executor over K: FloatDType with GemmComputeType mapping (bf16: CUDA_R_16BF + COMPUTE_32F tensor cores; f32 keeps current kernel); flip matmul=FLOAT_DTYPES; record f32 accumulation in OPERATION_SEMANTICS.
Risk: transmute-class unsoundness; alignment on K/N; no CPU bf16 oracle for #83.
Unblocks: #2; feeds #85/#94/#95.
