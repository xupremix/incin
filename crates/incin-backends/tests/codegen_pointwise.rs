//! Integration and unit tests for unified pointwise cross-backend codegen (PRF-007).

use incin_backends::codegen::{
    BinaryOp, LayoutKind, PointwiseExpr, PointwiseOpSpec, TernaryOp, UnaryOp, render_cuda,
    render_msl, render_wgsl,
};
use incin_core::prelude::DTypeId;

#[test]
fn test_unary_relu_codegen() {
    let spec = PointwiseOpSpec::unary("relu_f32", UnaryOp::Relu, DTypeId::F32);

    let cuda = render_cuda(&spec);
    assert!(cuda.contains("__global__ void relu_f32"));
    assert!(cuda.contains("fmaxf(0.0f, val0)"));

    let wgsl = render_wgsl(&spec);
    assert!(wgsl.contains("fn main"));
    assert!(wgsl.contains("max(0.0, val0)"));

    let msl = render_msl(&spec);
    assert!(msl.contains("kernel void relu_f32"));
    assert!(msl.contains("max(0.0f, val0)"));
}

#[test]
fn test_binary_add_f16_codegen() {
    let spec = PointwiseOpSpec::binary("add_f16", BinaryOp::Add, DTypeId::F16, DTypeId::F16);

    let cuda = render_cuda(&spec);
    assert!(cuda.contains("__half"));
    assert!(cuda.contains("(val0 + val1)"));

    let wgsl = render_wgsl(&spec);
    assert!(wgsl.contains("f16"));
    assert!(wgsl.contains("(val0 + val1)"));

    let msl = render_msl(&spec);
    assert!(msl.contains("half"));
    assert!(msl.contains("(val0 + val1)"));
}

#[test]
fn test_scalar_broadcast_layout() {
    let mut spec =
        PointwiseOpSpec::binary("add_scalar_f32", BinaryOp::Add, DTypeId::F32, DTypeId::F32);
    spec.layout = LayoutKind::ScalarBroadcast;

    let cuda = render_cuda(&spec);
    assert!(cuda.contains("in1[0]"));

    let wgsl = render_wgsl(&spec);
    assert!(wgsl.contains("in1[0]"));

    let msl = render_msl(&spec);
    assert!(msl.contains("in1[0]"));
}

#[test]
fn test_complex_fma_gelu_expression() {
    let expr = PointwiseExpr::Ternary(
        TernaryOp::Fma,
        Box::new(PointwiseExpr::Unary(
            UnaryOp::Gelu,
            Box::new(PointwiseExpr::Arg(0)),
        )),
        Box::new(PointwiseExpr::Arg(1)),
        Box::new(PointwiseExpr::Arg(2)),
    );

    let spec = PointwiseOpSpec {
        name: "fma_gelu_f32".to_string(),
        inputs: vec![DTypeId::F32, DTypeId::F32, DTypeId::F32],
        output: DTypeId::F32,
        expr,
        layout: LayoutKind::Contiguous,
        work_group_size: 256,
    };

    let cuda = render_cuda(&spec);
    assert!(cuda.contains("fmaf("));

    let wgsl = render_wgsl(&spec);
    assert!(wgsl.contains("fma("));

    let msl = render_msl(&spec);
    assert!(msl.contains("fma("));
}

#[test]
fn test_all_dtypes_type_mapping() {
    let dtypes = vec![
        DTypeId::F32,
        DTypeId::F64,
        DTypeId::F16,
        DTypeId::BF16,
        DTypeId::U8,
        DTypeId::U32,
        DTypeId::I64,
    ];

    for dt in dtypes {
        let spec = PointwiseOpSpec::unary(format!("abs_{}", dt.name()), UnaryOp::Abs, dt);
        let cuda = render_cuda(&spec);
        let wgsl = render_wgsl(&spec);
        let msl = render_msl(&spec);

        assert!(!cuda.is_empty());
        assert!(!wgsl.is_empty());
        assert!(!msl.is_empty());
    }
}

#[test]
fn test_vectorized_vec4_pointwise_codegen() {
    use incin_backends::codegen::{VectorWidth, VectorizedOpSpec};

    let spec = VectorizedOpSpec::unary(
        "gelu_vec4_f32",
        UnaryOp::Gelu,
        DTypeId::F32,
        VectorWidth::Vec4,
    );

    let cuda = spec.render_cuda();
    assert!(cuda.contains("const float4* __restrict__ in0"));
    assert!(cuda.contains("float4* __restrict__ out_data"));
    assert!(cuda.contains("make_float4(res_lane0, res_lane1, res_lane2, res_lane3)"));

    let wgsl = spec.render_wgsl();
    assert!(wgsl.contains("array<vec4<f32>>"));
    assert!(wgsl.contains("vec4(res_lane0, res_lane1, res_lane2, res_lane3)"));

    let msl = spec.render_msl();
    assert!(msl.contains("device const float4*"));
    assert!(msl.contains("float4(res_lane0, res_lane1, res_lane2, res_lane3)"));
}

#[test]
fn test_fused_gemm_epilogue_codegen() {
    use incin_backends::codegen::{FusedEpilogueKind, FusedEpilogueSpec};

    let spec = FusedEpilogueSpec::new(
        "fused_linear_bias_gelu",
        DTypeId::F32,
        FusedEpilogueKind::BiasResidualGelu,
    )
    .with_cols(768);

    let cuda = spec.render_cuda();
    assert!(cuda.contains("const float* __restrict__ matmul_out"));
    assert!(cuda.contains("const float* __restrict__ bias"));
    assert!(cuda.contains("const float* __restrict__ residual"));
    assert!(cuda.contains("bias[(idx % 768)]"));
    assert!(cuda.contains("tanhf("));

    let wgsl = spec.render_wgsl();
    assert!(wgsl.contains("var<storage, read> matmul_out: array<f32>"));
    assert!(wgsl.contains("var<storage, read> bias: array<f32>"));
    assert!(wgsl.contains("var<storage, read> residual: array<f32>"));
    assert!(wgsl.contains("bias[(idx % 768u)]"));
    assert!(wgsl.contains("tanh("));

    let msl = spec.render_msl();
    assert!(msl.contains("device const float* matmul_out"));
    assert!(msl.contains("device const float* bias"));
    assert!(msl.contains("bias[(idx % 768u)]"));
    assert!(msl.contains("tanh("));
}

#[test]
fn test_parallel_reduction_codegen() {
    use incin_backends::codegen::{ReductionOpKind, ReductionOpSpec};

    let spec = ReductionOpSpec::row_wise("warp_reduce_sum_f32", ReductionOpKind::Sum, DTypeId::F32)
        .with_reduction_size(512);

    let cuda = spec.render_cuda();
    assert!(cuda.contains("__global__ void warp_reduce_sum_f32"));
    assert!(cuda.contains("__shfl_down_sync"));
    assert!(cuda.contains("local_acc += __shfl_down_sync"));

    let wgsl = spec.render_wgsl();
    assert!(wgsl.contains("@compute @workgroup_size(256, 1, 1)"));
    assert!(wgsl.contains("var<workgroup> s_acc: array<f32, 256>"));
    assert!(wgsl.contains("workgroupBarrier()"));

    let msl = spec.render_msl();
    assert!(msl.contains("kernel void warp_reduce_sum_f32"));
    assert!(msl.contains("simd_sum(local_val)"));
}

#[test]
fn test_tiled_gemm_codegen() {
    use incin_backends::codegen::{GemmSpec, GemmTileConfig};

    let spec = GemmSpec::new("tiled_gemm_f32", DTypeId::F32).with_tile(GemmTileConfig {
        bm: 64,
        bn: 64,
        bk: 16,
        tm: 4,
        tn: 4,
    });

    let cuda = spec.render_cuda();
    assert!(cuda.contains("__global__ void tiled_gemm_f32"));
    assert!(cuda.contains("__shared__ float s_a[64][17]"));
    assert!(cuda.contains("__shared__ float s_b[16][65]"));
    assert!(cuda.contains("fmaf(r_a[i], r_b[j], r_c[i][j])"));

    let wgsl = spec.render_wgsl();
    assert!(wgsl.contains("var<workgroup> s_a: array<array<f32, 16>, 64>"));
    assert!(wgsl.contains("fma(r_a[i], r_b[j], r_c[i][j])"));

    let msl = spec.render_msl();
    assert!(msl.contains("threadgroup float s_a[64][16]"));
    assert!(msl.contains("fma(r_a[i], r_b[j], r_c[i][j])"));
}

#[test]
fn test_attention_sdpa_flash_codegen() {
    use incin_backends::codegen::AttentionSpec;

    let spec = AttentionSpec::new("sdpa_causal_f32", DTypeId::F32, 64, true);

    let cuda = spec.render_cuda();
    assert!(cuda.contains("__global__ void sdpa_causal_f32_forward"));
    assert!(cuda.contains("min(seq_len, q_seq_idx + 1)"));
    assert!(cuda.contains("m_curr = fmaxf(m_prev, dot)"));
    assert!(cuda.contains("p_prev_scale = expf(m_prev - m_curr)"));

    let wgsl = spec.render_wgsl();
    assert!(wgsl.contains("fn sdpa_causal_f32_forward"));
    assert!(wgsl.contains("min(params.seq_len, q_seq_idx + 1u)"));

    let msl = spec.render_msl();
    assert!(msl.contains("kernel void sdpa_causal_f32_forward"));
    assert!(msl.contains("min(seq_len, q_seq_idx + 1)"));
}

#[test]
fn test_normalization_fused_layernorm_and_rmsnorm() {
    use incin_backends::codegen::{NormKind, NormalizationSpec};

    let rms_spec = NormalizationSpec::new(
        "rms_norm_f32",
        NormKind::RmsNorm,
        DTypeId::F32,
        512,
        1e-5,
        true,
    );
    let cuda_rms = rms_spec.render_cuda_forward();
    assert!(cuda_rms.contains("__global__ void rms_norm_f32_forward"));
    assert!(cuda_rms.contains("__shfl_down_sync"));
    assert!(cuda_rms.contains("rsqrtf(s_mean_sq + eps)"));

    let cuda_rms_bwd = rms_spec.render_cuda_backward();
    assert!(cuda_rms_bwd.contains("__global__ void rms_norm_f32_backward"));
    assert!(cuda_rms_bwd.contains("atomicAdd"));

    let ln_spec = NormalizationSpec::new(
        "layer_norm_f32",
        NormKind::LayerNorm,
        DTypeId::F32,
        768,
        1e-5,
        true,
    );
    let cuda_ln = ln_spec.render_cuda_forward();
    assert!(cuda_ln.contains("__global__ void layer_norm_f32_forward"));
    assert!(cuda_ln.contains("normed * gamma[i] + beta[i]"));

    let cuda_ln_bwd = ln_spec.render_cuda_backward();
    assert!(cuda_ln_bwd.contains("__global__ void layer_norm_f32_backward"));
    assert!(cuda_ln_bwd.contains("s_sum_dy_g_hat"));
}

#[test]
fn test_fast_divisor_and_strided_indexing() {
    use incin_backends::codegen::{FastDivisor, StridedIndexSpec};

    let fdiv = FastDivisor::new(7);
    assert_eq!(fdiv.divisor, 7);
    assert!(fdiv.multiplier > 0);

    let spec = StridedIndexSpec::new(vec![2, 8, 64], vec![512, 64, 1]);
    let code = spec.render_cuda_offset_expression("tid");
    assert!(code.contains("Fast strided coordinate decomposition for rank 3"));
    assert!(code.contains("coord_2 = rem % 64"));
    assert!(code.contains("coord_1 = rem % 8"));
    assert!(code.contains("physical_offset"));
}

#[test]
fn test_triton_inductor_scheduler_and_autotuning() {
    use incin_backends::codegen::{
        AutotuneSpace, BlockTensorPtr, GpuArchProfile, KernelScheduler, LoopScheduleKind,
    };

    let ptr_a = BlockTensorPtr::global("a", DTypeId::F32, vec![32, 64], vec![64, 1]);
    assert!(ptr_a.is_innermost_contiguous());
    assert_eq!(ptr_a.optimal_vector_width(), 4);

    let scheduler = KernelScheduler::new(
        "custom_gemm",
        LoopScheduleKind::Tiled2D {
            block_m: 64,
            block_n: 64,
            block_k: 16,
        },
        vec![ptr_a],
        vec![],
    );

    let (gx, gy, gz) = scheduler.recommended_grid_dim();
    assert!(gx >= 1);
    assert!(gy >= 1);
    assert_eq!(gz, 1);

    let preamble = scheduler.render_cuda_preamble();
    assert!(preamble.contains("block_row = blockIdx.y * 64"));
    assert!(preamble.contains("block_col = blockIdx.x * 64"));

    let space =
        AutotuneSpace::for_matmul(1024, 1024, 1024, DTypeId::F32, GpuArchProfile::NvidiaModern);
    let candidates = space.generate_candidates();
    assert!(!candidates.is_empty());
    assert!(candidates[0].block_m >= 16);
    assert!(candidates[0].shared_memory_bytes > 0);

    let best = space.select_best_heuristic();
    assert!(best.block_m >= 16);
}

#[test]
fn test_tensor_core_mma_codegen() {
    use incin_backends::codegen::TensorCoreMmaSpec;

    let spec = TensorCoreMmaSpec::new("tensor_core_gemm_f16", DTypeId::F16, DTypeId::F32, 2, 2);
    let cuda = spec.render_cuda();
    assert!(cuda.contains("nvcuda::wmma"));
    assert!(cuda.contains("fragment<matrix_a, 16, 16, 16, __half, row_major>"));
    assert!(cuda.contains("mma_sync(c_frag, a_frag, b_frag, c_frag)"));
    assert!(cuda.contains("store_matrix_sync"));
}

#[test]
fn test_rope_embedding_codegen() {
    use incin_backends::codegen::RopeSpec;

    let spec_dyn = RopeSpec::new("rope_dyn_f32", DTypeId::F32, 64, 10000.0, false);
    let cuda_fwd = spec_dyn.render_cuda_forward();
    assert!(cuda_fwd.contains("__global__ void rope_dyn_f32_forward"));
    assert!(cuda_fwd.contains("sincosf(angle, &sin_v, &cos_v)"));
    assert!(cuda_fwd.contains("y0 = x0 * cos_v - x1 * sin_v"));

    let cuda_bwd = spec_dyn.render_cuda_backward();
    assert!(cuda_bwd.contains("__global__ void rope_dyn_f32_backward"));
    assert!(cuda_bwd.contains("dx0 = dy0 * cos_v + dy1 * sin_v"));

    let spec_pre = RopeSpec::new("rope_pre_f32", DTypeId::F32, 128, 500000.0, true);
    let cuda_pre_fwd = spec_pre.render_cuda_forward();
    assert!(cuda_pre_fwd.contains("cos_table"));
    assert!(cuda_pre_fwd.contains("sin_table"));
}

#[test]
fn test_fused_cross_entropy_codegen() {
    use incin_backends::codegen::CrossEntropySpec;

    let spec = CrossEntropySpec::new("cross_entropy_loss_f32", DTypeId::F32, 32000, 0.1);
    let cuda_fwd = spec.render_cuda_forward();
    assert!(cuda_fwd.contains("__global__ void cross_entropy_loss_f32_forward"));
    assert!(cuda_fwd.contains("__shfl_down_sync"));
    assert!(cuda_fwd.contains("smooth_loss"));

    let cuda_bwd = spec.render_cuda_backward();
    assert!(cuda_bwd.contains("__global__ void cross_entropy_loss_f32_backward"));
    assert!(cuda_bwd.contains("dlogits_sample[v] = static_cast<float>(grad)"));
}

#[test]
fn test_fused_optimizer_adamw_and_lion_codegen() {
    use incin_backends::codegen::FusedOptimizerSpec;

    let spec_adamw = FusedOptimizerSpec::adamw(
        "adamw_f32",
        DTypeId::F32,
        DTypeId::F32,
        1e-3,
        0.01,
        0.9,
        0.999,
        1e-8,
    );
    let cuda_adamw = spec_adamw.render_cuda();
    assert!(cuda_adamw.contains("__global__ void adamw_f32"));
    assert!(cuda_adamw.contains("p -= lr * weight_decay * p"));
    assert!(cuda_adamw.contains("step = m_hat / (sqrtf(v_hat) + eps)"));

    let spec_lion =
        FusedOptimizerSpec::lion("lion_f32", DTypeId::F32, DTypeId::F32, 1e-4, 0.1, 0.9, 0.99);
    let cuda_lion = spec_lion.render_cuda();
    assert!(cuda_lion.contains("__global__ void lion_f32"));
    assert!(
        cuda_lion.contains(
            "sign_val = (update_dir > 0.0f) ? 1.0f : ((update_dir < 0.0f) ? -1.0f : 0.0f)"
        )
    );
}

#[test]
fn test_quant_gemv_q8_0_codegen() {
    use incin_backends::codegen::QuantGemmSpec;

    let spec = QuantGemmSpec::q8_0_gemv("gemv_q8_0_f32", DTypeId::F32);
    let cuda = spec.render_cuda_gemv();
    assert!(cuda.contains("__global__ void gemv_q8_0_f32"));
    assert!(cuda.contains("struct __align__(2) BlockQ8_0"));
    assert!(cuda.contains("__shfl_down_sync"));
}

#[test]
fn test_prefix_scan_codegen() {
    use incin_backends::codegen::{PrefixScanSpec, ScanOpKind};

    let spec = PrefixScanSpec::new("cumsum_f32", ScanOpKind::Sum, DTypeId::F32, true);
    let cuda = spec.render_cuda();
    assert!(cuda.contains("__global__ void cumsum_f32"));
    assert!(cuda.contains("__shfl_up_sync"));
    assert!(cuda.contains("running_val += other"));
}

#[test]
fn test_composite_fusion_swiglu_codegen() {
    use incin_backends::codegen::CompositeFusionSpec;

    let spec = CompositeFusionSpec::swiglu_residual("swiglu_res_f32", DTypeId::F32);
    let cuda = spec.render_cuda();
    assert!(cuda.contains("__global__ void swiglu_res_f32"));
    assert!(cuda.contains("const float gate = static_cast<float>(in_0[idx])"));
    assert!(cuda.contains("const float silu_up = up / (1.0f + expf(-up))"));
    assert!(cuda.contains("const float gated = gate * silu_up"));
    assert!(cuda.contains("const float out = gated + res"));
    assert!(cuda.contains("Out[idx] = static_cast<float>(out)"));
}

#[test]
fn test_moe_topk_gating_codegen() {
    use incin_backends::codegen::MoeGatingSpec;

    let spec = MoeGatingSpec::new("moe_gating_top2_f32", DTypeId::F32, 8, 2);
    let cuda = spec.render_cuda_gating();
    assert!(cuda.contains("__global__ void moe_gating_top2_f32_gating"));
    assert!(cuda.contains("int top_idx[2]"));
    assert!(cuda.contains("float top_val[2]"));
    assert!(cuda.contains("out_indices[k] = top_idx[k]"));
    assert!(cuda.contains("out_weights[k] = expf(top_val[k] - max_val) * inv_sum"));
}

#[test]
fn test_implicit_gemm_conv2d_codegen() {
    use incin_backends::codegen::ImplicitConv2dSpec;

    let spec = ImplicitConv2dSpec::new("conv2d_k3s1p1_f32", DTypeId::F32, (3, 3), (1, 1), (1, 1));
    let cuda = spec.render_cuda();
    assert!(cuda.contains("__global__ void conv2d_k3s1p1_f32"));
    assert!(cuda.contains("const int w_out = out_idx % out_w"));
    assert!(cuda.contains("const int h_in = h_out * 1 - 1 + r * 1"));
    assert!(cuda.contains("fmaf(in_val, w_val, sum)"));
}
