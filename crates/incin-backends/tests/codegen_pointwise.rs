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
