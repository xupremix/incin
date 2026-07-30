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
