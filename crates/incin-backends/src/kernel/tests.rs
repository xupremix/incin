use super::*;

#[test]
fn canonical_keys_separate_binary_specializations_from_tuning_problems() {
    let scalar = KernelKey::cuda(
        OperationKind::Pointwise,
        KernelFamily::PointwiseUnary,
        "neg",
        DTypeId::F32,
        LayoutClass::Contiguous,
        KernelAccess::Scalar { unroll_width: 4 },
    )
    .unwrap();
    let mut packed = scalar.clone();
    packed.access = KernelAccess::Packed { vector_width: 4 };
    assert_ne!(scalar.cache_id(), packed.cache_id());
    assert_eq!(scalar.tuning_problem_id(), packed.tuning_problem_id());

    let mut strided = scalar.clone();
    strided.layout = LayoutClass::Strided;
    assert_ne!(scalar.cache_id(), strided.cache_id());
    assert_ne!(scalar.tuning_problem_id(), strided.tuning_problem_id());

    let mut wider_accumulator = scalar.clone();
    wider_accumulator.accumulator = KernelDType::F64;
    assert_ne!(scalar.cache_id(), wider_accumulator.cache_id());
    assert_ne!(
        scalar.tuning_problem_id(),
        wider_accumulator.tuning_problem_id()
    );
    assert!(
        scalar
            .cache_id()
            .starts_with("k1/cuda/pointwise-unary/neg/")
    );
}

#[test]
fn cuda_float_specializations_share_a_template_but_not_cache_keys() {
    let f16 = render_cuda_unary("relu", "x > 0.0f ? x : 0.0f", DTypeId::F16).unwrap();
    let f32 = render_cuda_unary("relu", "x > 0.0f ? x : 0.0f", DTypeId::F32).unwrap();
    let f64 = render_cuda_unary("relu", "x > 0.0 ? x : 0.0", DTypeId::F64).unwrap();

    assert_ne!(f16.cache_key, f32.cache_key);
    assert_ne!(f32.cache_key, f64.cache_key);
    assert_eq!(f16.dtype, DTypeId::F16);
    assert_eq!(f32.dtype, DTypeId::F32);
    assert_eq!(f64.dtype, DTypeId::F64);
    assert_eq!(
        f16.element_size,
        DTypeId::F16.encoding().scalar_bytes().unwrap()
    );
    assert_eq!(
        f32.element_size,
        DTypeId::F32.encoding().scalar_bytes().unwrap()
    );
    assert_eq!(
        f64.element_size,
        DTypeId::F64.encoding().scalar_bytes().unwrap()
    );
    assert!(f16.source.contains("const __half* input"));
    assert!(f16.source.contains("__half2float(input[flat_idx])"));
    assert!(f16.source.contains("__float2half_rn(out_val)"));
    assert!(f32.source.contains("const float* input"));
    assert!(f64.source.contains("const double* input"));
}

#[test]
fn cuda_bfloat16_uses_f32_compute_and_bfloat16_storage() {
    let rendered = render_cuda_binary("add", "a + b", DTypeId::BF16).unwrap();
    assert_eq!(rendered.element_size, 2);
    assert!(rendered.source.contains("#include <cuda_bf16.h>"));
    assert!(rendered.source.contains("const __nv_bfloat16* lhs"));
    assert!(rendered.source.contains("float a = __bfloat162float"));
    assert!(rendered.source.contains("__float2bfloat16_rn(out_val)"));
    assert!(!rendered.source.contains("lhs_shape"));
    assert!(!rendered.source.contains("rhs_shape"));
}

#[test]
fn renderer_rejects_non_float_dtypes_and_invalid_identifiers() {
    assert!(matches!(
        render_cuda_unary("relu", "x", DTypeId::U32),
        Err(Error::UnsupportedDType { .. })
    ));
    assert!(render_cuda_unary("relu;bad", "x", DTypeId::F32).is_err());
}

#[test]
fn layout_specializations_share_expressions_but_have_distinct_abis_and_keys() {
    let contiguous =
        render_cuda_binary_for_layout("sub", "a - b", DTypeId::F32, LayoutClass::Contiguous, 4)
            .unwrap();
    let scalar_left =
        render_cuda_binary_for_layout("sub", "a - b", DTypeId::F32, LayoutClass::ScalarLeft, 2)
            .unwrap();
    let strided =
        render_cuda_binary_for_layout("sub", "a - b", DTypeId::F32, LayoutClass::Strided, 1)
            .unwrap();

    assert_ne!(contiguous.cache_key, scalar_left.cache_key);
    assert_ne!(contiguous.cache_key, strided.cache_key);
    assert!(contiguous.source.contains("lhs[lhs_offset + idx]"));
    assert!(scalar_left.source.contains("lhs[lhs_offset]"));
    assert!(contiguous.source.contains("lane < 4"));
    assert!(contiguous.source.contains("if (idx < numel)"));
    assert_eq!(contiguous.unroll_width, 4);
    assert_eq!(contiguous.vector_width, 1);
    assert_eq!(contiguous.elements_per_thread(), 4);
    assert_eq!(
        contiguous.key.access,
        KernelAccess::Scalar { unroll_width: 4 }
    );
    assert!(contiguous.cache_key.contains("access=scalar-u4"));
    assert!(!contiguous.source.contains("out_shape"));
    assert!(strided.source.contains("out_shape"));

    let unary =
        render_cuda_unary_for_layout("neg", "-x", DTypeId::BF16, LayoutClass::Contiguous, 2)
            .unwrap();
    assert!(unary.source.contains("input[offset + idx]"));
    assert!(!unary.source.contains("strides"));
    assert_eq!(unary.unroll_width, 2);
    assert!(
        render_cuda_unary_for_layout("neg", "-x", DTypeId::F32, LayoutClass::Strided, 4,).is_err()
    );
}

#[test]
fn packed_templates_use_vector_storage_and_mask_scalar_tails() {
    let unary =
        render_cuda_unary_packed("neg", "-x", DTypeId::F32, LayoutClass::Contiguous).unwrap();
    assert_eq!(unary.unroll_width, 1);
    assert_eq!(unary.vector_width, 4);
    assert_eq!(unary.elements_per_thread(), 4);
    assert_eq!(unary.key.access, KernelAccess::Packed { vector_width: 4 });
    assert!(unary.cache_key.contains("access=packed-v4"));
    assert!(
        unary
            .source
            .contains("reinterpret_cast<const float4*>(input + offset)[packet_idx]")
    );
    assert!(unary.source.contains("if (base + 4 <= numel)"));
    assert!(unary.source.contains("input[offset + idx]"));

    let half =
        render_cuda_binary_packed("add", "a + b", DTypeId::F16, LayoutClass::Contiguous).unwrap();
    assert_eq!(half.vector_width, 2);
    assert!(half.source.contains("const __half2 lhs_storage"));
    assert!(half.source.contains("__half22float2(lhs_storage)"));
    assert!(
        half.source
            .contains("__floats2half2_rn(packed_output.x, packed_output.y)")
    );

    let scalar_left =
        render_cuda_binary_packed("sub", "a - b", DTypeId::BF16, LayoutClass::ScalarLeft).unwrap();
    assert!(scalar_left.source.contains("const float scalar_lhs"));
    assert!(
        scalar_left
            .source
            .contains("const __nv_bfloat162 rhs_storage")
    );
    assert!(scalar_left.source.contains("a = scalar_lhs"));
    assert!(
        render_cuda_binary_packed("add", "a + b", DTypeId::F32, LayoutClass::Strided,).is_err()
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a locally installed NVRTC shared library"]
fn packed_templates_compile_with_nvrtc_for_every_float_family() {
    for dtype in [DTypeId::F16, DTypeId::BF16, DTypeId::F32, DTypeId::F64] {
        let unary = render_cuda_unary_packed("neg", "-x", dtype, LayoutClass::Contiguous).unwrap();
        crate::cuda::gpu::compile_ptx_with_cuda_includes(&unary.source)
            .unwrap_or_else(|error| panic!("NVRTC rejected packed unary {dtype:?}: {error:?}"));

        for layout in [
            LayoutClass::Contiguous,
            LayoutClass::ScalarLeft,
            LayoutClass::ScalarRight,
        ] {
            let binary = render_cuda_binary_packed("add", "a + b", dtype, layout).unwrap();
            crate::cuda::gpu::compile_ptx_with_cuda_includes(&binary.source).unwrap_or_else(
                |error| panic!("NVRTC rejected packed binary {dtype:?}/{layout:?}: {error:?}"),
            );
        }
    }
}

#[test]
fn reduction_templates_share_structure_and_apply_accumulator_policy() {
    let half_fast = render_cuda_reduction("sum", DTypeId::F16, false, true).unwrap();
    assert_eq!(half_fast.key.layout, LayoutClass::ContiguousLastAxis);
    assert_eq!(half_fast.key.access, KernelAccess::WarpReduction);
    assert!(half_fast.cache_key.contains("/reduction/sum/s=f16"));
    assert!(half_fast.cache_key.contains("layout=contiguous-last-axis"));
    assert!(
        half_fast
            .source
            .contains("const __half* __restrict__ input")
    );
    assert!(half_fast.source.contains("float* shared"));
    assert!(
        half_fast
            .source
            .contains("__half2float(input[row_start + i])")
    );
    assert!(half_fast.source.contains("__float2half_rn(out_value)"));
    assert!(half_fast.source.contains("__shfl_down_sync"));
    assert!(half_fast.source.contains("shared[warp] = acc"));

    let double_mean = render_cuda_reduction("mean", DTypeId::F64, false, false).unwrap();
    assert_eq!(double_mean.key.layout, LayoutClass::Strided);
    assert!(double_mean.cache_key.contains("/reduction/mean/s=f64"));
    assert!(double_mean.source.contains("double acc"));
    assert!(double_mean.source.contains("acc / (double)reduce_dim_size"));
    assert!(!double_mean.source.contains("out_indices"));

    let indexed = render_cuda_reduction("max", DTypeId::BF16, true, false).unwrap();
    assert!(
        indexed
            .source
            .contains("unsigned int* __restrict__ out_indices")
    );
    assert!(indexed.source.contains("unsigned int best_idx"));
    assert!(indexed.source.contains("__bfloat162float(input[in_flat])"));
    assert!(indexed.source.contains("out_indices[out_flat] = best_idx"));
    assert!(render_cuda_reduction("sum", DTypeId::F32, true, false).is_err());
    assert!(render_cuda_reduction("unknown", DTypeId::F32, false, false).is_err());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a locally installed NVRTC shared library"]
fn reduction_templates_compile_with_nvrtc_for_every_float_family() {
    for dtype in [DTypeId::F16, DTypeId::BF16, DTypeId::F32, DTypeId::F64] {
        for op in ["sum", "mean", "max", "min"] {
            for fast in [false, true] {
                let kernel = render_cuda_reduction(op, dtype, false, fast).unwrap();
                crate::cuda::gpu::compile_ptx_with_cuda_includes(&kernel.source).unwrap_or_else(
                    |error| {
                        panic!("NVRTC rejected reduction {dtype:?}/{op}/fast={fast}: {error:?}")
                    },
                );
            }
        }
        for op in ["max", "min"] {
            let kernel = render_cuda_reduction(op, dtype, true, false).unwrap();
            crate::cuda::gpu::compile_ptx_with_cuda_includes(&kernel.source).unwrap_or_else(
                |error| panic!("NVRTC rejected indexed reduction {dtype:?}/{op}: {error:?}"),
            );
        }
    }
}

#[test]
fn normalization_templates_use_welford_and_dtype_specific_compute() {
    let half = render_cuda_normalization("layer_norm", DTypeId::F16).unwrap();
    assert!(half.source.contains("struct IncinWelford"));
    assert!(half.source.contains("__shfl_down_sync"));
    assert!(half.source.contains("float mean"));
    assert!(half.source.contains("__half2float(input[row_start + i])"));
    assert!(
        half.source
            .contains("__float2half_rn((normalized * scale + shift))")
    );

    let double = render_cuda_normalization("layer_norm", DTypeId::F64).unwrap();
    assert!(double.source.contains("1.0 / sqrt(variance + (double)eps)"));
    assert!(double.source.contains("double m2"));

    let bfloat_batch = render_cuda_normalization("batch_norm", DTypeId::BF16).unwrap();
    assert!(
        bfloat_batch
            .source
            .contains("const __nv_bfloat16* __restrict__ input")
    );
    assert!(
        bfloat_batch
            .source
            .contains("__float2bfloat16_rn((normalized * scale + shift))")
    );
    assert!(render_cuda_normalization("unknown", DTypeId::F32).is_err());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a locally installed NVRTC shared library"]
fn normalization_templates_compile_with_nvrtc_for_every_float_family() {
    for dtype in [DTypeId::F16, DTypeId::BF16, DTypeId::F32, DTypeId::F64] {
        for op in ["layer_norm", "batch_norm"] {
            let kernel = render_cuda_normalization(op, dtype).unwrap();
            crate::cuda::gpu::compile_ptx_with_cuda_includes(&kernel.source).unwrap_or_else(
                |error| panic!("NVRTC rejected normalization {dtype:?}/{op}: {error:?}"),
            );
        }
    }
}
