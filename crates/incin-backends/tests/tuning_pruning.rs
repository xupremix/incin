//! Integration coverage for `rank_class_conversions_and_tags` on the documented public surface.
#![cfg(feature = "autotune")]

use incin_backends::tuning::signature::{
    AlignmentClass, DTypePolicyId, KernelAccess, KernelFamily, KernelKey, KernelSignature,
    LaunchCandidate, MatMulTileCandidate, OperationKind, RankClass, ShapeBucket,
    prune_matmul_candidates, prune_pointwise_candidates, prune_reduction_candidates,
};
use incin_core::exec::LayoutClass;
use incin_core::prelude::DTypeId;

#[test]
fn rank_class_conversions_and_tags() {
    assert_eq!(RankClass::from_rank(0), RankClass::Scalar);
    assert_eq!(RankClass::from_rank(1), RankClass::Vector);
    assert_eq!(RankClass::from_rank(2), RankClass::Matrix);
    assert_eq!(RankClass::from_rank(3), RankClass::Volume);
    assert_eq!(RankClass::from_rank(4), RankClass::Tensor4);
    assert_eq!(RankClass::from_rank(8), RankClass::Higher(8));

    assert_eq!(RankClass::Scalar.tag(), "rank0");
    assert_eq!(RankClass::Vector.tag(), "rank1");
    assert_eq!(RankClass::Matrix.tag(), "rank2");
    assert_eq!(RankClass::Volume.tag(), "rank3");
    assert_eq!(RankClass::Tensor4.tag(), "rank4");
    assert_eq!(RankClass::Higher(5).tag(), "rankN");

    assert!(!RankClass::Scalar.is_supported_for_vectorization());
    assert!(RankClass::Vector.is_supported_for_vectorization());
    assert!(RankClass::Matrix.is_supported_for_vectorization());
}

#[test]
fn shape_bucket_log2_and_tags() {
    let b1 = ShapeBucket::from_numel(1024);
    assert_eq!(b1.numel_log2, 10);
    assert_eq!(b1.primary_dim_log2, 10);
    assert_eq!(b1.secondary_dim_log2, 0);
    assert_eq!(b1.tag(), "p10_s0");

    let b2 = ShapeBucket::from_matrix(64, 128);
    assert_eq!(b2.numel_log2, 13);
    assert_eq!(b2.primary_dim_log2, 6);
    assert_eq!(b2.secondary_dim_log2, 7);
    assert_eq!(b2.tag(), "p6_s7");

    let b3 = ShapeBucket::from_gemm(32, 64, 16);
    assert_eq!(b3.numel_log2, 11);
    assert_eq!(b3.primary_dim_log2, 5);
    assert_eq!(b3.secondary_dim_log2, 4);
    assert_eq!(b3.tag(), "p5_s4");
}

#[test]
fn alignment_class_detection_and_vector_compatibility() {
    assert_eq!(AlignmentClass::from_bytes(0), AlignmentClass::Align256);
    assert_eq!(AlignmentClass::from_bytes(256), AlignmentClass::Align256);
    assert_eq!(AlignmentClass::from_bytes(512), AlignmentClass::Align256);
    assert_eq!(AlignmentClass::from_bytes(16), AlignmentClass::Quad);
    assert_eq!(AlignmentClass::from_bytes(32), AlignmentClass::Quad);
    assert_eq!(AlignmentClass::from_bytes(4), AlignmentClass::Word);
    assert_eq!(AlignmentClass::from_bytes(2), AlignmentClass::Short);
    assert_eq!(AlignmentClass::from_bytes(1), AlignmentClass::Byte);
    assert_eq!(AlignmentClass::from_bytes(3), AlignmentClass::Byte);

    assert_eq!(AlignmentClass::Align256.bytes(), 256);
    assert_eq!(AlignmentClass::Quad.bytes(), 16);
    assert_eq!(AlignmentClass::Word.bytes(), 4);
    assert_eq!(AlignmentClass::Short.bytes(), 2);
    assert_eq!(AlignmentClass::Byte.bytes(), 1);

    // float4 (16 bytes required): Quad and Align256 are compatible, Word is not
    assert!(AlignmentClass::Quad.is_vector_compatible(4, 4));
    assert!(AlignmentClass::Align256.is_vector_compatible(4, 4));
    assert!(!AlignmentClass::Word.is_vector_compatible(4, 4));
    assert!(!AlignmentClass::Byte.is_vector_compatible(4, 4));

    // float2 (8 bytes required): Quad and Align256 compatible
    assert!(AlignmentClass::Quad.is_vector_compatible(2, 4));
    assert!(!AlignmentClass::Word.is_vector_compatible(2, 4));
}

#[test]
fn dtype_policy_id_formatting() {
    let policy = DTypePolicyId::new(DTypeId::F16, DTypeId::F32, DTypeId::F32, DTypeId::F16);
    assert_eq!(policy.storage, DTypeId::F16);
    assert_eq!(policy.compute, DTypeId::F32);
    assert_eq!(policy.tag(), "sF16_cF32_aF32_oF16");
}

#[test]
fn kernel_key_extension_with_signature() {
    let key = KernelKey::cuda_with_signature(
        OperationKind::Pointwise,
        KernelFamily::PointwiseUnary,
        "relu",
        DTypeId::F32,
        LayoutClass::Contiguous,
        KernelAccess::Packed { vector_width: 4 },
        RankClass::Matrix,
        ShapeBucket::from_matrix(64, 64),
        AlignmentClass::Align256,
    )
    .unwrap();

    let cache_id = key.cache_id();
    assert!(cache_id.contains("rank2"));
    assert!(cache_id.contains("p6_s6"));
    assert!(cache_id.contains("align256"));

    let tuning_id = key.tuning_problem_id();
    assert!(tuning_id.contains("rank2"));
    assert!(tuning_id.contains("p6_s6"));
    assert!(tuning_id.contains("align256"));
}

#[test]
fn pointwise_candidate_pruning_filters_illegal_candidates() {
    let policy = DTypePolicyId::new(DTypeId::F32, DTypeId::F32, DTypeId::F32, DTypeId::F32);
    let sig_contiguous = KernelSignature::new(
        policy,
        RankClass::Vector,
        ShapeBucket::from_numel(4096),
        AlignmentClass::Align256,
        LayoutClass::Contiguous,
        OperationKind::Pointwise,
    );

    let candidates = vec![
        LaunchCandidate {
            block_size: 256,
            access: KernelAccess::Scalar { unroll_width: 1 },
        },
        LaunchCandidate {
            block_size: 256,
            access: KernelAccess::Scalar { unroll_width: 4 },
        },
        LaunchCandidate {
            block_size: 256,
            access: KernelAccess::Packed { vector_width: 4 },
        },
        LaunchCandidate {
            block_size: 17, // Invalid block size (not power of two)
            access: KernelAccess::Scalar { unroll_width: 1 },
        },
    ];

    let pruned = prune_pointwise_candidates(&candidates, &sig_contiguous, 4096, 4);
    assert_eq!(pruned.len(), 3);
    assert!(!pruned.iter().any(|c| c.block_size == 17));

    // Strided layout: packed access and unrolled scalar access are pruned
    let sig_strided = KernelSignature::new(
        policy,
        RankClass::Vector,
        ShapeBucket::from_numel(4096),
        AlignmentClass::Align256,
        LayoutClass::Strided,
        OperationKind::Pointwise,
    );

    let pruned_strided = prune_pointwise_candidates(&candidates, &sig_strided, 4096, 4);
    assert_eq!(pruned_strided.len(), 1);
    assert_eq!(
        pruned_strided[0].access,
        KernelAccess::Scalar { unroll_width: 1 }
    );

    // Unaligned memory: packed access (16 bytes) is pruned if alignment is Byte (1 byte)
    let sig_unaligned = KernelSignature::new(
        policy,
        RankClass::Vector,
        ShapeBucket::from_numel(4096),
        AlignmentClass::Byte,
        LayoutClass::Contiguous,
        OperationKind::Pointwise,
    );

    let pruned_unaligned = prune_pointwise_candidates(&candidates, &sig_unaligned, 4096, 4);
    assert!(
        !pruned_unaligned
            .iter()
            .any(|c| matches!(c.access, KernelAccess::Packed { .. }))
    );
}

#[test]
fn reduction_candidate_pruning_filters_strided_warp_reduction() {
    let policy = DTypePolicyId::new(DTypeId::F32, DTypeId::F32, DTypeId::F32, DTypeId::F32);
    let sig_strided = KernelSignature::new(
        policy,
        RankClass::Matrix,
        ShapeBucket::from_matrix(64, 64),
        AlignmentClass::Align256,
        LayoutClass::Strided,
        OperationKind::Reduction,
    );

    let candidates = vec![
        LaunchCandidate {
            block_size: 256,
            access: KernelAccess::Scalar { unroll_width: 1 },
        },
        LaunchCandidate {
            block_size: 256,
            access: KernelAccess::WarpReduction,
        },
    ];

    let pruned = prune_reduction_candidates(&candidates, &sig_strided, 64);
    assert_eq!(pruned.len(), 1);
    assert_eq!(pruned[0].access, KernelAccess::Scalar { unroll_width: 1 });
}

#[test]
fn matmul_candidate_pruning_filters_excessive_shared_memory() {
    let policy = DTypePolicyId::new(DTypeId::F32, DTypeId::F32, DTypeId::F32, DTypeId::F32);
    let sig = KernelSignature::new(
        policy,
        RankClass::Matrix,
        ShapeBucket::from_gemm(1024, 1024, 1024),
        AlignmentClass::Align256,
        LayoutClass::Contiguous,
        OperationKind::MatMul,
    );

    let candidates = vec![
        MatMulTileCandidate::new(16, 16, 16, 256),
        MatMulTileCandidate::new(256, 256, 256, 256), // Shared memory: (256*256 + 256*256)*4 = 524,288 bytes > 49,152
    ];

    let pruned = prune_matmul_candidates(&candidates, &sig, 1024, 1024, 1024, 4);
    assert_eq!(pruned.len(), 1);
    assert_eq!(pruned[0].tile_m, 16);
}

#[test]
fn pruning_fallback_ensures_non_empty_candidate_set() {
    let policy = DTypePolicyId::new(DTypeId::F32, DTypeId::F32, DTypeId::F32, DTypeId::F32);
    let sig = KernelSignature::new(
        policy,
        RankClass::Vector,
        ShapeBucket::from_numel(10),
        AlignmentClass::Byte,
        LayoutClass::Strided,
        OperationKind::Pointwise,
    );

    let empty_set: Vec<LaunchCandidate> = vec![];
    let pruned = prune_pointwise_candidates(&empty_set, &sig, 10, 4);
    assert!(!pruned.is_empty());
    assert_eq!(pruned[0].access, KernelAccess::Scalar { unroll_width: 1 });
}

#[test]
fn signature_compile_fail_tests() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/tuning_pruning_compile_fail/*.rs");
}
