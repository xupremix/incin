#![cfg(feature = "cpu")]

use std::collections::BTreeSet;

use incin_backends::capability::{
    CPU_CAPABILITIES, CUDA_CAPABILITIES, WGPU_CAPABILITIES, registry, support,
};
use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::backend_authoring::{Backend, HostInterop};
#[cfg(feature = "cuda")]
use incin_core::backend_authoring::StorageBackend;
use incin_core::exec::catalog::{
    ArangeAttributes, AxisVarianceAttributes, ChunkAttributes, CreationAttributes, DataAttributes,
    DropoutAttributes, EpsilonAttributes, FullAttributes, LinearAttributes, LinspaceAttributes,
    LossAttributes, LossReduction, NoAttributes, NormAttributes, SplitAttributes,
    VarianceAttributes, op,
};
use incin_core::exec::{
    Capabilities, CapabilityQuery, DTypeRule, ExecutionContext, GradMode, ImplementationKind,
    LayoutClass, MathMode, OPERATION_CATALOG, OperationIdentity, SupportLevel, TensorHandle,
    UnsupportedReason, dispatch,
};
use incin_core::prelude::{
    Cpu, DType, DTypeDescriptor, DTypeId, DeviceId, DeviceKind, Dyn, Local, OperationKind, Q8_0,
    Reduction, ShapeBuf, ShapeValue,
};
use incin_core::__backend_compat::legacy::{FloatOps, ModuleOps, QuantizedOps, ReductionOps, TensorOps};

use incin_core::tensor::arg_into::ArgInto;

fn query(
    operation: OperationKind,
    dtype: impl ArgInto<DTypeDescriptor>,
    layout: LayoutClass,
    rank: usize,
) -> CapabilityQuery {
    CapabilityQuery {
        operation: OperationIdentity::Builtin(operation),
        dtype: dtype.into_arg(),
        layout,
        rank,
        training: false,
        math_mode: MathMode::Precise,
    }
}

#[test]
fn every_registration_generates_supported_boundary_cases_without_fallback() {
    #[cfg(feature = "metal")]
    use incin_backends::capability::METAL_CAPABILITIES;

    #[cfg(not(feature = "metal"))]
    let devices: Vec<(DeviceKind, &[incin_core::exec::CapabilityRule])> = vec![
        (DeviceKind::Cpu, CPU_CAPABILITIES),
        (DeviceKind::Cuda, CUDA_CAPABILITIES),
        (DeviceKind::Wgpu, WGPU_CAPABILITIES),
    ];
    #[cfg(feature = "metal")]
    let mut devices: Vec<(DeviceKind, &[incin_core::exec::CapabilityRule])> = vec![
        (DeviceKind::Cpu, CPU_CAPABILITIES),
        (DeviceKind::Cuda, CUDA_CAPABILITIES),
        (DeviceKind::Wgpu, WGPU_CAPABILITIES),
    ];
    #[cfg(feature = "metal")]
    devices.push((DeviceKind::Metal, METAL_CAPABILITIES));

    for (device, rules) in devices {
        assert!(!rules.is_empty());
        for rule in rules {
            assert!(!rule.dtypes.is_empty());
            assert!(!rule.layouts.is_empty());
            assert!(!rule.math_modes.is_empty());
            assert!(rule.min_rank <= rule.max_rank);
            assert_ne!(rule.implementation, ImplementationKind::Fallback);

            for &dtype in rule.dtypes {
                for &layout in rule.layouts {
                    for &math_mode in rule.math_modes {
                        for rank in [rule.min_rank, rule.max_rank] {
                            let mut case = query(rule.operation, dtype, layout, rank);
                            case.math_mode = math_mode;
                            assert_eq!(
                                support(device, &case),
                                rule.implementation.into(),
                                "{device:?} {case:?}"
                            );
                            if rule.training {
                                case.training = true;
                                assert_eq!(
                                    support(device, &case),
                                    rule.implementation.into(),
                                    "{device:?} training {case:?}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn unsupported_matrix_cells_return_the_documented_constraint() {
    let mut case = query(
        OperationKind::Reduction,
        DTypeId::F64,
        LayoutClass::Contiguous,
        2,
    );
    let level = support(DeviceKind::Cpu, &case);
    if let SupportLevel::Unsupported(UnsupportedReason::DType { dtype, .. }) = level {
        assert_eq!(dtype, DTypeId::F64.descriptor());
    } else {
        panic!("expected UnsupportedReason::DType, got {:?}", level);
    }

    case = query(
        OperationKind::Pointwise,
        DTypeId::F32,
        LayoutClass::Strided,
        2,
    );
    assert!(matches!(
        support(DeviceKind::Wgpu, &case),
        SupportLevel::Unsupported(UnsupportedReason::Layout {
            layout: LayoutClass::Strided,
            ..
        })
    ));

    case = query(
        OperationKind::Conv2d,
        DTypeId::F32,
        LayoutClass::Contiguous,
        2,
    );
    assert!(matches!(
        support(DeviceKind::Cuda, &case),
        SupportLevel::Unsupported(UnsupportedReason::Rank { min: 3, max: 4, .. })
    ));

    case = query(
        OperationKind::Pointwise,
        DTypeId::F32,
        LayoutClass::Contiguous,
        2,
    );
    case.math_mode = MathMode::Fast;
    assert!(matches!(
        support(DeviceKind::Cpu, &case),
        SupportLevel::Unsupported(UnsupportedReason::MathMode {
            math_mode: MathMode::Fast,
            ..
        })
    ));

    case = query(
        OperationKind::Fill,
        DTypeId::F32,
        LayoutClass::Contiguous,
        2,
    );
    case.training = true;
    assert!(matches!(
        support(DeviceKind::Cpu, &case),
        SupportLevel::Unsupported(UnsupportedReason::Training { .. })
    ));

    case = query(
        OperationKind::Reshape,
        DTypeId::Q8_0,
        LayoutClass::Strided,
        1,
    );
    assert_eq!(
        support(DeviceKind::Cpu, &case),
        SupportLevel::Unsupported(UnsupportedReason::Layout {
            operation: OperationKind::Reshape,
            layout: LayoutClass::Strided,
        })
    );

    case = query(
        OperationKind::Broadcast,
        DTypeId::Q8_0,
        LayoutClass::Contiguous,
        2,
    );
    case.training = true;
    assert_eq!(
        support(DeviceKind::Cpu, &case),
        SupportLevel::Unsupported(UnsupportedReason::Training {
            operation: OperationKind::Broadcast,
        })
    );

    case = query(
        OperationKind::Pointwise,
        DTypeId::F32,
        LayoutClass::Strided,
        2,
    );
    assert_eq!(
        support(DeviceKind::Cuda, &case),
        SupportLevel::Unsupported(UnsupportedReason::Layout {
            operation: OperationKind::Pointwise,
            layout: LayoutClass::Strided,
        })
    );

    case = query(
        OperationKind::Normalization,
        DTypeId::F32,
        LayoutClass::Contiguous,
        0,
    );
    assert_eq!(
        support(DeviceKind::Wgpu, &case),
        SupportLevel::Unsupported(UnsupportedReason::Rank {
            operation: OperationKind::Normalization,
            rank: 0,
            min: 1,
            max: usize::MAX,
        })
    );
}

fn f32_storage(shape: &[usize], values: &[f32]) -> CpuStorage {
    CpuStorage::try_from_contiguous(CpuBuffer::F32(values.to_vec()), shape.to_vec()).unwrap()
}

fn transpose_if_requested(storage: CpuStorage, layout: LayoutClass) -> CpuStorage {
    if layout == LayoutClass::Strided {
        storage.transpose(0, 1).unwrap()
    } else {
        storage
    }
}

/// Operand shapes for one tensor-family probe.
///
/// Declared once because the layout probe and the dtype probe must build the
/// same operands from different storage; two copies would let a row pass one
/// probe on a shape the other never tries.
fn cpu_tensor_operand_shapes(operation: OperationKind) -> &'static [&'static [usize]] {
    match operation {
        OperationKind::WhereCond
        | OperationKind::Scatter
        | OperationKind::Addmm
        | OperationKind::ScaledDotProductAttention => &[&[2, 2], &[2, 2], &[2, 2]],
        OperationKind::ConcatExact | OperationKind::StackExact | OperationKind::Gather => {
            &[&[2, 2], &[2, 2]]
        }
        OperationKind::IndexSelect => &[&[2, 2], &[1]],
        OperationKind::PixelShuffle => &[&[1, 4, 2, 2]],
        OperationKind::GroupNorm => &[&[1, 4, 2]],
        OperationKind::InstanceNorm => &[&[1, 4, 2, 2]],
        OperationKind::Unfold => &[&[2, 4]],
        OperationKind::SliceExact
        | OperationKind::Repeat
        | OperationKind::Pad
        | OperationKind::BroadcastLeft => &[&[2, 2]],
        OperationKind::Maximum
        | OperationKind::Minimum
        | OperationKind::AbsDiff
        | OperationKind::Lerp
        | OperationKind::MaskedFill
        | OperationKind::CmpEq
        | OperationKind::CmpNe
        | OperationKind::CmpLt
        | OperationKind::CmpLe
        | OperationKind::CmpGt
        | OperationKind::CmpGe
        | OperationKind::LogicalAnd
        | OperationKind::LogicalOr => &[&[2, 2], &[2, 2]],
        OperationKind::BatchedMatMul => &[&[1, 2, 2], &[1, 2, 2]],
        // `squeeze` needs an axis of extent one to remove, so its operand is
        // the one shape in this family that is not the probe matrix.
        OperationKind::SqueezeExact => &[&[1, 2]],
        OperationKind::LogicalNot
        | OperationKind::SubScalar
        | OperationKind::DivScalar
        | OperationKind::TransposeExact
        | OperationKind::Narrow
        | OperationKind::FlattenExact
        | OperationKind::UnsqueezeExact
        | OperationKind::Triu
        | OperationKind::Tril
        | OperationKind::Diag
        | OperationKind::ArgMax
        | OperationKind::ArgMin
        | OperationKind::Argsort
        | OperationKind::Cumsum => &[&[2, 2]],
        _ => panic!("missing CPU tensor operand shapes for {operation}"),
    }
}

/// Run one tensor-family operation over already-built operands.
///
/// Generic over the storage type parameter so the `f32` layout probe and the
/// `Dyn` dtype probe share one dispatch; the CPU storage handle is the same
/// type either way, and two copies of this match would drift.
fn cpu_tensor_probe<K: DType>(operation: OperationKind, operands: &[&CpuStorage]) -> CpuStorage {
    type B = CpuBackendImpl;
    let first = operands[0];
    match operation {
        OperationKind::Maximum => B::maximum::<K>(first, operands[1]).unwrap(),
        OperationKind::Minimum => B::minimum::<K>(first, operands[1]).unwrap(),
        OperationKind::AbsDiff => B::abs_diff::<K>(first, operands[1]).unwrap(),
        OperationKind::Lerp => B::lerp::<K>(first, operands[1], 0.5).unwrap(),
        OperationKind::MaskedFill => B::masked_fill::<K>(first, operands[1], 0.0).unwrap(),
        OperationKind::WhereCond => B::where_cond::<K>(first, operands[1], operands[2]).unwrap(),
        OperationKind::CmpEq => B::cmp_eq::<K>(first, operands[1]).unwrap(),
        OperationKind::CmpNe => B::cmp_ne::<K>(first, operands[1]).unwrap(),
        OperationKind::CmpLt => B::cmp_lt::<K>(first, operands[1]).unwrap(),
        OperationKind::CmpLe => B::cmp_le::<K>(first, operands[1]).unwrap(),
        OperationKind::CmpGt => B::cmp_gt::<K>(first, operands[1]).unwrap(),
        OperationKind::CmpGe => B::cmp_ge::<K>(first, operands[1]).unwrap(),
        OperationKind::LogicalAnd => B::logical_and(first, operands[1]).unwrap(),
        OperationKind::LogicalOr => B::logical_or(first, operands[1]).unwrap(),
        OperationKind::LogicalNot => B::logical_not(first).unwrap(),
        OperationKind::SubScalar => B::sub_scalar::<K>(first, 1.0).unwrap(),
        OperationKind::DivScalar => B::div_scalar::<K>(first, 2.0).unwrap(),
        OperationKind::TransposeExact => B::transpose::<K>(first, 0, 1).unwrap(),
        OperationKind::Narrow => B::narrow::<K>(first, 0, 0, 1).unwrap(),
        OperationKind::FlattenExact => B::flatten::<K>(first, 0, 1).unwrap(),
        OperationKind::SqueezeExact => B::squeeze::<K>(first, 0).unwrap(),
        OperationKind::UnsqueezeExact => B::unsqueeze::<K>(first, 0).unwrap(),
        OperationKind::Triu => B::triu::<K>(first, 0).unwrap(),
        OperationKind::Tril => B::tril::<K>(first, 0).unwrap(),
        OperationKind::Diag => B::diag::<K>(first, 0).unwrap(),
        OperationKind::BatchedMatMul => B::bmm::<K>(first, operands[1]).unwrap(),
        OperationKind::ConcatExact => B::concat::<K>(&[first, operands[1]], 0).unwrap(),
        OperationKind::StackExact => B::stack::<K>(&[first, operands[1]], 0).unwrap(),
        OperationKind::SliceExact => B::slice::<K>(first, &[(0, 1), (0, 2)]).unwrap(),
        OperationKind::Gather => B::gather::<K, K>(first, 1, operands[1]).unwrap(),
        OperationKind::Scatter => B::scatter::<K, K>(first, 1, operands[1], operands[2]).unwrap(),
        OperationKind::IndexSelect => B::index_select::<K, K>(first, 0, operands[1]).unwrap(),
        OperationKind::Repeat => B::repeat::<K>(first, &[2, 1]).unwrap(),
        OperationKind::Pad => B::pad::<K>(first, &[(1, 1), (0, 0)], 0.0).unwrap(),
        OperationKind::Unfold => B::unfold::<K>(first, 1, 2, 1).unwrap(),
        OperationKind::PixelShuffle => B::pixel_shuffle::<K>(first, 2).unwrap(),
        OperationKind::GroupNorm => B::group_norm::<K>(first, 2, 1e-5).unwrap(),
        OperationKind::InstanceNorm => B::instance_norm::<K>(first, 1e-5).unwrap(),
        OperationKind::BroadcastLeft => B::broadcast_left::<K>(first, &[3]).unwrap(),
        OperationKind::Addmm => B::addmm::<K>(first, operands[1], operands[2], 1.0, 1.0).unwrap(),
        OperationKind::ScaledDotProductAttention => {
            B::scaled_dot_product_attention::<K>(first, operands[1], operands[2], None, None)
                .unwrap()
        }
        OperationKind::ArgMax => B::argmax::<K, i64>(first, Some(1)).unwrap(),
        OperationKind::ArgMin => B::argmin::<K, i64>(first, Some(1)).unwrap(),
        OperationKind::Argsort => B::argsort::<K, i64>(first, 1, false).unwrap(),
        OperationKind::Cumsum => B::cumsum::<K>(first, 1).unwrap(),
        _ => panic!("missing CPU tensor probe for {operation}"),
    }
}

/// Build probe storage with `shape` under the requested layout class.
///
/// A strided probe has to carry the same logical shape as its contiguous
/// counterpart, or the two layout cases would be exercising different
/// operations. Materialising the last two axes swapped and transposing them
/// back keeps the shape and changes the strides.
/// Every element is `1.0`. These probes answer "does the advertised row
/// execute and produce the declared shape, dtype and device", not "is the
/// arithmetic right"; numerical agreement is `canonical_cpu`'s job. A uniform
/// fill also keeps the indexing operations' index operands inside their
/// extents without a second, position-aware value table.
fn laid_out(shape: &[usize], layout: LayoutClass) -> CpuStorage {
    let count: usize = shape.iter().product();
    let values = vec![1.0f32; count];
    let rank = shape.len();
    // A rank-one operand has no axis pair to permute, so it has no strided
    // form that keeps its shape. Those operands are index vectors, whose
    // layout is not what the row under test describes.
    if layout != LayoutClass::Strided || rank < 2 {
        return f32_storage(shape, &values);
    }
    let mut swapped = shape.to_vec();
    swapped.swap(rank - 2, rank - 1);
    f32_storage(&swapped, &values)
        .transpose(rank - 2, rank - 1)
        .unwrap()
}

fn cpu_probe_shape(operation: OperationKind) -> &'static [usize] {
    match operation {
        OperationKind::Storage
        | OperationKind::Fill
        | OperationKind::Random
        | OperationKind::Pointwise
        | OperationKind::MatMul => &[2, 2],
        OperationKind::Add | OperationKind::Sub | OperationKind::Mul | OperationKind::Div => {
            &[2, 2]
        }
        OperationKind::Reduction => &[],
        OperationKind::SumAll
        | OperationKind::MeanAll
        | OperationKind::MaxAll
        | OperationKind::MinAll
        | OperationKind::ProdAll => &[],
        OperationKind::SumDim
        | OperationKind::MeanDim
        | OperationKind::MaxDim
        | OperationKind::MinDim
        | OperationKind::ProdDim => &[2],
        OperationKind::SumKeepDim
        | OperationKind::MeanKeepDim
        | OperationKind::MaxKeepDim
        | OperationKind::MinKeepDim => &[2, 1],
        // The probe reduces a [2, 2] matrix along axis one. `argmax` and
        // `argmin` drop that axis, `topk` narrows it to k, and `argsort` and
        // `cumsum` keep the operand's shape.
        OperationKind::ArgMax | OperationKind::ArgMin => &[2],
        OperationKind::TopK => &[2, 1],
        OperationKind::Argsort | OperationKind::Cumsum => &[2, 2],
        OperationKind::Normalization => &[1, 2],
        OperationKind::Broadcast => &[3, 2],
        OperationKind::BroadcastAs => &[3, 2],
        OperationKind::Reshape => &[4],
        OperationKind::ReshapeExact => &[4],
        OperationKind::MatMulExact => &[2, 2],
        // The float family runs elementwise over the probe matrix, so each of
        // these preserves its operand's shape.
        OperationKind::Relu
        | OperationKind::Step
        | OperationKind::Mish
        | OperationKind::Elu
        | OperationKind::Gelu
        | OperationKind::Abs
        | OperationKind::Exp
        | OperationKind::Neg
        | OperationKind::Sqrt
        | OperationKind::Log
        | OperationKind::Tanh
        | OperationKind::Sigmoid
        | OperationKind::Swish
        | OperationKind::Sign
        | OperationKind::Floor
        | OperationKind::Ceil
        | OperationKind::Round
        | OperationKind::Log2
        | OperationKind::Log10
        | OperationKind::Sin
        | OperationKind::Cos
        | OperationKind::Tan
        | OperationKind::Asin
        | OperationKind::Acos
        | OperationKind::Atan
        | OperationKind::Sinh
        | OperationKind::Cosh
        | OperationKind::Asinh
        | OperationKind::Acosh
        | OperationKind::Atanh
        | OperationKind::Erf
        | OperationKind::Rsqrt
        | OperationKind::Trunc
        | OperationKind::Frac
        | OperationKind::AddScalar
        | OperationKind::MulScalar
        | OperationKind::Powf
        | OperationKind::Atan2
        | OperationKind::Fmod
        | OperationKind::Remainder
        | OperationKind::Clamp
        | OperationKind::Softmax => &[2, 2],
        // The tensor family, in the same order its capability groups declare
        // it. Every shape here is the one the probe above actually produces.
        OperationKind::Maximum
        | OperationKind::Minimum
        | OperationKind::AbsDiff
        | OperationKind::Lerp
        | OperationKind::MaskedFill
        | OperationKind::WhereCond
        | OperationKind::CmpEq
        | OperationKind::CmpNe
        | OperationKind::CmpLt
        | OperationKind::CmpLe
        | OperationKind::CmpGt
        | OperationKind::CmpGe
        | OperationKind::LogicalAnd
        | OperationKind::LogicalOr
        | OperationKind::LogicalNot
        | OperationKind::SubScalar
        | OperationKind::DivScalar
        | OperationKind::TransposeExact
        | OperationKind::Triu
        | OperationKind::Tril => &[2, 2],
        OperationKind::Narrow => &[1, 2],
        OperationKind::FlattenExact => &[4],
        OperationKind::SqueezeExact | OperationKind::Diag => &[2],
        OperationKind::UnsqueezeExact => &[1, 2, 2],
        OperationKind::BatchedMatMul => &[1, 2, 2],
        OperationKind::ConcatExact | OperationKind::Repeat | OperationKind::Pad => &[4, 2],
        OperationKind::StackExact => &[2, 2, 2],
        OperationKind::SliceExact | OperationKind::IndexSelect => &[1, 2],
        OperationKind::Gather
        | OperationKind::Scatter
        | OperationKind::Addmm
        | OperationKind::ScaledDotProductAttention => &[2, 2],
        OperationKind::Unfold => &[2, 3, 2],
        OperationKind::PixelShuffle => &[1, 1, 4, 4],
        OperationKind::GroupNorm => &[1, 4, 2],
        OperationKind::InstanceNorm => &[1, 4, 2, 2],
        OperationKind::BroadcastLeft => &[3, 2, 2],
        OperationKind::Conv2d => &[1, 1, 2, 2],
        OperationKind::Conv2dExact => &[1, 1, 2, 2],
        OperationKind::Pool2d => &[1, 1, 1, 1],
        OperationKind::MaxPool2d | OperationKind::AvgPool2d => &[1, 1, 1, 1],
        OperationKind::Conv1dExact => &[1, 1, 2],
        OperationKind::ConvTranspose2d => &[1, 1, 2, 2],
        OperationKind::AdaptiveAvgPool2dExact => &[1, 1, 1, 1],
        OperationKind::LayerNorm | OperationKind::RmsNorm | OperationKind::Linear => &[2, 2],
        OperationKind::Dropout => &[2, 2],
        OperationKind::Zeros
        | OperationKind::Ones
        | OperationKind::UniformRandom
        | OperationKind::NormalRandom
        | OperationKind::Full
        | OperationKind::VariableZeros
        | OperationKind::VariableOnes
        | OperationKind::VariableUniformRandom
        | OperationKind::VariableNormalRandom => &[2, 2],
        // The ranged fills are one-dimensional by definition.
        OperationKind::Arange | OperationKind::Linspace => &[4],
        OperationKind::TensorFromData | OperationKind::TensorFromBytes => &[2],
        OperationKind::ToDType => &[2, 2],
        // Probed with `Mean`, so the result is the scalar the reduction
        // produces rather than the elementwise buffer feeding it.
        OperationKind::MseLoss | OperationKind::L1Loss | OperationKind::BceWithLogitsLoss => &[],
        OperationKind::VarianceAll | OperationKind::StdAll | OperationKind::Norm => &[],
        OperationKind::Dot => &[],
        // Thirty-two elements is one Q8_0 block, the smallest buffer the
        // compression accepts.
        OperationKind::Quantize | OperationKind::Dequantize => &[32],
        OperationKind::QuantizedMatMul => &[1, 1],
        OperationKind::Outer => &[2, 2],
        // The first piece of the two each probe asks for.
        OperationKind::Chunk | OperationKind::Split => &[2, 1],
        OperationKind::VarianceDim | OperationKind::StdDim => &[2],
        OperationKind::VarianceKeepDim | OperationKind::StdKeepDim => &[2, 1],
        OperationKind::BatchNorm => &[1, 2, 2],
        // The probe's index vector is `[2]` and its weight table is `[3, 2]`;
        // the gather appends the weight's hidden axis to the index shape.
        OperationKind::EmbeddingExact => &[2, 2],
        // The probe reduces with `Mean`, which is a scalar. The `None` form
        // would be `[batch]`, but the reduction is an attribute rather than
        // part of the identity, so one shape is stated and the probe picks
        // the mode that produces it.
        OperationKind::CrossEntropyLoss => &[],
        _ => panic!("missing CPU expected shape for {operation}"),
    }
}

/// Run one float-family operation over `input`.
///
/// Split out because the same dispatch is needed by both the layout probe and
/// the dtype probe, and two copies of a 42-arm match would drift.
fn cpu_float_probe(operation: OperationKind, input: &CpuStorage) -> CpuStorage {
    type B = CpuBackendImpl;
    match operation {
        OperationKind::Relu => B::relu::<f32>(input).unwrap(),
        OperationKind::Step => B::step::<f32>(input).unwrap(),
        OperationKind::Mish => B::mish::<f32>(input).unwrap(),
        OperationKind::Elu => B::elu::<f32>(input).unwrap(),
        OperationKind::Gelu => B::gelu::<f32>(input).unwrap(),
        OperationKind::Abs => B::abs::<f32>(input).unwrap(),
        OperationKind::Exp => B::exp::<f32>(input).unwrap(),
        OperationKind::Neg => B::neg::<f32>(input).unwrap(),
        OperationKind::Sqrt => B::sqrt::<f32>(input).unwrap(),
        OperationKind::Log => B::log::<f32>(input).unwrap(),
        OperationKind::Tanh => B::tanh::<f32>(input).unwrap(),
        OperationKind::Sigmoid => B::sigmoid::<f32>(input).unwrap(),
        OperationKind::Swish => B::swish::<f32>(input).unwrap(),
        OperationKind::Sign => B::sign::<f32>(input).unwrap(),
        OperationKind::Floor => B::floor::<f32>(input).unwrap(),
        OperationKind::Ceil => B::ceil::<f32>(input).unwrap(),
        OperationKind::Round => B::round::<f32>(input).unwrap(),
        OperationKind::Log2 => B::log2::<f32>(input).unwrap(),
        OperationKind::Log10 => B::log10::<f32>(input).unwrap(),
        OperationKind::Sin => B::sin::<f32>(input).unwrap(),
        OperationKind::Cos => B::cos::<f32>(input).unwrap(),
        OperationKind::Tan => B::tan::<f32>(input).unwrap(),
        OperationKind::Asin => B::asin::<f32>(input).unwrap(),
        OperationKind::Acos => B::acos::<f32>(input).unwrap(),
        OperationKind::Atan => B::atan::<f32>(input).unwrap(),
        OperationKind::Sinh => B::sinh::<f32>(input).unwrap(),
        OperationKind::Cosh => B::cosh::<f32>(input).unwrap(),
        OperationKind::Asinh => B::asinh::<f32>(input).unwrap(),
        OperationKind::Acosh => B::acosh::<f32>(input).unwrap(),
        OperationKind::Atanh => B::atanh::<f32>(input).unwrap(),
        OperationKind::Erf => B::erf::<f32>(input).unwrap(),
        OperationKind::Rsqrt => B::rsqrt::<f32>(input).unwrap(),
        OperationKind::Trunc => B::trunc::<f32>(input).unwrap(),
        OperationKind::Frac => B::frac::<f32>(input).unwrap(),
        OperationKind::AddScalar => B::add_scalar_float::<f32>(input, 2.0).unwrap(),
        OperationKind::MulScalar => B::mul_scalar_float::<f32>(input, 2.0).unwrap(),
        OperationKind::Powf => B::powf::<f32>(input, 2.0).unwrap(),
        OperationKind::Clamp => B::clamp::<f32>(input, 0.0, 1.0).unwrap(),
        OperationKind::Softmax => B::softmax::<f32>(input, 1).unwrap(),
        OperationKind::Atan2 => B::atan2::<f32>(input, input).unwrap(),
        OperationKind::Fmod => B::fmod::<f32>(input, input).unwrap(),
        OperationKind::Remainder => B::remainder::<f32>(input, input).unwrap(),
        other => panic!("{other} is not a float-family operation"),
    }
}

/// Dispatch one allocation of `dtype`.
///
/// Separate from the probe below because it is the only family with no operand
/// to build, so it shares none of that function's setup, and because both the
/// layout probe and the dtype probe need it.
fn allocation_probe(operation: OperationKind, dtype: impl ArgInto<DTypeDescriptor>) -> CpuStorage {
    let dtype = dtype.into_arg();
    let context =
        ExecutionContext::new(CpuBackendImpl::<Cpu>::new()).with_grad_mode(GradMode::Disabled);
    let shape = cpu_probe_shape(operation).to_vec();
    let device = DeviceId::cpu();
    let plain = CreationAttributes {
        shape: shape.clone(),
        dtype,
        device,
    };
    match operation {
        OperationKind::Zeros => dispatch::execute::<op::Zeros, _>(&context, plain, &[]).unwrap(),
        OperationKind::Ones => dispatch::execute::<op::Ones, _>(&context, plain, &[]).unwrap(),
        OperationKind::UniformRandom => {
            dispatch::execute::<op::UniformRandom, _>(&context, plain, &[]).unwrap()
        }
        OperationKind::NormalRandom => {
            dispatch::execute::<op::NormalRandom, _>(&context, plain, &[]).unwrap()
        }
        OperationKind::Full => dispatch::execute::<op::Full, _>(
            &context,
            FullAttributes {
                shape,
                dtype,
                device,
                value: 1.0,
            },
            &[],
        )
        .unwrap(),
        OperationKind::Arange => dispatch::execute::<op::Arange, _>(
            &context,
            ArangeAttributes {
                shape,
                dtype,
                device,
                start: 0.0,
                step: 1.0,
            },
            &[],
        )
        .unwrap(),
        // The variable forms hand back a `CpuVar`, so their value is read out
        // before the shared assertions see it. That the executor's output type
        // differs at all is the point of `Execute` naming it as an associated
        // type; the probe just has to follow.
        OperationKind::VariableZeros => CpuBackendImpl::<Cpu>::var_as_tensor::<Dyn>(
            &dispatch::execute::<op::VariableZeros, _>(&context, plain, &[]).unwrap(),
        )
        .unwrap(),
        OperationKind::VariableOnes => CpuBackendImpl::<Cpu>::var_as_tensor::<Dyn>(
            &dispatch::execute::<op::VariableOnes, _>(&context, plain, &[]).unwrap(),
        )
        .unwrap(),
        OperationKind::VariableUniformRandom => CpuBackendImpl::<Cpu>::var_as_tensor::<Dyn>(
            &dispatch::execute::<op::VariableUniformRandom, _>(&context, plain, &[]).unwrap(),
        )
        .unwrap(),
        OperationKind::VariableNormalRandom => CpuBackendImpl::<Cpu>::var_as_tensor::<Dyn>(
            &dispatch::execute::<op::VariableNormalRandom, _>(&context, plain, &[]).unwrap(),
        )
        .unwrap(),
        _ => dispatch::execute::<op::Linspace, _>(
            &context,
            LinspaceAttributes {
                shape,
                dtype,
                device,
                start: 0.0,
                end: 3.0,
            },
            &[],
        )
        .unwrap(),
    }
}

fn execute_cpu_probe(operation: OperationKind, layout: LayoutClass) -> CpuStorage {
    type B = CpuBackendImpl;
    let device = DeviceId::cpu();
    match operation {
        OperationKind::Storage => {
            let storage =
                transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            assert_eq!(B::to_bytes::<f32>(&storage).unwrap().len(), 16);
            storage
        }
        OperationKind::Fill => {
            B::zeros::<f32>(&[2, 2], DTypeId::F32.descriptor(), &device).unwrap()
        }
        OperationKind::Random => {
            B::rand::<f32>(&[2, 2], DTypeId::F32.descriptor(), &device).unwrap()
        }
        OperationKind::Pointwise => {
            let lhs = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            let rhs = transpose_if_requested(f32_storage(&[2, 2], &[4.0, 3.0, 2.0, 1.0]), layout);
            B::add::<f32>(&lhs, &rhs).unwrap()
        }
        OperationKind::Add | OperationKind::Sub | OperationKind::Mul | OperationKind::Div => {
            let lhs = f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let rhs = f32_storage(&[2, 2], &[4.0, 3.0, 2.0, 1.0]);
            match operation {
                OperationKind::Add => B::add::<f32>(&lhs, &rhs).unwrap(),
                OperationKind::Sub => B::sub::<f32>(&lhs, &rhs).unwrap(),
                OperationKind::Mul => B::mul::<f32>(&lhs, &rhs).unwrap(),
                OperationKind::Div => B::div::<f32>(&lhs, &rhs).unwrap(),
                _ => unreachable!(),
            }
        }
        OperationKind::Reduction => {
            let input = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            B::sum_all::<f32>(&input).unwrap()
        }
        OperationKind::SumAll
        | OperationKind::MeanAll
        | OperationKind::MaxAll
        | OperationKind::MinAll
        | OperationKind::ProdAll
        | OperationKind::SumDim
        | OperationKind::SumKeepDim
        | OperationKind::MeanDim
        | OperationKind::MeanKeepDim
        | OperationKind::MaxDim
        | OperationKind::MaxKeepDim
        | OperationKind::MinDim
        | OperationKind::MinKeepDim
        | OperationKind::ProdDim => {
            let input = f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
            match operation {
                OperationKind::SumAll => B::sum_all::<f32>(&input).unwrap(),
                OperationKind::MeanAll => B::mean_all::<f32>(&input).unwrap(),
                OperationKind::MaxAll => B::max_all::<f32>(&input).unwrap(),
                OperationKind::MinAll => B::min_all::<f32>(&input).unwrap(),
                OperationKind::ProdAll => B::prod_all::<f32>(&input).unwrap(),
                OperationKind::SumDim => B::sum_dim::<f32>(&input, 1).unwrap(),
                OperationKind::SumKeepDim => B::sum_keepdim::<f32>(&input, 1).unwrap(),
                OperationKind::MeanDim => B::mean_dim::<f32>(&input, 1).unwrap(),
                OperationKind::MeanKeepDim => B::mean_keepdim::<f32>(&input, 1).unwrap(),
                OperationKind::MaxDim => B::max_dim::<f32>(&input, 1).unwrap(),
                OperationKind::MaxKeepDim => B::max_keepdim::<f32>(&input, 1).unwrap(),
                OperationKind::MinDim => B::min_dim::<f32>(&input, 1).unwrap(),
                OperationKind::MinKeepDim => B::min_keepdim::<f32>(&input, 1).unwrap(),
                OperationKind::ProdDim => B::prod_dim::<f32>(&input, 1).unwrap(),
                _ => unreachable!(),
            }
        }
        // Only the value tensor is probed here. The index tensor's dtype is not
        // the operand dtype the row advertises, so it is checked where that
        // distinction is made rather than folded into this shape probe.
        OperationKind::TopK => {
            let input = laid_out(&[2, 2], layout);
            B::topk::<f32, u32>(&input, 1, 1, true).unwrap().0
        }
        OperationKind::Normalization => {
            let input = f32_storage(&[1, 2], &[1.0, 3.0]);
            let weight = f32_storage(&[2], &[1.0, 1.0]);
            B::layer_norm::<f32>(&input, &weight, None, 1e-5).unwrap()
        }
        OperationKind::Broadcast | OperationKind::BroadcastAs => {
            let input = if layout == LayoutClass::Strided {
                f32_storage(&[2, 1], &[1.0, 2.0]).transpose(0, 1).unwrap()
            } else {
                f32_storage(&[1, 2], &[1.0, 2.0])
            };
            B::broadcast_as::<f32>(&input, &[3, 2]).unwrap()
        }
        OperationKind::Reshape | OperationKind::ReshapeExact => {
            let input = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            B::reshape::<f32>(&input, &[4]).unwrap()
        }
        OperationKind::MatMul | OperationKind::MatMulExact => {
            let lhs = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            let rhs = f32_storage(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
            B::matmul::<f32>(&lhs, &rhs).unwrap()
        }
        OperationKind::Conv2d | OperationKind::Conv2dExact => {
            let input = f32_storage(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let weight = f32_storage(&[1, 1, 1, 1], &[1.0]);
            B::conv2d::<f32>(&input, &weight, None, 1, 0, 1, 1).unwrap()
        }
        OperationKind::Pool2d | OperationKind::MaxPool2d => {
            let input = f32_storage(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            B::max_pool2d::<f32>(&input, (2, 2), (1, 1), (0, 0), (1, 1)).unwrap()
        }
        OperationKind::AvgPool2d => {
            let input = f32_storage(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            B::avg_pool2d::<f32>(&input, (2, 2), (1, 1), (0, 0)).unwrap()
        }
        OperationKind::Conv1dExact => {
            let input = f32_storage(&[1, 1, 2], &[1.0, 2.0]);
            let weight = f32_storage(&[1, 1, 1], &[1.0]);
            B::conv1d::<f32>(&input, &weight, None, 1, 0, 1, 1).unwrap()
        }
        OperationKind::ConvTranspose2d => {
            let input = f32_storage(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let weight = f32_storage(&[1, 1, 1, 1], &[1.0]);
            B::conv_transpose2d::<f32>(&input, &weight, None, 1, 0, 0, 1, 1).unwrap()
        }
        OperationKind::AdaptiveAvgPool2dExact => {
            let input = f32_storage(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            B::adaptive_avg_pool2d::<f32>(&input, (1, 1)).unwrap()
        }
        OperationKind::LayerNorm => {
            let input = f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let weight = f32_storage(&[2], &[1.0, 1.0]);
            B::layer_norm::<f32>(&input, &weight, None, 1e-5).unwrap()
        }
        // Allocation. The layout argument is ignored on purpose: these have no
        // operand to lay out, and a fresh allocation is contiguous whatever the
        // row is being probed for.
        OperationKind::Zeros
        | OperationKind::Ones
        | OperationKind::UniformRandom
        | OperationKind::NormalRandom
        | OperationKind::Full
        | OperationKind::Arange
        | OperationKind::Linspace
        | OperationKind::VariableZeros
        | OperationKind::VariableOnes
        | OperationKind::VariableUniformRandom
        | OperationKind::VariableNormalRandom => allocation_probe(operation, DTypeId::F32),
        // Probed in inference mode, where dropout is the identity. The
        // training path draws a random mask, and a probe that asserted a shape
        // and a dtype against a random result would be asserting the same two
        // things while pretending to have checked more.
        OperationKind::Dropout => {
            let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
            let input = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            let handle = TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&input);
            dispatch::execute::<op::Dropout, _>(
                &context,
                DropoutAttributes {
                    probability: 0.5,
                    training: false,
                },
                &[handle],
            )
            .unwrap()
        }
        // Both dispatch, because neither composition exists on a backend
        // trait: `Linear::forward` and `RMSNorm::forward` are module methods
        // over typed tensors, and the executor is the first place either one
        // exists over storage.
        OperationKind::Linear | OperationKind::RmsNorm => {
            let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
            // Square, so the strided case transposes without moving the
            // trailing extent that both weights are sized against.
            let input = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            let weight = f32_storage(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
            let input_handle =
                TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&input);
            let weight_handle =
                TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&weight);
            if operation == OperationKind::Linear {
                dispatch::execute::<op::Linear, _>(
                    &context,
                    LinearAttributes { has_bias: false },
                    &[input_handle, weight_handle],
                )
                .unwrap()
            } else {
                // The weight is per-feature for this one, so it is rank one.
                let scale = f32_storage(&[2], &[1.0, 1.0]);
                let scale_handle =
                    TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&scale);
                dispatch::execute::<op::RmsNorm, _>(
                    &context,
                    EpsilonAttributes { epsilon: 1e-5 },
                    &[input_handle, scale_handle],
                )
                .unwrap()
            }
        }
        OperationKind::ToDType => {
            let input = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            B::tensor_to_dtype::<f32, f32>(&input, DTypeId::F32.descriptor()).unwrap()
        }
        // The variance family and the p-norm have no method on any backend
        // trait: the composition exists in the canonical executor and in
        // `Tensor::var_all`, and nowhere in between. Probing them therefore
        // means dispatching them, which is the stronger check anyway because it
        // exercises the row being probed rather than a function beside it.
        // The sequence-returning rows. The probe checks the piece count and
        // hands back the first piece, because its caller asserts against one
        // storage.
        OperationKind::Chunk | OperationKind::Split => {
            let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
            let input = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            let handle = TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&input);
            let pieces = if operation == OperationKind::Chunk {
                dispatch::execute::<op::Chunk, _>(
                    &context,
                    ChunkAttributes { chunks: 2, axis: 1 },
                    &[handle],
                )
                .unwrap()
            } else {
                dispatch::execute::<op::Split, _>(
                    &context,
                    SplitAttributes {
                        split_size: 1,
                        axis: 1,
                    },
                    &[handle],
                )
                .unwrap()
            };
            assert_eq!(
                pieces.len(),
                2,
                "{operation} over a two-wide axis produces two pieces"
            );
            pieces.into_iter().next().unwrap()
        }
        // The block kernels read the buffer directly and never consult a
        // stride, so the row admits contiguous only and the probe never
        // transposes.
        OperationKind::Quantize | OperationKind::Dequantize | OperationKind::QuantizedMatMul => {
            let values: Vec<f32> = (0..32).map(|index| index as f32).collect();
            let blocks = B::quantize::<f32, Q8_0>(&f32_storage(&[32], &values)).unwrap();
            match operation {
                OperationKind::Quantize => blocks,
                OperationKind::Dequantize => B::dequantize::<Q8_0, f32>(&blocks).unwrap(),
                // The kernel reads `rhs` as `[N, K]` rather than `[K, N]`, so
                // both operands are one block wide and the result is `[1, 1]`.
                _ => {
                    let row = blocks.reshape(&[1, 32]).unwrap();
                    B::quantized_matmul::<Q8_0>(&row, &row).unwrap()
                }
            }
        }
        OperationKind::Dot | OperationKind::Outer => {
            let lhs = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            let rhs = f32_storage(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
            match operation {
                OperationKind::Dot => {
                    let product = B::mul::<f32>(&lhs, &rhs).unwrap();
                    B::sum_all::<f32>(&product).unwrap()
                }
                // Two vectors rather than the matrices above: an outer product
                // is defined on rank one.
                _ => {
                    let column = B::unsqueeze::<f32>(&f32_storage(&[2], &[1.0, 2.0]), 1).unwrap();
                    let row = B::unsqueeze::<f32>(&f32_storage(&[2], &[3.0, 4.0]), 0).unwrap();
                    B::mul::<f32>(&column, &row).unwrap()
                }
            }
        }
        OperationKind::VarianceAll
        | OperationKind::VarianceDim
        | OperationKind::VarianceKeepDim
        | OperationKind::StdAll
        | OperationKind::StdDim
        | OperationKind::StdKeepDim
        | OperationKind::Norm => {
            let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
            let input = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), layout);
            let handle = TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&input);
            let axis = AxisVarianceAttributes {
                axis: 1,
                unbiased: false,
            };
            let all = VarianceAttributes { unbiased: false };
            match operation {
                OperationKind::VarianceAll => {
                    dispatch::execute::<op::VarianceAll, _>(&context, all, &[handle]).unwrap()
                }
                OperationKind::VarianceDim => {
                    dispatch::execute::<op::VarianceDim, _>(&context, axis, &[handle]).unwrap()
                }
                OperationKind::VarianceKeepDim => {
                    dispatch::execute::<op::VarianceKeepDim, _>(&context, axis, &[handle]).unwrap()
                }
                OperationKind::StdAll => {
                    dispatch::execute::<op::StdAll, _>(&context, all, &[handle]).unwrap()
                }
                OperationKind::StdDim => {
                    dispatch::execute::<op::StdDim, _>(&context, axis, &[handle]).unwrap()
                }
                OperationKind::StdKeepDim => {
                    dispatch::execute::<op::StdKeepDim, _>(&context, axis, &[handle]).unwrap()
                }
                _ => dispatch::execute::<op::Norm, _>(
                    &context,
                    NormAttributes { order: 2.0 },
                    &[handle],
                )
                .unwrap(),
            }
        }
        OperationKind::MseLoss | OperationKind::L1Loss | OperationKind::BceWithLogitsLoss => {
            let prediction =
                transpose_if_requested(f32_storage(&[2, 2], &[0.5, 1.5, -0.5, 1.0]), layout);
            let target = f32_storage(&[2, 2], &[1.0, 1.0, 0.0, 0.0]);
            match operation {
                OperationKind::MseLoss => {
                    B::mse_loss::<f32>(&prediction, &target, Reduction::Mean).unwrap()
                }
                OperationKind::L1Loss => {
                    B::l1_loss::<f32>(&prediction, &target, Reduction::Mean).unwrap()
                }
                _ => B::bce_with_logits_loss::<f32>(&prediction, &target, Reduction::Mean).unwrap(),
            }
        }
        // Inference mode with running statistics, because that is the only mode
        // the CPU kernel implements and the only one the canonical executor
        // admits. A probe in training mode would be probing a refusal.
        OperationKind::BatchNorm => {
            let input = f32_storage(&[1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let running_mean = f32_storage(&[2], &[0.0, 0.0]);
            let running_variance = f32_storage(&[2], &[1.0, 1.0]);
            B::batch_norm::<f32>(
                &input,
                None,
                None,
                Some(&running_mean),
                Some(&running_variance),
                1e-5,
                0.1,
            )
            .unwrap()
        }
        // The float family. Every operand is 1.0 because several of these have
        // restricted domains - `acosh` needs at least 1, `asin` at most 1 - and
        // 1.0 is the one value inside all of them. The probe asserts shape,
        // dtype and device rather than values, so a non-finite result from a
        // domain edge would not make it pass falsely.
        OperationKind::Relu
        | OperationKind::Step
        | OperationKind::Mish
        | OperationKind::Elu
        | OperationKind::Gelu
        | OperationKind::Abs
        | OperationKind::Exp
        | OperationKind::Neg
        | OperationKind::Sqrt
        | OperationKind::Log
        | OperationKind::Tanh
        | OperationKind::Sigmoid
        | OperationKind::Swish
        | OperationKind::Sign
        | OperationKind::Floor
        | OperationKind::Ceil
        | OperationKind::Round
        | OperationKind::Log2
        | OperationKind::Log10
        | OperationKind::Sin
        | OperationKind::Cos
        | OperationKind::Tan
        | OperationKind::Asin
        | OperationKind::Acos
        | OperationKind::Atan
        | OperationKind::Sinh
        | OperationKind::Cosh
        | OperationKind::Asinh
        | OperationKind::Acosh
        | OperationKind::Atanh
        | OperationKind::Erf
        | OperationKind::Rsqrt
        | OperationKind::Trunc
        | OperationKind::Frac
        | OperationKind::AddScalar
        | OperationKind::MulScalar
        | OperationKind::Powf
        | OperationKind::Atan2
        | OperationKind::Fmod
        | OperationKind::Remainder
        | OperationKind::Clamp
        | OperationKind::Softmax => {
            let input = transpose_if_requested(f32_storage(&[2, 2], &[1.0, 1.0, 1.0, 1.0]), layout);
            cpu_float_probe(operation, &input)
        }
        OperationKind::Maximum
        | OperationKind::Minimum
        | OperationKind::AbsDiff
        | OperationKind::Lerp
        | OperationKind::MaskedFill
        | OperationKind::WhereCond
        | OperationKind::CmpEq
        | OperationKind::CmpNe
        | OperationKind::CmpLt
        | OperationKind::CmpLe
        | OperationKind::CmpGt
        | OperationKind::CmpGe
        | OperationKind::LogicalAnd
        | OperationKind::LogicalOr
        | OperationKind::LogicalNot
        | OperationKind::SubScalar
        | OperationKind::DivScalar
        | OperationKind::TransposeExact
        | OperationKind::Narrow
        | OperationKind::FlattenExact
        | OperationKind::SqueezeExact
        | OperationKind::UnsqueezeExact
        | OperationKind::Triu
        | OperationKind::Tril
        | OperationKind::Diag
        | OperationKind::BatchedMatMul
        | OperationKind::ConcatExact
        | OperationKind::StackExact
        | OperationKind::SliceExact
        | OperationKind::Gather
        | OperationKind::Scatter
        | OperationKind::IndexSelect
        | OperationKind::Repeat
        | OperationKind::Pad
        | OperationKind::Unfold
        | OperationKind::PixelShuffle
        | OperationKind::GroupNorm
        | OperationKind::InstanceNorm
        | OperationKind::BroadcastLeft
        | OperationKind::Addmm
        | OperationKind::ScaledDotProductAttention
        | OperationKind::ArgMax
        | OperationKind::ArgMin
        | OperationKind::Argsort
        | OperationKind::Cumsum => {
            let operands: Vec<CpuStorage> = cpu_tensor_operand_shapes(operation)
                .iter()
                .map(|shape| laid_out(shape, layout))
                .collect();
            let borrowed: Vec<&CpuStorage> = operands.iter().collect();
            cpu_tensor_probe::<f32>(operation, &borrowed)
        }
        // The index operand is not part of `layout`'s probe matrix — it is a
        // rank-one integer vector, not the strided-vs-contiguous tensor the
        // row's layout claim is actually about — so only the weight table is
        // built through `laid_out`.
        OperationKind::EmbeddingExact => {
            let indices = CpuStorage::try_from_contiguous(CpuBuffer::I64(vec![0, 1]), vec![2])
                .expect("a two-element i64 index vector must be constructible");
            let weight = laid_out(&[3, 2], layout);
            let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
            let handles = [
                TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&indices),
                TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&weight),
            ];
            dispatch::execute::<op::EmbeddingExact, _>(&context, NoAttributes, &handles).unwrap()
        }
        // Same split as `embedding` for the same reason: the class-target
        // vector is a rank-one integer operand, not the tensor whose layout
        // the row's claim is about, so only the logits go through `laid_out`.
        OperationKind::CrossEntropyLoss => {
            let logits = laid_out(&[2, 3], layout);
            let targets = CpuStorage::try_from_contiguous(CpuBuffer::I64(vec![0, 1]), vec![2])
                .expect("a two-element i64 target vector must be constructible");
            let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
            let handles = [
                TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&logits),
                TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&targets),
            ];
            dispatch::execute::<op::CrossEntropyLoss, _>(
                &context,
                LossAttributes {
                    reduction: LossReduction::Mean,
                },
                &handles,
            )
            .unwrap()
        }
        OperationKind::TensorFromData | OperationKind::TensorFromBytes => {
            let source = f32_storage(&[2], &[1.0, 2.0]);
            let bytes = B::to_bytes::<f32>(&source).unwrap();
            let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new())
                .with_grad_mode(GradMode::Disabled);
            let typed_attributes = DataAttributes {
                shape: vec![2],
                dtype: DTypeId::F32.descriptor(),
                device: DeviceId::cpu(),
                payload: incin_core::exec::catalog::CreationPayload::Typed {
                    byte_len: bytes.len(),
                    dtype: DTypeId::F32.descriptor(),
                },
            };
            match operation {
                OperationKind::TensorFromData => {
                    dispatch::execute_shaped_with_payload::<op::TensorFromData, _, Dyn>(
                        &context,
                        typed_attributes,
                        &[],
                        &ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&[2])).unwrap(),
                        Some(&bytes),
                    )
                    .unwrap()
                }
                OperationKind::TensorFromBytes => {
                    dispatch::execute_shaped_with_payload::<op::TensorFromBytes, _, Dyn>(
                        &context,
                        DataAttributes {
                            shape: vec![2],
                            dtype: DTypeId::F32.descriptor(),
                            device: DeviceId::cpu(),
                            payload: incin_core::exec::catalog::CreationPayload::Bytes {
                                byte_len: bytes.len(),
                            },
                        },
                        &[],
                        &ShapeValue::<Dyn>::try_new(ShapeBuf::from_slice(&[2])).unwrap(),
                        Some(&bytes),
                    )
                    .unwrap()
                }
                _ => unreachable!(),
            }
        }
        _ => panic!("missing CPU capability execution probe for {operation}"),
    }
}

/// Whether a row's executor answers with storage.
///
/// The readback rows do not: they return an `f64`, a `Vec<i64>` or a byte
/// buffer, because that is what reading a value back to the host means. The two
/// probes below assert a shape, a dtype and a device on the result, and none of
/// those is a question a host value has an answer to. Skipping them here is
/// narrower than weakening the assertions for every row, and
/// `the_readback_rows_return_host_values_through_dispatch` checks what they
/// actually produce.
fn produces_storage(operation: OperationKind) -> bool {
    !matches!(
        operation,
        OperationKind::ToHostFloatScalar
            | OperationKind::ToHostFloatVec
            | OperationKind::ToHostIntScalar
            | OperationKind::ToHostIntVec
            | OperationKind::TensorToBytes
    )
}

#[test]
fn generated_cpu_rows_match_real_execution_and_output_metadata() {
    for rule in CPU_CAPABILITIES {
        if !produces_storage(rule.operation) {
            continue;
        }
        for &layout in rule.layouts {
            // The probes build f32 operands wherever the row admits f32. The
            // rows over compressed storage do not admit it, and querying them
            // with f32 would assert support they never claimed, so the query
            // follows the row rather than the probe's usual choice.
            let probe_dtype = if rule.dtypes.contains(&DTypeId::F32.descriptor()) {
                DTypeId::F32.descriptor()
            } else {
                rule.dtypes[0]
            };
            let case = query(rule.operation, probe_dtype, layout, rule.min_rank);
            assert!(registry(DeviceKind::Cpu).support(&case).is_device_local());
            let output = execute_cpu_probe(rule.operation, layout);
            assert_eq!(&*output.shape, cpu_probe_shape(rule.operation));
            assert_eq!(
                output.dtype,
                expected_result_dtype(rule.operation, probe_dtype)
            );
            assert_eq!(output.device, DeviceId::cpu());
        }
    }
}

/// The dtype probe's counterpart to [`cpu_float_probe`], over `Dyn` storage.
///
/// `Dyn` rather than `f32` because the dtype probe walks every advertised dtype
/// and the storage handle is dtype-erased; the two helpers are otherwise the
/// same dispatch.
fn cpu_float_probe_dyn(operation: OperationKind, input: &CpuStorage) -> CpuStorage {
    type B = CpuBackendImpl;
    match operation {
        OperationKind::Relu => B::relu::<Dyn>(input).unwrap(),
        OperationKind::Step => B::step::<Dyn>(input).unwrap(),
        OperationKind::Mish => B::mish::<Dyn>(input).unwrap(),
        OperationKind::Elu => B::elu::<Dyn>(input).unwrap(),
        OperationKind::Gelu => B::gelu::<Dyn>(input).unwrap(),
        OperationKind::Abs => B::abs::<Dyn>(input).unwrap(),
        OperationKind::Exp => B::exp::<Dyn>(input).unwrap(),
        OperationKind::Neg => B::neg::<Dyn>(input).unwrap(),
        OperationKind::Sqrt => B::sqrt::<Dyn>(input).unwrap(),
        OperationKind::Log => B::log::<Dyn>(input).unwrap(),
        OperationKind::Tanh => B::tanh::<Dyn>(input).unwrap(),
        OperationKind::Sigmoid => B::sigmoid::<Dyn>(input).unwrap(),
        OperationKind::Swish => B::swish::<Dyn>(input).unwrap(),
        OperationKind::Sign => B::sign::<Dyn>(input).unwrap(),
        OperationKind::Floor => B::floor::<Dyn>(input).unwrap(),
        OperationKind::Ceil => B::ceil::<Dyn>(input).unwrap(),
        OperationKind::Round => B::round::<Dyn>(input).unwrap(),
        OperationKind::Log2 => B::log2::<Dyn>(input).unwrap(),
        OperationKind::Log10 => B::log10::<Dyn>(input).unwrap(),
        OperationKind::Sin => B::sin::<Dyn>(input).unwrap(),
        OperationKind::Cos => B::cos::<Dyn>(input).unwrap(),
        OperationKind::Tan => B::tan::<Dyn>(input).unwrap(),
        OperationKind::Asin => B::asin::<Dyn>(input).unwrap(),
        OperationKind::Acos => B::acos::<Dyn>(input).unwrap(),
        OperationKind::Atan => B::atan::<Dyn>(input).unwrap(),
        OperationKind::Sinh => B::sinh::<Dyn>(input).unwrap(),
        OperationKind::Cosh => B::cosh::<Dyn>(input).unwrap(),
        OperationKind::Asinh => B::asinh::<Dyn>(input).unwrap(),
        OperationKind::Acosh => B::acosh::<Dyn>(input).unwrap(),
        OperationKind::Atanh => B::atanh::<Dyn>(input).unwrap(),
        OperationKind::Erf => B::erf::<Dyn>(input).unwrap(),
        OperationKind::Rsqrt => B::rsqrt::<Dyn>(input).unwrap(),
        OperationKind::Trunc => B::trunc::<Dyn>(input).unwrap(),
        OperationKind::Frac => B::frac::<Dyn>(input).unwrap(),
        OperationKind::AddScalar => B::add_scalar_float::<Dyn>(input, 2.0).unwrap(),
        OperationKind::MulScalar => B::mul_scalar_float::<Dyn>(input, 2.0).unwrap(),
        OperationKind::Powf => B::powf::<Dyn>(input, 2.0).unwrap(),
        OperationKind::Clamp => B::clamp::<Dyn>(input, 0.0, 1.0).unwrap(),
        OperationKind::Softmax => B::softmax::<Dyn>(input, 1).unwrap(),
        OperationKind::Atan2 => B::atan2::<Dyn>(input, input).unwrap(),
        OperationKind::Fmod => B::fmod::<Dyn>(input, input).unwrap(),
        OperationKind::Remainder => B::remainder::<Dyn>(input, input).unwrap(),
        other => panic!("{other} is not a float-family operation"),
    }
}

fn cpu_zeros(dtype: DTypeDescriptor, shape: &[usize]) -> CpuStorage {
    type B = CpuBackendImpl;
    if dtype == DTypeId::Q8_0.descriptor() {
        assert_eq!(shape.iter().product::<usize>(), 32);
        B::from_bytes::<Dyn>(&[0u8; 34], shape, dtype, &DeviceId::cpu()).unwrap()
    } else {
        B::zeros::<Dyn>(shape, dtype, &DeviceId::cpu()).unwrap()
    }
}

#[test]
fn every_advertised_cpu_dtype_executes_its_registered_operation() {
    type B = CpuBackendImpl;
    for rule in CPU_CAPABILITIES {
        for &dtype in rule.dtypes {
            let output = match rule.operation {
                OperationKind::Storage => {
                    let shape = if dtype == DTypeId::Q8_0.descriptor() {
                        &[32][..]
                    } else {
                        &[2][..]
                    };
                    let storage = cpu_zeros(dtype, shape);
                    assert!(!B::to_bytes::<Dyn>(&storage).unwrap().is_empty());
                    storage
                }
                OperationKind::Fill => cpu_zeros(dtype, &[2]),
                OperationKind::Random => B::rand::<Dyn>(&[2], dtype, &DeviceId::cpu()).unwrap(),
                OperationKind::Pointwise => {
                    let lhs = B::ones::<Dyn>(&[2], dtype, &DeviceId::cpu()).unwrap();
                    let rhs = B::ones::<Dyn>(&[2], dtype, &DeviceId::cpu()).unwrap();
                    B::add::<Dyn>(&lhs, &rhs).unwrap()
                }
                OperationKind::Reduction => {
                    let input = cpu_zeros(dtype, &[2]);
                    B::sum_all::<Dyn>(&input).unwrap()
                }
                OperationKind::Normalization => {
                    let input = cpu_zeros(dtype, &[1, 2]);
                    let weight = B::ones::<Dyn>(&[2], dtype, &DeviceId::cpu()).unwrap();
                    B::layer_norm::<Dyn>(&input, &weight, None, 1e-5).unwrap()
                }
                OperationKind::Broadcast | OperationKind::BroadcastAs => {
                    let shape = if dtype == DTypeId::Q8_0.descriptor() {
                        &[1, 32][..]
                    } else {
                        &[1, 2][..]
                    };
                    let target = if dtype == DTypeId::Q8_0.descriptor() {
                        &[2, 32][..]
                    } else {
                        &[2, 2][..]
                    };
                    B::broadcast_as::<Dyn>(&cpu_zeros(dtype, shape), target).unwrap()
                }
                OperationKind::Reshape | OperationKind::ReshapeExact => {
                    let shape = if dtype == DTypeId::Q8_0.descriptor() {
                        &[32][..]
                    } else {
                        &[2, 2][..]
                    };
                    let target = if dtype == DTypeId::Q8_0.descriptor() {
                        &[1, 32][..]
                    } else {
                        &[4][..]
                    };
                    B::reshape::<Dyn>(&cpu_zeros(dtype, shape), target).unwrap()
                }
                OperationKind::Add
                | OperationKind::Sub
                | OperationKind::Mul
                | OperationKind::Div => {
                    let lhs = B::ones::<Dyn>(&[2], dtype, &DeviceId::cpu()).unwrap();
                    let rhs = B::ones::<Dyn>(&[2], dtype, &DeviceId::cpu()).unwrap();
                    match rule.operation {
                        OperationKind::Add => B::add::<Dyn>(&lhs, &rhs).unwrap(),
                        OperationKind::Sub => B::sub::<Dyn>(&lhs, &rhs).unwrap(),
                        OperationKind::Mul => B::mul::<Dyn>(&lhs, &rhs).unwrap(),
                        OperationKind::Div => B::div::<Dyn>(&lhs, &rhs).unwrap(),
                        _ => unreachable!(),
                    }
                }
                OperationKind::MatMul
                | OperationKind::Conv2d
                | OperationKind::Pool2d
                | OperationKind::MatMulExact
                | OperationKind::SumAll
                | OperationKind::MeanAll
                | OperationKind::MaxAll
                | OperationKind::MinAll
                | OperationKind::ProdAll
                | OperationKind::SumDim
                | OperationKind::SumKeepDim
                | OperationKind::MeanDim
                | OperationKind::MeanKeepDim
                | OperationKind::MaxDim
                | OperationKind::MaxKeepDim
                | OperationKind::MinDim
                | OperationKind::MinKeepDim
                | OperationKind::ProdDim
                | OperationKind::Conv2dExact
                | OperationKind::Conv1dExact
                | OperationKind::ConvTranspose2d
                | OperationKind::MaxPool2d
                | OperationKind::AvgPool2d
                | OperationKind::AdaptiveAvgPool2dExact
                | OperationKind::LayerNorm
                | OperationKind::RmsNorm
                | OperationKind::Linear
                | OperationKind::BatchNorm
                | OperationKind::MseLoss
                | OperationKind::L1Loss
                | OperationKind::BceWithLogitsLoss
                | OperationKind::VarianceAll
                | OperationKind::VarianceDim
                | OperationKind::VarianceKeepDim
                | OperationKind::StdAll
                | OperationKind::StdDim
                | OperationKind::StdKeepDim
                | OperationKind::Norm
                | OperationKind::Dot
                | OperationKind::Outer
                | OperationKind::Quantize
                | OperationKind::Dequantize
                | OperationKind::QuantizedMatMul
                | OperationKind::TopK => execute_cpu_probe(rule.operation, LayoutClass::Contiguous),
                // `chunk` and `split` are the two rows whose executor returns a
                // sequence. The probe asserts the piece count here, because the
                // shared assertion below can only speak for one storage, and
                // then hands back the first piece for it to check.
                OperationKind::Chunk | OperationKind::Split => {
                    let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
                    let input = B::ones::<Dyn>(&[2, 2], dtype, &DeviceId::cpu()).unwrap();
                    let handle =
                        TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&input);
                    let pieces = if rule.operation == OperationKind::Chunk {
                        dispatch::execute::<op::Chunk, _>(
                            &context,
                            ChunkAttributes { chunks: 2, axis: 1 },
                            &[handle],
                        )
                        .unwrap()
                    } else {
                        dispatch::execute::<op::Split, _>(
                            &context,
                            SplitAttributes {
                                split_size: 1,
                                axis: 1,
                            },
                            &[handle],
                        )
                        .unwrap()
                    };
                    assert_eq!(
                        pieces.len(),
                        2,
                        "{} over a two-wide axis produces two pieces",
                        rule.operation
                    );
                    pieces.into_iter().next().unwrap()
                }
                OperationKind::Zeros
                | OperationKind::Ones
                | OperationKind::UniformRandom
                | OperationKind::NormalRandom
                | OperationKind::Full
                | OperationKind::Arange
                | OperationKind::Linspace
                | OperationKind::VariableZeros
                | OperationKind::VariableOnes
                | OperationKind::VariableUniformRandom
                | OperationKind::VariableNormalRandom => allocation_probe(rule.operation, dtype),
                // Readback answers with a host value rather than storage, so
                // there is nothing here for the dtype assertion below to read.
                // Covered by `the_readback_rows_return_host_values_through_dispatch`.
                OperationKind::ToHostFloatScalar
                | OperationKind::ToHostFloatVec
                | OperationKind::ToHostIntScalar
                | OperationKind::ToHostIntVec
                | OperationKind::TensorToBytes => continue,
                // Inference-mode dropout hands the operand straight back, so it
                // is the one row here that has to answer for every float dtype
                // the elementwise group advertises rather than for f32 alone.
                OperationKind::Dropout => {
                    let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
                    let input = B::ones::<Dyn>(&[2, 2], dtype, &DeviceId::cpu()).unwrap();
                    let handle =
                        TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&input);
                    dispatch::execute::<op::Dropout, _>(
                        &context,
                        DropoutAttributes {
                            probability: 0.5,
                            training: false,
                        },
                        &[handle],
                    )
                    .unwrap()
                }
                // `to_dtype` is the one row whose result dtype is chosen by an
                // attribute rather than inherited from the operand, so it is
                // probed across every dtype it admits, converting each to f32.
                OperationKind::ToDType => {
                    let input = B::ones::<Dyn>(&[2, 2], dtype, &DeviceId::cpu()).unwrap();
                    B::tensor_to_dtype::<Dyn, Dyn>(&input, DTypeId::F32.descriptor()).unwrap()
                }
                // `INDEX_AND_F32_DTYPES` is the union of the index operand's
                // integer dtypes and the weight operand's f32-only one, so
                // `dtype` here names whichever position it actually belongs
                // to; the other operand takes a fixed dtype from its own real
                // set rather than the one under test.
                OperationKind::EmbeddingExact => {
                    let (index_dtype, weight_dtype) = if dtype.is_integer() {
                        (dtype, DTypeId::F32.descriptor())
                    } else {
                        (DTypeId::I64.descriptor(), dtype)
                    };
                    let indices = B::zeros::<Dyn>(&[2], index_dtype, &DeviceId::cpu()).unwrap();
                    let weight = B::ones::<Dyn>(&[3, 2], weight_dtype, &DeviceId::cpu()).unwrap();
                    let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
                    let handles = [
                        TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&indices),
                        TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&weight),
                    ];
                    dispatch::execute::<op::EmbeddingExact, _>(&context, NoAttributes, &handles)
                        .unwrap()
                }
                // The same union, read the other way round: here the float
                // position is the logits and the integer one is the class
                // target, so `dtype` selects whichever it names and the other
                // operand takes a fixed dtype from its own real set.
                OperationKind::CrossEntropyLoss => {
                    let (logit_dtype, target_dtype) = if dtype.is_integer() {
                        (DTypeId::F32.descriptor(), dtype)
                    } else {
                        (dtype, DTypeId::I64.descriptor())
                    };
                    let logits = B::ones::<Dyn>(&[2, 3], logit_dtype, &DeviceId::cpu()).unwrap();
                    let targets = B::zeros::<Dyn>(&[2], target_dtype, &DeviceId::cpu()).unwrap();
                    let context = ExecutionContext::new(CpuBackendImpl::<Cpu>::new());
                    let handles = [
                        TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&logits),
                        TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, Local>(&targets),
                    ];
                    dispatch::execute::<op::CrossEntropyLoss, _>(
                        &context,
                        LossAttributes {
                            reduction: LossReduction::Mean,
                        },
                        &handles,
                    )
                    .unwrap()
                }
                OperationKind::Relu
                | OperationKind::Step
                | OperationKind::Mish
                | OperationKind::Elu
                | OperationKind::Gelu
                | OperationKind::Abs
                | OperationKind::Exp
                | OperationKind::Neg
                | OperationKind::Sqrt
                | OperationKind::Log
                | OperationKind::Tanh
                | OperationKind::Sigmoid
                | OperationKind::Swish
                | OperationKind::Sign
                | OperationKind::Floor
                | OperationKind::Ceil
                | OperationKind::Round
                | OperationKind::Log2
                | OperationKind::Log10
                | OperationKind::Sin
                | OperationKind::Cos
                | OperationKind::Tan
                | OperationKind::Asin
                | OperationKind::Acos
                | OperationKind::Atan
                | OperationKind::Sinh
                | OperationKind::Cosh
                | OperationKind::Asinh
                | OperationKind::Acosh
                | OperationKind::Atanh
                | OperationKind::Erf
                | OperationKind::Rsqrt
                | OperationKind::Trunc
                | OperationKind::Frac
                | OperationKind::AddScalar
                | OperationKind::MulScalar
                | OperationKind::Powf
                | OperationKind::Atan2
                | OperationKind::Fmod
                | OperationKind::Remainder
                | OperationKind::Clamp
                | OperationKind::Softmax => {
                    let input = B::ones::<Dyn>(&[2, 2], dtype, &DeviceId::cpu()).unwrap();
                    cpu_float_probe_dyn(rule.operation, &input)
                }
                OperationKind::Maximum
                | OperationKind::Minimum
                | OperationKind::AbsDiff
                | OperationKind::Lerp
                | OperationKind::MaskedFill
                | OperationKind::WhereCond
                | OperationKind::CmpEq
                | OperationKind::CmpNe
                | OperationKind::CmpLt
                | OperationKind::CmpLe
                | OperationKind::CmpGt
                | OperationKind::CmpGe
                | OperationKind::LogicalAnd
                | OperationKind::LogicalOr
                | OperationKind::LogicalNot
                | OperationKind::SubScalar
                | OperationKind::DivScalar
                | OperationKind::TransposeExact
                | OperationKind::Narrow
                | OperationKind::FlattenExact
                | OperationKind::SqueezeExact
                | OperationKind::UnsqueezeExact
                | OperationKind::Triu
                | OperationKind::Tril
                | OperationKind::Diag
                | OperationKind::BatchedMatMul
                | OperationKind::ConcatExact
                | OperationKind::StackExact
                | OperationKind::SliceExact
                | OperationKind::Gather
                | OperationKind::Scatter
                | OperationKind::IndexSelect
                | OperationKind::Repeat
                | OperationKind::Pad
                | OperationKind::Unfold
                | OperationKind::PixelShuffle
                | OperationKind::GroupNorm
                | OperationKind::InstanceNorm
                | OperationKind::BroadcastLeft
                | OperationKind::Addmm
                | OperationKind::ScaledDotProductAttention
                | OperationKind::ArgMax
                | OperationKind::ArgMin
                | OperationKind::Argsort
                | OperationKind::Cumsum => {
                    let operands: Vec<CpuStorage> = cpu_tensor_operand_shapes(rule.operation)
                        .iter()
                        .map(|shape| B::ones::<Dyn>(shape, dtype, &DeviceId::cpu()).unwrap())
                        .collect();
                    let borrowed: Vec<&CpuStorage> = operands.iter().collect();
                    cpu_tensor_probe::<Dyn>(rule.operation, &borrowed)
                }
                OperationKind::TensorFromData | OperationKind::TensorFromBytes => {
                    let source = cpu_zeros(dtype, &[2]);
                    let bytes = B::to_bytes::<Dyn>(&source).unwrap();
                    B::from_bytes::<Dyn>(&bytes, &[2], dtype, &DeviceId::cpu()).unwrap()
                }
                _ => panic!("missing dtype conformance probe for {}", rule.operation),
            };
            assert_eq!(
                output.dtype,
                expected_result_dtype(rule.operation, dtype),
                "{} {dtype:?}",
                rule.operation
            );
        }
    }
}

/// What dtype the first result must carry, given the operand dtype the row
/// advertises.
///
/// A capability row's dtype set describes the *operand*, because that is what
/// the query in `dispatch::admit` is built from. For most operations the result
/// carries the same dtype, but an index-returning reduction does not, and the
/// catalog already says so: `DTypeRule::IndexResult` is declared once per
/// identity there. Reading it from the catalog rather than listing the
/// exceptions here means a future index-returning operation is covered by this
/// check the moment it is added to the catalog, instead of quietly asserting
/// the wrong thing until someone notices.
///
/// `topk` is the exception the catalog rule cannot express on its own. It is
/// declared `IndexResult` but returns two tensors, and only the second one is
/// an index tensor; the first carries the operand dtype. The descriptor layer
/// already makes exactly this distinction by output position, so this follows
/// it rather than inventing a second answer.
///
/// The concrete index dtype is the one the CPU kernel produces, which is not
/// uniform: `argmax` and `argmin` build `i64` buffers and `argsort` and `topk`
/// build `u32` ones. That inconsistency is the backend's, and it is recorded
/// rather than smoothed over.
fn expected_result_dtype(operation: OperationKind, operand: DTypeDescriptor) -> DTypeDescriptor {
    if operation == OperationKind::TopK {
        return operand;
    }
    // Not derived from the catalog, because the catalog cannot know it: the
    // result dtype of a conversion is whatever the caller asked for. This
    // mirrors the target the probes above pass.
    if operation == OperationKind::ToDType {
        return DTypeId::F32.descriptor();
    }
    // The three rows whose result dtype is not the operand's. `quantize`
    // compresses into blocks; the other two read blocks and answer in f32, so
    // for them the operand dtype is exactly the wrong answer.
    if operation == OperationKind::Quantize {
        return DTypeId::Q8_0.descriptor();
    }
    if matches!(
        operation,
        OperationKind::Dequantize | OperationKind::QuantizedMatMul
    ) {
        return DTypeId::F32.descriptor();
    }
    // `embedding`'s weight operand is f32 whatever `operand` names — it is the
    // probe's fixed choice when `operand` belongs to the index position
    // instead. The gathered result always carries the weight's dtype.
    //
    // `cross_entropy_loss` is the same shape of answer read the other way
    // round: its logits are the f32 position and its targets the integer one,
    // and the loss it computes is a float whichever of the two `operand`
    // happens to name.
    if matches!(
        operation,
        OperationKind::EmbeddingExact | OperationKind::CrossEntropyLoss
    ) {
        return DTypeId::F32.descriptor();
    }
    if matches!(
        operation,
        OperationKind::CmpEq
            | OperationKind::CmpNe
            | OperationKind::CmpLt
            | OperationKind::CmpLe
            | OperationKind::CmpGt
            | OperationKind::CmpGe
            | OperationKind::LogicalAnd
            | OperationKind::LogicalOr
            | OperationKind::LogicalNot
    ) {
        return DTypeId::Bool.descriptor();
    }
    let entry = OPERATION_CATALOG
        .iter()
        .find(|entry| entry.operation == operation);
    match entry.map(|entry| entry.dtype) {
        // An index result carries the index dtype the caller named, not the
        // operand's. The kernels used to ignore that parameter and hardcode a
        // buffer, `i64` for the extremum reductions and `u32` for the sorts,
        // so this used to have to split by operation to describe the
        // inconsistency. They honour it now, and every probe above asks for
        // `i64`, so that is what every index result has to be.
        Some(DTypeRule::IndexResult) => DTypeId::I64.descriptor(),
        _ => operand,
    }
}

/// The index half of `topk`, which the shared probes above do not reach.
///
/// `expected_result_dtype` answers for output zero, so without this the index
/// tensor would be the one migrated result nothing asserts a dtype for. It
/// used to be `u32` whatever the caller asked for; both halves now follow the
/// request, the values keeping the operand dtype and the indices the declared
/// index dtype, so both are asserted across the integer dtypes.
#[test]
fn the_topk_outputs_carry_the_dtypes_the_caller_named() {
    type B = CpuBackendImpl;
    let input = f32_storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);

    let (values, indices) = B::topk::<f32, u32>(&input, 1, 1, true).unwrap();
    assert_eq!(values.dtype, DTypeId::F32.descriptor());
    assert_eq!(indices.dtype, DTypeId::U32.descriptor());
    assert_eq!(&*indices.shape, &[2, 1]);

    let (_, indices) = B::topk::<f32, i64>(&input, 1, 1, true).unwrap();
    assert_eq!(indices.dtype, DTypeId::I64.descriptor());
    let (_, indices) = B::topk::<f32, u8>(&input, 1, 1, true).unwrap();
    assert_eq!(indices.dtype, DTypeId::U8.descriptor());

    // The value half used to be built as `f32` regardless of the operand, so
    // an `f64` operand came back relabelled and narrowed.
    let wide =
        CpuStorage::try_from_contiguous(CpuBuffer::F64(vec![1.0, 2.0, 3.0, 4.0]), vec![2, 2])
            .unwrap();
    let (values, _) = B::topk::<f64, i64>(&wide, 1, 1, true).unwrap();
    assert_eq!(values.dtype, DTypeId::F64.descriptor());
}

#[cfg(feature = "wgpu")]
type WgpuB =
    incin_backends::wgpu::WgpuBackendImpl<incin_core::prelude::WgpuN<incin_core::typenum::U0>>;

#[cfg(feature = "wgpu")]
fn wgpu_f32(
    shape: &[usize],
    values: &[f32],
) -> <WgpuB as incin_core::backend_authoring::StorageBackend>::Storage<f32> {
    WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::wgpu(0),
    )
    .unwrap()
}

#[cfg(feature = "wgpu")]
fn wgpu_bool(
    shape: &[usize],
    values: &[bool],
) -> <WgpuB as incin_core::backend_authoring::StorageBackend>::Storage<bool> {
    let floats: Vec<f32> = values.iter().map(|&b| if b { 1.0 } else { 0.0 }).collect();
    wgpu_f32(shape, &floats)
}

#[cfg(feature = "wgpu")]
fn wgpu_probe_shape(operation: OperationKind) -> &'static [usize] {
    match operation {
        OperationKind::Storage
        | OperationKind::Fill
        | OperationKind::Zeros
        | OperationKind::Ones
        | OperationKind::Full
        | OperationKind::Arange
        | OperationKind::Linspace
        | OperationKind::Random
        | OperationKind::Pointwise
        | OperationKind::Add
        | OperationKind::Sub
        | OperationKind::Mul
        | OperationKind::Div
        | OperationKind::WhereCond
        | OperationKind::MaskedFill => &[2],
        OperationKind::Reduction
        | OperationKind::SumAll
        | OperationKind::MeanAll
        | OperationKind::MaxAll
        | OperationKind::MinAll
        | OperationKind::ProdAll => &[],
        OperationKind::SumDim
        | OperationKind::MeanDim
        | OperationKind::MaxDim
        | OperationKind::MinDim
        | OperationKind::ProdDim => &[2],
        OperationKind::SumKeepDim
        | OperationKind::MeanKeepDim
        | OperationKind::MaxKeepDim
        | OperationKind::MinKeepDim => &[2, 1],
        OperationKind::Normalization => &[1, 2],
        OperationKind::Broadcast | OperationKind::BroadcastAs => &[2, 2],
        OperationKind::Reshape | OperationKind::ReshapeExact => &[4],
        OperationKind::MatMul | OperationKind::MatMulExact => &[2, 2],
        OperationKind::Conv2d | OperationKind::Conv2dExact => &[1, 1, 2, 2],
        OperationKind::Pool2d | OperationKind::MaxPool2d | OperationKind::AvgPool2d => {
            &[1, 1, 1, 1]
        }
        _ => panic!("missing WGPU expected shape for {operation}"),
    }
}

#[cfg(feature = "wgpu")]
fn execute_wgpu_probe(
    operation: OperationKind,
) -> <WgpuB as incin_core::backend_authoring::StorageBackend>::Storage<f32> {
    match operation {
        OperationKind::Storage => wgpu_f32(&[2], &[1.0, 2.0]),
        OperationKind::Fill | OperationKind::Zeros => {
            WgpuB::zeros::<f32>(&[2], DTypeId::F32.descriptor(), &DeviceId::wgpu(0)).unwrap()
        }
        OperationKind::Ones => {
            WgpuB::ones::<f32>(&[2], DTypeId::F32.descriptor(), &DeviceId::wgpu(0)).unwrap()
        }
        OperationKind::Full => {
            WgpuB::full::<f32>(1.0, &[2], DTypeId::F32.descriptor(), &DeviceId::wgpu(0)).unwrap()
        }
        OperationKind::Arange => WgpuB::arange::<f32>(
            0.0,
            1.0,
            &[2],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .unwrap(),
        OperationKind::Linspace => WgpuB::linspace::<f32>(
            0.0,
            1.0,
            &[2],
            DTypeId::F32.descriptor(),
            &DeviceId::wgpu(0),
        )
        .unwrap(),
        OperationKind::Random => {
            WgpuB::rand::<f32>(&[2], DTypeId::F32.descriptor(), &DeviceId::wgpu(0)).unwrap()
        }
        OperationKind::Pointwise => {
            let lhs = wgpu_f32(&[2], &[1.0, 2.0]);
            let rhs = wgpu_f32(&[2], &[3.0, 4.0]);
            WgpuB::add::<f32>(&lhs, &rhs).unwrap()
        }
        OperationKind::Add | OperationKind::Sub | OperationKind::Mul | OperationKind::Div => {
            let lhs = wgpu_f32(&[2], &[1.0, 2.0]);
            let rhs = wgpu_f32(&[2], &[3.0, 4.0]);
            match operation {
                OperationKind::Add => WgpuB::add::<f32>(&lhs, &rhs).unwrap(),
                OperationKind::Sub => WgpuB::sub::<f32>(&lhs, &rhs).unwrap(),
                OperationKind::Mul => WgpuB::mul::<f32>(&lhs, &rhs).unwrap(),
                OperationKind::Div => WgpuB::div::<f32>(&lhs, &rhs).unwrap(),
                _ => unreachable!(),
            }
        }
        OperationKind::WhereCond => {
            let mask = wgpu_bool(&[2], &[true, false]);
            let on_true = wgpu_f32(&[2], &[10.0, 20.0]);
            let on_false = wgpu_f32(&[2], &[-1.0, -2.0]);
            WgpuB::where_cond::<f32>(&mask, &on_true, &on_false).unwrap()
        }
        OperationKind::MaskedFill => {
            let input = wgpu_f32(&[2], &[10.0, 20.0]);
            let mask = wgpu_bool(&[2], &[true, false]);
            WgpuB::masked_fill::<f32>(&input, &mask, 0.0).unwrap()
        }
        OperationKind::Reduction => WgpuB::sum_all::<f32>(&wgpu_f32(&[2], &[1.0, 2.0])).unwrap(),
        OperationKind::SumAll
        | OperationKind::MeanAll
        | OperationKind::MaxAll
        | OperationKind::MinAll
        | OperationKind::ProdAll
        | OperationKind::SumDim
        | OperationKind::SumKeepDim
        | OperationKind::MeanDim
        | OperationKind::MeanKeepDim
        | OperationKind::MaxDim
        | OperationKind::MaxKeepDim
        | OperationKind::MinDim
        | OperationKind::MinKeepDim
        | OperationKind::ProdDim => {
            let input = wgpu_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
            match operation {
                OperationKind::SumAll => WgpuB::sum_all::<f32>(&input).unwrap(),
                OperationKind::MeanAll => WgpuB::mean_all::<f32>(&input).unwrap(),
                OperationKind::MaxAll => WgpuB::max_all::<f32>(&input).unwrap(),
                OperationKind::MinAll => WgpuB::min_all::<f32>(&input).unwrap(),
                OperationKind::ProdAll => WgpuB::prod_all::<f32>(&input).unwrap(),
                OperationKind::SumDim => WgpuB::sum_dim::<f32>(&input, 1).unwrap(),
                OperationKind::SumKeepDim => WgpuB::sum_keepdim::<f32>(&input, 1).unwrap(),
                OperationKind::MeanDim => WgpuB::mean_dim::<f32>(&input, 1).unwrap(),
                OperationKind::MeanKeepDim => WgpuB::mean_keepdim::<f32>(&input, 1).unwrap(),
                OperationKind::MaxDim => WgpuB::max_dim::<f32>(&input, 1).unwrap(),
                OperationKind::MaxKeepDim => WgpuB::max_keepdim::<f32>(&input, 1).unwrap(),
                OperationKind::MinDim => WgpuB::min_dim::<f32>(&input, 1).unwrap(),
                OperationKind::MinKeepDim => WgpuB::min_keepdim::<f32>(&input, 1).unwrap(),
                OperationKind::ProdDim => WgpuB::prod_dim::<f32>(&input, 1).unwrap(),
                _ => unreachable!(),
            }
        }
        OperationKind::Normalization => {
            let input = wgpu_f32(&[1, 2], &[1.0, 3.0]);
            let weight = wgpu_f32(&[2], &[1.0, 1.0]);
            WgpuB::layer_norm::<f32>(&input, &weight, None, 1e-5).unwrap()
        }
        OperationKind::Broadcast | OperationKind::BroadcastAs => {
            WgpuB::broadcast_as::<f32>(&wgpu_f32(&[1, 2], &[1.0, 2.0]), &[2, 2]).unwrap()
        }
        OperationKind::Reshape | OperationKind::ReshapeExact => {
            WgpuB::reshape::<f32>(&wgpu_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), &[4]).unwrap()
        }
        OperationKind::MatMul | OperationKind::MatMulExact => {
            let lhs = wgpu_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let rhs = wgpu_f32(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
            WgpuB::matmul::<f32>(&lhs, &rhs).unwrap()
        }
        OperationKind::Conv2d | OperationKind::Conv2dExact => {
            let input = wgpu_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let weight = wgpu_f32(&[1, 1, 1, 1], &[1.0]);
            WgpuB::conv2d::<f32>(&input, &weight, None, 1, 0, 1, 1).unwrap()
        }
        OperationKind::Pool2d | OperationKind::MaxPool2d => {
            let input = wgpu_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            WgpuB::max_pool2d::<f32>(&input, (2, 2), (1, 1), (0, 0), (1, 1)).unwrap()
        }
        OperationKind::AvgPool2d => {
            let input = wgpu_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            WgpuB::avg_pool2d::<f32>(&input, (2, 2), (1, 1), (0, 0)).unwrap()
        }
        _ => panic!("missing WGPU capability execution probe for {operation}"),
    }
}

#[cfg(feature = "wgpu")]
#[test]
fn every_generated_wgpu_row_matches_real_execution() {
    for rule in WGPU_CAPABILITIES {
        let case = query(
            rule.operation,
            DTypeId::F32.descriptor(),
            LayoutClass::Contiguous,
            rule.min_rank,
        );
        assert_eq!(support(DeviceKind::Wgpu, &case), rule.implementation.into());
        let output = execute_wgpu_probe(rule.operation);
        assert_eq!(&*output.shape, wgpu_probe_shape(rule.operation));
        assert_eq!(output.dtype, DTypeId::F32.descriptor());
        assert_eq!(output.device, DeviceId::wgpu(0));
    }
}

#[cfg(feature = "cuda")]
type CudaB =
    incin_backends::cuda::CudaBackendImpl<incin_core::prelude::CudaN<incin_core::typenum::U0>>;

#[cfg(feature = "cuda")]
fn cuda_f32(shape: &[usize], values: &[f32]) -> <CudaB as StorageBackend>::Storage<f32> {
    CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32.descriptor(),
        &DeviceId::cuda(0),
    )
    .unwrap()
}

#[cfg(feature = "cuda")]
fn execute_cuda_probe(
    operation: OperationKind,
    layout: LayoutClass,
) -> <CudaB as StorageBackend>::Storage<f32> {
    let storage = |shape: &[usize], values: &[f32]| {
        let value = cuda_f32(shape, values);
        if layout == LayoutClass::Strided && shape.len() == 2 {
            CudaB::transpose::<f32>(&value, 0, 1).unwrap()
        } else {
            value
        }
    };

    match operation {
        OperationKind::Storage => storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]),
        OperationKind::Fill => {
            CudaB::zeros::<f32>(&[2], DTypeId::F32.descriptor(), &DeviceId::cuda(0)).unwrap()
        }
        OperationKind::Random => {
            CudaB::rand::<f32>(&[2], DTypeId::F32.descriptor(), &DeviceId::cuda(0)).unwrap()
        }
        OperationKind::Pointwise => {
            let lhs = storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let rhs = storage(&[2, 2], &[4.0, 3.0, 2.0, 1.0]);
            CudaB::add::<f32>(&lhs, &rhs).unwrap()
        }
        OperationKind::Reduction => {
            CudaB::sum_all::<f32>(&storage(&[2, 2], &[1.0, 2.0, 3.0, 4.0])).unwrap()
        }
        OperationKind::Normalization => {
            let input = cuda_f32(&[1, 2], &[1.0, 3.0]);
            let weight = cuda_f32(&[2], &[1.0, 1.0]);
            CudaB::layer_norm::<f32>(&input, &weight, None, 1e-5).unwrap()
        }
        OperationKind::Broadcast => {
            CudaB::broadcast_as::<f32>(&cuda_f32(&[1, 2], &[1.0, 2.0]), &[2, 2]).unwrap()
        }
        OperationKind::Reshape => {
            CudaB::reshape::<f32>(&cuda_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]), &[4]).unwrap()
        }
        OperationKind::MatMul => {
            let lhs = cuda_f32(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let rhs = cuda_f32(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
            CudaB::matmul::<f32>(&lhs, &rhs).unwrap()
        }
        OperationKind::Conv2d => {
            let input = cuda_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            let weight = cuda_f32(&[1, 1, 1, 1], &[1.0]);
            CudaB::conv2d::<f32>(&input, &weight, None, 1, 0, 1, 1).unwrap()
        }
        OperationKind::Pool2d => {
            let input = cuda_f32(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
            CudaB::max_pool2d::<f32>(&input, (2, 2), (1, 1), (0, 0), (1, 1)).unwrap()
        }
        _ => panic!("missing CUDA capability execution probe for {operation}"),
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a CUDA device and driver"]
fn every_generated_cuda_row_matches_real_execution_on_hardware() {
    for rule in CUDA_CAPABILITIES {
        for &layout in rule.layouts {
            let case = query(
                rule.operation,
                DTypeId::F32.descriptor(),
                layout,
                rule.min_rank,
            );
            assert_eq!(support(DeviceKind::Cuda, &case), rule.implementation.into());
            let output = execute_cuda_probe(rule.operation, layout);
            assert_eq!(output.dtype, DTypeId::F32.descriptor());
            assert_eq!(output.device, DeviceId::cuda(0));
        }
    }
}

/// Every layout class an exact row does *not* advertise must be refused with
/// the documented typed reason rather than silently admitted.
///
/// The companion to `generated_cpu_rows_match_real_execution_and_output_metadata`:
/// that test proves each advertised layout really executes, this one proves the
/// advertisement is exhaustive, so a row cannot quietly widen to a layout no
/// kernel handles.
#[test]
fn an_unadvertised_exact_layout_returns_the_documented_typed_reason() {
    const ALL_LAYOUTS: &[LayoutClass] = &[
        LayoutClass::Strided,
        LayoutClass::Contiguous,
        LayoutClass::ScalarLeft,
        LayoutClass::ScalarRight,
        LayoutClass::ContiguousLastAxis,
        LayoutClass::RowWise,
        LayoutClass::ChannelWise,
    ];

    for rule in CPU_CAPABILITIES
        .iter()
        .filter(|rule| rule.operation.is_exact())
    {
        // A second row for the same identity may advertise a layout this one
        // omits; the claim under test is the union across rows.
        let advertised = |layout: LayoutClass| {
            CPU_CAPABILITIES
                .iter()
                .any(|other| other.operation == rule.operation && other.layouts.contains(&layout))
        };
        for &layout in ALL_LAYOUTS.iter().filter(|&&layout| !advertised(layout)) {
            let case = query(rule.operation, DTypeId::F32, layout, rule.min_rank);
            let level = registry(DeviceKind::Cpu).support(&case);
            assert!(
                matches!(
                    level,
                    SupportLevel::Unsupported(
                        UnsupportedReason::Layout { .. }
                            | UnsupportedReason::DType { .. }
                            | UnsupportedReason::Operation { .. }
                    )
                ),
                "{} must refuse the unadvertised {layout:?} layout, got {level:?}",
                rule.operation,
            );
        }
    }
}

/// An exact identity with no exact row is unsupported even when its family row
/// would match the same dtype, layout, and rank.
///
/// This is the regression guard for the removed family fallback: `Relu` sits
/// under the `Pointwise` family and `Softmax` under `Normalization`, both of
/// which CPU registers natively for f32/contiguous. Before the exact-identity
/// rule those queries resolved through the family and reported support the CPU
/// descriptor path never admitted.
#[test]
fn an_exact_query_never_resolves_through_a_broad_family_row() {
    // The pairs are derived rather than listed. An earlier version named `relu`
    // and `softmax`, and FND-005 migrated both, which silently turned two of
    // the five cases into assertions about registered operations. Deriving them
    // means the test keeps testing the thing it is named after as the migration
    // proceeds, and fails loudly if it ever runs out of unmigrated examples.
    let registered: BTreeSet<OperationKind> =
        CPU_CAPABILITIES.iter().map(|rule| rule.operation).collect();

    let mut checked = 0;
    for row in OPERATION_CATALOG {
        let operation = row.operation;
        if registered.contains(&operation) {
            continue;
        }
        let family = operation.family();
        let family_case = query(family, DTypeId::F32, LayoutClass::Contiguous, 2);
        if !registry(DeviceKind::Cpu)
            .support(&family_case)
            .is_device_local()
        {
            continue;
        }

        let exact_case = query(operation, DTypeId::F32, LayoutClass::Contiguous, 2);
        assert!(
            matches!(
                registry(DeviceKind::Cpu).support(&exact_case),
                SupportLevel::Unsupported(UnsupportedReason::Operation { operation: reported })
                    if reported == operation
            ),
            "{operation} must not inherit support from its {family} family row",
        );
        checked += 1;
    }

    assert!(
        checked > 0,
        "no unmigrated operation has a registered family row, so this test proves \
         nothing; either the migration is complete and the family rows should be \
         gone, or the derivation above is wrong"
    );
}

/// Every readback row runs, and answers with the host value its type promises.
///
/// This is what the two generated probes cannot check, because their
/// assertions are about storage metadata and these rows produce none. The
/// values are hand-computed so that a readback wired to the wrong operand or
/// the wrong element would fail rather than merely return something.
#[test]
fn the_readback_rows_return_host_values_through_dispatch() {
    // Inference, because a host value is off the tape by definition: the rows
    // carry `training = false` and refusing a training request is the claim,
    // not an inconvenience around it.
    let context =
        ExecutionContext::new(CpuBackendImpl::<Cpu>::new()).with_grad_mode(GradMode::Disabled);
    let scalar = f32_storage(&[1], &[7.5]);
    let vector = f32_storage(&[3], &[1.0, 2.0, 3.0]);
    let scalar_handle = || TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&scalar);
    let vector_handle = || TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&vector);

    let value = dispatch::execute::<op::ToHostFloatScalar, _>(
        &context,
        incin_core::exec::catalog::NoAttributes,
        &[scalar_handle()],
    )
    .unwrap();
    assert!((value - 7.5).abs() < 1e-6);

    let values = dispatch::execute::<op::ToHostFloatVec, _>(
        &context,
        incin_core::exec::catalog::NoAttributes,
        &[vector_handle()],
    )
    .unwrap();
    assert_eq!(values, vec![1.0, 2.0, 3.0]);

    // The integer readbacks refuse a fractional source rather than rounding it,
    // which is the FND-003 conversion policy reaching through the canonical
    // path. Asserted here because it is the behaviour most easily lost by
    // routing a readback to a cast.
    let refused = dispatch::execute::<op::ToHostIntScalar, _>(
        &context,
        incin_core::exec::catalog::NoAttributes,
        &[scalar_handle()],
    );
    assert!(
        refused.is_err(),
        "7.5 has no integer value, so reading it as one must fail"
    );

    let integral = f32_storage(&[1], &[7.0]);
    let whole = dispatch::execute::<op::ToHostIntScalar, _>(
        &context,
        incin_core::exec::catalog::NoAttributes,
        &[TensorHandle::from_storage::<CpuBackendImpl<Cpu>, f32, Local>(&integral)],
    )
    .unwrap();
    assert_eq!(whole, 7);

    let indices = dispatch::execute::<op::ToHostIntVec, _>(
        &context,
        incin_core::exec::catalog::NoAttributes,
        &[vector_handle()],
    )
    .unwrap();
    assert_eq!(indices, vec![1, 2, 3]);

    // Three f32 elements, four bytes each.
    let bytes = dispatch::execute::<op::TensorToBytes, _>(
        &context,
        incin_core::exec::catalog::NoAttributes,
        &[vector_handle()],
    )
    .unwrap();
    assert_eq!(bytes.len(), 12);
}
