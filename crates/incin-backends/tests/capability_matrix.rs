#![cfg(feature = "cpu")]

use incin_backends::capability::{
    CPU_CAPABILITIES, CUDA_CAPABILITIES, WGPU_CAPABILITIES, registry, support,
};
use incin_backends::cpu::{CpuBackendImpl, CpuBuffer, CpuStorage};
use incin_core::backend_authoring::{
    Backend, CreationOps, ModuleOps, NumericOps, ReductionOps, TensorOps,
};
use incin_core::exec::{
    Capabilities, CapabilityQuery, ImplementationKind, LayoutClass, MathMode, SupportLevel,
    UnsupportedReason,
};
use incin_core::prelude::{DTypeId, DeviceId, DeviceKind, Dyn, OperationKind};

fn query(
    operation: OperationKind,
    dtype: DTypeId,
    layout: LayoutClass,
    rank: usize,
) -> CapabilityQuery {
    CapabilityQuery {
        operation,
        dtype,
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
    assert!(matches!(
        support(DeviceKind::Cpu, &case),
        SupportLevel::Unsupported(UnsupportedReason::DType {
            dtype: DTypeId::F64,
            ..
        })
    ));

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
            max: incin_core::prelude::MAX_RANK,
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
        OperationKind::Normalization => &[1, 2],
        OperationKind::Broadcast => &[3, 2],
        OperationKind::BroadcastAs => &[3, 2],
        OperationKind::Reshape => &[4],
        OperationKind::ReshapeExact => &[4],
        OperationKind::MatMulExact => &[2, 2],
        OperationKind::Conv2d => &[1, 1, 2, 2],
        OperationKind::Conv2dExact => &[1, 1, 2, 2],
        OperationKind::Pool2d => &[1, 1, 1, 1],
        OperationKind::MaxPool2d | OperationKind::AvgPool2d => &[1, 1, 1, 1],
        _ => panic!("missing CPU expected shape for {operation}"),
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
        OperationKind::Fill => B::zeros::<f32>(&[2, 2], DTypeId::F32, &device).unwrap(),
        OperationKind::Random => B::rand::<f32>(&[2, 2], DTypeId::F32, &device).unwrap(),
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
        _ => panic!("missing CPU capability execution probe for {operation}"),
    }
}

#[test]
fn generated_cpu_rows_match_real_execution_and_output_metadata() {
    for rule in CPU_CAPABILITIES {
        for &layout in rule.layouts {
            let case = query(rule.operation, DTypeId::F32, layout, rule.min_rank);
            assert!(registry(DeviceKind::Cpu).support(&case).is_device_local());
            let output = execute_cpu_probe(rule.operation, layout);
            assert_eq!(&*output.shape, cpu_probe_shape(rule.operation));
            assert_eq!(output.dtype, DTypeId::F32);
            assert_eq!(output.device, DeviceId::cpu());
        }
    }
}

fn cpu_zeros(dtype: DTypeId, shape: &[usize]) -> CpuStorage {
    type B = CpuBackendImpl;
    if dtype == DTypeId::Q8_0 {
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
                    let shape = if dtype == DTypeId::Q8_0 {
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
                    let shape = if dtype == DTypeId::Q8_0 {
                        &[1, 32][..]
                    } else {
                        &[1, 2][..]
                    };
                    let target = if dtype == DTypeId::Q8_0 {
                        &[2, 32][..]
                    } else {
                        &[2, 2][..]
                    };
                    B::broadcast_as::<Dyn>(&cpu_zeros(dtype, shape), target).unwrap()
                }
                OperationKind::Reshape | OperationKind::ReshapeExact => {
                    let shape = if dtype == DTypeId::Q8_0 {
                        &[32][..]
                    } else {
                        &[2, 2][..]
                    };
                    let target = if dtype == DTypeId::Q8_0 {
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
                | OperationKind::MaxPool2d
                | OperationKind::AvgPool2d => {
                    execute_cpu_probe(rule.operation, LayoutClass::Contiguous)
                }
                _ => panic!("missing dtype conformance probe for {}", rule.operation),
            };
            assert_eq!(output.dtype, dtype, "{} {dtype:?}", rule.operation);
        }
    }
}

#[cfg(feature = "wgpu")]
type WgpuB =
    incin_backends::wgpu::WgpuBackendImpl<f32, incin_core::prelude::WgpuN<incin_core::typenum::U0>>;

#[cfg(feature = "wgpu")]
fn wgpu_f32(shape: &[usize], values: &[f32]) -> <WgpuB as Backend>::Storage<f32> {
    WgpuB::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32,
        &DeviceId::wgpu(0),
    )
    .unwrap()
}

#[cfg(feature = "wgpu")]
fn wgpu_probe_shape(operation: OperationKind) -> &'static [usize] {
    match operation {
        OperationKind::Storage
        | OperationKind::Fill
        | OperationKind::Random
        | OperationKind::Pointwise => &[2],
        OperationKind::Add | OperationKind::Sub | OperationKind::Mul | OperationKind::Div => &[2],
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
fn execute_wgpu_probe(operation: OperationKind) -> <WgpuB as Backend>::Storage<f32> {
    match operation {
        OperationKind::Storage => wgpu_f32(&[2], &[1.0, 2.0]),
        OperationKind::Fill => WgpuB::zeros::<f32>(&[2], DTypeId::F32, &DeviceId::wgpu(0)).unwrap(),
        OperationKind::Random => {
            WgpuB::rand::<f32>(&[2], DTypeId::F32, &DeviceId::wgpu(0)).unwrap()
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
            DTypeId::F32,
            LayoutClass::Contiguous,
            rule.min_rank,
        );
        assert_eq!(support(DeviceKind::Wgpu, &case), rule.implementation.into());
        let output = execute_wgpu_probe(rule.operation);
        assert_eq!(&*output.shape, wgpu_probe_shape(rule.operation));
        assert_eq!(output.dtype, DTypeId::F32);
        assert_eq!(output.device, DeviceId::wgpu(0));
    }
}

#[cfg(feature = "cuda")]
type CudaB =
    incin_backends::cuda::CudaBackendImpl<f32, incin_core::prelude::CudaN<incin_core::typenum::U0>>;

#[cfg(feature = "cuda")]
fn cuda_f32(shape: &[usize], values: &[f32]) -> <CudaB as Backend>::Storage<f32> {
    CudaB::from_bytes::<f32>(
        bytemuck::cast_slice(values),
        shape,
        DTypeId::F32,
        &DeviceId::cuda(0),
    )
    .unwrap()
}

#[cfg(feature = "cuda")]
fn execute_cuda_probe(
    operation: OperationKind,
    layout: LayoutClass,
) -> <CudaB as Backend>::Storage<f32> {
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
        OperationKind::Fill => CudaB::zeros::<f32>(&[2], DTypeId::F32, &DeviceId::cuda(0)).unwrap(),
        OperationKind::Random => {
            CudaB::rand::<f32>(&[2], DTypeId::F32, &DeviceId::cuda(0)).unwrap()
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
            let case = query(rule.operation, DTypeId::F32, layout, rule.min_rank);
            assert_eq!(support(DeviceKind::Cuda, &case), rule.implementation.into());
            let output = execute_cuda_probe(rule.operation, layout);
            assert_eq!(output.dtype, DTypeId::F32);
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
    for (operation, family) in [
        (OperationKind::Relu, OperationKind::Pointwise),
        (OperationKind::Softmax, OperationKind::Normalization),
        (OperationKind::Gather, OperationKind::Storage),
        (OperationKind::Dot, OperationKind::Reduction),
        (OperationKind::TransposeExact, OperationKind::Storage),
    ] {
        assert_eq!(operation.family(), family, "{operation} family drifted");

        let family_case = query(family, DTypeId::F32, LayoutClass::Contiguous, 2);
        assert!(
            registry(DeviceKind::Cpu)
                .support(&family_case)
                .is_device_local(),
            "{family} must stay registered for this test to prove anything",
        );

        let exact_case = query(operation, DTypeId::F32, LayoutClass::Contiguous, 2);
        assert!(
            matches!(
                registry(DeviceKind::Cpu).support(&exact_case),
                SupportLevel::Unsupported(UnsupportedReason::Operation { operation: reported })
                    if reported == operation
            ),
            "{operation} must not inherit support from its {family} family row",
        );
    }
}
