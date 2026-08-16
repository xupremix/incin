//! GOV-004's stable single-device baseline surface.
//!
//! Benchmark IDs are part of the regression-data contract consumed by
//! GOV-005. Keep the family/workload/shape spelling stable; add a new ID when
//! semantics change instead of silently reusing an old series.

use criterion::{BatchSize, BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use incin::prelude::*;
use std::time::Duration;

type B = incin::DefaultBackend;

#[cfg(feature = "wgpu")]
type WgpuB = incin_backends::wgpu::WgpuBackendImpl<incin::WgpuN<incin::typenum::U0>>;

#[cfg(feature = "cuda")]
type CudaB = incin_backends::cuda::CudaBackendImpl<incin::CudaN<incin::typenum::U0>>;

#[cfg(feature = "metal")]
type MetalB = incin_backends::metal::MetalBackendImpl<incin::MetalN<incin::typenum::U0>>;

fn capability_baselines(c: &mut Criterion) {
    let mut group = c.benchmark_group("capability");
    group.bench_function("cpu/f32_create", |b| {
        b.iter(|| black_box(Tensor::<s![1], B>::zeros(()).unwrap()))
    });
    group.bench_function("cpu/u32_create", |b| {
        b.iter(|| black_box(Tensor::<s![1], B, u32>::zeros(()).unwrap()))
    });
    group.finish();
}

fn eager_baselines(c: &mut Criterion) {
    let mut group = c.benchmark_group("eager");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    for elements in [1_024usize, 65_536] {
        let input = Tensor::<Dyn, B>::ones(vec![elements]).unwrap();
        group.bench_with_input(BenchmarkId::new("add_f32", elements), &elements, |b, _| {
            b.iter(|| black_box(&input) + black_box(&input))
        });
        group.bench_with_input(BenchmarkId::new("sum_f32", elements), &elements, |b, _| {
            b.iter_batched(
                || input.clone(),
                |owned| black_box(owned.sum_all().unwrap()),
                BatchSize::SmallInput,
            )
        });
    }

    for size in [16usize, 64] {
        let lhs = Tensor::<Dyn, B>::ones(vec![size, size]).unwrap();
        let rhs = Tensor::<Dyn, B>::ones(vec![size, size]).unwrap();
        group.bench_with_input(BenchmarkId::new("matmul_f32", size), &size, |b, _| {
            b.iter(|| black_box(lhs.matmul(black_box(&rhs)).unwrap()))
        });
    }

    // The batched path, which reaches an entirely different driver than the
    // rank-2 series above and is what an attention or batched-linear layer
    // actually runs. Both operands carry the same batch axis because the
    // typed `Tensor::matmul` still requires the output to keep the left
    // operand's shape, so a broadcast batch axis is not reachable from here.
    {
        let lhs = Tensor::<Dyn, B>::ones(vec![32, 64, 64]).unwrap();
        let rhs = Tensor::<Dyn, B>::ones(vec![32, 64, 64]).unwrap();
        group.bench_function("batched_matmul_f32/32x64", |b| {
            b.iter(|| black_box(lhs.matmul(black_box(&rhs)).unwrap()))
        });
    }
    group.finish();
}

#[cfg(feature = "wgpu")]
fn gpu_baselines(c: &mut Criterion) {
    let mut capability = c.benchmark_group("capability");
    capability.bench_function("wgpu/f32_create", |b| {
        b.iter(|| black_box(Tensor::<s![1], WgpuB>::zeros(()).unwrap()))
    });
    capability.finish();

    let mut group = c.benchmark_group("gpu");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);

    let input = Tensor::<Dyn, WgpuB>::ones(vec![65_536]).unwrap();
    group.bench_function("add_f32/65536", |b| {
        b.iter(|| black_box(&input) + black_box(&input))
    });

    let lhs = Tensor::<Dyn, WgpuB>::ones(vec![64, 64]).unwrap();
    let rhs = Tensor::<Dyn, WgpuB>::ones(vec![64, 64]).unwrap();
    group.bench_function("matmul_f32/64", |b| {
        b.iter(|| black_box(lhs.matmul(black_box(&rhs)).unwrap()))
    });
    group.finish();
}

#[cfg(not(feature = "wgpu"))]
fn gpu_baselines(_: &mut Criterion) {}

/// CUDA counterparts of the WGPU series.
///
/// The IDs carry the backend even inside the `gpu` group, unlike the WGPU rows
/// whose spelling is frozen by GOV-005. Two accelerators sharing one Criterion
/// ID would collide the moment a build enabled both features, and the budget
/// key is `(backend, id)` rather than `id` alone, so the collision would be
/// silent in `target/criterion` and invisible to the gate.
#[cfg(feature = "cuda")]
fn cuda_baselines(c: &mut Criterion) {
    let mut capability = c.benchmark_group("capability");
    capability.bench_function("cuda/f32_create", |b| {
        b.iter(|| black_box(Tensor::<s![1], CudaB>::zeros(()).unwrap()))
    });
    capability.finish();

    let mut group = c.benchmark_group("gpu");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);

    let input = Tensor::<Dyn, CudaB>::ones(vec![65_536]).unwrap();
    group.bench_function("cuda/add_f32/65536", |b| {
        b.iter(|| black_box(input.add(black_box(&input)).unwrap()))
    });

    let lhs = Tensor::<Dyn, CudaB>::ones(vec![64, 64]).unwrap();
    let rhs = Tensor::<Dyn, CudaB>::ones(vec![64, 64]).unwrap();
    group.bench_function("cuda/matmul_f32/64", |b| {
        b.iter(|| black_box(lhs.matmul(black_box(&rhs)).unwrap()))
    });
    group.finish();
}

#[cfg(not(feature = "cuda"))]
fn cuda_baselines(_: &mut Criterion) {}

/// Metal counterparts of the GPU series.
#[cfg(feature = "metal")]
fn metal_baselines(c: &mut Criterion) {
    let mut capability = c.benchmark_group("capability");
    capability.bench_function("metal/f32_create", |b| {
        b.iter(|| black_box(Tensor::<s![1], MetalB>::zeros(()).unwrap()))
    });
    capability.finish();

    let mut group = c.benchmark_group("gpu");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(20);

    let input = Tensor::<Dyn, MetalB>::ones(vec![65_536]).unwrap();
    group.bench_function("metal/add_f32/65536", |b| {
        b.iter(|| black_box(input.add(black_box(&input)).unwrap()))
    });

    let lhs = Tensor::<Dyn, MetalB>::ones(vec![64, 64]).unwrap();
    let rhs = Tensor::<Dyn, MetalB>::ones(vec![64, 64]).unwrap();
    group.bench_function("metal/matmul_f32/64", |b| {
        b.iter(|| black_box(lhs.matmul(black_box(&rhs)).unwrap()))
    });
    group.finish();
}

#[cfg(not(feature = "metal"))]
fn metal_baselines(_: &mut Criterion) {}

criterion_group!(
    baselines,
    capability_baselines,
    eager_baselines,
    gpu_baselines,
    cuda_baselines,
    metal_baselines
);
criterion_main!(baselines);
