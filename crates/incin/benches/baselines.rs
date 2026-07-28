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
type WgpuB = incin_backends::wgpu::WgpuBackendImpl<f32, incin::WgpuN<incin::typenum::U0>>;

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
            b.iter(|| black_box(input.add(black_box(&input)).unwrap()))
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
        b.iter(|| black_box(input.add(black_box(&input)).unwrap()))
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

criterion_group!(
    baselines,
    capability_baselines,
    eager_baselines,
    gpu_baselines
);
criterion_main!(baselines);
