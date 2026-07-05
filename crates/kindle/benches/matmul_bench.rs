use criterion::{black_box, criterion_group, criterion_main, Criterion};
use kindle::prelude::*;

fn bench_matmul_kindle(c: &mut Criterion) {
    // Generate static shapes: A(128, 256), B(256, 512) -> C(128, 512)
    type AType = Tensor<(Const<128>, Const<256>), CandleBackend<f32, Cpu>>;
    type BType = Tensor<(Const<256>, Const<512>), CandleBackend<f32, Cpu>>;

    let a = AType::static_ones().unwrap();
    let b = BType::static_ones().unwrap();

    c.bench_function("kindle_static_matmul_128_256_512", |bencher| {
        bencher.iter(|| {
            let result = a.matmul(&b).unwrap();
            black_box(result);
        })
    });
}

fn bench_matmul_candle_raw(c: &mut Criterion) {
    let device = candle_core::Device::Cpu;
    let a = candle_core::Tensor::ones((128, 256), candle_core::DType::F32, &device).unwrap();
    let b = candle_core::Tensor::ones((256, 512), candle_core::DType::F32, &device).unwrap();

    c.bench_function("candle_raw_matmul_128_256_512", |bencher| {
        bencher.iter(|| {
            let result = a.matmul(&b).unwrap();
            black_box(result);
        })
    });
}

criterion_group!(benches, bench_matmul_kindle, bench_matmul_candle_raw);
criterion_main!(benches);
