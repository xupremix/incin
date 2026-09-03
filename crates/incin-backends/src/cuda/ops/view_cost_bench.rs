//! Does materialising a transpose cost more than reading it strided?
//!
//! This is the measurement that gates issue #113 and, behind it, whether typed
//! non-copying views are worth pursuing. The two backends currently disagree:
//! CPU `transpose` returns a view over permuted strides, CUDA's runs a
//! permutation kernel into a fresh contiguous buffer. Both are defensible in
//! the abstract, so the question is empirical.
//!
//! What is compared, for the same logical work -- a transpose followed by one
//! pointwise pass over the result:
//!
//! * **materialise**: permute into a fresh buffer, then run the *dense* kernel
//!   over it. Two kernels, one extra `numel`-sized allocation, and every
//!   subsequent read is contiguous.
//! * **view**: relabel the metadata with permuted strides, then run the
//!   *strided* kernel over it. One kernel, no allocation, and every read pays
//!   an index computation and an uncoalesced access pattern.
//!
//! The view is built through `try_from_parts` rather than through `transpose`,
//! because on CUDA `transpose` is precisely the thing being measured against.
//!
//! Reported rather than asserted. A threshold would encode this machine's
//! memory system as a contract, and the useful output is the ratio and how it
//! moves with size, not a pass or fail. Run with
//! `cargo test -p incin-backends --features cuda --lib -- --ignored --nocapture view_cost`.

use super::elementwise::launch_unary_body;
use crate::codegen::catalog;
use crate::cuda::backend::{cuda_from_f32, download_f32_host};
use crate::cuda::storage::CudaStorage;
use crate::kernel::{KernelSpecialization, lower_unary_body};
use alloc::vec::Vec;
use incin_core::tensor::device::DeviceId;
use incin_core::tensor::dtype::DTypeId;
use std::time::Instant;

/// Aborts unless a CUDA device is present.
///
/// Replaces a `has_cuda() -> bool` predicate that callers used to skip with an
/// early `return`. Every caller is `#[ignore]`d, so reaching one is a deliberate
/// request for the hardware run, and returning early there reports `ok` for a
/// test that launched nothing -- the pattern that kept three real CUDA defects
/// green for as long as they existed.
///
/// # Panics
///
/// If no CUDA device can be opened on ordinal 0.
fn require_cuda() {
    assert!(
        cudarc::driver::CudaContext::new(0).is_ok(),
        "no CUDA device, but this test is #[ignore]d -- running it is an explicit request for hardware. Skipping here would report `ok` for a test that launched nothing."
    );
}

fn storage(shape: &[usize], values: Vec<f32>) -> CudaStorage {
    cuda_from_f32(
        shape,
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
        values,
        "bench",
    )
    .unwrap()
}

/// Runs `iterations` of a closure and returns the mean wall time in
/// microseconds, synchronising the stream once at the end.
///
/// An earlier version forced completion with a device-to-host copy of the
/// output every round. That is a barrier, but it is also four megabytes of
/// transfer at a million elements -- a large constant added to both arms, which
/// compresses every ratio toward one and hides exactly the differences the
/// benchmark exists to find. Synchronising the stream instead measures the work
/// and nothing else.
fn timed(iterations: u32, mut body: impl FnMut() -> CudaStorage) -> f64 {
    let context = cudarc::driver::CudaContext::new(0).unwrap();
    let stream = context.default_stream();

    // Warm up: the first call compiles and caches the kernel, which would
    // otherwise dominate the first measurement entirely.
    let warm = body();
    let _ = download_f32_host(&warm).unwrap();

    let start = Instant::now();
    for _ in 0..iterations {
        let _out = body();
    }
    stream.synchronize().unwrap();
    start.elapsed().as_secs_f64() * 1e6 / f64::from(iterations)
}

/// The counter-hypothesis: a copy amortises when the result is read more than
/// once.
///
/// The single-pass comparison is the view's best case -- it pays one strided
/// read against the copy's read-plus-write. If the transposed result is
/// consumed `k` times, the copy pays its cost once and every subsequent read is
/// coalesced, while the view pays the strided penalty `k` times over. This
/// measures where the two cross.
#[test]
#[ignore = "requires CUDA hardware"]
fn view_cost_amortised_over_repeated_reads() {
    require_cuda();
    let body = lower_unary_body(&catalog::unary_forward("neg").unwrap(), DTypeId::F32).unwrap();
    let iterations = 100;
    let (rows, cols) = (1024usize, 1024usize);
    let numel = rows * cols;
    let values: Vec<f32> = (0..numel).map(|index| index as f32 * 0.001).collect();
    let source = storage(&[rows, cols], values);

    println!();
    println!("{numel} elements, transpose then k pointwise passes, mean microseconds");
    println!(
        "{:>4} {:>14} {:>14} {:>10}",
        "k", "materialise", "strided view", "view/mat"
    );

    for &passes in &[1u32, 2, 4, 8] {
        let materialise = timed(iterations, || {
            let transposed = crate::cuda::backend::CudaBackendImpl::<
                incin_core::tensor::device::Cuda,
            >::transpose::<f32>(&source, 0, 1)
            .unwrap();
            let mut last =
                launch_unary_body("amort_mat", &body, &transposed, KernelSpecialization::NONE)
                    .unwrap();
            for _ in 1..passes {
                last =
                    launch_unary_body("amort_mat", &body, &transposed, KernelSpecialization::NONE)
                        .unwrap();
            }
            last
        });

        let view = timed(iterations, || {
            let viewed = CudaStorage::try_from_parts(
                source.buffer.clone(),
                alloc::vec![cols, rows],
                alloc::vec![1, cols],
                0,
            )
            .unwrap();
            let mut last =
                launch_unary_body("amort_view", &body, &viewed, KernelSpecialization::NONE)
                    .unwrap();
            for _ in 1..passes {
                last = launch_unary_body("amort_view", &body, &viewed, KernelSpecialization::NONE)
                    .unwrap();
            }
            last
        });

        println!(
            "{passes:>4} {materialise:>14.1} {view:>14.1} {:>10.2}",
            view / materialise
        );
    }
    println!();
}

/// Does folding proven extents into the strided walk actually pay?
///
/// The folding replaces the per-axis modulo and division -- both by values read
/// from device memory -- with literal divisors the compiler lowers to
/// multiply-and-shift, and drops the `shape` upload that fed them. It was
/// written before anything could reach the strided path on CUDA, because every
/// operation materialised; `transpose_view` makes it reachable, so this is the
/// first time the work can be measured rather than argued for.
///
/// Both arms run the identical kernel body over the identical strided view. The
/// only difference is whether the extents were proven.
#[test]
#[ignore = "requires CUDA hardware"]
fn extent_folding_on_the_strided_path() {
    require_cuda();
    let body = lower_unary_body(&catalog::unary_forward("neg").unwrap(), DTypeId::F32).unwrap();
    let iterations = 200;

    println!();
    println!("strided walk over a transposed view, mean microseconds over {iterations} iterations");
    println!(
        "{:>12} {:>16} {:>14} {:>12}",
        "elements", "loaded divisors", "folded", "folded/loaded"
    );

    for &(rows, cols) in &[(16usize, 16usize), (64, 64), (256, 256), (1024, 1024)] {
        let numel = rows * cols;
        let values: Vec<f32> = (0..numel).map(|index| index as f32 * 0.001).collect();
        let source = storage(&[rows, cols], values);
        let view = || {
            CudaStorage::try_from_parts(
                source.buffer.clone(),
                alloc::vec![cols, rows],
                alloc::vec![1, cols],
                0,
            )
            .unwrap()
        };

        // The extents the shape type would settle for the transposed view.
        let extents: &'static [Option<usize>] =
            alloc::boxed::Box::leak(alloc::vec![Some(cols), Some(rows)].into_boxed_slice());
        let proven = KernelSpecialization {
            static_numel: Some(numel),
            static_extents: extents,
        };

        let loaded = timed(iterations, || {
            launch_unary_body("fold_dyn", &body, &view(), KernelSpecialization::NONE).unwrap()
        });
        let folded = timed(iterations, || {
            launch_unary_body("fold_static", &body, &view(), proven).unwrap()
        });

        println!(
            "{numel:>12} {loaded:>16.1} {folded:>14.1} {:>12.2}",
            folded / loaded
        );
    }
    println!();
}

#[test]
#[ignore = "requires CUDA hardware"]
fn view_cost_materialise_versus_strided_read() {
    require_cuda();
    let body = lower_unary_body(&catalog::unary_forward("neg").unwrap(), DTypeId::F32).unwrap();
    let iterations = 200;

    println!();
    println!("transpose + one pointwise pass, mean microseconds over {iterations} iterations");
    println!(
        "{:>12} {:>14} {:>14} {:>10}",
        "elements", "materialise", "strided view", "view/mat"
    );

    for &(rows, cols) in &[(64usize, 64usize), (256, 256), (1024, 1024), (2048, 2048)] {
        let numel = rows * cols;
        let values: Vec<f32> = (0..numel).map(|index| index as f32 * 0.001).collect();
        let source = storage(&[rows, cols], values);

        // Materialise: permutation kernel into a fresh contiguous buffer, then
        // the dense pointwise kernel over it.
        let materialise = timed(iterations, || {
            let transposed = crate::cuda::backend::CudaBackendImpl::<
                incin_core::tensor::device::Cuda,
            >::transpose::<f32>(&source, 0, 1)
            .unwrap();
            launch_unary_body("bench_mat", &body, &transposed, KernelSpecialization::NONE).unwrap()
        });

        // View: relabel to [cols, rows] over strides [1, cols], no copy, then
        // the strided pointwise kernel over it.
        let view = timed(iterations, || {
            let viewed = CudaStorage::try_from_parts(
                source.buffer.clone(),
                alloc::vec![cols, rows],
                alloc::vec![1, cols],
                0,
            )
            .unwrap();
            launch_unary_body("bench_view", &body, &viewed, KernelSpecialization::NONE).unwrap()
        });

        println!(
            "{numel:>12} {materialise:>14.1} {view:>14.1} {:>10.2}",
            view / materialise
        );
    }
    println!();
}
