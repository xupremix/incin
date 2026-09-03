//! Tests for the `#[autotune(...)]` proc-macro attribute.

use incin_macros::autotune;

#[autotune(
    key = "test_kernel_tile",
    params = [(16, 16), (32, 32), (64, 64)],
    policy = heuristic
)]
fn sample_tiled_kernel(m: usize, n: usize) -> usize {
    m * n
}

#[autotune(
    key = "test_kernel_array",
    params = [32, 64, 128, 256],
    policy = warmup
)]
fn sample_chunk_kernel(len: usize) -> usize {
    len / 2
}

#[test]
fn test_autotune_macro_expansion() {
    assert_eq!(sample_tiled_kernel(4, 5), 20);
    assert_eq!(sample_chunk_kernel(100), 50);
}
