//! Integration coverage for `test_simd_lanes_positive_and_divisible` on the documented public surface.
#![allow(clippy::assertions_on_constants)]

use half::{bf16, f16};
use incin_backends::simd::{simd_lanes, simd_vector_bytes};

#[test]
fn test_simd_lanes_positive_and_divisible() {
    let lanes_f32 = simd_lanes::<f32>();
    let lanes_f64 = simd_lanes::<f64>();
    let lanes_f16 = simd_lanes::<f16>();
    let lanes_bf16 = simd_lanes::<bf16>();
    let lanes_i32 = simd_lanes::<i32>();
    let lanes_u8 = simd_lanes::<u8>();

    assert!(lanes_f32 >= 1);
    assert!(lanes_f64 >= 1);
    assert!(lanes_f16 >= 1);
    assert!(lanes_bf16 >= 1);
    assert!(lanes_i32 >= 1);
    assert!(lanes_u8 >= 1);

    let vector_bytes = simd_vector_bytes();
    assert!(vector_bytes >= 1);
    assert!(vector_bytes.is_power_of_two());

    // Verify relationship between dtype sizes and lane counts
    assert_eq!(lanes_f32 * 4, vector_bytes);
    assert_eq!(lanes_f64 * 8, vector_bytes);
    assert_eq!(lanes_f16 * 2, vector_bytes);
    assert_eq!(lanes_bf16 * 2, vector_bytes);
    assert_eq!(lanes_i32 * 4, vector_bytes);
    assert_eq!(lanes_u8, vector_bytes);
}

#[test]
fn test_simd_lanes_const_eval() {
    const LANES_F32: usize = simd_lanes::<f32>();
    const LANES_F64: usize = simd_lanes::<f64>();
    assert!(LANES_F32 >= 1);
    assert!(LANES_F64 >= 1);
}
