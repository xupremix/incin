//! Runtime half of issue #93's tensor-level quantized contract (Decisions 3-5).
//!
//! The compile-fail fixtures in `tests/compile_fail/` pin the static half:
//! call-site dtype refusal and the const block-divisibility assert. These
//! tests pin what is only knowable at runtime instead -- `Dyn` shapes, `Dyn`
//! dtypes, the typed error text, and the `Linear<..., K = Q8_0>` type path
//! (Decision 8) -- each asserting a typed error, never a panic and never a
//! silent wrong result.

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::prelude::*;

type B = CpuBackendImpl;

#[test]
fn roundtrip_1d_preserves_values_within_block_tolerance() {
    let data: Vec<f32> = (0..64).map(|i| (i as f32) - 32.0).collect();
    let t = Tensor::<s![64], B>::from_slice(&data, ()).unwrap();
    let q = t.quantize(-1).unwrap();
    assert_eq!(q.dtype(), DTypeId::Q8_0.descriptor());
    let back = q.dequantize::<f32>().unwrap();
    let out = back.to_vec1::<f32>().unwrap();
    assert_eq!(out.len(), 64);
    for (original, decoded) in data.iter().zip(out.iter()) {
        // One f16 block scale over 32 values: error is bounded by half the
        // per-block step, far below this tolerance.
        assert!(
            (original - decoded).abs() < 0.5,
            "roundtrip drift: {original} vs {decoded}"
        );
    }
}

#[test]
fn roundtrip_2d_blocks_along_the_last_axis() {
    let data: Vec<f32> = (0..128).map(|i| (i as f32) * 0.1 - 6.0).collect();
    let t = Tensor::<s![2, 64], B>::from_slice(&data, ()).unwrap();
    let q = t.quantize(-1).unwrap();
    let back = q.dequantize::<f32>().unwrap();
    let out = back.to_vec1::<f32>().unwrap();
    assert_eq!(out.len(), 128);
    for (original, decoded) in data.iter().zip(out.iter()) {
        assert!(
            (original - decoded).abs() < 0.5,
            "roundtrip drift: {original} vs {decoded}"
        );
    }
}

#[test]
fn quantize_refuses_a_non_last_axis() {
    let t = Tensor::<s![2, 64], B>::zeros(()).unwrap();
    let err = t.quantize(0).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("last axis"),
        "expected the block-axis rule in the message, got: {msg}"
    );
    let err = t.quantize(-2).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("last axis"),
        "expected the block-axis rule in the message, got: {msg}"
    );
}

#[test]
fn quantize_refuses_an_axis_out_of_bounds() {
    let t = Tensor::<s![2, 64], B>::zeros(()).unwrap();
    let err = t.quantize(5).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("out of bounds"),
        "expected an axis-bounds error, got: {msg}"
    );
}

#[test]
fn quantize_on_dyn_extent_reports_the_block_rule() {
    // 48 is not a multiple of 32; only the runtime shape knows that here.
    let t = Tensor::<Dyn, B>::zeros(vec![2, 48]).unwrap();
    let err = t.quantize(-1).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains('4') && msg.contains('8') && msg.contains("block size 32"),
        "expected axis, extent, and block size in the message, got: {msg}"
    );
}

#[test]
fn quantize_roundtrip_on_dyn_shape() {
    let t = Tensor::<Dyn, B>::zeros(vec![2, 64]).unwrap();
    let q = t.quantize(-1).unwrap();
    assert_eq!(q.dtype(), DTypeId::Q8_0.descriptor());
    let back = q.dequantize::<f32>().unwrap();
    let out = back.to_vec1::<f32>().unwrap();
    assert_eq!(out.len(), 128);
    assert!(out.iter().all(|v| *v == 0.0));
}

#[test]
fn quantize_on_integer_dyn_dtype_is_refused_at_runtime() {
    // `Dyn` admits the call so the catalog's runtime check carries Decision
    // 3's runtime half: a typed descriptor error, not a panic.
    let t = Tensor::<Dyn, B, Dyn>::zeros((vec![64], DTypeId::U8)).unwrap();
    let err = t.quantize(-1).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("floating"),
        "expected the float-input rule in the message, got: {msg}"
    );
}

#[test]
fn dequantize_on_integer_dyn_dtype_is_refused_at_runtime() {
    let t = Tensor::<Dyn, B, Dyn>::zeros((vec![64], DTypeId::U8)).unwrap();
    let err = t.dequantize::<f32>().unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("q8_0"),
        "expected the q8_0-input rule in the message, got: {msg}"
    );
}

#[test]
fn quantize_accepts_dyn_dtype_that_resolves_to_float() {
    let t = Tensor::<Dyn, B, Dyn>::zeros((vec![64], DTypeId::F32)).unwrap();
    let q = t.quantize(-1).unwrap();
    assert_eq!(q.dtype(), DTypeId::Q8_0.descriptor());
    let back = q.dequantize::<f32>().unwrap();
    assert_eq!(back.to_vec1::<f32>().unwrap().len(), 64);
}

#[test]
fn linear_q8_0_type_path_compiles_and_refuses_init_descriptively() {
    // Decision 8: `Linear<..., K = Q8_0>` must be a spelling the type system
    // accepts end to end -- parameter type, `build` bounds, and forward
    // bounds all monomorphize. Runtime then refuses Q8_0 draws at the
    // capability row with a typed error naming the dtype, not a panic.
    use incin_core::nn::Linear;
    use incin_core::nn::optional::True;

    type QLinear = Linear<s![32, 4], B, True, Q8_0>;

    let err = match QLinear::build(()) {
        Ok(_) => panic!("Linear<Q8_0>::build must refuse Q8_0 init at the capability row"),
        Err(err) => err,
    };
    let msg = err.to_string();
    assert!(
        msg.contains("q8_0") && msg.contains("unsupported"),
        "expected a typed capability refusal naming q8_0, got: {msg}"
    );
}
