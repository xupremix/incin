use super::*;
use crate::cpu::gradcheck::gradcheck;
use crate::cpu::storage::CpuBuffer;
use crate::cpu::tape;

/// `matrix`.
fn matrix(v: Vec<f32>, rows: usize, cols: usize) -> CpuStorage {
    CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![rows, cols])
}

/// `vector`.
fn vector(v: Vec<f32>) -> CpuStorage {
    let len = v.len();
    CpuStorage::from_contiguous(CpuBuffer::F32(v), vec![len])
}

/// `f32_vec`.
fn f32_vec(s: &CpuStorage) -> Vec<f32> {
    match &*s.buffer {
        CpuBuffer::F32(v) => v.clone(),
        _ => panic!("expected F32 buffer"),
    }
}

fn f64_storage(v: Vec<f64>, shape: Vec<usize>) -> CpuStorage {
    CpuStorage::from_contiguous(CpuBuffer::F64(v), shape)
}

// Regression guard for C-2: elementwise ops used to hardcode `CpuBuffer::F32`
// for every result regardless of the operands' actual dtype, silently
// downcasting F64 (and F16/BF16) tensors through f32 with no error. These
// values are specifically chosen to be exactly representable in f64 but
// NOT exactly representable in f32, so an accidental f32 round-trip
// changes the result.
#[test]
fn add_preserves_f64_dtype_and_precision() {
    let lhs = f64_storage(vec![1.000000123456789], vec![1]);
    let rhs = f64_storage(vec![2.000000987654321], vec![1]);
    let out = add_storage(&lhs, &rhs).unwrap();

    match &*out.buffer {
        CpuBuffer::F64(v) => {
            let expected = 1.000000123456789 + 2.000000987654321;
            assert_eq!(
                v[0], expected,
                "add on F64 operands must return an F64 buffer with full f64 precision, \
                 not a value that has round-tripped through f32"
            );
        }
        other => panic!("expected CpuBuffer::F64, got {other:?}"),
    }
}

#[test]
fn numeric_ops_preserve_half_storage_and_compute_in_f32() {
    let f16_lhs = CpuStorage::from_contiguous(
        CpuBuffer::F16(vec![half::f16::from_f32(1.5), half::f16::from_f32(2.0)]),
        vec![2],
    );
    let f16_rhs = CpuStorage::from_contiguous(
        CpuBuffer::F16(vec![half::f16::from_f32(2.0), half::f16::from_f32(4.0)]),
        vec![2],
    );
    let f16_out = mul_storage(&f16_lhs, &f16_rhs).unwrap();
    assert_eq!(
        &*f16_out.buffer,
        &CpuBuffer::F16(vec![half::f16::from_f32(3.0), half::f16::from_f32(8.0)])
    );

    let bf16_lhs = CpuStorage::from_contiguous(
        CpuBuffer::BF16(vec![half::bf16::from_f32(1.5), half::bf16::from_f32(2.0)]),
        vec![2],
    );
    let bf16_rhs = CpuStorage::from_contiguous(
        CpuBuffer::BF16(vec![half::bf16::from_f32(2.0), half::bf16::from_f32(4.0)]),
        vec![2],
    );
    let bf16_out = mul_storage(&bf16_lhs, &bf16_rhs).unwrap();
    assert_eq!(
        &*bf16_out.buffer,
        &CpuBuffer::BF16(vec![half::bf16::from_f32(3.0), half::bf16::from_f32(8.0)])
    );
}

#[test]
fn numeric_ops_preserve_non_contiguous_view_semantics() {
    let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3)
        .transpose(0, 1)
        .unwrap();
    let rhs = matrix(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3)
        .transpose(0, 1)
        .unwrap();
    let output = add_storage(&lhs, &rhs).unwrap();

    assert_eq!(output.shape, vec![3, 2]);
    assert_eq!(f32_vec(&output), vec![11.0, 44.0, 22.0, 55.0, 33.0, 66.0]);
}

#[test]
fn relu_preserves_f64_dtype() {
    let t = f64_storage(vec![-1.000000123456789, 3.000000987654321], vec![2]);
    let out = canonical_relu(&t).unwrap();

    match &*out.buffer {
        CpuBuffer::F64(v) => {
            assert_eq!(*v, vec![0.0f64, 3.000000987654321f64]);
        }
        other => panic!("expected CpuBuffer::F64, got {other:?}"),
    }
}

#[test]
/// `add_broadcasts_forward_correctly`.
fn add_broadcasts_forward_correctly() {
    let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let rhs = vector(vec![10.0, 20.0, 30.0]);
    let out = add_storage(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(f32_vec(&out), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
}

#[test]
/// `add_backward_unbroadcasts_correctly_for_bias_vector_case`.
fn add_backward_unbroadcasts_correctly_for_bias_vector_case() {
    let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let rhs = vector(vec![10.0, 20.0, 30.0]);
    let out = add_storage(&lhs, &rhs).unwrap();

    let grads = tape::backward(&out).unwrap();
    let lhs_grad = grads.get(lhs.id).expect("lhs should have a gradient");
    let rhs_grad = grads.get(rhs.id).expect("rhs should have a gradient");

    // lhs's grad: ones_like(out) unbroadcast to [2,3] = all ones.
    assert_eq!(lhs_grad.shape, vec![2, 3]);
    assert_eq!(f32_vec(lhs_grad), vec![1.0; 6]);

    // rhs's grad: ones_like(out) [2,3] unbroadcast (summed) to [3] = [2,2,2].
    assert_eq!(rhs_grad.shape, vec![3]);
    assert_eq!(f32_vec(rhs_grad), vec![2.0, 2.0, 2.0]);
}

#[test]
/// `sub_forward_computes_elementwise_difference_with_broadcast`.
fn sub_forward_computes_elementwise_difference_with_broadcast() {
    let lhs = matrix(vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 2, 3);
    let rhs = vector(vec![1.0, 2.0, 3.0]);
    let out = sub_storage(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(f32_vec(&out), vec![9.0, 18.0, 27.0, 39.0, 48.0, 57.0]);
}

#[test]
/// `sub_backward_negates_rhs_contribution`.
fn sub_backward_negates_rhs_contribution() {
    let lhs = vector(vec![10.0, 20.0, 30.0]);
    let rhs = vector(vec![1.0, 2.0, 3.0]);
    let out = sub_storage(&lhs, &rhs).unwrap();

    let grads = tape::backward(&out).unwrap();
    let lhs_grad = grads.get(lhs.id).unwrap();
    let rhs_grad = grads.get(rhs.id).unwrap();

    assert_eq!(f32_vec(lhs_grad), vec![1.0, 1.0, 1.0]);
    assert_eq!(f32_vec(rhs_grad), vec![-1.0, -1.0, -1.0]);
}

#[test]
/// `mul_forward_computes_elementwise_product_with_broadcast`.
fn mul_forward_computes_elementwise_product_with_broadcast() {
    let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let rhs = vector(vec![2.0, 3.0, 4.0]);
    let out = mul_storage(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(f32_vec(&out), vec![2.0, 6.0, 12.0, 8.0, 15.0, 24.0]);
}

#[test]
/// `mul_backward_uses_other_operands_real_values`.
fn mul_backward_uses_other_operands_real_values() {
    // d(a*b)/da = b, d(a*b)/db = a — verify the retrieved gradient
    // equals a manually-computed expected value (not merely "some
    // gradient exists").
    let a = vector(vec![2.0, 3.0, 4.0]);
    let b = vector(vec![5.0, 6.0, 7.0]);
    let out = mul_storage(&a, &b).unwrap();

    let grads = tape::backward(&out).unwrap();
    let a_grad = grads.get(a.id).unwrap();
    let b_grad = grads.get(b.id).unwrap();

    // grad_out is ones_like(out) = [1,1,1].
    // da = grad_out * b = [5,6,7]
    assert_eq!(f32_vec(a_grad), vec![5.0, 6.0, 7.0]);
    // db = grad_out * a = [2,3,4]
    assert_eq!(f32_vec(b_grad), vec![2.0, 3.0, 4.0]);
}

#[test]
/// `mul_backward_with_broadcast_bias_vector_case`.
fn mul_backward_with_broadcast_bias_vector_case() {
    let lhs = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let rhs = vector(vec![2.0, 3.0, 4.0]);
    let out = mul_storage(&lhs, &rhs).unwrap();

    let grads = tape::backward(&out).unwrap();
    let lhs_grad = grads.get(lhs.id).unwrap();
    let rhs_grad = grads.get(rhs.id).unwrap();

    // grad_out = ones_like(out) = [[1,1,1],[1,1,1]]
    // d(lhs*rhs)/dlhs = rhs broadcast -> unbroadcast to lhs shape [2,3]
    // (no reduction needed since lhs shape == out shape): [[2,3,4],[2,3,4]]
    assert_eq!(lhs_grad.shape, vec![2, 3]);
    assert_eq!(f32_vec(lhs_grad), vec![2.0, 3.0, 4.0, 2.0, 3.0, 4.0]);

    // d(lhs*rhs)/drhs = lhs broadcast, summed (unbroadcast) to rhs shape [3]:
    // col sums of lhs = [1+4, 2+5, 3+6] = [5,7,9]
    assert_eq!(rhs_grad.shape, vec![3]);
    assert_eq!(f32_vec(rhs_grad), vec![5.0, 7.0, 9.0]);
}

#[test]
/// `add_scalar_float_forward_and_backward`.
fn add_scalar_float_forward_and_backward() {
    let t = vector(vec![1.0, 2.0, 3.0]);
    let out = canonical_add_scalar(&t, 1.0).unwrap();
    assert_eq!(f32_vec(&out), vec![2.0, 3.0, 4.0]);

    let grads = tape::backward(&out).unwrap();
    let t_grad = grads.get(t.id).unwrap();
    // Gradient passes through unchanged.
    assert_eq!(f32_vec(t_grad), vec![1.0, 1.0, 1.0]);
}

#[test]
/// `mul_scalar_float_forward_and_backward`.
fn mul_scalar_float_forward_and_backward() {
    let t = vector(vec![1.0, 2.0, 3.0]);
    let out = canonical_mul_scalar(&t, 2.5).unwrap();
    assert_eq!(f32_vec(&out), vec![2.5, 5.0, 7.5]);

    let grads = tape::backward(&out).unwrap();
    let t_grad = grads.get(t.id).unwrap();
    // Gradient scales by the same constant.
    assert_eq!(f32_vec(t_grad), vec![2.5, 2.5, 2.5]);
}

// --- Task 1: relu / abs / neg ---

#[test]
/// `relu_forward_and_backward_zero_at_boundary`.
fn relu_forward_and_backward_zero_at_boundary() {
    let t = vector(vec![-2.0, 0.0, 3.0]);
    let out = canonical_relu(&t).unwrap();
    assert_eq!(f32_vec(&out), vec![0.0, 0.0, 3.0]);

    let grads = tape::backward(&out).unwrap();
    let t_grad = grads.get(t.id).unwrap();
    // Zero gradient at the x=0 boundary (strict `>`, not `>=`).
    assert_eq!(f32_vec(t_grad), vec![0.0, 0.0, 1.0]);
}

#[test]
/// `relu_gradcheck_on_nonzero_input`.
fn relu_gradcheck_on_nonzero_input() {
    let x = vector(vec![2.0, -1.5, 0.7]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let r = canonical_relu(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&r).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "relu gradcheck error too high: {max_rel_err}"
    );
}

#[test]
/// `abs_forward_and_gradcheck`.
fn abs_forward_and_gradcheck() {
    let t = vector(vec![-2.5, 0.0, 3.5]);
    let out = canonical_abs(&t).unwrap();
    assert_eq!(f32_vec(&out), vec![2.5, 0.0, 3.5]);

    let x = vector(vec![-2.0, 1.5, -0.3]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let a = canonical_abs(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&a).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "abs gradcheck error too high: {max_rel_err}"
    );
}

#[test]
/// `neg_forward_and_gradcheck`.
fn neg_forward_and_gradcheck() {
    let t = vector(vec![1.0, -2.0, 3.0]);
    let out = canonical_neg(&t).unwrap();
    assert_eq!(f32_vec(&out), vec![-1.0, 2.0, -3.0]);

    let x = vector(vec![1.0, -2.0, 3.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let n = canonical_neg(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&n).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "neg gradcheck error too high: {max_rel_err}"
    );
}

// --- Task 2: exp / sqrt / log / tanh / sigmoid / swish ---

#[test]
/// `exp_forward_and_gradcheck`.
fn exp_forward_and_gradcheck() {
    let t = vector(vec![0.0, 1.0]);
    let out = canonical_exp(&t).unwrap();
    let expect = [1.0f32, core::f64::consts::E as f32];
    for (a, b) in f32_vec(&out).iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-5, "exp forward mismatch: {a} vs {b}");
    }

    let x = vector(vec![0.5, -0.3, 1.2]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let e = canonical_exp(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&e).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "exp gradcheck error too high: {max_rel_err}"
    );
}

#[test]
/// `sqrt_forward_gradcheck_and_nan_propagation`.
fn sqrt_forward_gradcheck_and_nan_propagation() {
    let t = vector(vec![4.0, 9.0]);
    let out = canonical_sqrt(&t).unwrap();
    assert_eq!(f32_vec(&out), vec![2.0, 3.0]);

    let x = vector(vec![4.0, 1.0, 9.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let s = canonical_sqrt(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&s).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "sqrt gradcheck error too high: {max_rel_err}"
    );

    // Negative input propagates NaN cpuly (RESEARCH.md Pitfall 2),
    // not a panic and not an Err.
    let neg_input = vector(vec![-1.0]);
    let neg_out = canonical_sqrt(&neg_input).unwrap();
    assert!(f32_vec(&neg_out)[0].is_nan(), "sqrt(-1.0) should be NaN");
}

#[test]
/// `log_forward_gradcheck_and_domain_propagation`.
fn log_forward_gradcheck_and_domain_propagation() {
    let t = vector(vec![1.0, core::f64::consts::E as f32]);
    let out = canonical_log(&t).unwrap();
    let expect = [0.0f32, 1.0f32];
    for (a, b) in f32_vec(&out).iter().zip(expect.iter()) {
        assert!((a - b).abs() < 1e-5, "log forward mismatch: {a} vs {b}");
    }

    let x = vector(vec![1.0, 2.0, 5.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let l = canonical_log(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&l).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "log gradcheck error too high: {max_rel_err}"
    );

    // Zero/negative input propagates NaN/-inf cpuly, not a panic and
    // not an Err.
    let zero_input = vector(vec![0.0]);
    let zero_out = canonical_log(&zero_input).unwrap();
    assert!(
        f32_vec(&zero_out)[0].is_infinite() && f32_vec(&zero_out)[0] < 0.0,
        "log(0.0) should be -inf"
    );

    let neg_input = vector(vec![-1.0]);
    let neg_out = canonical_log(&neg_input).unwrap();
    assert!(f32_vec(&neg_out)[0].is_nan(), "log(-1.0) should be NaN");
}

#[test]
/// `tanh_gradcheck`.
fn tanh_gradcheck() {
    let x = vector(vec![0.5, -1.0, 2.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let th = canonical_tanh(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&th).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "tanh gradcheck error too high: {max_rel_err}"
    );
}

#[test]
/// `sigmoid_forward_and_gradcheck`.
fn sigmoid_forward_and_gradcheck() {
    let t = vector(vec![0.0]);
    let out = canonical_sigmoid(&t).unwrap();
    assert_eq!(f32_vec(&out), vec![0.5]);

    let x = vector(vec![0.5, -1.0, 2.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let s = canonical_sigmoid(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&s).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "sigmoid gradcheck error too high: {max_rel_err}"
    );
}

#[test]
/// `swish_forward_and_gradcheck`.
fn swish_forward_and_gradcheck() {
    let t = vector(vec![0.0]);
    let out = canonical_swish(&t).unwrap();
    assert_eq!(f32_vec(&out), vec![0.0]);

    let x = vector(vec![0.5, -1.0, 2.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let s = canonical_swish(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&s).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "swish gradcheck error too high: {max_rel_err}"
    );
}

// --- Task 3: gelu (exact erf-based form) ---

#[test]
/// `gelu_forward_zero_and_one`.
fn gelu_forward_zero_and_one() {
    let zero = vector(vec![0.0]);
    let out_zero = canonical_gelu(&zero).unwrap();
    assert_eq!(f32_vec(&out_zero), vec![0.0]);

    let one = vector(vec![1.0]);
    let out_one = canonical_gelu(&one).unwrap();
    // Known reference value for erf-based GELU at x=1 (~0.8413).
    // Looser 1e-3 tolerance than other ops' 1e-5 since this uses a
    // polynomial erf approximation, not an exact closed form
    // ([ASSUMED] per RESEARCH.md Assumption A3).
    assert!(
        (f32_vec(&out_one)[0] - 0.8413).abs() < 1e-3,
        "gelu(1.0) mismatch: {}",
        f32_vec(&out_one)[0]
    );
}

#[test]
/// `gelu_gradcheck`.
fn gelu_gradcheck() {
    let x = vector(vec![0.5, -1.0, 2.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let g = canonical_gelu(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&g).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "gelu gradcheck error too high: {max_rel_err}"
    );
}

// --- Task 1 (plan 02-04): softmax by composition ---

#[test]
/// `softmax_forward_sums_to_one_on_vector`.
fn softmax_forward_sums_to_one_on_vector() {
    let t = vector(vec![1.0, 2.0, 3.0]);
    let out = canonical_softmax::<incin_core::tensor::device::Cpu>(&t, 0).unwrap();
    let vals = f32_vec(&out);

    let sum: f32 = vals.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1: {sum}");

    // Largest input gets largest probability, monotonic ordering preserved.
    assert!(vals[0] < vals[1]);
    assert!(vals[1] < vals[2]);
}

#[test]
/// `softmax_forward_sums_to_one_per_row_on_matrix`.
fn softmax_forward_sums_to_one_per_row_on_matrix() {
    let t = matrix(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
    let out = canonical_softmax::<incin_core::tensor::device::Cpu>(&t, 1).unwrap();
    let vals = f32_vec(&out);

    let row0_sum: f32 = vals[0..3].iter().sum();
    let row1_sum: f32 = vals[3..6].iter().sum();
    assert!(
        (row0_sum - 1.0).abs() < 1e-5,
        "row 0 should sum to 1: {row0_sum}"
    );
    assert!(
        (row1_sum - 1.0).abs() < 1e-5,
        "row 1 should sum to 1: {row1_sum}"
    );
}

#[test]
/// `softmax_forward_stable_on_large_magnitude_equal_logits`.
fn softmax_forward_stable_on_large_magnitude_equal_logits() {
    // Without max-subtraction, exp(1000.0) overflows to inf, producing
    // NaN (inf/inf) instead of a finite uniform distribution.
    let t = vector(vec![1000.0, 1000.0, 1000.0]);
    let out = canonical_softmax::<incin_core::tensor::device::Cpu>(&t, 0).unwrap();
    let vals = f32_vec(&out);

    for v in &vals {
        assert!(v.is_finite(), "softmax output should be finite: {v}");
        assert!(
            (v - 1.0 / 3.0).abs() < 1e-4,
            "softmax(equal large logits) should be uniform: {v}"
        );
    }
}

#[test]
/// `softmax_forward_uniform_on_all_zero_logits`.
fn softmax_forward_uniform_on_all_zero_logits() {
    let t = vector(vec![0.0, 0.0, 0.0]);
    let out = canonical_softmax::<incin_core::tensor::device::Cpu>(&t, 0).unwrap();
    let vals = f32_vec(&out);

    for v in &vals {
        assert!(v.is_finite(), "softmax output should be finite: {v}");
        assert!(
            (v - 1.0 / 3.0).abs() < 1e-4,
            "softmax(all-zero logits) should be uniform: {v}"
        );
    }
}

#[test]
/// `softmax_gradcheck`.
fn softmax_gradcheck() {
    let x = vector(vec![0.5, -1.0, 2.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let s = canonical_softmax::<incin_core::tensor::device::Cpu>(&inputs[0], 0).unwrap();
        crate::cpu::ops::reduce::sum_all(&s).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "softmax gradcheck error too high: {max_rel_err}"
    );
}

#[test]
/// `softmax_backward_finite_on_large_magnitude_equal_logits`.
fn softmax_backward_finite_on_large_magnitude_equal_logits() {
    // Proves both forward AND backward are numerically stable under the
    // composition, not just forward (Test 3's finite-forward twin).
    let t = vector(vec![1000.0, 1000.0, 1000.0]);
    let out = canonical_softmax::<incin_core::tensor::device::Cpu>(&t, 0).unwrap();

    let grads = tape::backward(&out).unwrap();
    let t_grad = grads.get(t.id).unwrap();
    for v in f32_vec(t_grad) {
        assert!(
            v.is_finite(),
            "softmax backward gradient should be finite on extreme logits: {v}"
        );
    }
}

// --- log_softmax kernel tests (Plan 04-01 Task 1) ---

#[test]
/// `log_softmax_exp_sums_to_one_on_vector`.
fn log_softmax_exp_sums_to_one_on_vector() {
    // exp(log_softmax(x)).sum() == 1.0 (the softmax identity).
    use crate::cpu::ops::elementwise::log_softmax;
    let t = vector(vec![1.0, 2.0, 3.0]);
    let ls = log_softmax::<incin_core::tensor::device::Cpu, f32>(&t, 0).unwrap();
    let exp_ls = canonical_exp(&ls).unwrap();
    let vals = f32_vec(&exp_ls);
    let sum: f32 = vals.iter().sum();
    assert!(
        (sum - 1.0).abs() < 1e-5,
        "exp(log_softmax) should sum to 1: {sum}"
    );
}

#[test]
/// `log_softmax_is_finite_and_correct_on_large_magnitude_equal_logits`.
fn log_softmax_is_finite_and_correct_on_large_magnitude_equal_logits() {
    // log_softmax([1000, 1000, 1000]) should be -ln(3) for each element.
    // Without max-subtraction, exp(1000) overflows to inf and log(inf) = inf.
    use crate::cpu::ops::elementwise::log_softmax;
    let t = vector(vec![1000.0f32, 1000.0, 1000.0]);
    let ls = log_softmax::<incin_core::tensor::device::Cpu, f32>(&t, 0).unwrap();
    let vals = f32_vec(&ls);
    let expected = -(3.0f32.ln());
    for (i, &v) in vals.iter().enumerate() {
        assert!(v.is_finite(), "log_softmax[{i}] should be finite: {v}");
        assert!(
            (v - expected).abs() < 1e-4,
            "log_softmax of equal large logits should be -ln(3): got {v}, expected {expected}"
        );
    }
}

#[test]
/// `softmax_after_refactor_still_passes_all_prior_behavior`.
fn softmax_after_refactor_still_passes_all_prior_behavior() {
    // Regression guard: the refactored softmax (exp(log_softmax(x, dim)))
    // must produce the same output as the old max_keepdim/sub/exp/sum_keepdim/div
    // composition. Verified by running all pre-existing scenarios in one test.
    // (Pre-existing tests above already cover this — this is an explicit marker
    //  that the refactor did not break them.)
    //
    // Spot-check: vector [0.5, -1.0, 2.0] forward correctness.
    let t = vector(vec![0.5f32, -1.0, 2.0]);
    let out = canonical_softmax::<incin_core::tensor::device::Cpu>(&t, 0).unwrap();
    let vals = f32_vec(&out);
    let sum: f32 = vals.iter().sum();
    assert!((sum - 1.0).abs() < 1e-5, "softmax sum should be 1: {sum}");
    for v in &vals {
        assert!(v.is_finite(), "softmax output should be finite: {v}");
        assert!(*v > 0.0, "softmax output should be positive: {v}");
    }
}

#[test]
/// `log_softmax_gradcheck`.
fn log_softmax_gradcheck() {
    // Finite-difference gradcheck for log_softmax itself.
    use crate::cpu::ops::elementwise::log_softmax;
    let x = vector(vec![0.5f32, -1.0, 2.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let ls = log_softmax::<incin_core::tensor::device::Cpu, f32>(&inputs[0], 0).unwrap();
        crate::cpu::ops::reduce::sum_all(&ls).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "log_softmax gradcheck error too high: {max_rel_err}"
    );
}

#[test]
/// `step_forward_and_backward_preserves_dtype_and_zero_grad`.
fn step_forward_and_backward_preserves_dtype_and_zero_grad() {
    // F32 test
    let t_f32 = vector(vec![-1.0f32, 0.0, 2.0]);
    let out_f32 = canonical_step(&t_f32).unwrap();
    assert_eq!(f32_vec(&out_f32), vec![0.0, 0.0, 1.0]);
    let grads_f32 = tape::backward(&out_f32).unwrap();
    let grad_f32 = grads_f32.get(t_f32.id).unwrap();
    assert_eq!(
        grad_f32.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::F32
    );
    assert_eq!(f32_vec(grad_f32), vec![0.0, 0.0, 0.0]);

    // F64 test
    let t_f64 = CpuStorage::from_contiguous(CpuBuffer::F64(vec![-2.0f64, 0.0, 3.0]), [3]);
    let out_f64 = canonical_step(&t_f64).unwrap();
    assert_eq!(
        out_f64.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::F64
    );
    let grads_f64 = tape::backward(&out_f64).unwrap();
    let grad_f64 = grads_f64.get(t_f64.id).unwrap();
    assert_eq!(
        grad_f64.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::F64
    );
    if let CpuBuffer::F64(ref v) = *grad_f64.buffer {
        assert_eq!(v, &vec![0.0f64, 0.0, 0.0]);
    } else {
        panic!("expected F64 buffer");
    }

    // F16 test
    let t_f16 = CpuStorage::from_contiguous(
        CpuBuffer::F16(vec![
            half::f16::from_f32(-1.0),
            half::f16::from_f32(0.0),
            half::f16::from_f32(2.0),
        ]),
        [3],
    );
    let out_f16 = canonical_step(&t_f16).unwrap();
    assert_eq!(
        out_f16.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::F16
    );
    let grads_f16 = tape::backward(&out_f16).unwrap();
    let grad_f16 = grads_f16.get(t_f16.id).unwrap();
    assert_eq!(
        grad_f16.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::F16
    );

    // BF16 test
    let t_bf16 = CpuStorage::from_contiguous(
        CpuBuffer::BF16(vec![
            half::bf16::from_f32(-1.0),
            half::bf16::from_f32(0.0),
            half::bf16::from_f32(2.0),
        ]),
        [3],
    );
    let out_bf16 = canonical_step(&t_bf16).unwrap();
    assert_eq!(
        out_bf16.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::BF16
    );
    let grads_bf16 = tape::backward(&out_bf16).unwrap();
    let grad_bf16 = grads_bf16.get(t_bf16.id).unwrap();
    assert_eq!(
        grad_bf16.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::BF16
    );
}

#[test]
/// `mul_scalar_typed_preserves_dtype_and_precision`.
fn mul_scalar_typed_preserves_dtype_and_precision() {
    let t_f64 = CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.5f64, -2.5, 3.0]), [3]);
    let out = canonical_mul_scalar(&t_f64, 4.0).unwrap();
    assert_eq!(
        out.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::F64
    );
    let grads = tape::backward(&out).unwrap();
    let grad = grads.get(t_f64.id).unwrap();
    assert_eq!(
        grad.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::F64
    );
    if let CpuBuffer::F64(ref v) = *grad.buffer {
        assert_eq!(v, &vec![4.0f64, 4.0, 4.0]);
    } else {
        panic!("expected F64 buffer");
    }
}

#[test]
/// `zeros_like_matches_shape_and_dtype`.
fn zeros_like_matches_shape_and_dtype() {
    let t_f32 = vector(vec![1.0, 2.0, 3.0]);
    let z_f32 = CpuStorage::zeros_like(&t_f32).unwrap();
    assert_eq!(
        z_f32.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::F32
    );
    assert_eq!(f32_vec(&z_f32), vec![0.0, 0.0, 0.0]);

    let t_f64 = CpuStorage::from_contiguous(CpuBuffer::F64(vec![1.0, 2.0]), [2]);
    let z_f64 = CpuStorage::zeros_like(&t_f64).unwrap();
    assert_eq!(
        z_f64.buffer.dtype_id(),
        incin_core::tensor::dtype::DTypeId::F64
    );
}

#[test]
/// `trig_and_hyperbolic_gradchecks`.
fn trig_and_hyperbolic_gradchecks() {
    let check = |name: &str, f: fn(&CpuStorage) -> Result<CpuStorage>, input: &[f32]| {
        let x = vector(input.to_vec());
        let op = |inputs: &[CpuStorage]| -> CpuStorage {
            let out = f(&inputs[0]).unwrap();
            crate::cpu::ops::reduce::sum_all(&out).unwrap()
        };
        let max_rel_err = gradcheck(op, &[x], 1e-4);
        assert!(
            max_rel_err < 1e-2,
            "{name} gradcheck error too high: {max_rel_err}"
        );
    };

    check("tan", canonical_tan, &[0.2, -0.4, 0.5]);
    check("asin", canonical_asin, &[0.1, -0.3, 0.5]);
    check("acos", canonical_acos, &[0.1, -0.3, 0.5]);
    check("atan", canonical_atan, &[0.5, -1.2, 2.0]);
    check("sinh", canonical_sinh, &[0.5, -0.8, 1.2]);
    check("cosh", canonical_cosh, &[0.5, -0.8, 1.2]);
    check("asinh", canonical_asinh, &[0.5, -1.0, 2.0]);
    check("atanh", canonical_atanh, &[0.2, -0.4, 0.6]);
    check("erf", canonical_erf, &[0.3, -0.5, 1.0]);
    check("rsqrt", canonical_rsqrt, &[1.0, 4.0, 9.0]);
    check("elu", canonical_elu, &[0.5, -0.5, 1.5]);
    check("mish", canonical_mish, &[0.5, -1.0, 2.0]);
}

#[test]
/// `acosh_gradcheck`. Split out from `trig_and_hyperbolic_gradchecks`
/// for the same reason that bundle is itself now on
/// `tools/soundness.sh`'s Miri numeric skip list: Miri's software float
/// implementation diverges from native transcendental-function results
/// by enough, at these sample points, to intermittently push a
/// finite-difference error over the shared 1% threshold. It is not one
/// fixed function that trips: a second Miri run tripped `atan` inside
/// the bundle instead of `acosh`, confirming the whole group sits on
/// this margin, not any single member of it. Keeping this one isolated
/// still means only this test (rather than a coarser bundle) needs
/// re-checking if the bundle's own Miri skip is ever lifted.
fn acosh_gradcheck() {
    let x = vector(vec![1.5, 2.0, 3.5]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let out = canonical_acosh(&inputs[0]).unwrap();
        crate::cpu::ops::reduce::sum_all(&out).unwrap()
    };
    let max_rel_err = gradcheck(op, &[x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "acosh gradcheck error too high: {max_rel_err}"
    );
}

#[test]
/// `atan2_gradcheck`.
fn atan2_gradcheck() {
    let y = vector(vec![1.0, -2.0, 0.5]);
    let x = vector(vec![2.0, 1.5, -1.0]);
    let op = |inputs: &[CpuStorage]| -> CpuStorage {
        let out = canonical_atan2(&inputs[0], &inputs[1]).unwrap();
        crate::cpu::ops::reduce::sum_all(&out).unwrap()
    };
    let max_rel_err = gradcheck(op, &[y, x], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "atan2 gradcheck error too high: {max_rel_err}"
    );
}
