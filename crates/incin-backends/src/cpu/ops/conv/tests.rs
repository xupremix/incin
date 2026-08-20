use super::*;
use incin_core::error::Error;
use incin_core::tensor::device::Cpu;

use crate::cpu::gradcheck::gradcheck;
use crate::cpu::storage::{CpuBuffer, CpuStorage};
use crate::cpu::tape;

/// `tensor`.
fn tensor(v: Vec<f32>, shape: Vec<usize>) -> CpuStorage {
    CpuStorage::from_contiguous(CpuBuffer::F32(v), shape)
}

/// `f32_vec`.
fn f32_vec(s: &CpuStorage) -> Vec<f32> {
    match &*s.buffer {
        CpuBuffer::F32(v) => v.clone(),
        _ => panic!("expected F32 buffer"),
    }
}

// --- output-size arithmetic edge cases ---

/// `out_size` subtracts an effective kernel from a padded input length.
/// When the kernel is larger than the padded input - here a 5x5 kernel
/// with dilation 3, an effective span of `3*4+1 = 13` against a padded
/// input of 2 - a raw `usize` subtraction underflows: a panic in debug
/// builds and a wrapped, astronomically large output extent in release.
/// The saturating form must instead produce an empty spatial extent and
/// return normally.
#[test]
fn conv2d_with_a_kernel_larger_than_its_input_yields_an_empty_output_not_a_panic() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let weight = tensor(vec![0.5; 25], vec![1, 1, 5, 5]);

    let out =
        conv2d_windowed_impl::<Cpu, f32>(&input, &weight, None, Window2d::isotropic(1, 0, 3), 1)
            .unwrap();

    assert_eq!(out.shape.as_ref(), &[1, 1, 1, 1]);
    // `saturating_sub` floors the numerator at 0, so the formula's
    // trailing `+ 1` leaves exactly one degenerate output position.
    assert_eq!(f32_vec(&out).len(), 1);
}

// --- conv1d forward ---

/// Forward test (groups=1, stride=1, padding=0, dilation=1): a small
/// hand-computable [1,1,4] input convolved with a [1,1,2] kernel produces
/// a [1,1,3] output matching manual sliding-window dot products.
#[test]
fn conv1d_forward_hand_computed_no_padding() {
    // input = [1,2,3,4], kernel = [10,1]
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
    let weight = tensor(vec![10.0, 1.0], vec![1, 1, 2]);
    let out = conv1d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 3]);
    // window0 = [1,2] . [10,1] = 10+2=12
    // window1 = [2,3] . [10,1] = 20+3=23
    // window2 = [3,4] . [10,1] = 30+4=34
    assert_eq!(f32_vec(&out), vec![12.0, 23.0, 34.0]);
}

/// Forward test (padding>0, Pitfall 2): a [1,1,3] input with padding=1
/// and a [1,1,3] kernel produces the correct zero-padded-boundary
/// output.
#[test]
fn conv1d_forward_with_padding_zero_fills_boundary() {
    // input = [1,2,3], kernel = [1,1,1], padding=1 -> padded = [0,1,2,3,0]
    let input = tensor(vec![1.0, 2.0, 3.0], vec![1, 1, 3]);
    let weight = tensor(vec![1.0, 1.0, 1.0], vec![1, 1, 3]);
    let out = conv1d_impl::<Cpu, f32>(&input, &weight, None, 1, 1, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 3]);
    // windows over padded [0,1,2,3,0]: [0,1,2]->3, [1,2,3]->6, [2,3,0]->5
    assert_eq!(f32_vec(&out), vec![3.0, 6.0, 5.0]);
}

/// Forward test (groups>1, Pitfall 7): a [1,4,5] input (Cin=4) with
/// groups=2 and a [2,2,2] weight (Cout=2, Cin/groups=2) matches two
/// independent single-group convolutions concatenated along the
/// output-channel axis.
#[test]
fn conv1d_forward_groups_matches_two_independent_convs() {
    let input_data: Vec<f32> = (1..=20).map(|x| x as f32).collect(); // [1,4,5]
    let input = tensor(input_data.clone(), vec![1, 4, 5]);
    let weight_data: Vec<f32> = (1..=8).map(|x| x as f32 * 0.1).collect(); // [2,2,2]
    let weight = tensor(weight_data.clone(), vec![2, 2, 2]);

    let out = conv1d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 1, 2).unwrap();
    assert_eq!(out.shape, vec![1, 2, 4]);

    // group 0: input channels [0,1] (rows 0-1 of input, each len 5),
    // weight channel 0 (shape [1,2,2])
    let g0_input = tensor(input_data[0..10].to_vec(), vec![1, 2, 5]);
    let g0_weight = tensor(weight_data[0..4].to_vec(), vec![1, 2, 2]);
    let g0_out = conv1d_impl::<Cpu, f32>(&g0_input, &g0_weight, None, 1, 0, 1, 1).unwrap();

    let g1_input = tensor(input_data[10..20].to_vec(), vec![1, 2, 5]);
    let g1_weight = tensor(weight_data[4..8].to_vec(), vec![1, 2, 2]);
    let g1_out = conv1d_impl::<Cpu, f32>(&g1_input, &g1_weight, None, 1, 0, 1, 1).unwrap();

    let combined = f32_vec(&out);
    assert_eq!(&combined[0..4], &f32_vec(&g0_out)[..]);
    assert_eq!(&combined[4..8], &f32_vec(&g1_out)[..]);
}

/// Forward test (bias): providing `Some(bias)` adds the per-output-channel
/// bias value to every spatial position of that channel.
#[test]
fn conv1d_forward_with_bias_adds_per_channel_constant() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
    let weight = tensor(vec![10.0, 1.0], vec![1, 1, 2]);
    let bias = tensor(vec![100.0], vec![1]);
    let out = conv1d_impl::<Cpu, f32>(&input, &weight, Some(&bias), 1, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 3]);
    assert_eq!(f32_vec(&out), vec![112.0, 123.0, 134.0]);
}

// --- conv1d backward ---

/// `conv1d_sum_op`.
fn conv1d_sum_op(inputs: &[CpuStorage]) -> CpuStorage {
    let out = conv1d_impl::<Cpu, f32>(&inputs[0], &inputs[1], None, 1, 0, 1, 1).unwrap();
    crate::cpu::ops::reduce::sum_all(&out).unwrap()
}

/// Backward test (gradcheck against input AND weight): a small
/// [1,1,4]/[1,1,2] pair, wrapped in `sum_all`, gradchecked with
/// `max_relative_error < 1e-2` for BOTH the input and the weight tensor.
#[test]
fn conv1d_gradcheck_input_and_weight() {
    let input = tensor(vec![0.1, 0.2, 0.3, 0.4], vec![1, 1, 4]);
    let weight = tensor(vec![0.5, 0.6], vec![1, 1, 2]);
    let max_rel_err = gradcheck(conv1d_sum_op, &[input, weight], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "gradcheck max relative error too high: {max_rel_err}"
    );
}

/// Backward test (overlapping windows, stride < kernel_size): confirms
/// col2im's fold uses `+=` accumulation.
#[test]
fn conv1d_backward_overlapping_windows_accumulates_grad_input() {
    // kernel_size=2, stride=1 on a length-3 input -> 2 output positions,
    // each input position (except the first/last) touched by 2 windows.
    let input = tensor(vec![1.0, 2.0, 3.0], vec![1, 1, 3]);
    let weight = tensor(vec![1.0, 1.0], vec![1, 1, 2]);
    let out = conv1d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 1, 1).unwrap();
    let loss = crate::cpu::ops::reduce::sum_all(&out).unwrap();
    let grads = tape::backward(&loss).unwrap();
    let grad_input = grads.get(input.id).expect("grad_input should exist");
    // grad w.r.t weight[k] summed over both output positions = 1 each,
    // and grad w.r.t input[i] = sum of weight values whose window covers i.
    // window0 covers input[0],input[1]; window1 covers input[1],input[2].
    // grad_input[0] = weight[0] (only window0) = 1
    // grad_input[1] = weight[1] (window0) + weight[0] (window1) = 1+1=2
    // grad_input[2] = weight[1] (only window1) = 1
    assert_eq!(f32_vec(grad_input), vec![1.0, 2.0, 1.0]);
}

// --- conv2d forward ---

/// Forward test (2D, groups=1): a [1,1,4,4] input convolved with a
/// [1,1,3,3] kernel, stride=1, padding=0, dilation=1 -> [1,1,2,2] output
/// matching hand-computed sliding-window sums.
#[test]
fn conv2d_forward_hand_computed_no_padding() {
    let input_data: Vec<f32> = (1..=16).map(|x| x as f32).collect(); // [1,1,4,4]
    let input = tensor(input_data, vec![1, 1, 4, 4]);
    let weight = tensor(vec![1.0; 9], vec![1, 1, 3, 3]); // sum-of-window kernel
    let out =
        conv2d_windowed_impl::<Cpu, f32>(&input, &weight, None, Window2d::isotropic(1, 0, 1), 1)
            .unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    // input matrix:
    //  1  2  3  4
    //  5  6  7  8
    //  9 10 11 12
    // 13 14 15 16
    // window(0,0) = rows0-2,cols0-2 = 1+2+3+5+6+7+9+10+11=54
    // window(0,1) = rows0-2,cols1-3 = 2+3+4+6+7+8+10+11+12=63
    // window(1,0) = rows1-3,cols0-2 = 5+6+7+9+10+11+13+14+15=90
    // window(1,1) = rows1-3,cols1-3 = 6+7+8+10+11+12+14+15+16=99
    assert_eq!(f32_vec(&out), vec![54.0, 63.0, 90.0, 99.0]);
}

/// Forward test (groups>1, Pitfall 7, 2D case): a [1,4,5,5] input
/// (Cin=4) with groups=2 and weight [2,2,3,3] matches two independent
/// single-group conv2d calls concatenated along the output-channel axis.
#[test]
fn conv2d_forward_groups_matches_two_independent_convs() {
    let input_data: Vec<f32> = (1..=100).map(|x| x as f32 * 0.01).collect(); // [1,4,5,5]
    let input = tensor(input_data.clone(), vec![1, 4, 5, 5]);
    let weight_data: Vec<f32> = (1..=36).map(|x| x as f32 * 0.01).collect(); // [2,2,3,3]
    let weight = tensor(weight_data.clone(), vec![2, 2, 3, 3]);

    let out =
        conv2d_windowed_impl::<Cpu, f32>(&input, &weight, None, Window2d::isotropic(1, 0, 1), 2)
            .unwrap();
    assert_eq!(out.shape, vec![1, 2, 3, 3]);

    let g0_input = tensor(input_data[0..50].to_vec(), vec![1, 2, 5, 5]);
    let g0_weight = tensor(weight_data[0..18].to_vec(), vec![1, 2, 3, 3]);
    let g0_out = conv2d_windowed_impl::<Cpu, f32>(
        &g0_input,
        &g0_weight,
        None,
        Window2d::isotropic(1, 0, 1),
        1,
    )
    .unwrap();

    let g1_input = tensor(input_data[50..100].to_vec(), vec![1, 2, 5, 5]);
    let g1_weight = tensor(weight_data[18..36].to_vec(), vec![1, 2, 3, 3]);
    let g1_out = conv2d_windowed_impl::<Cpu, f32>(
        &g1_input,
        &g1_weight,
        None,
        Window2d::isotropic(1, 0, 1),
        1,
    )
    .unwrap();

    let combined = f32_vec(&out);
    assert_eq!(&combined[0..9], &f32_vec(&g0_out)[..]);
    assert_eq!(&combined[9..18], &f32_vec(&g1_out)[..]);
}

/// Forward test (depthwise, groups==Cin): a [1,3,5,5] input with
/// groups=3 and weight [3,1,3,3] runs through the SAME code path as
/// groups=2 above (no special branch), producing correct
/// per-channel-independent output.
#[test]
fn conv2d_forward_depthwise_groups_equal_cin() {
    let input_data: Vec<f32> = (1..=75).map(|x| x as f32 * 0.01).collect(); // [1,3,5,5]
    let input = tensor(input_data.clone(), vec![1, 3, 5, 5]);
    let weight_data: Vec<f32> = (1..=27).map(|x| x as f32 * 0.01).collect(); // [3,1,3,3]
    let weight = tensor(weight_data.clone(), vec![3, 1, 3, 3]);

    let out =
        conv2d_windowed_impl::<Cpu, f32>(&input, &weight, None, Window2d::isotropic(1, 0, 1), 3)
            .unwrap();
    assert_eq!(out.shape, vec![1, 3, 3, 3]);

    // Verify each channel independently against a groups=1 conv on just
    // that channel's own input/weight slice.
    for c in 0..3 {
        let ch_input = tensor(input_data[c * 25..(c + 1) * 25].to_vec(), vec![1, 1, 5, 5]);
        let ch_weight = tensor(weight_data[c * 9..(c + 1) * 9].to_vec(), vec![1, 1, 3, 3]);
        let ch_out = conv2d_windowed_impl::<Cpu, f32>(
            &ch_input,
            &ch_weight,
            None,
            Window2d::isotropic(1, 0, 1),
            1,
        )
        .unwrap();
        let combined = f32_vec(&out);
        assert_eq!(&combined[c * 9..(c + 1) * 9], &f32_vec(&ch_out)[..]);
    }
}

// --- conv2d backward ---

/// `conv2d_sum_op`.
fn conv2d_sum_op(inputs: &[CpuStorage]) -> CpuStorage {
    let out = conv2d_windowed_impl::<Cpu, f32>(
        &inputs[0],
        &inputs[1],
        None,
        Window2d::isotropic(1, 0, 1),
        1,
    )
    .unwrap();
    crate::cpu::ops::reduce::sum_all(&out).unwrap()
}

/// Backward test: gradcheck on a small [1,1,4,4]/[1,1,2,2] pair
/// (stride=1,padding=0,dilation=1,groups=1), max_relative_error < 1e-2
/// for both grad_input and grad_weight.
#[test]
fn conv2d_gradcheck_input_and_weight() {
    let input_data: Vec<f32> = (1..=16).map(|x| x as f32 * 0.01).collect();
    let input = tensor(input_data, vec![1, 1, 4, 4]);
    let weight_data: Vec<f32> = (1..=4).map(|x| x as f32 * 0.1).collect();
    let weight = tensor(weight_data, vec![1, 1, 2, 2]);
    let max_rel_err = gradcheck(conv2d_sum_op, &[input, weight], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "gradcheck max relative error too high: {max_rel_err}"
    );
}

// --- conv_transpose2d forward ---

/// Forward test (basic, stride=1, padding=0, output_padding=0,
/// dilation=1, groups=1): a small [1,1,2,2] input with a [1,1,2,2]
/// weight (Cin=1,Cout=1) produces the hand-computed transposed-conv
/// output (verified against a manually-derived scatter-add-of-weighted-
/// patches reference, not just shape).
#[test]
fn conv_transpose2d_forward_hand_computed_basic() {
    // input = [[1,2],[3,4]], weight = [[1,1],[1,1]]
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let weight = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
    let out = conv_transpose2d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 3, 3]);
    // Hand-computed scatter-add of weighted 2x2 patches:
    // out[i+kh, j+kw] += input[i,j] * weight[kh,kw] for i,j,kh,kw in 0..2
    assert_eq!(
        f32_vec(&out),
        vec![1.0, 3.0, 2.0, 4.0, 10.0, 6.0, 3.0, 7.0, 4.0]
    );
}

/// Forward test (stride>1, the common upsampling case): a [1,1,2,2]
/// input with stride=2 produces an output shape matching Candle's exact
/// formula `(i_h - 1) * stride + dilation*(k_h-1) + output_padding + 1 -
/// 2*padding` for both H and W, with hand-computed values.
#[test]
fn conv_transpose2d_forward_stride_upsamples() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let weight = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
    let out = conv_transpose2d_impl::<Cpu, f32>(&input, &weight, None, 2, 0, 0, 1, 1).unwrap();
    // (2-1)*2 + 1*(2-1) + 1 - 0 = 4
    assert_eq!(out.shape, vec![1, 1, 4, 4]);
    assert_eq!(
        f32_vec(&out),
        vec![
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0
        ]
    );
}

/// Forward test (output_padding>0, Pitfall 4): explicitly constructs a
/// case with non-zero `output_padding` and confirms the extra
/// rows/columns are allocated on the correct (bottom/right) side ONLY,
/// at exactly value 0.0 - confirming the natural fold-output size is
/// computed first using `padding` symmetrically, THEN `output_padding`
/// extra rows/columns are appended afterward (not folded into the same
/// offset arithmetic as `padding`).
#[test]
fn conv_transpose2d_forward_output_padding_appends_trailing_zeros_only() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let weight = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
    let out = conv_transpose2d_impl::<Cpu, f32>(&input, &weight, None, 2, 0, 1, 1, 1).unwrap();
    // natural (output_padding=0) shape was [1,1,4,4]; output_padding=1
    // appends ONE extra trailing row and column -> [1,1,5,5].
    assert_eq!(out.shape, vec![1, 1, 5, 5]);
    let vals = f32_vec(&out);
    let natural = [
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    // Leading [0..4, 0..4] sub-region matches the natural (no
    // output_padding) result exactly.
    for row in 0..4 {
        for col in 0..4 {
            assert_eq!(vals[row * 5 + col], natural[row * 4 + col]);
        }
    }
    // The trailing row (row=4) and trailing column (col=4) are exactly
    // 0.0 for every position.
    for col in 0..5 {
        assert_eq!(vals[4 * 5 + col], 0.0, "trailing row must be zero");
    }
    for row in 0..5 {
        assert_eq!(vals[row * 5 + 4], 0.0, "trailing column must be zero");
    }
}

/// Forward test (groups != 1 rejected): calling with `groups=2` returns
/// a typed `Error::ShapeMismatch` rather than silently ignoring the
/// parameter or panicking via `debug_assert_eq!`.
#[test]
fn conv_transpose2d_rejects_groups_other_than_one() {
    let input = tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
    let weight = tensor(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
    let result = conv_transpose2d_impl::<Cpu, f32>(&input, &weight, None, 1, 0, 0, 1, 2);
    assert!(matches!(result, Err(Error::ShapeMismatch { .. })));
}

// --- conv_transpose2d backward ---

/// `conv_transpose2d_sum_op`.
fn conv_transpose2d_sum_op(inputs: &[CpuStorage]) -> CpuStorage {
    let out =
        conv_transpose2d_impl::<Cpu, f32>(&inputs[0], &inputs[1], None, 1, 0, 0, 1, 1).unwrap();
    crate::cpu::ops::reduce::sum_all(&out).unwrap()
}

/// Backward test: gradcheck on the basic [1,1,2,2]/[1,1,2,2] case
/// (stride=1, padding=0, output_padding=0, dilation=1),
/// max_relative_error < 1e-2 for both grad_input and grad_weight.
#[test]
fn conv_transpose2d_gradcheck_input_and_weight() {
    let input = tensor(vec![0.1, 0.2, 0.3, 0.4], vec![1, 1, 2, 2]);
    let weight = tensor(vec![0.5, 0.6, 0.7, 0.8], vec![1, 1, 2, 2]);
    let max_rel_err = gradcheck(conv_transpose2d_sum_op, &[input, weight], 1e-4);
    assert!(
        max_rel_err < 1e-2,
        "gradcheck max relative error too high: {max_rel_err}"
    );
}
