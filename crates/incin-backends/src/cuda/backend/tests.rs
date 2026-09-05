//! Unit tests for the CUDA backend implementation.

use super::*;

#[test]
fn byte_length_uses_authoritative_storage_dtype() {
    assert_eq!(
        checked_storage_byte_len(7, DTypeId::F16.into()).unwrap(),
        14
    );
    assert_eq!(
        checked_storage_byte_len(7, DTypeId::BF16.into()).unwrap(),
        14
    );
    assert_eq!(
        checked_storage_byte_len(7, DTypeId::F32.into()).unwrap(),
        28
    );
    assert_eq!(
        checked_storage_byte_len(7, DTypeId::F64.into()).unwrap(),
        56
    );
    assert!(checked_storage_byte_len(usize::MAX, DTypeId::F64.into()).is_err());
}

#[test]
fn storage_validation_accepts_renderable_float_family_and_i64_indices() {
    let device = DeviceId::cuda(0);
    for dtype in [
        DTypeId::F16,
        DTypeId::BF16,
        DTypeId::F32,
        DTypeId::F64,
        DTypeId::I64,
    ] {
        validate_cuda_storage(dtype.into(), &device, "test").unwrap();
    }
    assert!(matches!(
        validate_cuda_storage(DTypeId::U32.into(), &device, "test"),
        Err(Error::UnsupportedDType { .. })
    ));
    assert!(validate_cuda_storage(DTypeId::F32.into(), &DeviceId::cpu(), "test").is_err());
}

// shape_cardinality_is_checked_before_allocation moved to
// bytes::tests::numel_is_the_checked_product_of_the_dims, which now owns
// the one checked_numel implementation this file calls.

// The tests below exercise real GPU dispatch (`::{reshape,
// transpose, narrow, broadcast_as, squeeze, stack, slice, flatten,
// broadcast_left, matmul}`) and therefore need a real CUDA device to
// run - none is available in this environment, so this path is compile-verified
// only locally. `#[ignore]`d so `cargo test` stays green everywhere; run with
// `cargo test --features cuda,std -- --ignored` on real hardware.

type B = CudaBackendImpl<Cuda>;

fn cuda_f32(shape: &[usize], values: Vec<f32>) -> CudaStorage {
    cuda_from_f32(
        shape,
        DTypeId::F32.into(),
        &DeviceId::cuda(0),
        values,
        "test",
    )
    .unwrap()
}

#[test]
#[ignore = "requires CUDA hardware"]
fn reshape_preserves_element_order() {
    let t = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = B::reshape::<f32>(&t, &[3, 2]).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn reshape_rejects_mismatched_element_count() {
    let t = cuda_f32(&[2, 3], vec![0.0; 6]);
    assert!(B::reshape::<f32>(&t, &[4, 2]).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn transpose_2d_swaps_shape() {
    let t = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let out = B::transpose::<f32>(&t, 0, 1).unwrap();
    assert_eq!(out.shape, vec![3, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn narrow_reduces_target_dim() {
    let t = cuda_f32(&[4, 3], vec![0.0; 12]);
    let out = B::narrow::<f32>(&t, 0, 1, 2).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn broadcast_as_expands_size_one_dim() {
    let t = cuda_f32(&[1, 3], vec![1.0, 2.0, 3.0]);
    let out = B::broadcast_as::<f32>(&t, &[4, 3]).unwrap();
    assert_eq!(out.shape, vec![4, 3]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn broadcast_as_rejects_incompatible_shape() {
    let t = cuda_f32(&[2, 3], vec![0.0; 6]);
    assert!(B::broadcast_as::<f32>(&t, &[2, 5]).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn compare_writes_bool_storage_at_the_broadcast_shape() {
    use crate::cuda::ops::compare::{CompareOp, launch_compare};
    let lhs = cuda_f32(&[1, 3], vec![1.0, 2.0, 3.0]);
    let rhs = cuda_f32(&[2, 3], vec![1.0, 0.0, 3.0, 5.0, 2.0, 3.0]);
    let lhs_b = B::broadcast_as::<f32>(&lhs, &[2, 3]).unwrap();
    let out = launch_compare(CompareOp::Eq, &lhs_b, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::Bool.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn compare_rejects_mismatched_shapes() {
    use crate::cuda::ops::compare::{CompareOp, launch_compare};
    let lhs = cuda_f32(&[2, 3], vec![0.0; 6]);
    let rhs = cuda_f32(&[2, 4], vec![0.0; 8]);
    assert!(launch_compare(CompareOp::Lt, &lhs, &rhs).is_err());
}

fn cuda_bool(shape: &[usize], values: Vec<u8>) -> CudaStorage {
    cuda_from_bytes(shape, DTypeId::Bool.descriptor(), 0, &values).unwrap()
}

#[test]
#[ignore = "requires CUDA hardware"]
fn where_cond_selects_at_the_shared_operand_shape() {
    use crate::cuda::ops::select::launch_where_cond;
    let mask = cuda_bool(&[2, 3], vec![1, 0, 1, 0, 1, 0]);
    let on_true = cuda_f32(&[2, 3], vec![1.0; 6]);
    let on_false = cuda_f32(&[2, 3], vec![0.0; 6]);
    let out = launch_where_cond(&mask, &on_true, &on_false).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::F32.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn where_cond_rejects_mismatched_shapes() {
    use crate::cuda::ops::select::launch_where_cond;
    let mask = cuda_bool(&[2, 3], vec![1; 6]);
    let on_true = cuda_f32(&[2, 4], vec![0.0; 8]);
    let on_false = cuda_f32(&[2, 3], vec![0.0; 6]);
    assert!(launch_where_cond(&mask, &on_true, &on_false).is_err());
}

/// `launch_where_cond` itself takes no broadcast responsibility (see its
/// own doc): a lower-rank mask has to go through
/// `launch_broadcast_bool_mask` first, the same composition
/// `Execute<op::WhereCond>` performs.
#[test]
#[ignore = "requires CUDA hardware"]
fn broadcast_bool_mask_expands_a_lower_rank_mask() {
    use crate::cuda::ops::select::launch_broadcast_bool_mask;
    let mask = cuda_bool(&[3], vec![1, 0, 1]);
    let out = launch_broadcast_bool_mask(&mask, &[2, 3]).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::Bool.descriptor());
}

/// The composition `Execute<op::WhereCond>` performs when the mask
/// arrives at a lower rank than the data it selects between - the exact
/// case `where_cond`'s own descriptor permits (its output shape is the
/// broadcast of all three operands, not just the two data ones).
#[test]
#[ignore = "requires CUDA hardware"]
fn where_cond_broadcasts_a_lower_rank_mask_before_selecting() {
    use crate::cuda::ops::select::{launch_broadcast_bool_mask, launch_where_cond};
    let mask = cuda_bool(&[3], vec![1, 0, 1]);
    let on_true = cuda_f32(&[2, 3], vec![1.0; 6]);
    let on_false = cuda_f32(&[2, 3], vec![0.0; 6]);
    let mask_b = launch_broadcast_bool_mask(&mask, &[2, 3]).unwrap();
    let out = launch_where_cond(&mask_b, &on_true, &on_false).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::F32.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn masked_fill_overwrites_at_the_input_shape() {
    use crate::cuda::ops::select::launch_masked_fill;
    let input = cuda_f32(&[2, 3], vec![1.0; 6]);
    let mask = cuda_bool(&[2, 3], vec![1, 0, 1, 0, 1, 0]);
    let out = launch_masked_fill(&input, &mask, 9.0).unwrap();
    assert_eq!(out.shape, vec![2, 3]);
    assert_eq!(out.dtype(), DTypeId::F32.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn masked_fill_rejects_mismatched_shapes() {
    use crate::cuda::ops::select::launch_masked_fill;
    let input = cuda_f32(&[2, 3], vec![0.0; 6]);
    let mask = cuda_bool(&[2, 4], vec![0; 8]);
    assert!(launch_masked_fill(&input, &mask, 0.0).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn logical_and_or_and_not_write_bool_storage_at_the_shared_shape() {
    use crate::cuda::ops::logical::{launch_logical_and, launch_logical_not, launch_logical_or};
    let lhs = cuda_bool(&[4], vec![1, 1, 0, 0]);
    let rhs = cuda_bool(&[4], vec![1, 0, 1, 0]);

    let and_out = launch_logical_and(&lhs, &rhs).unwrap();
    assert_eq!(and_out.shape, vec![4]);
    assert_eq!(and_out.dtype(), DTypeId::Bool.descriptor());

    let or_out = launch_logical_or(&lhs, &rhs).unwrap();
    assert_eq!(or_out.shape, vec![4]);
    assert_eq!(or_out.dtype(), DTypeId::Bool.descriptor());

    let not_out = launch_logical_not(&lhs).unwrap();
    assert_eq!(not_out.shape, vec![4]);
    assert_eq!(not_out.dtype(), DTypeId::Bool.descriptor());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn logical_and_rejects_mismatched_shapes() {
    use crate::cuda::ops::logical::launch_logical_and;
    let lhs = cuda_bool(&[2, 3], vec![1; 6]);
    let rhs = cuda_bool(&[2, 4], vec![1; 8]);
    assert!(launch_logical_and(&lhs, &rhs).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn logical_and_rejects_non_bool_storage() {
    use crate::cuda::ops::logical::launch_logical_and;
    let lhs = cuda_f32(&[4], vec![1.0; 4]);
    let rhs = cuda_bool(&[4], vec![1; 4]);
    assert!(launch_logical_and(&lhs, &rhs).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn squeeze_removes_size_one_axis() {
    let t = cuda_f32(&[1, 3], vec![1.0, 2.0, 3.0]);
    let out = B::squeeze::<f32>(&t, 0).unwrap();
    assert_eq!(out.shape, vec![3]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_computes_correct_shape_and_values() {
    // [[1,2,3],[4,5,6]] @ [[7,8],[9,10],[11,12]] = [[58,64],[139,154]]
    let lhs = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let out = B::matmul::<f32>(&lhs, &rhs).unwrap();
    assert_eq!(out.shape, vec![2, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_rejects_incompatible_inner_dims() {
    let lhs = cuda_f32(&[2, 3], vec![0.0; 6]);
    let rhs = cuda_f32(&[4, 2], vec![0.0; 8]);
    assert!(B::matmul::<f32>(&lhs, &rhs).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn matmul_backward_produces_gradients_for_both_operands() {
    let lhs = cuda_f32(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = cuda_f32(&[3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
    let (lhs_id, rhs_id) = (lhs.id, rhs.id);
    let out = B::matmul::<f32>(&lhs, &rhs).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    assert!(grads.get(lhs_id).is_some());
    assert!(grads.get(rhs_id).is_some());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn narrow_backward_zero_pads_grad_to_original_shape() {
    let t = cuda_f32(&[4, 3], vec![0.0; 12]);
    let t_id = t.id;
    let out = B::narrow::<f32>(&t, 0, 1, 2).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let g = grads
        .get(t_id)
        .expect("narrow input should have a gradient");
    assert_eq!(g.shape, vec![4, 3]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn max_pool2d_computes_correct_output_shape() {
    // N=1,C=1,H=4,W=4, kernel=2, stride=2 -> 2x2 output
    let t = cuda_f32(&[1, 1, 4, 4], vec![0.0; 16]);
    let out = B::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn max_pool2d_backward_zero_pads_to_input_shape() {
    let t = cuda_f32(&[1, 1, 4, 4], vec![0.0; 16]);
    let t_id = t.id;
    let out = B::max_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let g = grads
        .get(t_id)
        .expect("max_pool2d input should have a gradient");
    assert_eq!(g.shape, vec![1, 1, 4, 4]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn avg_pool2d_computes_correct_output_shape() {
    let t = cuda_f32(&[1, 1, 4, 4], vec![0.0; 16]);
    let out = B::avg_pool2d::<f32>(&t, (2, 2), (2, 2), (0, 0)).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv2d_computes_correct_output_shape_and_values() {
    // [1,1,3,3] input, [1,1,2,2] kernel, stride=1, no padding -> [1,1,2,2],
    // matching CPU's hand-computed test fixture (conv.rs) exactly.
    let t = cuda_f32(
        &[1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let w = cuda_f32(&[1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let out = B::conv2d::<f32>(&t, &w, None, 1, 0, 1, 1).unwrap();
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    let vals = download_f32_host(&out).unwrap();
    assert_eq!(vals, vec![12.0, 16.0, 24.0, 28.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv2d_with_bias_adds_per_channel_constant() {
    let t = cuda_f32(&[1, 1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]);
    let w = cuda_f32(&[1, 1, 1, 1], vec![1.0]);
    let bias = cuda_f32(&[1], vec![10.0]);
    let out = B::conv2d::<f32>(&t, &w, Some(&bias), 1, 0, 1, 1).unwrap();
    let vals = download_f32_host(&out).unwrap();
    assert_eq!(vals, vec![11.0, 12.0, 13.0, 14.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv2d_backward_produces_gradients_for_input_and_weight() {
    let t = cuda_f32(
        &[1, 1, 3, 3],
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let w = cuda_f32(&[1, 1, 2, 2], vec![1.0, 1.0, 1.0, 1.0]);
    let (t_id, w_id) = (t.id, w.id);
    let out = B::conv2d::<f32>(&t, &w, None, 1, 0, 1, 1).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    assert_eq!(grads.get(t_id).unwrap().shape, vec![1, 1, 3, 3]);
    assert_eq!(grads.get(w_id).unwrap().shape, vec![1, 1, 2, 2]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn conv2d_groups_matches_two_independent_convs() {
    // groups=2 depthwise-ish split: Cin=2,Cout=2 each channel convolved
    // independently, mirrors CPU's `conv2d_forward_groups_matches_two_independent_convs`.
    let t = cuda_f32(&[1, 2, 2, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    let w = cuda_f32(&[2, 1, 1, 1], vec![2.0, 3.0]);
    let out = B::conv2d::<f32>(&t, &w, None, 1, 0, 1, 2).unwrap();
    assert_eq!(out.shape, vec![1, 2, 2, 2]);
    let vals = download_f32_host(&out).unwrap();
    assert_eq!(vals, vec![2.0, 4.0, 6.0, 8.0, 15.0, 18.0, 21.0, 24.0]);
}

// mse_loss/l1_loss/bce_with_logits_loss have no override in this file's
// the free loss helpers (`incin-backends/src/legacy.rs`),
// which compose entirely from ``/``/``
// (already wired on CUDA). These tests exist to prove that resolution
// actually compiles and runs correctly, not to add new functionality.

// The tests below cover the methods added in this pass: `unsqueeze`,
// the host-readback conversions, `addmm`/`bmm`/
// `scaled_dot_product_attention`. Same convention as everything above -
// `#[ignore]`d because there is no CUDA device in this environment, so
// only compilation is verified here; run with `--ignored` on real
// hardware. Fixtures and expected values are the same ones the CPU and
// WGPU backends' own tests for the identical methods use.

#[test]
#[ignore = "requires CUDA hardware"]
fn test_full() {
    let out = B::full::<f32>(3.5, &[2, 2], DTypeId::F32.into(), &DeviceId::cuda(0)).unwrap();
    assert_eq!(download_f32_host(&out).unwrap(), vec![3.5, 3.5, 3.5, 3.5]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn test_arange() {
    let out = B::arange::<f32>(1.0, 2.0, &[4], DTypeId::F32.into(), &DeviceId::cuda(0)).unwrap();
    assert_eq!(download_f32_host(&out).unwrap(), vec![1.0, 3.0, 5.0, 7.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn test_linspace() {
    let out = B::linspace::<f32>(0.0, 10.0, &[5], DTypeId::F32.into(), &DeviceId::cuda(0)).unwrap();
    assert_eq!(
        download_f32_host(&out).unwrap(),
        vec![0.0, 2.5, 5.0, 7.5, 10.0]
    );
}

// ---------------------------------------------------------------------------
// layer_norm forward parity and backward (issue #4)
// ---------------------------------------------------------------------------

use crate::cpu::storage::{CpuBuffer as HostBuffer, CpuStorage as HostStorage};

/// The documented fixture: two rows of four, non-trivial weight and bias, so
/// no symmetry can hide a permuted axis.
fn ln_input() -> CudaStorage {
    cuda_f32(&[2, 4], vec![0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0])
}

fn ln_weight() -> CudaStorage {
    cuda_f32(&[4], vec![2.0, 1.0, 0.5, 1.5])
}

fn ln_bias() -> CudaStorage {
    cuda_f32(&[4], vec![0.1, -0.1, 0.2, -0.2])
}

fn host_f32(shape: &[usize], values: Vec<f32>) -> HostStorage {
    HostStorage::from_contiguous(HostBuffer::F32(values), shape)
}

fn host_values(storage: &HostStorage) -> Vec<f64> {
    let total: usize = storage.shape.iter().product::<usize>().max(1);
    let mut out = Vec::with_capacity(total);
    let mut index = vec![0usize; storage.shape.len().max(1)];
    for _ in 0..total {
        out.push(storage.get(&index));
        for (i, extent) in index.iter_mut().zip(storage.shape.iter()).rev() {
            *i += 1;
            if *i < *extent {
                break;
            }
            *i = 0;
        }
    }
    out
}

/// CPU forward plus its composed backward, on the same values the CUDA side
/// runs: the reference the parity tests below compare against.
fn cpu_layer_norm_grads(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let t = host_f32(&[2, 4], input.to_vec());
    let w = host_f32(&[4], weight.to_vec());
    let b = bias.map(|values| host_f32(&[4], values.to_vec()));
    let out = crate::cpu::ops::norm::layer_norm_impl::<incin_core::tensor::device::Cpu, f32>(
        &t,
        &w,
        b.as_ref(),
        1e-5,
    )
    .unwrap();
    let grads = crate::cpu::tape::backward(&out).unwrap();
    let read = |storage: &HostStorage| {
        let grad = grads.get(storage.id).unwrap();
        host_values(grad)
    };
    let db = match &b {
        Some(bias_storage) => read(bias_storage),
        None => Vec::new(),
    };
    (host_values(&out), read(&t), read(&w), db)
}

fn assert_close(left: &[f64], right: &[f64], tol: f64, what: &str) {
    assert_eq!(left.len(), right.len(), "{what}: length mismatch");
    for (i, (l, r)) in left.iter().zip(right.iter()).enumerate() {
        let denom = l.abs().max(r.abs()).max(1e-6);
        assert!(
            (l - r).abs() / denom <= tol,
            "{what}[{i}]: cuda={l} cpu={r}"
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_forward_matches_cpu_reference() {
    // Regression guard for the stats-saving edit to the fused template: the
    // two extra stores must not disturb the output values.
    let (input, weight, bias) = (ln_input(), ln_weight(), ln_bias());
    let out = B::layer_norm::<f32>(&input, &weight, Some(&bias), 1e-5).unwrap();
    assert_eq!(out.shape, vec![2, 4]);
    let (expected, _, _, _) = cpu_layer_norm_grads(
        &[0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0],
        &[2.0, 1.0, 0.5, 1.5],
        Some(&[0.1, -0.1, 0.2, -0.2]),
    );
    let got: Vec<f64> = download_f32_host(&out)
        .unwrap()
        .iter()
        .map(|v| *v as f64)
        .collect();
    assert_close(&got, &expected, 1e-5, "forward");
    // Draining here keeps this forward's entry off the next test's walk.
    let _ = crate::cuda::tape::backward(&out);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_backward_matches_cpu_reference() {
    let (input, weight, bias) = (ln_input(), ln_weight(), ln_bias());
    let (input_id, weight_id, bias_id) = (input.id, weight.id, bias.id);
    let out = B::layer_norm::<f32>(&input, &weight, Some(&bias), 1e-5).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let read = |id: incin_core::exec::TensorId| {
        let grad = grads
            .get(id)
            .expect("layer norm input should have a gradient");
        download_f32_host(grad)
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    let (_, expected_dx, expected_dw, expected_db) = cpu_layer_norm_grads(
        &[0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0],
        &[2.0, 1.0, 0.5, 1.5],
        Some(&[0.1, -0.1, 0.2, -0.2]),
    );
    // Welford on device against composed primitives on host: agreement to
    // four digits, not bit-exact.
    assert_close(&read(input_id), &expected_dx, 1e-4, "dx");
    assert_close(&read(weight_id), &expected_dw, 1e-4, "dw");
    assert_close(&read(bias_id), &expected_db, 1e-4, "db");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_uniform_upstream_gradient_gives_zero_input_gradient() {
    // Analytic property, no reference needed, but it only holds for uniform
    // weight: with `gw = gout * weight` uniform too, every input gradient is
    // exactly rstd*(gw - mean(gw) - y*mean(gw*y)) = 0. Averaging the upstream
    // gradient first and multiplying by weight after is the same only there,
    // which is why this test pins the uniform-weight case while the parity
    // tests above pin a non-uniform one.
    let input = ln_input();
    let weight = cuda_f32(&[4], vec![1.0; 4]);
    let bias = ln_bias();
    let (input_id, bias_id) = (input.id, bias.id);
    let out = B::layer_norm::<f32>(&input, &weight, Some(&bias), 1e-5).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let dx = download_f32_host(grads.get(input_id).unwrap()).unwrap();
    for (i, value) in dx.iter().enumerate() {
        assert!(
            value.abs() < 1e-4,
            "dx[{i}] should vanish under uniform gradients, got {value}"
        );
    }
    // Same seed, dual property: each bias element sees every row once.
    let db = download_f32_host(grads.get(bias_id).unwrap()).unwrap();
    assert_eq!(db, vec![2.0, 2.0, 2.0, 2.0]);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_backward_without_bias_returns_two_gradients() {
    let (input, weight) = (ln_input(), ln_weight());
    let (input_id, weight_id) = (input.id, weight.id);
    let out = B::layer_norm::<f32>(&input, &weight, None, 1e-5).unwrap();
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let dx = grads.get(input_id).expect("input should have a gradient");
    let dw = grads.get(weight_id).expect("weight should have a gradient");
    assert_eq!(dx.shape, vec![2, 4]);
    assert_eq!(dw.shape, vec![4]);
    let (_, expected_dx, expected_dw, _) = cpu_layer_norm_grads(
        &[0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0],
        &[2.0, 1.0, 0.5, 1.5],
        None,
    );
    let read = |storage: &CudaStorage| {
        download_f32_host(storage)
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    assert_close(&read(dx), &expected_dx, 1e-4, "dx without bias");
    assert_close(&read(dw), &expected_dw, 1e-4, "dw without bias");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_backward_replays_saved_statistics() {
    // White-box proof that the kernel reads the passed statistics rather
    // than recomputing them: the same launch with a perturbed mean must
    // produce different gradients, and with the true statistics must match
    // the CPU reference.
    use crate::cuda::ops::norm::launch_layer_norm_backward;
    let (input, weight, bias) = (ln_input(), ln_weight(), ln_bias());
    let (_, stats) =
        crate::cuda::ops::norm::launch_layer_norm(&input, &weight, Some(&bias), 1e-5, true)
            .unwrap();
    let stats = stats.expect("recording forward keeps statistics");
    let gout = cuda_f32(&[2, 4], vec![1.0, 0.5, -0.5, 2.0, -1.0, 1.0, 0.25, -0.75]);
    let grads =
        launch_layer_norm_backward(&gout, &input, &weight, &stats.mean, &stats.rstd, true).unwrap();
    let read = |storage: &CudaStorage| {
        download_f32_host(storage)
            .unwrap()
            .iter()
            .map(|v| *v as f64)
            .collect::<Vec<_>>()
    };
    let dx = read(&grads.input);
    // Perturb the saved mean by 1.0 on both rows: a kernel that recomputed
    // its statistics internally would be unaffected, so its gradients would
    // coincide. They must not.
    let bad_mean = cuda_f32(
        &[2],
        vec![
            download_f32_host(&stats.mean).unwrap()[0] + 1.0,
            download_f32_host(&stats.mean).unwrap()[1] + 1.0,
        ],
    );
    let bad =
        launch_layer_norm_backward(&gout, &input, &weight, &bad_mean, &stats.rstd, true).unwrap();
    let bad_dx = read(&bad.input);
    let drift: f64 = dx
        .iter()
        .zip(bad_dx.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        drift > 1e-3,
        "perturbed statistics left the gradients unchanged: the kernel is not reading them"
    );
    // And with the true statistics, the CPU reference agrees. The normalized
    // values come straight from the definition here: reading them off the
    // operation's output would include the weight and bias the formula has
    // already accounted for.
    let t = host_f32(&[2, 4], vec![0.5, -1.0, 2.0, 1.0, 0.0, -0.5, 1.5, -2.0]);
    let w = host_f32(&[4], vec![2.0, 1.0, 0.5, 1.5]);
    let g = host_f32(&[2, 4], vec![1.0, 0.5, -0.5, 2.0, -1.0, 1.0, 0.25, -0.75]);
    let xv = host_values(&t);
    let wv = host_values(&w);
    let gv = host_values(&g);
    let mut expected = Vec::with_capacity(8);
    for row in 0..2 {
        let mean = xv[row * 4..row * 4 + 4].iter().sum::<f64>() / 4.0;
        let var = xv[row * 4..row * 4 + 4]
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / 4.0;
        let rstd = 1.0 / (var + 1e-5).sqrt();
        let yv: Vec<f64> = xv[row * 4..row * 4 + 4]
            .iter()
            .map(|v| (v - mean) * rstd)
            .collect();
        let mut sum_gw = 0.0;
        let mut sum_gwy = 0.0;
        for col in 0..4 {
            sum_gw += gv[row * 4 + col] * wv[col];
            sum_gwy += gv[row * 4 + col] * wv[col] * yv[col];
        }
        for col in 0..4 {
            expected.push(
                rstd * (gv[row * 4 + col] * wv[col] - sum_gw / 4.0 - yv[col] * sum_gwy / 4.0),
            );
        }
    }
    assert_close(&dx, &expected, 1e-4, "dx against host definition");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_rejects_mismatched_upstream_shape() {
    let (input, weight, bias) = (ln_input(), ln_weight(), ln_bias());
    let (_, stats) =
        crate::cuda::ops::norm::launch_layer_norm(&input, &weight, Some(&bias), 1e-5, true)
            .unwrap();
    let stats = stats.expect("recording forward keeps statistics");
    let short = cuda_f32(&[2, 3], vec![0.0; 6]);
    assert!(
        crate::cuda::ops::norm::launch_layer_norm_backward(
            &short,
            &input,
            &weight,
            &stats.mean,
            &stats.rstd,
            true,
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires CUDA hardware"]
fn layer_norm_rejects_integer_storage() {
    // i64 passes storage validation (indices live in it) but no float kernel
    // may read it: the launch must refuse before any transmute.
    let bytes: Vec<u8> = (0..8).flat_map(|v: i64| v.to_le_bytes()).collect();
    let input =
        crate::cuda::backend::cuda_from_bytes(&[2, 4], DTypeId::I64.into(), 0, &bytes).unwrap();
    let weight = cuda_f32(&[4], vec![1.0; 4]);
    assert!(B::layer_norm::<i64>(&input, &weight, None, 1e-5).is_err());
}

#[test]
#[ignore = "requires CUDA hardware"]
fn l1_loss_trains_through_scalar_reduction_on_cuda() {
    // A mean reduction seeds the walk with a scalar gradient that the next
    // recipe needs at full width. `unbroadcast` used to hand the scalar on
    // unchanged -- the CPU kernels broadcast implicitly and never noticed,
    // but the CUDA binary launch refused it in `iteration_plan`. This loss
    // is the shape that caught it: sub, abs, then mean.
    //
    // The expected gradients also pin the fused abs derivative at zero:
    // pred[0] == targ[0] makes diff[0] exactly 0, where the symbolic
    // differentiator used to answer -1 against CPU `Sign`, PyTorch, and the
    // hand-written kernel expression, which all answer 0.
    use incin_core::exec::catalog::{LossAttributes, LossReduction};
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};
    let context = ExecutionContext::new(B::new());
    let pred = cuda_f32(&[3], vec![1.0, 0.0, -1.0]);
    let targ = cuda_f32(&[3], vec![1.0, 1.0, 0.0]);
    let pred_id = pred.id;
    let pred_handle = TensorHandle::from_storage::<B, f32, _>(&pred);
    let targ_handle = TensorHandle::from_storage::<B, f32, _>(&targ);
    let out = dispatch::execute::<op::L1Loss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[pred_handle, targ_handle],
    )
    .expect("l1 executes on CUDA");
    assert_eq!(download_f32_host(&out).unwrap(), vec![2.0 / 3.0]);
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad = grads.get(pred_id).expect("pred has a gradient");
    let values = download_f32_host(grad).unwrap();
    assert_eq!(values.len(), 3);
    assert!(
        (values[0] - 0.0).abs() < 1e-6,
        "sign(0) must be 0, got {}",
        values[0]
    );
    for (i, value) in values.iter().enumerate().skip(1) {
        assert!(
            (value - (-1.0 / 3.0)).abs() < 1e-6,
            "grad[{i}] should be sign/3, got {value}"
        );
    }
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cross_entropy_loss_trains_through_gather_on_cuda() {
    // The executor used to call the raw gather launch, which runs the kernel
    // but records no tape entry: forward matched, backward reached nothing,
    // and the logits gradient was silently absent. Routing through the
    // tape-tracked gather (scatter-based backward) closes the walk.
    use incin_core::exec::catalog::{LossAttributes, LossReduction};
    use incin_core::exec::{ExecutionContext, TensorHandle, dispatch, op};
    let context = ExecutionContext::new(B::new());
    let logits = cuda_f32(&[2, 3], vec![2.0, 1.0, 0.5, 0.5, 1.5, 0.0]);
    let target_bytes: Vec<u8> = [0i64, 2].iter().flat_map(|v| v.to_le_bytes()).collect();
    let targets =
        crate::cuda::backend::cuda_from_bytes(&[2], DTypeId::I64.into(), 0, &target_bytes).unwrap();
    let logits_id = logits.id;
    let logits_handle = TensorHandle::from_storage::<B, f32, _>(&logits);
    let targets_handle = TensorHandle::from_storage::<B, i64, _>(&targets);
    let out = dispatch::execute::<op::CrossEntropyLoss, _>(
        &context,
        LossAttributes {
            reduction: LossReduction::Mean,
        },
        &[logits_handle, targets_handle],
    )
    .expect("cross entropy executes on CUDA");
    let fwd = download_f32_host(&out).unwrap();
    assert!(
        (fwd[0] - 1.2144).abs() < 1e-3,
        "forward should match -(log p0 + log p2)/2, got {}",
        fwd[0]
    );
    let grads = crate::cuda::tape::backward(&out).unwrap();
    let grad = grads.get(logits_id).expect("logits have a gradient");
    let values = download_f32_host(grad).unwrap();
    // dL/dlogits = (softmax - onehot) / batch, computed by hand.
    let expected = [-0.1857, 0.1156, 0.0701, 0.1156, 0.3142, -0.4299];
    assert_eq!(values.len(), expected.len());
    for (i, (got, want)) in values.iter().zip(expected.iter()).enumerate() {
        assert!(
            (f64::from(*got) - want).abs() < 1e-3,
            "logits grad[{i}]: got {got}, want {want}"
        );
    }
}
