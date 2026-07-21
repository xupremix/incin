#[cfg(test)]
/// Auto-generated documentation for tests.
mod tests {
    use crate::wgpu::storage::{WgpuBuffer, WgpuStorage};
    use crate::wgpu::{WgpuBackend, WgpuVar};
    use kindle_core::prelude::*;

    // Helper: create a WgpuStorage from a flat vec and shape
    /// Auto-generated documentation for storage.
    fn storage(data: Vec<f32>, shape: Vec<usize>) -> WgpuStorage {
        WgpuStorage::new(WgpuBuffer::from_slice(&data), shape)
    }

    /// Auto-generated documentation for readback.
    fn readback(s: &WgpuStorage) -> Vec<f32> {
        s.buffer.to_vec::<f32>()
    }

    /// Auto-generated documentation for approx_eq.
    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    /// Auto-generated documentation for vec_approx_eq.
    fn vec_approx_eq(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| approx_eq(*x, *y, tol))
    }

    /// Auto-generated documentation for B.
    type B = WgpuBackend<f32, Cpu>;

    // ── Creation ──────────────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_zeros.
    fn test_zeros() {
        let s =
            <B as CreationOps<B>>::zeros::<f32>(&[2, 3], KindleDType::F32, &KindleDevice::cpu())
                .unwrap();
        assert_eq!(s.shape, vec![2, 3]);
        assert!(readback(&s).iter().all(|&x| x == 0.0));
    }

    #[test]
    /// Auto-generated documentation for test_ones.
    fn test_ones() {
        let s = <B as CreationOps<B>>::ones::<f32>(&[3, 2], KindleDType::F32, &KindleDevice::cpu())
            .unwrap();
        assert!(readback(&s).iter().all(|&x| x == 1.0));
    }

    #[test]
    /// Auto-generated documentation for test_rand_shape.
    fn test_rand_shape() {
        let s = <B as CreationOps<B>>::rand::<f32>(&[4, 4], KindleDType::F32, &KindleDevice::cpu())
            .unwrap();
        assert_eq!(s.shape, vec![4, 4]);
        let data = readback(&s);
        // All values should be in [0, 1)
        assert!(data.iter().all(|&x| x >= 0.0 && x < 1.0));
    }

    #[test]
    /// Auto-generated documentation for test_randn_shape.
    fn test_randn_shape() {
        let s = <B as CreationOps<B>>::randn::<f32>(&[100], KindleDType::F32, &KindleDevice::cpu())
            .unwrap();
        assert_eq!(s.shape, vec![100]);
    }

    // ── Binary ops ────────────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_add.
    fn test_add() {
        let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = storage(vec![10.0, 20.0, 30.0, 40.0], vec![4]);
        let out = <B as NumericOps<B>>::add::<f32>(&a, &b).unwrap();
        assert_eq!(readback(&out), vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    /// Auto-generated documentation for test_sub.
    fn test_sub() {
        let a = storage(vec![10.0, 20.0, 30.0], vec![3]);
        let b = storage(vec![1.0, 2.0, 3.0], vec![3]);
        let out = <B as NumericOps<B>>::sub::<f32>(&a, &b).unwrap();
        assert_eq!(readback(&out), vec![9.0, 18.0, 27.0]);
    }

    #[test]
    /// Auto-generated documentation for test_mul.
    fn test_mul() {
        let a = storage(vec![2.0, 3.0, 4.0], vec![3]);
        let b = storage(vec![5.0, 6.0, 7.0], vec![3]);
        let out = <B as NumericOps<B>>::mul::<f32>(&a, &b).unwrap();
        assert_eq!(readback(&out), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    /// Auto-generated documentation for test_div.
    fn test_div() {
        let a = storage(vec![10.0, 20.0, 30.0], vec![3]);
        let b = storage(vec![2.0, 4.0, 5.0], vec![3]);
        let out = <B as NumericOps<B>>::div::<f32>(&a, &b).unwrap();
        assert!(vec_approx_eq(&readback(&out), &[5.0, 5.0, 6.0], 1e-5));
    }

    // ── Matmul ────────────────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_matmul_2x3_3x2.
    fn test_matmul_2x3_3x2() {
        let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = storage(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);
        let out = <B as TensorOps<B>>::matmul::<f32>(&a, &b).unwrap();
        assert_eq!(out.shape, vec![2, 2]);
        assert!(vec_approx_eq(
            &readback(&out),
            &[58.0, 64.0, 139.0, 154.0],
            1e-4
        ));
    }

    #[test]
    /// Auto-generated documentation for test_matmul_square.
    fn test_matmul_square() {
        let a = storage(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]); // identity
        let b = storage(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]);
        let out = <B as TensorOps<B>>::matmul::<f32>(&a, &b).unwrap();
        assert!(vec_approx_eq(&readback(&out), &[3.0, 4.0, 5.0, 6.0], 1e-4));
    }

    // ── Float / Unary ops ─────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_relu.
    fn test_relu() {
        let a = storage(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5]);
        let out = <B as FloatOps<B>>::relu::<f32>(&a).unwrap();
        assert_eq!(readback(&out), vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    /// Auto-generated documentation for test_neg.
    fn test_neg() {
        let a = storage(vec![1.0, -2.0, 3.0], vec![3]);
        let out = <B as FloatOps<B>>::neg::<f32>(&a).unwrap();
        assert_eq!(readback(&out), vec![-1.0, 2.0, -3.0]);
    }

    #[test]
    /// Auto-generated documentation for test_abs.
    fn test_abs() {
        let a = storage(vec![-3.0, 0.0, 4.0], vec![3]);
        let out = <B as FloatOps<B>>::abs::<f32>(&a).unwrap();
        assert_eq!(readback(&out), vec![3.0, 0.0, 4.0]);
    }

    #[test]
    /// Auto-generated documentation for test_sqrt.
    fn test_sqrt() {
        let a = storage(vec![4.0, 9.0, 16.0], vec![3]);
        let out = <B as FloatOps<B>>::sqrt::<f32>(&a).unwrap();
        assert!(vec_approx_eq(&readback(&out), &[2.0, 3.0, 4.0], 1e-5));
    }

    #[test]
    /// Auto-generated documentation for test_exp_log.
    fn test_exp_log() {
        let a = storage(vec![0.0, 1.0, 2.0], vec![3]);
        let exp_out = <B as FloatOps<B>>::exp::<f32>(&a).unwrap();
        let expected_exp = [1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)];
        assert!(vec_approx_eq(&readback(&exp_out), &expected_exp, 1e-5));

        let log_out = <B as FloatOps<B>>::log::<f32>(&exp_out).unwrap();
        assert!(vec_approx_eq(&readback(&log_out), &[0.0, 1.0, 2.0], 1e-5));
    }

    #[test]
    /// Auto-generated documentation for test_sigmoid.
    fn test_sigmoid() {
        let a = storage(vec![0.0], vec![1]);
        let out = <B as FloatOps<B>>::sigmoid::<f32>(&a).unwrap();
        assert!(approx_eq(readback(&out)[0], 0.5, 1e-5));
    }

    #[test]
    /// Auto-generated documentation for test_tanh.
    fn test_tanh() {
        let a = storage(vec![0.0], vec![1]);
        let out = <B as FloatOps<B>>::tanh::<f32>(&a).unwrap();
        assert!(approx_eq(readback(&out)[0], 0.0, 1e-5));
    }

    #[test]
    /// Auto-generated documentation for test_swish.
    fn test_swish() {
        // swish(x) = x * sigmoid(x); swish(0) = 0
        let a = storage(vec![0.0, 1.0], vec![2]);
        let out = <B as FloatOps<B>>::swish::<f32>(&a).unwrap();
        let data = readback(&out);
        assert!(approx_eq(data[0], 0.0, 1e-5));
        // swish(1) = 1 * sigmoid(1) ≈ 0.7311
        assert!(approx_eq(data[1], 0.7310586, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_gelu.
    fn test_gelu() {
        let a = storage(vec![0.0], vec![1]);
        let out = <B as FloatOps<B>>::gelu::<f32>(&a).unwrap();
        assert!(approx_eq(readback(&out)[0], 0.0, 1e-5));
    }

    #[test]
    /// Auto-generated documentation for test_add_scalar.
    fn test_add_scalar() {
        let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
        let out = <B as FloatOps<B>>::add_scalar_float::<f32>(&a, 10.0).unwrap();
        assert!(vec_approx_eq(&readback(&out), &[11.0, 12.0, 13.0], 1e-5));
    }

    #[test]
    /// Auto-generated documentation for test_mul_scalar.
    fn test_mul_scalar() {
        let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
        let out = <B as FloatOps<B>>::mul_scalar_float::<f32>(&a, 3.0).unwrap();
        assert!(vec_approx_eq(&readback(&out), &[3.0, 6.0, 9.0], 1e-5));
    }

    #[test]
    /// Auto-generated documentation for test_softmax.
    fn test_softmax() {
        let a = storage(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out = <B as FloatOps<B>>::softmax::<f32>(&a, 1).unwrap();
        let data = readback(&out);
        // Sum should be 1
        let sum: f32 = data.iter().sum();
        assert!(approx_eq(sum, 1.0, 1e-5));
        // Max should be at index 2
        let max_idx = data
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        assert_eq!(max_idx, 2);
    }

    // ── Reduction ops ─────────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_sum_all.
    fn test_sum_all() {
        let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let out = <B as ReductionOps<B>>::sum_all::<f32>(&a).unwrap();
        assert!(approx_eq(readback(&out)[0], 10.0, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_mean_all.
    fn test_mean_all() {
        let a = storage(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = <B as ReductionOps<B>>::mean_all::<f32>(&a).unwrap();
        assert!(approx_eq(readback(&out)[0], 2.5, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_max_all.
    fn test_max_all() {
        let a = storage(vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0], vec![6]);
        let out = <B as ReductionOps<B>>::max_all::<f32>(&a).unwrap();
        assert!(approx_eq(readback(&out)[0], 9.0, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_min_all.
    fn test_min_all() {
        let a = storage(vec![3.0, 1.0, 4.0, -2.0, 5.0], vec![5]);
        let out = <B as ReductionOps<B>>::min_all::<f32>(&a).unwrap();
        assert!(approx_eq(readback(&out)[0], -2.0, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_sum_dim.
    fn test_sum_dim() {
        // [[1,2,3],[4,5,6]] sum along dim 0 -> [5,7,9]
        let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = <B as ReductionOps<B>>::sum_dim::<f32>(&a, 0).unwrap();
        assert!(vec_approx_eq(&readback(&out), &[5.0, 7.0, 9.0], 1e-4));
        assert_eq!(out.shape, vec![3]);
    }

    #[test]
    /// Auto-generated documentation for test_sum_keepdim.
    fn test_sum_keepdim() {
        let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = <B as ReductionOps<B>>::sum_keepdim::<f32>(&a, 0).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
    }

    #[test]
    /// Auto-generated documentation for test_mean_dim.
    fn test_mean_dim() {
        let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = <B as ReductionOps<B>>::mean_dim::<f32>(&a, 0).unwrap();
        assert!(vec_approx_eq(&readback(&out), &[2.5, 3.5, 4.5], 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_max_dim.
    fn test_max_dim() {
        let a = storage(vec![1.0, 5.0, 3.0, 4.0, 2.0, 6.0], vec![2, 3]);
        let out = <B as ReductionOps<B>>::max_dim::<f32>(&a, 0).unwrap();
        assert!(vec_approx_eq(&readback(&out), &[4.0, 5.0, 6.0], 1e-4));
    }
    #[test]
    /// Auto-generated documentation for test_argmax_flat.
    fn test_argmax_flat() {
        let a = storage(vec![1.0, 5.0, 3.0, 9.0, 2.0], vec![5]);
        let out = <B as ReductionOps<B>>::argmax::<f32, u32>(&a, None).unwrap();
        assert_eq!(out.buffer.to_vec::<u32>()[0] as usize, 3);
    }

    #[test]
    /// Auto-generated documentation for test_argmin_flat.
    fn test_argmin_flat() {
        let a = storage(vec![3.0, -1.0, 5.0], vec![3]);
        let out = <B as ReductionOps<B>>::argmin::<f32, u32>(&a, None).unwrap();
        assert_eq!(out.buffer.to_vec::<u32>()[0] as usize, 1);
    }

    // ── Tensor ops ────────────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_reshape.
    fn test_reshape() {
        let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = <B as TensorOps<B>>::reshape::<f32>(&a, &[3, 2]).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert_eq!(readback(&out), readback(&a)); // same buffer, same data
    }

    #[test]
    /// Auto-generated documentation for test_transpose_2d.
    fn test_transpose_2d() {
        // [[1,2,3],[4,5,6]] -> [[1,4],[2,5],[3,6]]
        let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = <B as TensorOps<B>>::transpose::<f32>(&a, 0, 1).unwrap();
        assert_eq!(out.shape, vec![3, 2]);
        assert!(vec_approx_eq(
            &readback(&out),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0],
            1e-5
        ));
    }

    #[test]
    /// Auto-generated documentation for test_flatten.
    fn test_flatten() {
        let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = <B as TensorOps<B>>::flatten::<f32>(&a, 0, 1).unwrap();
        assert_eq!(out.shape, vec![6]);
    }

    #[test]
    /// Auto-generated documentation for test_squeeze.
    fn test_squeeze() {
        let a = storage(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out = <B as TensorOps<B>>::squeeze::<f32>(&a, 0).unwrap();
        assert_eq!(out.shape, vec![3]);
    }

    #[test]
    /// Auto-generated documentation for test_narrow.
    fn test_narrow() {
        // [[1,2,3],[4,5,6]] narrow dim=0, start=1, len=1 -> [[4,5,6]]
        let a = storage(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let out = <B as TensorOps<B>>::narrow::<f32>(&a, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 3]);
        assert!(vec_approx_eq(&readback(&out), &[4.0, 5.0, 6.0], 1e-5));
    }

    #[test]
    /// Auto-generated documentation for test_concat_dim0.
    fn test_concat_dim0() {
        let a = storage(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let b = storage(vec![4.0, 5.0, 6.0], vec![1, 3]);
        let out = <B as TensorOps<B>>::concat::<f32>(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(readback(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    /// Auto-generated documentation for test_stack.
    fn test_stack() {
        let a = storage(vec![1.0, 2.0, 3.0], vec![3]);
        let b = storage(vec![4.0, 5.0, 6.0], vec![3]);
        let out = <B as TensorOps<B>>::stack::<f32>(&[&a, &b], 0).unwrap();
        assert_eq!(out.shape, vec![2, 3]);
        assert_eq!(readback(&out), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    /// Auto-generated documentation for test_float_to_scalar.
    fn test_float_to_scalar() {
        let a = storage(vec![42.0], vec![1]);
        let val = <B as TensorOps<B>>::float_to_scalar::<f32>(&a).unwrap();
        assert!(approx_eq(val as f32, 42.0, 1e-5));
    }

    // ── Module ops ────────────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_embedding.
    fn test_embedding() {
        // vocab=3, dim=4; pick indices [0, 2]
        let weight = storage(
            vec![
                0.1, 0.2, 0.3, 0.4, // token 0
                0.5, 0.6, 0.7, 0.8, // token 1
                0.9, 1.0, 1.1, 1.2, // token 2
            ],
            vec![3, 4],
        );
        let indices = storage(vec![0.0, 2.0], vec![2]);
        let out = <B as ModuleOps<B>>::embedding::<f32, f32>(&indices, &weight).unwrap();
        assert_eq!(out.shape, vec![2, 4]);
        assert!(vec_approx_eq(
            &readback(&out),
            &[0.1, 0.2, 0.3, 0.4, 0.9, 1.0, 1.1, 1.2],
            1e-5
        ));
    }

    #[test]
    /// Auto-generated documentation for test_layer_norm.
    fn test_layer_norm() {
        let x = storage(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let gamma = storage(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let beta = storage(vec![0.0, 0.0, 0.0, 0.0], vec![4]);
        let out = <B as ModuleOps<B>>::layer_norm::<f32>(&x, &gamma, Some(&beta), 1e-5).unwrap();
        let data = readback(&out);
        // After layer norm, mean≈0, std≈1
        let mean: f32 = data.iter().sum::<f32>() / 4.0;
        assert!(approx_eq(mean, 0.0, 1e-4));
        let var: f32 = data.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / 4.0;
        assert!(approx_eq(var, 1.0, 1e-3));
    }

    #[test]
    /// Auto-generated documentation for test_adaptive_avg_pool2d.
    fn test_adaptive_avg_pool2d() {
        // 1x1x4x4 -> 1x1x2x2
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let inp = storage(data, vec![1, 1, 4, 4]);
        let out = <B as ModuleOps<B>>::adaptive_avg_pool2d::<f32>(&inp, (2, 2)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        let result = readback(&out);
        // Top-left 2x2 avg = (1+2+5+6)/4 = 3.5
        assert!(approx_eq(result[0], 3.5, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_max_pool2d.
    fn test_max_pool2d() {
        // 1x1x4x4, kernel 2x2, stride 2
        let data: Vec<f32> = vec![
            1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0, 9.0, 11.0, 10.0, 12.0, 13.0, 15.0, 14.0, 16.0,
        ];
        let inp = storage(data, vec![1, 1, 4, 4]);
        let out =
            <B as ModuleOps<B>>::max_pool2d::<f32>(&inp, (2, 2), (2, 2), (0, 0), (1, 1)).unwrap();
        assert_eq!(out.shape, vec![1, 1, 2, 2]);
        let result = readback(&out);
        assert!(approx_eq(result[0], 7.0, 1e-4));
        assert!(approx_eq(result[1], 8.0, 1e-4));
        assert!(approx_eq(result[2], 15.0, 1e-4));
        assert!(approx_eq(result[3], 16.0, 1e-4));
    }

    // ── Loss ops ──────────────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_cross_entropy_mean.
    fn test_cross_entropy_mean() {
        // 2 classes, batch=2; pred logits
        let pred = storage(vec![2.0, 1.0, 0.5, 3.0], vec![2, 2]);
        let target = storage(vec![0.0, 1.0], vec![2]); // class 0 and class 1
        let out = <B as LossOps<B>>::cross_entropy_loss::<f32, f32>(
            &pred,
            &target,
            kindle_core::prelude::Reduction::Mean,
        )
        .unwrap();
        let loss = readback(&out)[0];
        assert!(loss > 0.0, "Loss should be positive");
        assert!(loss < 5.0, "Loss should be reasonable");
    }

    // ── Optimizer ─────────────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_adamw_step.
    fn test_adamw_step() {
        let param = storage(vec![1.0, 2.0, 3.0], vec![3]);
        let grad = storage(vec![0.1, 0.2, 0.3], vec![3]);
        let mut m = storage(vec![0.0, 0.0, 0.0], vec![3]);
        let mut v = storage(vec![0.0, 0.0, 0.0], vec![3]);

        let old_param = readback(&param);
        let mut var = WgpuVar {
            storage: param.clone(),
        };

        <B as OptimizerOps<B>>::adamw_step::<f32>(
            &mut var, &grad, &mut m, &mut v, 1e-3, 0.9, 0.999, 1e-8, 0.01, 1,
        )
        .unwrap();

        let new_param = readback(&var.storage);
        // Parameters should have moved
        assert!(
            new_param
                .iter()
                .zip(old_param.iter())
                .any(|(n, o)| (n - o).abs() > 1e-6)
        );
    }

    // ── GPU AdamW parity ─────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_adamw_gpu_matches_reference.
    fn test_adamw_gpu_matches_reference() {
        // Reference CPU calculation for a single AdamW step
        let lr = 0.001f32;
        let beta1 = 0.9f32;
        let beta2 = 0.999f32;
        let eps = 1e-8f32;
        let wd = 0.01f32;
        let step = 1;

        let p_init = vec![1.0f32, -0.5, 2.0];
        let g_init = vec![0.1f32, 0.2, -0.3];

        // Reference calculation
        let mut ref_p = p_init.clone();
        let mut ref_m = vec![0.0f32; 3];
        let mut ref_v = vec![0.0f32; 3];
        for i in 0..3 {
            let p_val = p_init[i] - lr * wd * p_init[i];
            let g = g_init[i];

            ref_m[i] = beta1 * ref_m[i] + (1.0 - beta1) * g;
            ref_v[i] = beta2 * ref_v[i] + (1.0 - beta2) * g * g;

            ref_p[i] = p_val - lr * ref_m[i] / (ref_v[i].sqrt() + eps);
        }

        // GPU calculation
        let param = storage(p_init.clone(), vec![3]);
        let grad = storage(g_init.clone(), vec![3]);
        let mut m = storage(vec![0.0, 0.0, 0.0], vec![3]);
        let mut v = storage(vec![0.0, 0.0, 0.0], vec![3]);
        let mut var = WgpuVar {
            storage: param.clone(),
        };

        <B as OptimizerOps<B>>::adamw_step::<f32>(
            &mut var,
            &grad,
            &mut m,
            &mut v,
            lr as f64,
            beta1 as f64,
            beta2 as f64,
            eps as f64,
            wd as f64,
            step as usize,
        )
        .unwrap();

        let gpu_p = readback(&var.storage);
        assert!(
            vec_approx_eq(&gpu_p, &ref_p, 1e-5),
            "GPU AdamW mismatch: got {:?}, expected {:?}",
            gpu_p,
            ref_p
        );
    }

    // ── Conv ops ──────────────────────────────────────────────────────────────

    #[test]
    /// Auto-generated documentation for test_conv2d_identity_kernel.
    fn test_conv2d_identity_kernel() {
        // 3x3 identity filter: output should equal the center pixel (no padding, stride=1)
        // Input: 1x1x3x3, Weight: 1x1x1x1 = [[1.0]] -> output = input
        let inp = storage(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![1, 1, 3, 3],
        );
        let weight = storage(vec![1.0], vec![1, 1, 1, 1]);
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, None, 1, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
        assert!(vec_approx_eq(
            &readback(&out),
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            1e-5
        ));
    }

    #[test]
    /// Auto-generated documentation for test_conv2d_known_output.
    fn test_conv2d_known_output() {
        // Input 1x1x4x4, weight 1x1x2x2, stride=1, padding=0
        // Using weight [[1,0],[0,1]] (sum of diagonal)
        let inp_data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let inp = storage(inp_data, vec![1, 1, 4, 4]);
        let weight = storage(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, None, 1, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
        let result = readback(&out);
        // out[0,0] = inp[0,0]*1 + inp[0,1]*0 + inp[1,0]*0 + inp[1,1]*1 = 1 + 6 = 7
        assert!(approx_eq(result[0], 7.0, 1e-4));
        // out[0,1] = inp[0,1]*1 + inp[0,2]*0 + inp[1,1]*0 + inp[1,2]*1 = 2 + 7 = 9
        assert!(approx_eq(result[1], 9.0, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_conv2d_with_bias.
    fn test_conv2d_with_bias() {
        let inp = storage(vec![1.0, 2.0, 2.0, 1.0], vec![1, 1, 2, 2]);
        let weight = storage(vec![1.0, 1.0, 1.0, 1.0], vec![1, 1, 2, 2]);
        let bias = storage(vec![10.0], vec![1]);
        let out =
            <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, Some(&bias), 1, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 1, 1]);
        // sum(1+2+2+1) = 6, + bias 10 = 16
        assert!(approx_eq(readback(&out)[0], 16.0, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_conv2d_padding.
    fn test_conv2d_padding() {
        // 1x1x1x1 input with 1-padding, 1x1x3x3 kernel -> 1x1x1x1 output
        let inp = storage(vec![1.0], vec![1, 1, 1, 1]);
        let weight = storage(
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            vec![1, 1, 3, 3],
        );
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, None, 1, 1, 1, 1).unwrap();
        // Input padded: only center = 1.0, matches weight center (index 4)
        assert_eq!(out.shape, vec![1, 1, 1, 1]);
        assert!(approx_eq(readback(&out)[0], 1.0, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_conv2d_two_output_channels.
    fn test_conv2d_two_output_channels() {
        // 1 input, 2 output channels: each filter reads the same input
        let inp = storage(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        // Filter 1: sum all    Filter 2: negate sum
        let weight = storage(
            vec![
                1.0, 1.0, 1.0, 1.0, // C_out=0
                -1.0, -1.0, -1.0, -1.0, // C_out=1
            ],
            vec![2, 1, 2, 2],
        );
        let out = <B as ModuleOps<B>>::conv2d::<f32>(&inp, &weight, None, 1, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 2, 1, 1]);
        let data = readback(&out);
        assert!(approx_eq(data[0], 10.0, 1e-4)); // 1+2+3+4=10
        assert!(approx_eq(data[1], -10.0, 1e-4));
    }

    #[test]
    /// Auto-generated documentation for test_conv1d_basic.
    fn test_conv1d_basic() {
        // Input: 1 batch, 1 channel, 4 elements; Weight: 1 out, 1 in, 2 kernel
        let inp = storage(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 4]);
        let weight = storage(vec![1.0, 1.0], vec![1, 1, 2]);
        let out = <B as ModuleOps<B>>::conv1d::<f32>(&inp, &weight, None, 1, 0, 1, 1).unwrap();
        assert_eq!(out.shape, vec![1, 1, 3]);
        // out[0] = 1+2=3, out[1] = 2+3=5, out[2] = 3+4=7
        assert!(vec_approx_eq(&readback(&out), &[3.0, 5.0, 7.0], 1e-4));
    }

    #[test]
    fn test_quantize_dequantize() {
        let mut data = vec![0.0f32; 64];
        for i in 0..64 {
            data[i] = (i as f32 - 32.0) * 0.1; // ranging -3.2 to +3.1
        }
        let s = storage(data.clone(), vec![2, 32]);
        let q_storage = <B as QuantizedOps<B>>::quantize::<f32, kindle_core::prelude::Q8_0>(&s).unwrap();
        let deq_storage = <B as QuantizedOps<B>>::dequantize::<kindle_core::prelude::Q8_0, f32>(&q_storage).unwrap();
        let deq_data = readback(&deq_storage);

        for (orig, deq) in data.iter().zip(deq_data.iter()) {
            let diff = (orig - deq).abs();
            assert!(diff < 0.05, "Diff too large: {} vs {}", orig, deq);
        }
    }
}
