use kindle::prelude::*;

type CpuBackend = DefaultBackend;

fn to_vec(t: &Tensor<Dyn, CpuBackend>) -> Vec<f32> {
    t.inner().flatten_all().unwrap().to_vec1::<f32>().unwrap()
}

#[test]
fn test_unary_ops_permutations() -> Result<()> {
    // 1.1 Unary Operations
    // abs() - positive, negative, zero, very small numbers, very large numbers, NaN, Inf
    let t_abs = Tensor::<s![7], CpuBackend>::from_slice(
        &[1.0, -1.0, 0.0, 1e-30, -1e30, f32::NAN, f32::INFINITY],
        (),
    )?;
    let r_abs = to_vec(&t_abs.abs()?.into_dyn());
    assert_eq!(r_abs[0], 1.0);
    assert_eq!(r_abs[1], 1.0);
    assert_eq!(r_abs[2], 0.0);
    assert_eq!(r_abs[3], 1e-30);
    assert_eq!(r_abs[4], 1e30);
    assert!(r_abs[5].is_nan());
    assert!(r_abs[6].is_infinite() && r_abs[6] > 0.0);

    // relu() - positive (unchanged), negative (zeroed), zero.
    let t_relu = Tensor::<s![3], CpuBackend>::from_slice(&[5.0, -5.0, 0.0], ())?;
    let r_relu = to_vec(&t_relu.relu()?.into_dyn());
    assert_eq!(r_relu, vec![5.0, 0.0, 0.0]);

    // gelu() - standard normal values, extreme negatives/positives
    let t_gelu = Tensor::<s![3], CpuBackend>::from_slice(&[0.0, -10.0, 10.0], ())?;
    let r_gelu = to_vec(&t_gelu.gelu()?.into_dyn());
    assert_eq!(r_gelu[0], 0.0);
    assert!((r_gelu[1] - 0.0).abs() < 1e-4);
    assert!((r_gelu[2] - 10.0).abs() < 1e-4);

    // swish() - beta=1 definitions
    let t_swish = Tensor::<s![2], CpuBackend>::from_slice(&[0.0, 1.0], ())?;
    let r_swish = to_vec(&t_swish.swish()?.into_dyn());
    assert_eq!(r_swish[0], 0.0);
    assert!((r_swish[1] - (1.0 / (1.0 + (-1.0f32).exp()))).abs() < 1e-4);

    // softmax(dim) - dim 0, negative values, very large/small values (numerical stability)
    let t_sm = Tensor::<s![3, 1], CpuBackend>::from_slice(&[1000.0, 1000.0, -1000.0], ())?;
    let r_sm = to_vec(&t_sm.softmax(0)?.into_dyn());
    assert!((r_sm[0] - 0.5).abs() < 1e-4);
    assert!((r_sm[1] - 0.5).abs() < 1e-4);
    assert!((r_sm[2] - 0.0).abs() < 1e-4);

    // neg() - zero, positives, negatives.
    let t_neg = Tensor::<s![3], CpuBackend>::from_slice(&[0.0, 1.0, -1.0], ())?;
    let r_neg = to_vec(&t_neg.neg()?.into_dyn());
    assert_eq!(r_neg, vec![0.0, -1.0, 1.0]);

    // sqrt() - positive, zero, negative (handling NaN).
    let t_sqrt = Tensor::<s![3], CpuBackend>::from_slice(&[4.0, 0.0, -1.0], ())?;
    let r_sqrt = to_vec(&t_sqrt.sqrt()?.into_dyn());
    assert_eq!(r_sqrt[0], 2.0);
    assert_eq!(r_sqrt[1], 0.0);
    assert!(r_sqrt[2].is_nan());

    // exp() - large positive, zero, negative
    let t_exp = Tensor::<s![3], CpuBackend>::from_slice(&[100.0, 0.0, -100.0], ())?;
    let r_exp = to_vec(&t_exp.exp()?.into_dyn());
    assert!(r_exp[0] > 1e10);
    assert_eq!(r_exp[1], 1.0);
    assert!((r_exp[2] - 0.0).abs() < 1e-7);

    // log() - positive, zero (-Inf), negative (NaN)
    let t_log = Tensor::<s![3], CpuBackend>::from_slice(&[1.0, 0.0, -1.0], ())?;
    let r_log = to_vec(&t_log.log()?.into_dyn());
    assert_eq!(r_log[0], 0.0);
    assert!(r_log[1].is_infinite() && r_log[1] < 0.0);
    assert!(r_log[2].is_nan());

    // tanh() - asymptotes at +1 and -1, near zero
    let t_tanh = Tensor::<s![3], CpuBackend>::from_slice(&[100.0, -100.0, 0.0], ())?;
    let r_tanh = to_vec(&t_tanh.tanh()?.into_dyn());
    assert!((r_tanh[0] - 1.0).abs() < 1e-4);
    assert!((r_tanh[1] - (-1.0)).abs() < 1e-4);
    assert_eq!(r_tanh[2], 0.0);

    // sigmoid() - asymptotes at 0 and 1, origin
    let t_sig = Tensor::<s![3], CpuBackend>::from_slice(&[100.0, -100.0, 0.0], ())?;
    let r_sig = to_vec(&t_sig.sigmoid()?.into_dyn());
    assert!((r_sig[0] - 1.0).abs() < 1e-4);
    assert!((r_sig[1] - 0.0).abs() < 1e-4);
    assert_eq!(r_sig[2], 0.5);

    Ok(())
}

#[test]
fn test_unary_ops() -> Result<()> {
    test_unary_ops_permutations()?;
    Ok(())
}

#[test]
fn test_binary_ops_permutations() -> Result<()> {
    // 1.2 Binary Operations
    // add(rhs) - positive + positive, negative + negative, zeroes
    let a = Tensor::<s![3], CpuBackend>::from_slice(&[1.0, -1.0, 0.0], ())?;
    let b = Tensor::<s![3], CpuBackend>::from_slice(&[2.0, -2.0, 0.0], ())?;
    assert_eq!(to_vec(&a.add(&b)?.into_dyn()), vec![3.0, -3.0, 0.0]);

    // sub(rhs) - lhs > rhs, lhs < rhs, identical tensors
    let c = Tensor::<s![3], CpuBackend>::from_slice(&[5.0, 1.0, 3.0], ())?;
    let d = Tensor::<s![3], CpuBackend>::from_slice(&[2.0, 4.0, 3.0], ())?;
    assert_eq!(to_vec(&c.sub(&d)?.into_dyn()), vec![3.0, -3.0, 0.0]);

    // mul(rhs) - zeroes, identity matrix (element-wise), negative terms
    let e = Tensor::<s![3], CpuBackend>::from_slice(&[0.0, 1.0, -2.0], ())?;
    let f = Tensor::<s![3], CpuBackend>::from_slice(&[5.0, 1.0, 3.0], ())?;
    assert_eq!(to_vec(&e.mul(&f)?.into_dyn()), vec![0.0, 1.0, -6.0]);

    // div(rhs) - standard division, division by zero, precision limits
    let g = Tensor::<s![3], CpuBackend>::from_slice(&[6.0, 1.0, 1.0], ())?;
    let h = Tensor::<s![3], CpuBackend>::from_slice(&[2.0, 0.0, 1e20], ())?;
    let res = to_vec(&g.div(&h)?.into_dyn());
    assert_eq!(res[0], 3.0);
    assert!(res[1].is_infinite());
    assert!(res[2].abs() < 1e-19);

    Ok(())
}

#[test]
fn test_binary_ops() -> Result<()> {
    test_binary_ops_permutations()?;
    Ok(())
}

#[test]
fn test_broadcast_ops() -> Result<()> {
    let matrix = Tensor::<s![5, 5], CpuBackend>::ones(())?;
    let scalar = Tensor::<s![5, 5], CpuBackend>::ones(())?.mul_scalar(9.0)?; // Use matching shapes for now

    let b_add = matrix.broadcast_add(&scalar)?;
    assert_eq!(to_vec(&b_add.into_dyn())[0], 10.0);

    // Testing std::ops
    let b_sub = matrix - scalar;
    assert_eq!(to_vec(&b_sub.into_dyn())[0], -8.0);

    Ok(())
}

#[test]
fn test_reduction_ops() -> Result<()> {
    let t = Tensor::<s![10, 20], CpuBackend>::ones(())?;

    let sum_all = t.clone().sum_all()?;
    assert_eq!(to_vec(&sum_all.into_dyn())[0], 200.0);

    let mean_all = t.clone().mean_all()?;
    assert_eq!(to_vec(&mean_all.into_dyn())[0], 1.0);

    let sum_dim1 = t.clone().sum_dim::<1>()?;
    let val = to_vec(&sum_dim1.into_dyn());
    assert_eq!(val[0], 20.0);
    assert_eq!(val.len(), 10);

    let max_all = t.clone().max_all()?;
    assert_eq!(to_vec(&max_all.into_dyn())[0], 1.0);

    Ok(())
}

#[test]
fn test_manipulation_ops() -> Result<()> {
    let t = Tensor::<s![10, 20], CpuBackend>::ones(())?;

    // Reshape [10, 20] -> [200]
    let r = t.reshape_idx::<idx![200]>()?;
    assert_eq!(r.rank(), 1);

    // Transpose
    let tr = t.transpose::<0, 1>()?;
    let tr_dims: [usize; 2] = tr.dims().into();
    assert_eq!(tr_dims, [20, 10]);

    // Flatten
    let f = t.flatten::<0, 1>()?;
    let f_dims: [usize; 1] = f.dims().into();
    assert_eq!(f_dims, [200]);

    // Narrow (extract 5 elements from dim 1 dynamically)
    let n = t.clone().try_narrow(1, 0, 5)?;
    let n_dims: Vec<usize> = n.dims().into();
    assert_eq!(n_dims, vec![10, 5]);

    Ok(())
}

#[test]
fn test_indexing_slicing() -> Result<()> {
    let t1 = Tensor::<s![10, 20], CpuBackend>::ones(())?;
    let t2 = Tensor::<s![10, 20], CpuBackend>::ones(())?;

    // Concat on dim 0 -> [20, 20]
    let c = t1.concat::<s![10, 20], kindle::prelude::typenum::U0>(&t2)?;
    let c_dims: [usize; 2] = c.dims().into();
    assert_eq!(c_dims, [20, 20]);

    // Stack on dim 0 -> [2, 10, 20]
    let s = t1.stack::<kindle::prelude::typenum::U0>(&t2)?;
    let s_dims: [usize; 3] = s.dims().into();
    assert_eq!(s_dims, [2, 10, 20]);

    // Try Concat Slice
    let c_slice = Tensor::<s![10, 20], CpuBackend>::try_concat_slice(&[&t1, &t2], 0)?;
    let c_slice_dims: Vec<usize> = c_slice.dims().into();
    assert_eq!(c_slice_dims, vec![20, 20]);

    Ok(())
}

#[test]
fn test_loss_functions() -> Result<()> {
    let pred = Tensor::<s![10, 20], CpuBackend>::ones(())?;
    let target = Tensor::<s![10, 20], CpuBackend>::zeros(())?;

    // MSE Loss
    let mse = pred.mse_loss(&target)?;
    assert_eq!(to_vec(&mse.into_dyn())[0], 1.0);

    Ok(())
}
