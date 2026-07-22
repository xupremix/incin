use kindle::prelude::*;

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = kindle_backends::cpu::CpuBackendImpl;

fn to_vec(t: &Tensor<Dyn, CpuBackendImpl>) -> Vec<f32> {
    t.to_vec1::<f32>().unwrap()
}

// -----------------------------------------------------------------------------
// 1.1 Unary Operations
// -----------------------------------------------------------------------------
#[test]
/// Test unary abs.
fn test_unary_abs() -> Result<()> {
    // permutations: positive, negative, zero, very small numbers, very large numbers, NaN, Inf
    let t = Tensor::<s![7], CpuBackendImpl>::from_slice(
        &[1.0, -1.0, 0.0, 1e-30, -1e30, f32::NAN, f32::INFINITY],
        (),
    )?;
    let r = to_vec(&t.abs()?.into_dyn());
    assert_eq!(r[0], 1.0);
    assert_eq!(r[1], 1.0);
    assert_eq!(r[2], 0.0);
    assert_eq!(r[3], 1e-30);
    assert_eq!(r[4], 1e30);
    assert!(r[5].is_nan());
    assert!(r[6].is_infinite() && r[6] > 0.0);
    Ok(())
}

#[test]
/// Test unary relu.
fn test_unary_relu() -> Result<()> {
    // positive (unchanged), negative (zeroed), zero
    let t = Tensor::<s![3], CpuBackendImpl>::from_slice(&[5.0, -5.0, 0.0], ())?;
    let r = to_vec(&t.relu()?.into_dyn());
    assert_eq!(r, vec![5.0, 0.0, 0.0]);
    Ok(())
}

#[test]
/// Test unary gelu.
fn test_unary_gelu() -> Result<()> {
    // standard normal values, extreme negatives/positives
    let t = Tensor::<s![3], CpuBackendImpl>::from_slice(&[0.0, -10.0, 10.0], ())?;
    let r = to_vec(&t.gelu()?.into_dyn());
    assert_eq!(r[0], 0.0);
    assert!((r[1] - 0.0).abs() < 1e-4); // gelu(-10) is practically 0
    assert!((r[2] - 10.0).abs() < 1e-4); // gelu(10) is practically 10
    Ok(())
}

#[test]
/// Test unary swish.
fn test_unary_swish() -> Result<()> {
    // beta=1 definitions
    let t = Tensor::<s![2], CpuBackendImpl>::from_slice(&[0.0, 1.0], ())?;
    let r = to_vec(&t.swish()?.into_dyn());
    assert_eq!(r[0], 0.0);
    assert!((r[1] - (1.0 / (1.0 + (-1.0f32).exp()))).abs() < 1e-4);
    Ok(())
}

#[test]
/// Test unary softmax.
fn test_unary_softmax() -> Result<()> {
    // dim 0, dim 1, very large/small values
    // Softmax along dim 1
    let t_2d = Tensor::<s![2, 3], CpuBackendImpl>::from_slice(
        &[
            1000.0, 1000.0, 1000.0, // Should be 0.333, 0.333, 0.333
            -1000.0, 0.0, 1000.0, // Should be 0, 0, 1
        ],
        (),
    )?;
    let r_dim1 = to_vec(&t_2d.softmax(1)?.into_dyn());
    assert!((r_dim1[0] - 0.3333).abs() < 1e-3);
    assert!((r_dim1[4] - 0.0).abs() < 1e-4);
    assert!((r_dim1[5] - 1.0).abs() < 1e-4);

    // Softmax along dim 0
    let r_dim0 = to_vec(&t_2d.softmax(0)?.into_dyn());
    assert!((r_dim0[0] - 1.0).abs() < 1e-4);
    assert!((r_dim0[3] - 0.0).abs() < 1e-4);
    Ok(())
}

#[test]
/// Test unary misc.
fn test_unary_misc() -> Result<()> {
    // neg
    let t_neg = Tensor::<s![3], CpuBackendImpl>::from_slice(&[0.0, 1.0, -1.0], ())?;
    assert_eq!(to_vec(&t_neg.neg()?.into_dyn()), vec![0.0, -1.0, 1.0]);

    // sqrt (NaN on negative)
    let t_sqrt = Tensor::<s![3], CpuBackendImpl>::from_slice(&[4.0, 0.0, -1.0], ())?;
    let r_sqrt = to_vec(&t_sqrt.sqrt()?.into_dyn());
    assert_eq!(r_sqrt[0], 2.0);
    assert_eq!(r_sqrt[1], 0.0);
    assert!(r_sqrt[2].is_nan());

    // exp (large positive, zero, negative)
    let t_exp = Tensor::<s![3], CpuBackendImpl>::from_slice(&[100.0, 0.0, -100.0], ())?;
    let r_exp = to_vec(&t_exp.exp()?.into_dyn());
    assert!(r_exp[0] > 1e10);
    assert_eq!(r_exp[1], 1.0);
    assert!((r_exp[2] - 0.0).abs() < 1e-7);

    // log (positive, zero -> -Inf, negative -> NaN)
    let t_log = Tensor::<s![3], CpuBackendImpl>::from_slice(&[1.0, 0.0, -1.0], ())?;
    let r_log = to_vec(&t_log.log()?.into_dyn());
    assert_eq!(r_log[0], 0.0);
    assert!(r_log[1].is_infinite() && r_log[1] < 0.0);
    assert!(r_log[2].is_nan());

    // tanh
    let t_tanh = Tensor::<s![3], CpuBackendImpl>::from_slice(&[100.0, -100.0, 0.0], ())?;
    let r_tanh = to_vec(&t_tanh.tanh()?.into_dyn());
    assert!((r_tanh[0] - 1.0).abs() < 1e-4);
    assert!((r_tanh[1] - (-1.0)).abs() < 1e-4);
    assert_eq!(r_tanh[2], 0.0);

    // sigmoid
    let t_sig = Tensor::<s![3], CpuBackendImpl>::from_slice(&[100.0, -100.0, 0.0], ())?;
    let r_sig = to_vec(&t_sig.sigmoid()?.into_dyn());
    assert!((r_sig[0] - 1.0).abs() < 1e-4);
    assert!((r_sig[1] - 0.0).abs() < 1e-4);
    assert_eq!(r_sig[2], 0.5);

    Ok(())
}

// -----------------------------------------------------------------------------
// 1.2 Binary Operations
// -----------------------------------------------------------------------------
#[test]
/// Test binary add.
fn test_binary_add() -> Result<()> {
    // positive + positive, negative + negative, zeroes, very large (overflow potential but f32 handles it)
    let a = Tensor::<s![4], CpuBackendImpl>::from_slice(&[1.0, -1.0, 0.0, 3e38], ())?;
    let b = Tensor::<s![4], CpuBackendImpl>::from_slice(&[2.0, -2.0, 0.0, 3e38], ())?;
    let res = to_vec(&a.add(&b)?.into_dyn());
    assert_eq!(res[0], 3.0);
    assert_eq!(res[1], -3.0);
    assert_eq!(res[2], 0.0);
    assert!(res[3].is_infinite()); // f32 overflow
    Ok(())
}

#[test]
/// Test binary sub.
fn test_binary_sub() -> Result<()> {
    // lhs > rhs, lhs < rhs, identical tensors
    let a = Tensor::<s![3], CpuBackendImpl>::from_slice(&[5.0, 1.0, 3.0], ())?;
    let b = Tensor::<s![3], CpuBackendImpl>::from_slice(&[2.0, 4.0, 3.0], ())?;
    let res = to_vec(&a.sub(&b)?.into_dyn());
    assert_eq!(res, vec![3.0, -3.0, 0.0]);
    Ok(())
}

#[test]
/// Test binary mul.
fn test_binary_mul() -> Result<()> {
    // zeroes, element-wise identity, negative terms
    let a = Tensor::<s![3], CpuBackendImpl>::from_slice(&[0.0, 1.0, -2.0], ())?;
    let b = Tensor::<s![3], CpuBackendImpl>::from_slice(&[5.0, 1.0, 3.0], ())?;
    let res = to_vec(&a.mul(&b)?.into_dyn());
    assert_eq!(res, vec![0.0, 1.0, -6.0]);
    Ok(())
}

#[test]
/// Test binary div.
fn test_binary_div() -> Result<()> {
    // standard division, division by zero, precision limits
    let a = Tensor::<s![3], CpuBackendImpl>::from_slice(&[6.0, 1.0, 1.0], ())?;
    let b = Tensor::<s![3], CpuBackendImpl>::from_slice(&[2.0, 0.0, 1e20], ())?;
    let res = to_vec(&a.div(&b)?.into_dyn());
    assert_eq!(res[0], 3.0);
    assert!(res[1].is_infinite()); // div by zero
    assert!(res[2].abs() < 1e-19);
    Ok(())
}

// -----------------------------------------------------------------------------
// 1.3 Broadcasting Operations
// -----------------------------------------------------------------------------
#[test]
/// Test broadcast scalar.
fn test_broadcast_scalar() -> Result<()> {
    let t = Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?.into_dyn();
    let s = Tensor::<s![1], CpuBackendImpl>::from_slice(&[10.0], ())?.into_dyn();
    // Add scalar
    let r = t.broadcast_add(&s)?;
    assert_eq!(to_vec(&r.into_dyn()), vec![11.0, 12.0, 13.0, 14.0]);
    Ok(())
}

#[test]
/// Test broadcast 1d to 2d.
fn test_broadcast_1d_to_2d() -> Result<()> {
    let t_2d = Tensor::<s![2, 3], CpuBackendImpl>::from_slice(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], ())?
        .into_dyn();
    let t_1d = Tensor::<s![3], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0], ())?.into_dyn();
    let r = t_2d.broadcast_mul(&t_1d)?;
    assert_eq!(to_vec(&r.into_dyn()), vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
    Ok(())
}

#[test]
/// Test broadcast trailing dims.
fn test_broadcast_trailing_dims() -> Result<()> {
    let t_3d = Tensor::<s![2, 2, 2], CpuBackendImpl>::ones(())?.into_dyn();
    let t_2d = Tensor::<s![2, 2], CpuBackendImpl>::ones(())?.into_dyn();
    let r = t_3d.broadcast_sub(&t_2d)?;
    assert_eq!(to_vec(&r.into_dyn()), vec![0.0; 8]);
    Ok(())
}

// -----------------------------------------------------------------------------
// 1.4 Reduction Operations
// -----------------------------------------------------------------------------
#[test]
/// Test reduction sum.
fn test_reduction_sum() -> Result<()> {
    let t = Tensor::<s![2, 3], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ())?;
    // sum_all
    assert_eq!(to_vec(&t.clone().sum_all()?.into_dyn())[0], 21.0);
    // sum_dim (0)
    let s0 = t.clone().sum_dim::<0>()?;
    assert_eq!(s0.rank(), 1);
    assert_eq!(to_vec(&s0.into_dyn()), vec![5.0, 7.0, 9.0]);
    // sum_keepdim (1)
    let s1 = t.sum_keepdim::<1>()?;
    assert_eq!(s1.rank(), 2);
    let s1_dims: [usize; 2] = s1.dims();
    assert_eq!(s1_dims, [2, 1]);
    assert_eq!(to_vec(&s1.into_dyn()), vec![6.0, 15.0]);
    Ok(())
}

#[test]
/// Test reduction mean.
fn test_reduction_mean() -> Result<()> {
    let t = Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?;
    assert_eq!(to_vec(&t.clone().mean_all()?.into_dyn())[0], 2.5);
    let m0 = t.mean_dim::<0>()?;
    assert_eq!(to_vec(&m0.into_dyn()), vec![2.0, 3.0]);
    Ok(())
}

#[test]
/// Test reduction max min.
fn test_reduction_max_min() -> Result<()> {
    let t = Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&[-1.0, 5.0, 0.0, 3.0], ())?;
    // max
    assert_eq!(to_vec(&t.clone().max_all()?.into_dyn())[0], 5.0);
    assert_eq!(
        to_vec(&t.clone().max_dim::<0>()?.into_dyn()),
        vec![0.0, 5.0]
    );
    assert_eq!(
        to_vec(&t.clone().max_dim::<1>()?.into_dyn()),
        vec![5.0, 3.0]
    );
    // min
    assert_eq!(to_vec(&t.clone().min_all()?.into_dyn())[0], -1.0);
    assert_eq!(
        to_vec(&t.clone().min_dim::<0>()?.into_dyn()),
        vec![-1.0, 3.0]
    );
    Ok(())
}

// -----------------------------------------------------------------------------
// 1.5 Manipulation Operations
// -----------------------------------------------------------------------------
#[test]
/// Test manipulation reshape flatten.
fn test_manipulation_reshape_flatten() -> Result<()> {
    let t = Tensor::<s![2, 3], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ())?;

    // reshape
    let r = t.clone().reshape_idx::<idx![3, 2]>()?;
    let r_dims: [usize; 2] = r.dims();
    assert_eq!(r_dims, [3, 2]);

    // flatten all (using 0 and 1 since it's 2D)
    let f_all = t.clone().flatten::<0, 1>()?;
    let f_all_dims: [usize; 1] = f_all.dims();
    assert_eq!(f_all_dims, [6]);

    // flatten partial
    let t3 = Tensor::<s![2, 2, 2], CpuBackendImpl>::ones(())?;
    let f_part = t3.flatten::<1, 2>()?;
    let f_part_dims: [usize; 2] = f_part.dims();
    assert_eq!(f_part_dims, [2, 4]);

    Ok(())
}

#[test]
/// Test manipulation transpose squeeze.
fn test_manipulation_transpose_squeeze() -> Result<()> {
    let t = Tensor::<s![2, 3], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ())?;

    // transpose
    let tr = t.clone().transpose::<0, 1>()?;
    let tr_dims: [usize; 2] = tr.dims();
    assert_eq!(tr_dims, [3, 2]);
    assert_eq!(to_vec(&tr.into_dyn()), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    // squeeze (must be size 1)
    let t_sq = Tensor::<s![1, 3], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0], ())?;
    let sq = t_sq.try_squeeze(0)?;
    let sq_dims: Vec<usize> = sq.dims();
    assert_eq!(sq_dims, vec![3]);

    Ok(())
}

// -----------------------------------------------------------------------------
// 1.6 Indexing & Slicing
// -----------------------------------------------------------------------------
#[test]
/// Test indexing concat.
fn test_indexing_concat() -> Result<()> {
    let t1 = Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?;
    let t2 = Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&[5.0, 6.0, 7.0, 8.0], ())?;

    // concat dim 0
    let c0 = t1
        .clone()
        .concat::<s![2, 2], kindle::prelude::typenum::U0>(&t2)?;
    let c0_dims: [usize; 2] = c0.dims();
    assert_eq!(c0_dims, [4, 2]);
    assert_eq!(to_vec(&c0.into_dyn()), vec![1., 2., 3., 4., 5., 6., 7., 8.]);

    // concat dim 1
    let c1 = t1.concat::<s![2, 2], kindle::prelude::typenum::U1>(&t2)?;
    let c1_dims: [usize; 2] = c1.dims();
    assert_eq!(c1_dims, [2, 4]);
    assert_eq!(to_vec(&c1.into_dyn()), vec![1., 2., 5., 6., 3., 4., 7., 8.]);

    Ok(())
}

#[test]
/// Test indexing stack.
fn test_indexing_stack() -> Result<()> {
    let t1 = Tensor::<s![2], CpuBackendImpl>::from_slice(&[1.0, 2.0], ())?;
    let t2 = Tensor::<s![2], CpuBackendImpl>::from_slice(&[3.0, 4.0], ())?;

    // stack dim 0
    let s0 = t1.clone().stack::<kindle::prelude::typenum::U0>(&t2)?;
    let s0_dims: [usize; 2] = s0.dims();
    assert_eq!(s0_dims, [2, 2]);
    assert_eq!(to_vec(&s0.into_dyn()), vec![1., 2., 3., 4.]);

    // stack dim 1
    let s1 = t1.clone().stack::<kindle::prelude::typenum::U1>(&t2)?;
    let s1_dims: [usize; 2] = s1.dims();
    assert_eq!(s1_dims, [2, 2]);
    assert_eq!(to_vec(&s1.into_dyn()), vec![1., 3., 2., 4.]);

    // stack > 2 tensors (via dynamic API or future static variadic if available)
    // currently we test `try_concat_slice` which is dynamic
    let c_slice = Tensor::<s![2], CpuBackendImpl>::try_concat_slice(&[&t1, &t2, &t1], 0)?;
    assert_eq!(to_vec(&c_slice.into_dyn()), vec![1., 2., 3., 4., 1., 2.]);

    Ok(())
}

#[test]
/// Test indexing narrow.
fn test_indexing_narrow() -> Result<()> {
    let t = Tensor::<s![3, 3], CpuBackendImpl>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        (),
    )?;

    // Narrow dim 0
    let n0 = t.clone().try_narrow(0, 1, 2)?; // elements from index 1, len 2
    assert_eq!(to_vec(&n0.into_dyn()), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // Narrow dim 1
    let n1 = t.clone().try_narrow(1, 1, 2)?; // elements from index 1, len 2
    assert_eq!(to_vec(&n1.into_dyn()), vec![2.0, 3.0, 5.0, 6.0, 8.0, 9.0]);

    // Out of bounds
    let err = t.try_narrow(0, 2, 5);
    assert!(err.is_err()); // should fail

    Ok(())
}

// -----------------------------------------------------------------------------
// 1.7 Loss Functions
// -----------------------------------------------------------------------------
#[test]
/// Test loss mse.
fn test_loss_mse() -> Result<()> {
    let pred = Tensor::<s![2], CpuBackendImpl>::from_slice(&[1.0, 2.0], ())?;
    let target1 = Tensor::<s![2], CpuBackendImpl>::from_slice(&[1.0, 2.0], ())?; // identical
    let target2 = Tensor::<s![2], CpuBackendImpl>::from_slice(&[-1.0, -2.0], ())?; // different
    let target3 = Tensor::<s![2], CpuBackendImpl>::from_slice(&[1.1, 1.9], ())?; // small deltas

    assert_eq!(to_vec(&pred.mse_loss(&target1)?.into_dyn())[0], 0.0);
    assert_eq!(to_vec(&pred.mse_loss(&target2)?.into_dyn())[0], 10.0); // ((2)^2 + (4)^2)/2 = (4+16)/2 = 10

    let loss3 = to_vec(&pred.mse_loss(&target3)?.into_dyn())[0];
    assert!((loss3 - 0.01).abs() < 1e-4); // (0.01 + 0.01)/2 = 0.01

    Ok(())
}

#[test]
/// Test loss cross entropy.
fn test_loss_cross_entropy() -> Result<()> {
    // 2 samples, 3 classes
    let logits = Tensor::<s![2, 3], CpuBackendImpl>::from_slice(
        &[
            10.0, 0.0, 0.0, // confident class 0
            0.0, 10.0, 0.0, // confident class 1
        ],
        (),
    )?;

    // target integers: class 0, class 1
    // The framework uses one-hot float targets or indices?
    // Standard float targets for one-hot cross entropy
    let targets = Tensor::<s![2], CpuBackendImpl>::from_slice(&[0.0, 1.0], ())?;

    let loss = logits.cross_entropy_loss(&targets)?;
    let val = to_vec(&loss.into_dyn())[0];
    // With such high confidence, cross entropy should be ~0
    assert!(val < 1e-3);

    // Uniform distribution
    let uniform =
        Tensor::<s![2, 3], CpuBackendImpl>::from_slice(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], ())?;
    let loss_u = uniform.cross_entropy_loss(&targets)?;
    let val_u = to_vec(&loss_u.into_dyn())[0];
    // -log(1/3) = 1.0986
    assert!((val_u - 1.0986).abs() < 1e-3);

    Ok(())
}
