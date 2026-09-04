//! Integration coverage for `to_vec` on the documented public surface.
#![cfg(feature = "cpu")]
// These explicit types are the assertions: they verify that selector-based
// operations retain known dimensions instead of erasing them to `Dyn`.
#![allow(clippy::type_complexity)]
use incin::prelude::*;

/// Implementation of `CpuBackendImpl` for the respective backend.
type CpuBackendImpl = incin_backends::cpu::CpuBackendImpl;

/// Generic over the layout: `into_dyn` erases the *shape* proof and keeps the
/// layout one, so callers reach here holding either.
fn to_vec<L: Layout>(t: &Tensor<Dyn, CpuBackendImpl, f32, NoGrad, Local, L>) -> Vec<f32> {
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

/// `log_softmax` is not `softmax` then `log`, and this is the input that shows
/// it.
///
/// The second row spans two thousand in a single axis. Softmax normalises
/// against the row maximum, so the two entries below it exponentiate to values
/// no `f32` can hold apart from zero, and taking the logarithm of that zero
/// yields negative infinity: the composition destroys exactly the two numbers a
/// router would compare. Subtracting the log of the summed exponentials instead
/// never exponentiates the large negative difference at all, so it comes back as
/// itself.
///
/// The assertions are therefore split. The first row, whose entries are equal,
/// pins the value: three equal logits share the mass, so each log-probability is
/// `ln(1/3)`, and that holds for either implementation. The second row pins the
/// difference between the two, and it is stated as `is_finite` plus a wide
/// tolerance rather than an exact comparison, because the claim under test is
/// "a usable number survives here", not "this bit pattern survives here".
#[test]
fn log_softmax_survives_the_span_that_breaks_softmax_then_log() -> Result<()> {
    let logits = Tensor::<s![2, 3], CpuBackendImpl>::from_slice(
        &[
            1000.0, 1000.0, 1000.0, // equal, so each is ln(1/3)
            -1000.0, 0.0, 1000.0, // spans 2000, so the composition collapses
        ],
        (),
    )?;

    let direct = to_vec(&logits.log_softmax(1)?.into_dyn());
    let composed = to_vec(&logits.softmax(1)?.log()?.into_dyn());

    let ln_third = (1.0f32 / 3.0).ln();
    for index in 0..3 {
        assert!((direct[index] - ln_third).abs() < 1e-3);
        assert!((composed[index] - ln_third).abs() < 1e-3);
    }

    assert!(direct[3].is_finite() && (direct[3] + 2000.0).abs() < 1.0);
    assert!(direct[4].is_finite() && (direct[4] + 1000.0).abs() < 1.0);
    assert!((direct[5] - 0.0).abs() < 1e-4);

    // The composition's verdict on the same two entries, asserted rather than
    // left as a comment, so that a change which makes the composition safe is
    // reported here instead of quietly making this test pointless.
    assert!(composed[3].is_infinite() && composed[4].is_infinite());

    Ok(())
}

/// `logsumexp` holds at magnitudes where the naive spelling has no answer.
///
/// The first row sits at 300, where a single `exp` already overflows f32, and
/// the second at -300, where one underflows to zero. Both are stated against a
/// closed form that is a maximum plus the log of a small integer, so the
/// expected value is arithmetic rather than a recorded output.
///
/// The composed spelling is executed beside it and asserted to be useless on
/// both rows, in opposite directions: infinity where the entries are large and
/// negative infinity where they are small. That the failure has two directions
/// is the reason the shift is by the maximum rather than by a constant.
#[test]
fn logsumexp_holds_where_the_naive_spelling_overflows_and_underflows() -> Result<()> {
    let logits = Tensor::<s![2, 3], CpuBackendImpl>::from_slice(
        &[
            300.0, 300.0, 300.0, // exp overflows: 300 + ln(3)
            -300.0, -300.0, -400.0, // exp underflows: -300 + ln(2)
        ],
        (),
    )?;

    let direct = to_vec(&logits.logsumexp(1)?.into_dyn());
    assert!((direct[0] - (300.0 + 3.0f32.ln())).abs() < 1e-2);
    assert!((direct[1] - (-300.0 + 2.0f32.ln())).abs() < 1e-2);

    let composed = to_vec(&logits.exp()?.sum(1)?.log()?.into_dyn());
    assert!(composed[0].is_infinite() && composed[0].is_sign_positive());
    assert!(composed[1].is_infinite() && composed[1].is_sign_negative());

    // The keepdim spelling reduces the same axis and differs only in whether it
    // survives, which is what makes it usable as a broadcast operand.
    let kept = logits.logsumexp_keepdim(1)?;
    assert_eq!(kept.dims().dims(), &[2, 1]);
    assert_eq!(to_vec(&kept.into_dyn()), direct);

    Ok(())
}

#[test]
fn scatter_add_keeps_every_contribution_where_scatter_keeps_one() -> Result<()> {
    // Four tokens' worth of contributions aimed at three slots. Slot 0 is
    // written three times and slot 2 once, which is the shape a top-k router
    // produces when several tokens pick the same expert.
    let base = Tensor::<s![4], CpuBackendImpl>::zeros(())?;
    let index = Tensor::<s![4], CpuBackendImpl, u32>::from_slice(&[0, 0, 0, 2], ())?;
    let src = Tensor::<s![4], CpuBackendImpl>::from_slice(&[1.0, 2.0, 4.0, 8.0], ())?;

    let summed = to_vec(&base.scatter_add(0, &index, &src)?.into_dyn());
    assert_eq!(summed, vec![7.0, 0.0, 8.0, 0.0]);

    // The overwriting form on the same operands, asserted rather than described,
    // so that a change making `scatter` accumulate is reported here instead of
    // quietly making this test redundant. It keeps the last write, 4.0, and the
    // 1.0 and 2.0 are gone with no error to say so.
    let overwritten = to_vec(&base.scatter(0, &index, &src)?.into_dyn());
    assert_eq!(overwritten, vec![4.0, 0.0, 8.0, 0.0]);

    // Adding onto a non-zero target accumulates rather than replacing, which is
    // what makes the target's own gradient a pass-through.
    let occupied = Tensor::<s![4], CpuBackendImpl>::from_slice(&[10.0, 20.0, 30.0, 40.0], ())?;
    let onto = to_vec(&occupied.scatter_add(0, &index, &src)?.into_dyn());
    assert_eq!(onto, vec![17.0, 20.0, 38.0, 40.0]);

    Ok(())
}

#[test]
fn signed_axis_selectors_cover_runtime_and_axis_macro_paths() -> Result<()> {
    let tensor =
        Tensor::<s![2, 3], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ())?;

    let runtime = tensor.sum(-1isize)?;
    let macro_selected = tensor.sum(axis!(-1))?;
    let compile_time = tensor.sum(axis!(-1))?;
    let mean = tensor.mean(axis!(-1))?;
    let max = tensor.max_keepdim(axis!(0))?;
    let min = tensor.min(0isize)?;
    let argmax = tensor.argmax(axis!(-1))?;

    assert_eq!(runtime.to_vec1::<f32>()?, vec![6.0, 15.0]);
    assert_eq!(macro_selected.to_vec1::<f32>()?, vec![6.0, 15.0]);
    assert_eq!(compile_time.to_vec1::<f32>()?, vec![6.0, 15.0]);
    assert_eq!(mean.to_vec1::<f32>()?, vec![2.0, 5.0]);
    assert_eq!(max.to_vec1::<f32>()?, vec![4.0, 5.0, 6.0]);
    assert_eq!(min.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0]);
    assert_eq!(argmax.to_vec1::<u32>()?, vec![2, 2]);
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
    let res = to_vec(&a.try_add(&b)?.into_dyn());
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
    let res = to_vec(&a.try_sub(&b)?.into_dyn());
    assert_eq!(res, vec![3.0, -3.0, 0.0]);
    Ok(())
}

#[test]
/// Test binary mul.
fn test_binary_mul() -> Result<()> {
    // zeroes, element-wise identity, negative terms
    let a = Tensor::<s![3], CpuBackendImpl>::from_slice(&[0.0, 1.0, -2.0], ())?;
    let b = Tensor::<s![3], CpuBackendImpl>::from_slice(&[5.0, 1.0, 3.0], ())?;
    let res = to_vec(&a.try_mul(&b)?.into_dyn());
    assert_eq!(res, vec![0.0, 1.0, -6.0]);
    Ok(())
}

#[test]
/// Test binary div.
fn test_binary_div() -> Result<()> {
    // standard division, division by zero, precision limits
    let a = Tensor::<s![3], CpuBackendImpl>::from_slice(&[6.0, 1.0, 1.0], ())?;
    let b = Tensor::<s![3], CpuBackendImpl>::from_slice(&[2.0, 0.0, 1e20], ())?;
    let res = to_vec(&a.try_div(&b)?.into_dyn());
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
    let s0 = t.clone().sum(axis!(0))?;
    assert_eq!(s0.rank(), 1);
    assert_eq!(to_vec(&s0.into_dyn()), vec![5.0, 7.0, 9.0]);
    // sum_keepdim (1)
    let s1 = t.sum_keepdim(axis!(1))?;
    assert_eq!(s1.rank(), 2);
    assert_eq!(s1.dims().dims(), &[2, 1]);
    assert_eq!(to_vec(&s1.into_dyn()), vec![6.0, 15.0]);
    Ok(())
}

#[test]
/// Test reduction mean.
fn test_reduction_mean() -> Result<()> {
    let t = Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?;
    assert_eq!(to_vec(&t.clone().mean_all()?.into_dyn())[0], 2.5);
    // Axis reductions use the structural selector API; mean has no separate
    // axis-specific frontend method, so the all-elements path remains the
    // canonical mean coverage here.
    Ok(())
}

#[test]
/// Test reduction max min.
fn test_reduction_max_min() -> Result<()> {
    let t = Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&[-1.0, 5.0, 0.0, 3.0], ())?;
    // max
    assert_eq!(to_vec(&t.clone().max_all()?.into_dyn())[0], 5.0);
    assert_eq!(to_vec(&t.clone().min_all()?.into_dyn())[0], -1.0);
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
    let r = t.clone().reshape(shape![3, 2])?;
    assert_eq!(r.dims().as_ref(), &[3, 2]);
    let inferred: Tensor<s![6, usize], CpuBackendImpl> =
        t.clone().reshape_infer(shape![6, infer])?;
    assert_eq!(inferred.dims().as_ref(), &[6, 1]);
    let indexed = t.get(i![-1, ..])?;
    assert_eq!(indexed.dims().as_ref(), &[3]);
    let tail = t.get(i![.., -2..])?;
    assert_eq!(tail.dims().as_ref(), &[2, 2]);
    assert!(t.get(i![-3, ..]).is_err());

    // flatten all (using 0 and 1 since it's 2D)
    let f_all = t.clone().flatten(axis!(0), axis!(1))?;
    assert_eq!(f_all.dims().as_ref(), &[6]);

    // flatten partial
    let t3 = Tensor::<s![2, 2, 2], CpuBackendImpl>::ones(())?;
    let f_part = t3.flatten(axis!(1), axis!(2))?;
    assert_eq!(f_part.dims().as_ref(), &[2, 4]);
    let f_runtime = t3.flatten(-2isize, -1isize)?;
    assert_eq!(f_runtime.dims().as_ref(), &[2, 4]);

    let t4 = Tensor::<s![2, 3, 4], CpuBackendImpl>::ones(())?;
    let f_negative = t4.flatten(axis!(1), axis!(-1))?;
    assert_eq!(f_negative.dims().as_ref(), &[2, 12]);

    Ok(())
}

#[test]
/// Test manipulation transpose squeeze.
#[allow(clippy::type_complexity)]
fn test_manipulation_transpose_squeeze() -> Result<()> {
    let t = Tensor::<s![2, 3], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ())?;

    // transpose
    let tr_static: Tensor<s![3, 2], CpuBackendImpl> = t.clone().transpose(axis!(0), axis!(1))?;
    assert_eq!(tr_static.dims().dims(), &[3, 2]);
    let tr = t.clone().transpose(0isize, 1isize)?;
    assert_eq!(tr.dims().dims(), &[3, 2]);
    assert_eq!(to_vec(&tr.into_dyn()), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);

    let t3 = Tensor::<s![2, 3, 4], CpuBackendImpl>::ones(())?;
    let neg_axis: incin::advanced::ReverseAxis<incin::advanced::Here> = axis!(-1);
    let tr_negative: Tensor<s![4, 3, 2], CpuBackendImpl> = t3.transpose(axis!(0), neg_axis)?;
    assert_eq!(tr_negative.dims().as_ref(), &[4, 3, 2]);

    // squeeze (must be size 1)
    let t_sq = Tensor::<s![1, 3], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0], ())?;
    let sq = t_sq.try_squeeze(0isize)?;
    let sq_dims: Vec<usize> = sq.dims().as_ref().to_vec();
    assert_eq!(sq_dims, vec![3]);

    Ok(())
}

#[test]
/// Static selectors support arbitrary rank and negative positions without a
/// generated rank table, while `i![]` accepts an arbitrary number of entries.
fn test_arbitrary_rank_axis_and_index_selectors() -> Result<()> {
    type Rank18 = s![1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 3];
    let tensor = Tensor::<Rank18, CpuBackendImpl>::ones(())?;
    let transposed = tensor.transpose(axis!(16), axis!(-1))?;
    assert_eq!(
        transposed.dims().as_ref(),
        &[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 2]
    );

    let indexed = transposed.get(i![
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..,
        ..
    ])?;
    assert_eq!(indexed.rank(), 18);
    Ok(())
}

// -----------------------------------------------------------------------------
// 1.6 Indexing & Slicing
// -----------------------------------------------------------------------------
#[test]
/// Test indexing concat.
#[allow(clippy::type_complexity)]
fn test_indexing_concat() -> Result<()> {
    let t1 = Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?;
    let t2 = Tensor::<s![2, 2], CpuBackendImpl>::from_slice(&[5.0, 6.0, 7.0, 8.0], ())?;

    // concat dim 0
    let c0: Tensor<s![4, 2], CpuBackendImpl> = t1.clone().concat(&t2, axis!(0))?;
    assert_eq!(c0.dims().dims(), &[4, 2]);
    assert_eq!(to_vec(&c0.into_dyn()), vec![1., 2., 3., 4., 5., 6., 7., 8.]);

    // concat dim 1
    let c1: Tensor<s![2, 4], CpuBackendImpl> = t1.concat(&t2, axis!(1))?;
    assert_eq!(c1.dims().dims(), &[2, 4]);
    assert_eq!(to_vec(&c1.into_dyn()), vec![1., 2., 5., 6., 3., 4., 7., 8.]);

    let c_runtime: Tensor<Ranked<incin::typenum::U2>, CpuBackendImpl> = t1.concat(&t2, -1isize)?;
    assert_eq!(c_runtime.dims().as_ref(), &[2, 4]);

    let c_negative: Tensor<s![2, 4], CpuBackendImpl> = t1.concat(&t2, axis!(-1))?;
    assert_eq!(c_negative.dims().as_ref(), &[2, 4]);

    Ok(())
}

#[test]
/// Test indexing stack.
fn test_indexing_stack() -> Result<()> {
    let t1 = Tensor::<s![2], CpuBackendImpl>::from_slice(&[1.0, 2.0], ())?;
    let t2 = Tensor::<s![2], CpuBackendImpl>::from_slice(&[3.0, 4.0], ())?;

    // stack dim 0
    let s0 = t1.clone().stack(&t2, axis!(0))?;
    assert_eq!(s0.dims().dims(), &[2, 2]);
    assert_eq!(to_vec(&s0.into_dyn()), vec![1., 2., 3., 4.]);

    // stack dim 1
    let s1 = t1.clone().stack(&t2, axis!(1))?;
    assert_eq!(s1.dims().dims(), &[2, 2]);
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
    let n0 = t.clone().try_narrow(0isize, 1, 2)?; // elements from index 1, len 2
    assert_eq!(to_vec(&n0.into_dyn()), vec![4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

    // Narrow dim 1
    let n1 = t.clone().try_narrow(1isize, 1, 2)?; // elements from index 1, len 2
    assert_eq!(to_vec(&n1.into_dyn()), vec![2.0, 3.0, 5.0, 6.0, 8.0, 9.0]);

    // Out of bounds
    let err = t.clone().try_narrow(0isize, 2, 5);
    assert!(err.is_err()); // should fail

    let invalid_axis = t.clone().try_narrow(2isize, 0, 1);
    assert!(invalid_axis.is_err());

    let invalid_squeeze = t.clone().try_squeeze(0isize);
    assert!(invalid_squeeze.is_err());

    let invalid_topk = t.topk(4, 1isize, true);
    assert!(invalid_topk.is_err());

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
    let targets = Tensor::<s![2], CpuBackendImpl, i64>::from_slice(&[0, 1], ())?;

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

// -----------------------------------------------------------------------------
// to_scalar::<bool>() / to_vec1::<bool>() require DTypeId::Bool
// -----------------------------------------------------------------------------

#[test]
fn to_scalar_bool_rejects_numeric_u8_tensor() -> Result<()> {
    let t = Tensor::<s![1], CpuBackendImpl, u8>::from_bytes(&[5u8], ())?;
    assert!(t.to_scalar::<bool>().is_err());
    Ok(())
}

#[test]
fn to_vec1_bool_rejects_numeric_u8_tensor() -> Result<()> {
    let t = Tensor::<s![4], CpuBackendImpl, u8>::from_bytes(&[0u8, 1u8, 5u8, 255u8], ())?;
    assert!(t.to_vec1::<bool>().is_err());
    Ok(())
}

#[test]
fn dimension_selecting_operations_accept_axis_selectors() -> Result<()> {
    let tensor = Tensor::<s![2, 3], CpuBackendImpl>::ones(())?;
    let indices = Tensor::<s![2, 3], CpuBackendImpl, u32>::zeros(())?;
    let select_indices = Tensor::<s![1], CpuBackendImpl, u32>::zeros(())?;

    let _ = tensor.gather(axis!(1), &indices)?;
    let selected: Tensor<s![2, usize], CpuBackendImpl> =
        tensor.index_select(axis!(1), &select_indices)?;
    let narrowed: Tensor<s![2, usize], CpuBackendImpl> =
        tensor.clone().try_narrow(axis!(1), 0, 1)?;
    let _ = tensor.scatter(axis!(-1), &indices, &tensor)?;
    let chunks: Vec<Tensor<s![2, usize], CpuBackendImpl>> = tensor.chunk(2, axis!(1))?;
    let splits: Vec<Tensor<s![2, usize], CpuBackendImpl>> = tensor.split(2, axis!(1))?;
    let _: Tensor<Ranked<typenum::U2>, CpuBackendImpl> =
        tensor.clone().try_narrow(-1isize, 0, 1)?;
    assert_eq!(selected.dims().as_ref(), &[2, 1]);
    assert_eq!(narrowed.dims().as_ref(), &[2, 1]);
    assert_eq!(chunks.len(), 2);
    assert_eq!(splits.len(), 2);
    // `Dense`, not `Tensor`: topk writes two fresh buffers and states so.
    let _: (
        Dense<s![2, usize], CpuBackendImpl, f32, NoGrad>,
        Dense<s![2, usize], CpuBackendImpl, u32, NoGrad>,
    ) = tensor.topk(1, axis!(1), true)?;
    Ok(())
}
