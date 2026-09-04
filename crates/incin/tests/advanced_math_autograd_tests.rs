//! Integration coverage for `test_trigonometric_and_transcendental_ops` on the documented public surface.
#![cfg(feature = "cpu")]

use incin::prelude::*;

#[test]
fn test_trigonometric_and_transcendental_ops() -> Result<()> {
    let t = Tensor::<s![3], DefaultBackend>::from_slice(&[0.1, 0.5, 0.8], ())?;

    let tan_val = t.tan()?;
    assert_eq!(tan_val.dims().to_vec(), vec![3]);

    let asin_val = t.asin()?;
    assert_eq!(asin_val.dims().to_vec(), vec![3]);

    let acos_val = t.acos()?;
    assert_eq!(acos_val.dims().to_vec(), vec![3]);

    let atan_val = t.atan()?;
    assert_eq!(atan_val.dims().to_vec(), vec![3]);

    let sinh_val = t.sinh()?;
    assert_eq!(sinh_val.dims().to_vec(), vec![3]);

    let cosh_val = t.cosh()?;
    assert_eq!(cosh_val.dims().to_vec(), vec![3]);

    let erf_val = t.erf()?;
    assert_eq!(erf_val.dims().to_vec(), vec![3]);

    let rsqrt_val = t.rsqrt()?;
    assert_eq!(rsqrt_val.dims().to_vec(), vec![3]);

    let trunc_val = t.trunc()?;
    assert_eq!(trunc_val.dims().to_vec(), vec![3]);

    let frac_val = t.frac()?;
    assert_eq!(frac_val.dims().to_vec(), vec![3]);

    Ok(())
}

#[test]
fn scatter_add_routes_gradient_to_every_colliding_source() -> Result<()> {
    // Slot 0 is written by sources 0, 1 and 2; slot 2 by source 3. This is the
    // collision an overwriting scatter resolves by discarding, and the reason
    // its catalog row cannot state a gradient at all.
    let base = Tensor::<s![4], DefaultBackend, f32, Grad>::zeros(())?;
    let index = Tensor::<s![4], DefaultBackend, u32>::from_slice(&[0, 0, 0, 2], ())?;
    let src = Tensor::<s![4], DefaultBackend, f32, Grad>::from_slice(&[1.0, 2.0, 4.0, 8.0], ())?;

    let out = base.scatter_add(0, &index, &src)?;
    assert_eq!(out.to_vec1::<f32>()?, vec![7.0, 0.0, 8.0, 0.0]);

    let grads = out.sum_all()?.backward()?;

    // Every source reached the output exactly once and the objective weights
    // each output slot by one, so each source's cotangent is one. Under
    // last-write-wins only source 2 would have earned anything, and which one
    // that is would be a fact about traversal order rather than about values.
    assert_eq!(
        grads.require(&src)?.to_vec1::<f32>()?,
        vec![1.0, 1.0, 1.0, 1.0]
    );

    // Addition passes the target through untouched, so its cotangent is the
    // output's, including at the slots that were written into.
    assert_eq!(
        grads.require(&base)?.to_vec1::<f32>()?,
        vec![1.0, 1.0, 1.0, 1.0]
    );

    Ok(())
}

#[test]
fn test_autograd_tape_closures() -> Result<()> {
    let mask = Tensor::<s![3], DefaultBackend, bool>::from_slice(&[true, false, true], ())?;
    let true_val = Tensor::<s![3], DefaultBackend, f32, Grad>::from_slice(&[10.0, 20.0, 30.0], ())?;
    let false_val = Tensor::<s![3], DefaultBackend, f32, Grad>::from_slice(&[1.0, 2.0, 3.0], ())?;

    let out = Tensor::where_cond(&mask, &true_val, &false_val)?;
    assert_eq!(out.to_vec1::<f32>()?, vec![10.0, 2.0, 30.0]);

    let sum = out.sum_all()?;
    let grads = sum.backward()?;

    let grad_true_vec = grads.require(&true_val)?.to_vec1::<f32>()?;
    assert_eq!(grad_true_vec, vec![1.0, 0.0, 1.0]);

    let grad_false_vec = grads.require(&false_val)?.to_vec1::<f32>()?;
    assert_eq!(grad_false_vec, vec![0.0, 1.0, 0.0]);

    Ok(())
}
