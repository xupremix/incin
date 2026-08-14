#![cfg(feature = "cpu")]

use incin::backend_authoring::*;
use incin::prelude::*;
use incin_core::__backend_compat::legacy::;

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
fn test_inplace_operations() -> Result<()> {
    let mut t1 = Tensor::<s![2, 2], DefaultBackend>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?;
    let t2 = Tensor::<s![2, 2], DefaultBackend>::from_slice(&[10.0, 20.0, 30.0, 40.0], ())?;

    t1.add_(&t2)?;
    assert_eq!(t1.to_vec1::<f32>()?, vec![11.0, 22.0, 33.0, 44.0]);

    t1.sub_(&t2)?;
    assert_eq!(t1.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

    t1.mul_(&t2)?;
    assert_eq!(t1.to_vec1::<f32>()?, vec![10.0, 40.0, 90.0, 160.0]);

    t1.div_(&t2)?;
    assert_eq!(t1.to_vec1::<f32>()?, vec![1.0, 2.0, 3.0, 4.0]);

    t1.zero_()?;
    assert_eq!(t1.to_vec1::<f32>()?, vec![0.0, 0.0, 0.0, 0.0]);

    t1.fill_(7.0)?;
    assert_eq!(t1.to_vec1::<f32>()?, vec![7.0, 7.0, 7.0, 7.0]);

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

    let grad_true_storage = DefaultBackend::get_grad::<f32>(true_val.inner(), grads.as_backend())?
        .expect("grad_true should exist");
    let grad_true_vec = DefaultBackend::float_to_vec1::<f32>(&grad_true_storage)?;
    assert_eq!(grad_true_vec, vec![1.0, 0.0, 1.0]);

    let grad_false_storage =
        DefaultBackend::get_grad::<f32>(false_val.inner(), grads.as_backend())?
            .expect("grad_false should exist");
    let grad_false_vec = DefaultBackend::float_to_vec1::<f32>(&grad_false_storage)?;
    assert_eq!(grad_false_vec, vec![0.0, 1.0, 0.0]);

    Ok(())
}
