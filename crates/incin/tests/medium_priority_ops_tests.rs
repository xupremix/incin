//! Integration coverage for `test_comparisons_and_logical` on the documented public surface.
#![cfg(feature = "cpu")]

use incin::prelude::*;

#[test]
fn test_comparisons_and_logical() -> Result<()> {
    let a = Tensor::<s![3], DefaultBackend>::from_slice(&[1.0, 2.0, 3.0], ())?;
    let b = Tensor::<s![3], DefaultBackend>::from_slice(&[1.0, 5.0, 2.0], ())?;

    let eq_t = a.eq(&b)?;
    assert_eq!(eq_t.to_vec1::<bool>()?, vec![true, false, false]);

    let lt_t = a.lt(&b)?;
    assert_eq!(lt_t.to_vec1::<bool>()?, vec![false, true, false]);

    let gt_t = a.gt(&b)?;
    assert_eq!(gt_t.to_vec1::<bool>()?, vec![false, false, true]);

    let not_lt = lt_t.logical_not()?;
    assert_eq!(not_lt.to_vec1::<bool>()?, vec![true, false, true]);

    Ok(())
}

#[test]
fn test_asymmetric_scalars_and_extremes() -> Result<()> {
    let a = Tensor::<s![3], DefaultBackend>::from_slice(&[10.0, 20.0, 30.0], ())?;

    let sub_s = a.sub_scalar(5.0)?;
    assert_eq!(sub_s.to_vec1::<f32>()?, vec![5.0, 15.0, 25.0]);

    let div_s = a.div_scalar(2.0)?;
    assert_eq!(div_s.to_vec1::<f32>()?, vec![5.0, 10.0, 15.0]);

    let b = Tensor::<s![3], DefaultBackend>::from_slice(&[15.0, 15.0, 15.0], ())?;
    let max_t = a.maximum(&b)?;
    assert_eq!(max_t.to_vec1::<f32>()?, vec![15.0, 20.0, 30.0]);

    let min_t = a.minimum(&b)?;
    assert_eq!(min_t.to_vec1::<f32>()?, vec![10.0, 15.0, 15.0]);

    let lerp_t = a.lerp(&b, 0.5)?;
    assert_eq!(lerp_t.to_vec1::<f32>()?, vec![12.5, 17.5, 22.5]);

    Ok(())
}

#[test]
fn test_fused_matmul_and_sdpa() -> Result<()> {
    let mat = Tensor::<s![2, 2], DefaultBackend>::from_slice(&[1.0, 1.0, 1.0, 1.0], ())?;
    let mat1 = Tensor::<s![2, 2], DefaultBackend>::from_slice(&[1.0, 2.0, 3.0, 4.0], ())?;
    let mat2 = Tensor::<s![2, 2], DefaultBackend>::from_slice(&[5.0, 6.0, 7.0, 8.0], ())?;

    let fused = mat.addmm(&mat1, &mat2, 1.0, 1.0)?;
    assert_eq!(fused.dims().to_vec(), vec![2, 2]);

    let q = Tensor::<s![1, 2, 4], DefaultBackend>::ones(())?;
    let k = Tensor::<s![1, 2, 4], DefaultBackend>::ones(())?;
    let v = Tensor::<s![1, 2, 4], DefaultBackend>::ones(())?;
    let attn = Tensor::scaled_dot_product_attention(
        &q,
        &k,
        &v,
        None::<&Tensor<Dyn, DefaultBackend>>,
        None,
    )?;
    assert_eq!(attn.dims().to_vec(), vec![1, 2, 4]);

    Ok(())
}

#[test]
fn test_spatial_and_norm_ops() -> Result<()> {
    let t = Tensor::<s![6], DefaultBackend>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], ())?;
    let win = t.unfold(0, 3, 1)?;
    assert_eq!(win.dims().to_vec(), vec![4, 3]);

    let img = Tensor::<s![1, 4, 2, 2], DefaultBackend>::ones(())?;
    let p_shuf = img.pixel_shuffle(2)?;
    assert_eq!(p_shuf.dims().to_vec(), vec![1, 1, 4, 4]);

    let g_norm = img.group_norm(2, 1e-5)?;
    assert_eq!(g_norm.dims().to_vec(), vec![1, 4, 2, 2]);

    Ok(())
}
