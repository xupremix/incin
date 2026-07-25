use incin::prelude::*;

#[test]
fn test_elementwise_math_ops() -> Result<()> {
    let t = Tensor::<s![4], DefaultBackend>::from_slice(&[-2.5, 0.0, 1.5, 4.0], ())?;
    
    let abs_t = t.abs()?;
    assert_eq!(abs_t.to_vec1::<f32>()?, vec![2.5, 0.0, 1.5, 4.0]);

    let sign_t = t.sign()?;
    assert_eq!(sign_t.to_vec1::<f32>()?, vec![-1.0, 0.0, 1.0, 1.0]);

    let floor_t = t.floor()?;
    assert_eq!(floor_t.to_vec1::<f32>()?, vec![-3.0, 0.0, 1.0, 4.0]);

    let ceil_t = t.ceil()?;
    assert_eq!(ceil_t.to_vec1::<f32>()?, vec![-2.0, 0.0, 2.0, 4.0]);

    let round_t = t.round()?;
    assert_eq!(round_t.to_vec1::<f32>()?, vec![-3.0, 0.0, 2.0, 4.0]);

    let clamp_t = t.clamp(-1.0, 2.0)?;
    assert_eq!(clamp_t.to_vec1::<f32>()?, vec![-1.0, 0.0, 1.5, 2.0]);

    let pos = Tensor::<s![3], DefaultBackend>::from_slice(&[1.0, 4.0, 100.0], ())?;
    let pow_t = pos.powf(2.0)?;
    assert_eq!(pow_t.to_vec1::<f32>()?, vec![1.0, 16.0, 10000.0]);

    let log2_t = pos.log2()?;
    assert_eq!(log2_t.to_vec1::<f32>()?, vec![0.0, 2.0, 6.643856]);

    let log10_t = pos.log10()?;
    let vec_log10 = log10_t.to_vec1::<f32>()?;
    assert_eq!(vec_log10[0], 0.0);
    assert!((vec_log10[1] - 0.60205999).abs() < 1e-4);
    assert_eq!(vec_log10[2], 2.0);

    Ok(())
}

#[test]
fn test_creation_factory_ops() -> Result<()> {
    let full_t = Tensor::<s![2, 3], DefaultBackend>::full(7.0, ())?;
    assert_eq!(full_t.dims().as_ref(), &[2, 3]);
    assert_eq!(full_t.to_vec1::<f32>()?, vec![7.0; 6]);

    let arange_t = Tensor::<s![5], DefaultBackend>::arange(0.0, 2.0, ())?;
    assert_eq!(arange_t.to_vec1::<f32>()?, vec![0.0, 2.0, 4.0, 6.0, 8.0]);

    let linspace_t = Tensor::<s![5], DefaultBackend>::linspace(0.0, 1.0, ())?;
    assert_eq!(linspace_t.to_vec1::<f32>()?, vec![0.0, 0.25, 0.5, 0.75, 1.0]);

    Ok(())
}

#[test]
fn test_selection_and_indexing() -> Result<()> {
    let mask = Tensor::<s![4], DefaultBackend>::from_slice(&[1.0, 0.0, 1.0, 0.0], ())?;
    let on_true = Tensor::<s![4], DefaultBackend>::from_slice(&[10.0, 20.0, 30.0, 40.0], ())?;
    let on_false = Tensor::<s![4], DefaultBackend>::from_slice(&[-1.0, -2.0, -3.0, -4.0], ())?;

    let selected = mask.where_cond(&on_true, &on_false)?;
    assert_eq!(selected.to_vec1::<f32>()?, vec![10.0, -2.0, 30.0, -4.0]);

    let filled = on_true.masked_fill(&mask, 99.0)?;
    assert_eq!(filled.to_vec1::<f32>()?, vec![99.0, 20.0, 99.0, 40.0]);

    Ok(())
}

#[test]
fn test_matrix_and_reductions() -> Result<()> {
    let v1 = Tensor::<s![3], DefaultBackend>::from_slice(&[1.0, 2.0, 3.0], ())?;
    let v2 = Tensor::<s![3], DefaultBackend>::from_slice(&[4.0, 5.0, 6.0], ())?;

    let d = v1.dot(&v2)?;
    assert_eq!(d.to_scalar::<f32>()?, 32.0);

    let l1_norm = v1.norm(1.0)?;
    assert_eq!(l1_norm.to_scalar::<f32>()?, 6.0);

    let l2_norm = v1.norm(2.0)?;
    assert!((l2_norm.to_scalar::<f32>()? - 3.74165738).abs() < 1e-4);

    let prod = v1.clone().prod_all()?;
    assert_eq!(prod.to_scalar::<f32>()?, 6.0);

    let cum = v1.cumsum::<0>()?;
    assert_eq!(cum.to_vec1::<f32>()?, vec![1.0, 3.0, 6.0]);

    Ok(())
}
