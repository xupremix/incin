#![cfg(feature = "cpu")]

use incin::prelude::*;

#[test]
fn one_d_float_tensor_prints_like_pytorch() -> Result<()> {
    let t = Tensor::<s![3], DefaultBackend>::from_slice(&[1.0, 2.0, 3.0], ())?;
    assert_eq!(t.to_string(), "tensor([1., 2., 3.])");
    Ok(())
}

#[test]
fn two_d_float_tensor_aligns_negative_columns() -> Result<()> {
    let t = Tensor::<s![2, 3], DefaultBackend>::from_slice(
        &[0.3171, -0.9524, 0.1331, -0.6189, 0.4829, -0.2168],
        (),
    )?;
    assert_eq!(
        t.to_string(),
        "tensor([[ 0.3171, -0.9524,  0.1331],\n        [-0.6189,  0.4829, -0.2168]])"
    );
    Ok(())
}

#[test]
fn no_grad_tensor_omits_the_requires_grad_footer() -> Result<()> {
    let t = Tensor::<s![2], DefaultBackend, f32, NoGrad>::from_slice(&[1.0, 2.0], ())?;
    assert_eq!(t.to_string(), "tensor([1., 2.])");
    Ok(())
}

#[test]
fn non_default_dtype_is_named_in_the_footer() -> Result<()> {
    let t = Tensor::<s![3], DefaultBackend, i64>::from_slice(&[1, 2, 3], ())?;
    assert_eq!(t.to_string(), "tensor([1, 2, 3], dtype=i64)");
    Ok(())
}

#[test]
fn debug_still_carries_shape_and_placement_alongside_real_values() -> Result<()> {
    let t = Tensor::<s![2], DefaultBackend, f32, NoGrad>::from_slice(&[1.0, 2.0], ())?;
    let rendered = format!("{t:?}");
    assert!(rendered.contains("global_shape=[2]"), "got {rendered}");
    assert!(rendered.contains("[1., 2.]"), "got {rendered}");
    Ok(())
}
