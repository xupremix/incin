//! Zero-sized operand semantics: the catalog's `EmptyRule` decides which
//! operations accept an empty axis, and the CPU execution path must agree
//! with that decision for every family a 0.1 user can reach.
//!
//! * elementwise/broadcast families declare `EmptyRule::Allowed`, so a
//!   zero-sized axis flows through and the result is empty in turn;
//! * `sum`/`prod` declare `IdentityOrDefined`: reducing nothing yields the
//!   identity element (0 and 1 respectively) rather than an error;
//! * every other reduction (`mean`, `max`, `min`) declares
//!   `RejectedWhenReductionIsEmpty`, because averaging or extremizing
//!   nothing has no defined answer - it must fail explicitly, never fabricate;
//! * matmul contracting over an empty `K` defines the product as zeros.
#![cfg(feature = "cpu")]

use incin::prelude::*;

/// B.
type B = incin_backends::cpu::CpuBackendImpl;

#[test]
fn elementwise_over_a_zero_sized_axis_yields_an_empty_result() -> Result<()> {
    let a: Tensor<s![2, 0], B> = Tensor::zeros(())?;
    let b: Tensor<s![2, 0], B> = Tensor::zeros(())?;
    let out = a.add_exact(&b)?;
    assert_eq!(out.dims().to_vec(), vec![2, 0]);
    Ok(())
}

#[test]
fn broadcast_over_a_zero_sized_axis_yields_an_empty_result() -> Result<()> {
    let a: Tensor<s![1], B> = Tensor::ones(())?;
    let b: Tensor<s![3, 0], B> = Tensor::zeros(())?;
    let out = a.broadcast_add(&b)?;
    assert_eq!(out.dims().to_vec(), vec![3, 0]);
    Ok(())
}

#[test]
fn sum_and_prod_over_nothing_yield_their_identity_elements() -> Result<()> {
    let t: Tensor<s![2, 0], B> = Tensor::zeros(())?;
    let summed = t.sum_all()?;
    assert!((summed.to_scalar::<f32>()? - 0.0).abs() < f32::EPSILON);

    let u: Tensor<s![0], B> = Tensor::zeros(())?;
    let product = u.prod_all()?;
    assert!((product.to_scalar::<f32>()? - 1.0).abs() < f32::EPSILON);
    Ok(())
}

#[test]
fn mean_max_min_over_nothing_are_rejected_explicitly() -> Result<()> {
    let t: Tensor<s![2, 0], B> = Tensor::zeros(())?;
    let attempts = [
        ("mean_all", t.clone().mean_all().err()),
        ("max_all", t.clone().max_all().err()),
        ("min_all", t.min_all().err()),
    ];
    for (name, err) in attempts {
        let err = err.unwrap_or_else(|| panic!("{name} over [] must fail"));
        assert!(
            !matches!(err, Error::Msg(_)),
            "{name} over an empty tensor must fail with a typed error, got: {err:?}"
        );
    }
    Ok(())
}

#[test]
fn matmul_contracting_over_an_empty_axis_produces_zeros() -> Result<()> {
    let lhs: Tensor<s![2, 0], B> = Tensor::ones(())?;
    let rhs: Tensor<s![0, 3], B> = Tensor::ones(())?;
    let out = lhs.matmul(&rhs)?;
    assert_eq!(out.dims().to_vec(), vec![2, 3]);
    assert_eq!(out.to_vec1::<f32>()?, vec![0.0; 6]);
    Ok(())
}

#[test]
fn autograd_over_a_zero_sized_operand_stays_consistent() -> Result<()> {
    // An empty forward contributes its identity, records tape entries like
    // any other shape, and must not corrupt later passes: an ordinary
    // backward afterwards still reaches its inputs with exact values.
    let empty: Tensor<s![2, 0], B, f32, Grad> = Tensor::zeros(())?;
    let doubled = empty.add_exact(&empty)?;
    let total = doubled.sum_all()?;
    assert!((total.to_scalar::<f32>()? - 0.0).abs() < f32::EPSILON);
    let grads = total.backward()?;
    assert_eq!(grads.require(&empty)?.to_vec1::<f32>()?, Vec::<f32>::new());

    let live = Tensor::<s![3], B, f32, Grad>::from_slice(&[1.0, 2.0, 3.0], ())?;
    let loss = live.mul_exact(&live)?.sum_all()?;
    let grads = loss.backward()?;
    assert_eq!(grads.require(&live)?.to_vec1::<f32>()?, vec![2.0, 4.0, 6.0]);
    Ok(())
}
