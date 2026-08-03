use incin::{
    BackendError, ConversionFailure, DTypeId, Error, ErrorMessage, FloatToIntPolicy,
    convert_f64_to_i64,
};

#[test]
fn stable_facade_exposes_typed_bounded_failure_contract() {
    let conversion = convert_f64_to_i64(
        "consumer_conversion",
        DTypeId::F64,
        f64::NAN,
        FloatToIntPolicy::Exact,
    )
    .unwrap_err();
    assert!(matches!(
        conversion,
        Error::InvalidConversion {
            operation: "consumer_conversion",
            reason: ConversionFailure::NonFinite,
            ..
        }
    ));

    let bounded = ErrorMessage::new("x".repeat(4096));
    assert!(bounded.as_str().len() < 4096);

    fn accepts_backend_error(_: Option<BackendError>) {}
    accepts_backend_error(None);
}

#[cfg(feature = "cpu")]
#[test]
fn tensor_operators_propagate_results_instead_of_panicking() -> incin::Result<()> {
    use incin::prelude::{DefaultBackend, Tensor, s};

    let lhs = Tensor::<s![2], DefaultBackend>::ones(())?;
    let rhs = Tensor::<s![2], DefaultBackend>::ones(())?;
    let sum: incin::Result<Tensor<s![2], DefaultBackend>> = lhs + rhs;
    assert_eq!(sum?.to_vec1::<f32>()?, vec![2.0, 2.0]);
    Ok(())
}
