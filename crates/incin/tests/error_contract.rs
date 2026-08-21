use incin::{
    BackendError, ConversionFailure, DTypeId, Error, ErrorMessage, FloatToIntPolicy,
    convert_f64_to_i64,
};

#[test]
fn stable_facade_exposes_typed_bounded_failure_contract() {
    let conversion = convert_f64_to_i64(
        "consumer_conversion",
        DTypeId::F64.descriptor(),
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
fn tensor_operators_are_infallible_syntax_over_fallible_named_methods() -> incin::Result<()> {
    use incin::prelude::{DefaultBackend, Tensor, s};

    let lhs = Tensor::<s![2], DefaultBackend>::ones(())?;
    let rhs = Tensor::<s![2], DefaultBackend>::ones(())?;
    let sum = lhs + rhs;
    assert_eq!(sum.to_vec1::<f32>()?, vec![2.0, 2.0]);
    Ok(())
}

/// `to_scalar` and `to_vec1` read the tensor's bytes through a `*const E`, so
/// a target type that merely agrees on width silently reinterprets the bit
/// pattern. `f32` and `u32` are both four bytes, and `1.0f32` came back as
/// `1065353216` with no error. Extraction has to name the stored dtype.
#[cfg(feature = "cpu")]
#[test]
fn extracting_a_tensor_as_a_same_width_but_different_type_is_refused() {
    use incin::prelude::{DefaultBackend, Tensor, s};

    let t = Tensor::<s![2], DefaultBackend>::ones(()).unwrap();

    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0, 1.0]);

    let reinterpreted = t.to_vec1::<u32>();
    assert!(
        reinterpreted.is_err(),
        "an f32 tensor must not extract as u32, got {:?}",
        reinterpreted.map(|v| v.first().copied())
    );

    let scalar = Tensor::<s![1], DefaultBackend>::ones(()).unwrap();
    assert_eq!(scalar.to_scalar::<f32>().unwrap(), 1.0);
    assert!(
        scalar.to_scalar::<i32>().is_err(),
        "an f32 tensor must not extract as i32"
    );
}
