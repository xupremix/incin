extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::backend_authoring::VariableBackend;
use incin_core::prelude::*;
use incin_macros::s;

type B = CpuBackendImpl;

#[test]
fn checked_construction_accepts_matching_storage_metadata() {
    let source = Tensor::<s![2, 3], B>::ones(()).unwrap();
    let rebuilt = Tensor::<s![2, 3], B>::try_from_storage(
        source.into_inner(),
        <s![2, 3] as Shape>::try_from_dims(&[2, 3]).unwrap(),
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .unwrap();

    assert_eq!(rebuilt.dims().as_ref(), &[2, 3]);
}

#[test]
fn checked_construction_rejects_storage_shape_mismatch() {
    let storage = Tensor::<s![2, 3], B>::zeros(()).unwrap().into_inner();
    let err = Tensor::<s![2, 2], B>::try_from_storage(
        storage,
        <s![2, 2] as Shape>::try_from_dims(&[2, 2]).unwrap(),
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        Error::ShapeMismatch { expected, got, .. }
            if expected == vec![2, 2] && got == vec![2, 3]
    ));
}

#[test]
fn checked_construction_rejects_static_shape_contract_mismatch() {
    let storage = Tensor::<s![2, 3], B>::zeros(()).unwrap().into_inner();
    let err = Tensor::<s![2, 2], B>::try_from_storage(
        storage,
        ShapeBuf::from_slice(&[2, 3]),
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .unwrap_err();

    assert!(matches!(err, Error::Shape(_)));
}

#[test]
fn parameter_checked_construction_validates_static_shape_contract() {
    let source = Tensor::<s![2], B>::zeros(()).unwrap().into_inner();
    let storage = B::var_from_tensor::<f32>(&source).unwrap();
    let err = Param::<s![3], B>::from_parts_checked(
        storage,
        ShapeBuf::from_slice(&[2]),
        Default::default(),
        Default::default(),
    )
    .unwrap_err();

    assert!(matches!(err, Error::Shape(_)));
}

#[test]
fn buffer_checked_construction_validates_static_shape_contract() {
    let source = Tensor::<s![2], B>::zeros(()).unwrap().into_inner();
    let storage = B::var_from_tensor::<f32>(&source).unwrap();
    let err = Buffer::<s![3], B>::from_parts_checked(
        storage,
        ShapeBuf::from_slice(&[2]),
        Default::default(),
        Default::default(),
    )
    .unwrap_err();

    assert!(matches!(err, Error::Shape(_)));
}

#[test]
fn parameter_raw_construction_validates_dtype_contract() {
    let source = Tensor::<s![2], B>::zeros(()).unwrap().into_inner();
    let raw = B::var_from_tensor::<f32>(&source).unwrap();
    let err = Param::<s![2], B, u32>::from_raw(raw, ()).unwrap_err();

    assert!(matches!(err, Error::DTypeStorageMismatch { .. }));
}

#[test]
fn checked_construction_rejects_storage_dtype_mismatch() {
    let storage = Tensor::<s![2], B>::zeros(()).unwrap().into_inner();
    let err = Tensor::<Dyn, B, u32>::try_from_storage(
        storage,
        ShapeBuf::from_slice(&[2]),
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .unwrap_err();

    if let Error::DTypeStorageMismatch { expected, got } = err {
        assert_eq!(expected, DTypeId::U32.descriptor());
        assert_eq!(got, DTypeId::F32.descriptor());
    } else {
        panic!("expected Error::DTypeStorageMismatch, got {:?}", err);
    }
}

#[test]
fn checked_construction_rejects_integer_gradient_tracking() {
    let source = Tensor::<s![2], B, i64>::zeros(()).unwrap();
    let err = Tensor::<s![2], B, i64, Grad>::try_from_storage(
        source.into_inner(),
        ShapeBuf::from_slice(&[2]),
        Default::default(),
        Default::default(),
        Default::default(),
    )
    .unwrap_err();

    assert!(matches!(
        err,
        Error::UnsupportedDType {
            op: "gradient tracking",
            ..
        }
    ));
}

#[test]
fn metadata_only_retagging_preserves_validated_storage() {
    let tensor = Tensor::<s![2, 3], B, f32, Grad>::ones(()).unwrap();
    let dynamic = tensor.into_dyn();
    assert_eq!(dynamic.dims(), vec![2, 3]);

    let tracked = dynamic.detach().require_grad();
    assert_eq!(tracked.dims(), vec![2, 3]);
    assert!(tracked.requires_grad());
}

#[test]
fn dynamic_flatten_reports_invalid_ranges_without_panicking() {
    let tensor = Tensor::<Dyn, B>::ones(vec![2, 3, 4]).unwrap();
    let err = tensor.flatten_runtime(2, 1).unwrap_err();
    assert_eq!(
        err.to_string(),
        "flatten: axis range 2..1 is invalid for rank 3"
    );

    let err = tensor.flatten_runtime(1, 3).unwrap_err();
    assert_eq!(err.to_string(), "axis 3 is invalid for rank 3");
}

#[test]
fn raw_unchecked_constructor_is_absent_from_the_crate() {
    let source_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut pending = vec![source_root];
    while let Some(path) = pending.pop() {
        for entry in std::fs::read_dir(path).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                pending.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                let source = std::fs::read_to_string(&path).unwrap();
                assert!(
                    !source.contains("from_parts_unchecked"),
                    "{} still bypasses construction witnesses",
                    path.display()
                );
            }
        }
    }
}
