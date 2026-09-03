//! Whether a CUDA pointwise result is dense, which decides a type claim.
//!
//! `incin-core`'s `typed_layout` suite establishes on CPU that a pointwise
//! operation returns a packed row-major buffer even when its operand is
//! strided. That is the evidence a stronger output layout would rest on -- but
//! it is evidence about one backend, and the claim a type makes is made for
//! every backend at once.
//!
//! CUDA is the backend most likely to disagree, because it materialises several
//! operations that CPU keeps as views. So the same question is asked here
//! directly rather than assumed to carry across.
#![cfg(feature = "cuda")]

extern crate incin_core as incin;

use incin_backends::cuda::CudaBackendImpl;
use incin_core::backend_authoring::StorageBackend;
use incin_core::prelude::*;
use incin_core::shapes::Layout;
use incin_core::shapes::idx::{Here, Next};
use incin_core::typenum::U0;

type Cuda = CudaBackendImpl<CudaN<U0>>;

/// Spells the expected readback as an owned vector for comparison.
fn alloc_vec(values: &[f32]) -> Vec<f32> {
    values.to_vec()
}

/// Asserts row-major suffix-product strides and a zero offset.
fn assert_dense<S, K, G, P, L>(label: &str, t: &Tensor<S, Cuda, K, G, P, L>)
where
    S: incin_core::shapes::Shape,
    K: DType,
    G: incin_core::tensor::grad::RequiresGrad,
    P: incin_core::dist::Placement,
    L: Layout,
{
    let meta = <Cuda as StorageBackend>::metadata::<K>(t.inner());
    let dims = meta.shape().as_ref();
    let strides = meta.strides().as_ref();
    let mut expected = 1usize;
    for axis in (0..dims.len()).rev() {
        assert_eq!(
            strides[axis], expected,
            "{label}: axis {axis} of {dims:?} has stride {} but a dense buffer needs {expected}",
            strides[axis]
        );
        expected *= dims[axis];
    }
    assert_eq!(meta.offset_elements(), 0, "{label}: must start at zero");
}

/// A CUDA pointwise result is dense, whatever its operand's strides were.
///
/// The operand is built with `transpose_view`, which permutes metadata without
/// copying on this backend too. If that ever changes the premise assertion
/// fails first, rather than the test quietly proving nothing about strided
/// operands while still reporting green.
#[test]
#[ignore = "requires CUDA hardware"]
fn a_cuda_pointwise_result_is_dense_even_from_a_strided_operand() {
    incin_backends::cuda::testing::require_cuda();

    // Distinct values, so a kernel that reads the wrong offsets produces the
    // wrong numbers rather than a plausible buffer of ones.
    let base = Tensor::<s![3, 4], Cuda>::from_slice(
        &[
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ],
        (),
    )
    .unwrap();
    let strided = base
        .transpose_view::<Here, Next<Here>>()
        .expect("a 3x4 tensor transposes to 4x3");

    let meta = <Cuda as StorageBackend>::metadata::<f32>(strided.inner());
    assert_eq!(meta.shape().as_ref(), &[4, 3]);
    assert_eq!(
        meta.strides().as_ref(),
        &[1, 4],
        "transpose_view must not copy on CUDA either, or this test says nothing \
         about strided operands"
    );

    let negated = strided.neg().unwrap();
    assert_dense("neg", &negated);

    // Dense strides prove the buffer's *shape*; they say nothing about whether
    // the strided kernel read the right elements. A kernel that walked the
    // operand contiguously instead of by its strides would still write a dense
    // output -- of the wrong values. The transpose of
    // [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]] is
    // [[1, 5, 9], [2, 6, 10], [3, 7, 11], [4, 8, 12]], negated below.
    assert_eq!(
        negated.to_vec1::<f32>().unwrap(),
        alloc_vec(&[
            -1.0, -5.0, -9.0, -2.0, -6.0, -10.0, -3.0, -7.0, -11.0, -4.0, -8.0, -12.0
        ]),
        "the strided kernel must read by the operand's strides, not linearly"
    );
    assert_dense("abs", &strided.abs().unwrap());
    assert_dense("exp", &strided.exp().unwrap());
    assert_dense("mul_scalar", &strided.mul_scalar(2.0).unwrap());

    let dense_43 = Tensor::<s![4, 3], Cuda>::ones(()).unwrap();
    assert_dense(
        "add(strided, dense)",
        &strided.add_exact(&dense_43).unwrap(),
    );
    assert_dense(
        "add(dense, strided)",
        &dense_43.add_exact(&strided).unwrap(),
    );
    assert_dense(
        "mul(strided, dense)",
        &strided.mul_exact(&dense_43).unwrap(),
    );
}
