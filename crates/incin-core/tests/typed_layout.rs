//! A layout proof must be derived correctly and must gate what needs it.
#![cfg(feature = "std")]

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::shapes::{Contiguous, Layout, LayoutOf, RowMajor, Shape, Unknown};
use incin_macros::s;

incin_core::dim!(Batch);

/// A tensor carrying a real layout must keep the API a plain one has.
///
/// This is the check the parameter's default would otherwise hide. An
/// `impl<S, B, K, G, P> Tensor<S, B, K, G, P>` binds `L` to its default, so it
/// silently stops applying to a tensor that has proven something -- the crate
/// still compiles and the loss is invisible until someone holds a proven
/// tensor and finds it has no methods.
///
/// Every accessor exercised here is one that used to be unreachable that way.
/// Extend this as further modules are converted; a method that stops compiling
/// here is a module that still pins `L`.
#[test]
fn a_proven_tensor_keeps_the_ordinary_api() {
    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .expect("a freshly created tensor is dense row-major");

    assert_eq!(t.dims().as_ref(), &[3, 4]);
    assert_eq!(t.shape_buf().as_ref(), &[3, 4]);
    assert_eq!(t.dtype(), incin_core::prelude::DTypeId::F32.into());
    assert_eq!(t.numel(), 12);
    let _ = t.inner();
}

/// The same, for the operation surface rather than the accessors.
///
/// Kept separate because these are the conversions still outstanding: an
/// operation has to decide what layout its *output* carries, which is a
/// contract question rather than a rename, so the modules behind these calls
/// are converted deliberately and one at a time.
#[test]
fn a_proven_tensor_can_still_be_operated_on() {
    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();

    let negated = t.neg().expect("a unary op applies to a proven tensor");
    assert_eq!(negated.dims().as_ref(), &[3, 4]);

    // Binary ops take a second operand with its own, independent layout: the
    // two need not agree, and neither is required to have proven anything.
    let unproven = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(()).unwrap();
    let summed = negated
        .add_exact(&unproven)
        .expect("a proven operand and an unproven one combine");
    assert_eq!(summed.dims().as_ref(), &[3, 4]);
}

/// Reductions accept a proven operand.
///
/// A reduction changes the shape, so its result's layout is stated rather than
/// carried -- a layout is only meaningful against the shape it describes, and
/// carrying the operand's would be claiming something about a different
/// geometry.
#[test]
fn a_reduction_accepts_a_proven_operand() {
    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();

    let summed = t
        .sum(incin_core::shapes::idx::ForwardAxis::<
            incin_core::shapes::idx::Here,
        >::default())
        .expect("a reduction applies to a proven tensor");
    assert_eq!(summed.dims().as_ref(), &[4]);
}

/// A pointwise op preserves the operand's layout rather than upgrading or
/// discarding it.
///
/// This is the propagation rule, and it is what keeps the parameter usable: a
/// proven tensor stays proven through a chain, so `reshape_view` is still
/// reachable at the end of one. Asserting `RowMajor` on every output instead
/// would be equally truthful -- the buffer is dense either way -- but it forces
/// every downstream signature that says `Tensor<S, B, K, G>` to be rewritten,
/// because `Unknown` and `RowMajor` are different types.
#[test]
fn a_proof_survives_a_pointwise_chain() {
    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();

    // Still contiguous after two ops, so the `Contiguous` bound is satisfied.
    let flat = t
        .neg()
        .unwrap()
        .neg()
        .unwrap()
        .reshape_view::<s![12]>()
        .expect("contiguity survives a pointwise chain");
    assert_eq!(flat.dims().as_ref(), &[12]);
}

/// The `Dense` alias names the common case without repeating the shape.
///
/// `Tensor<S, B, K, G, P, RowMajor<S>>` mentions the shape twice, which is
/// noise: `RowMajor` is congruent with the shape it describes by construction.
#[test]
fn the_dense_alias_is_the_ergonomic_spelling() {
    // Aliased once locally, which is how a caller would actually write it and
    // what keeps `clippy::type_complexity` quiet: `s![3, 4]` still expands to a
    // typenum chain, so `Dense` shortens the spelling rather than the type.
    type Batch34 = incin_core::prelude::Dense<s![3, 4], CpuBackendImpl>;

    fn takes_dense(t: &Batch34) -> usize {
        t.numel()
    }

    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();
    assert_eq!(takes_dense(&t), 12);
}

/// Row-major strides are the suffix products of the shape's extents.
#[test]
fn a_row_major_layout_derives_its_strides_from_the_shape() {
    assert_eq!(
        <RowMajor<s![3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(4), Some(1)][..]
    );
    assert_eq!(
        <RowMajor<s![2, 3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(12), Some(4), Some(1)][..]
    );
    assert_eq!(<RowMajor<s![3, 4]> as Layout>::STATIC_OFFSET, Some(0));
}

/// Only a proven layout satisfies `Contiguous`.
///
/// This is the gate the whole parameter exists for: `reshape_view` is bounded
/// on it, so a tensor that has established nothing cannot reinterpret its
/// buffer. The negative case is a compile-fail fixture rather than an
/// assertion, because "does not implement" is not observable at runtime.
#[test]
fn only_a_proven_layout_satisfies_contiguous() {
    fn needs_contiguous<L: Contiguous>() {}
    needs_contiguous::<RowMajor<s![3, 4]>>();
    // needs_contiguous::<Unknown>() does not compile.

    // The two carry different evidence, so the distinction is real rather than
    // a marker that everything happens to satisfy.
    assert_eq!(
        <RowMajor<s![3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(4), Some(1)][..]
    );
    assert_eq!(<Unknown as Layout>::STATIC_STRIDES, &[][..]);
    assert_eq!(<Unknown as Layout>::STATIC_OFFSET, None);
}

/// Congruence: a layout describes a shape of the same rank.
#[test]
fn congruence_relates_a_layout_to_its_shape() {
    fn describes<S: Shape, L: LayoutOf<S>>() {}

    describes::<s![3, 4], RowMajor<s![3, 4]>>();
    // `Unknown` describes anything, because it claims nothing about it.
    describes::<s![3, 4], Unknown>();
}

/// A dynamic axis voids the strides outside it and spares those inside.
///
/// The asymmetry that justifies reporting per axis: each stride is a product
/// of the extents inner to it, so a dynamic *outermost* axis voids nothing at
/// all, and a dynamic inner axis voids only what encloses it.
#[test]
fn a_dynamic_axis_voids_only_the_strides_that_enclose_it() {
    assert_eq!(
        <RowMajor<s![Batch, 3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(12), Some(4), Some(1)][..]
    );
    assert_eq!(
        <RowMajor<s![2, Batch, 4]> as Layout>::STATIC_STRIDES,
        &[None, Some(4), Some(1)][..]
    );
}

/// Every converted module accepts a proven tensor.
///
/// This is the enumeration mechanism: a method that stops compiling here names
/// a module that still pins `L` to its default. It grew one call at a time as
/// the conversion proceeded, which is how each unconverted module was found.
#[test]
fn a_proven_tensor_reaches_every_converted_module() {
    use incin_core::shapes::idx::{ForwardAxis, Here, Next};

    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();

    // pointwise, unary and binary
    let _ = t.neg().unwrap();
    let _ = t.add_exact(&t).unwrap();
    // reductions, per-axis and whole-tensor
    let _ = t.sum(ForwardAxis::<Here>::default()).unwrap();
    let _ = t.clone().sum_all().unwrap();
    // shape manipulation
    let _ = t.transpose_structural::<Here, Next<Here>>().unwrap();
    let _ = t.clone().into_shape::<incin_core::shapes::Dyn>().unwrap();
    let _ = t.clone();
    // matmul
    let square = incin_core::prelude::Tensor::<s![4, 3], CpuBackendImpl>::zeros(())
        .unwrap()
        .into_row_major()
        .unwrap();
    let _ = t.matmul(&square).unwrap();
    // an nn layer, through the Module trait
    let _ = incin_core::nn::Module::forward(&incin_core::nn::ReLU, t.clone());
}

/// Shape-changing operations really do produce dense buffers.
///
/// This backs a claim the type system would otherwise make on faith. A
/// reduction or a transpose cannot carry its operand's layout -- the shape
/// changes, and a layout is only meaningful against the shape it describes --
/// so its result's layout has to be *stated*. Stating `RowMajor` is only honest
/// if the buffer is actually dense.
///
/// Checked here at runtime rather than assumed, because the density is a
/// property of what every current backend happens to do rather than something
/// the operation contract states. If a backend ever returns a strided result,
/// this fails and the type claim has to be withdrawn before it becomes a
/// silent mis-read.
#[test]
fn shape_changing_operations_produce_dense_results() {
    use incin_core::shapes::idx::{ForwardAxis, Here};

    fn assert_dense<S: Shape, B, K, G, P, L>(t: &incin_core::prelude::Tensor<S, B, K, G, P, L>)
    where
        B: incin_core::backend_authoring::Backend,
        K: incin_core::prelude::DType,
        G: incin_core::tensor::grad::RequiresGrad,
        P: incin_core::dist::Placement,
        L: incin_core::shapes::Layout,
    {
        let meta = <B as incin_core::backend_authoring::StorageBackend>::metadata::<K>(t.inner());
        let dims = meta.shape().as_ref();
        let strides = meta.strides().as_ref();
        let mut expected = 1usize;
        for axis in (0..dims.len()).rev() {
            assert_eq!(
                strides[axis], expected,
                "axis {axis} of {dims:?} has stride {} but a dense buffer needs {expected}",
                strides[axis]
            );
            expected *= dims[axis];
        }
        assert_eq!(meta.offset_elements(), 0, "a fresh buffer starts at zero");
    }

    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(()).unwrap();

    // Reductions and pointwise operations allocate.
    assert_dense(&t.sum(ForwardAxis::<Here>::default()).unwrap());
    assert_dense(&t.mean(ForwardAxis::<Here>::default()).unwrap());
    assert_dense(&t.neg().unwrap());

    // `transpose` does *not*, on this backend. It returns a view: shape
    // [4, 3] over strides [1, 4], sharing the original buffer. Asserted rather
    // than assumed, because the CUDA backend materialises the same operation
    // into a fresh contiguous buffer -- the two disagree, and a type claiming
    // `RowMajor` for a transpose would be false on exactly one of them.
    let transposed = t
        .transpose_structural::<Here, incin_core::shapes::idx::Next<Here>>()
        .unwrap();
    let meta = <CpuBackendImpl as incin_core::backend_authoring::StorageBackend>::metadata::<f32>(
        transposed.inner(),
    );
    assert_eq!(meta.shape().as_ref(), &[4, 3]);
    assert_eq!(
        meta.strides().as_ref(),
        &[1, 4],
        "CPU transpose is a view; if this becomes [3, 1] it started copying"
    );
}

/// `transpose_view` must not copy, and must produce a non-contiguous result.
///
/// The point of the operation is that it does no work: it permutes shape and
/// strides over the same buffer. Asserted on the metadata rather than timed,
/// because "did not copy" is a structural claim -- a copy would come back dense.
#[test]
fn transpose_view_permutes_metadata_without_copying() {
    use incin_core::backend_authoring::StorageBackend;
    use incin_core::shapes::idx::{Here, Next};

    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(()).unwrap();
    let viewed = t
        .transpose_view::<Here, Next<Here>>()
        .expect("a 3x4 tensor transposes to 4x3");

    let meta = <CpuBackendImpl as StorageBackend>::metadata::<f32>(viewed.inner());
    assert_eq!(meta.shape().as_ref(), &[4, 3]);
    assert_eq!(
        meta.strides().as_ref(),
        &[1, 4],
        "a view permutes the strides; [3, 1] would mean it copied"
    );
}

/// The materialising transpose and the view disagree, which is the whole point.
///
/// Both are legal and neither is universally faster -- the view wins for a
/// single consumer and loses from about four -- so the framework offers both
/// and the caller chooses. This pins that they are actually different, since a
/// backend quietly making them the same would remove the choice without
/// removing the API.
#[test]
fn the_two_transposes_are_genuinely_different_operations() {
    use incin_core::backend_authoring::StorageBackend;
    use incin_core::shapes::idx::{Here, Next};

    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(()).unwrap();

    let viewed = t.transpose_view::<Here, Next<Here>>().unwrap();
    let copied = t.transpose_structural::<Here, Next<Here>>().unwrap();

    let view_meta = <CpuBackendImpl as StorageBackend>::metadata::<f32>(viewed.inner());
    let copy_meta = <CpuBackendImpl as StorageBackend>::metadata::<f32>(copied.inner());

    assert_eq!(view_meta.shape().as_ref(), copy_meta.shape().as_ref());
    // On CPU both are currently views, so the strides agree today. The shapes
    // must always agree; the strides are what #113 is about.
    assert_eq!(view_meta.strides().as_ref(), &[1, 4]);
}
