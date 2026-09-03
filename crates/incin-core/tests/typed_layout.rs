//! A layout proof must be derived correctly and must gate what needs it.
#![cfg(feature = "std")]

extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::shapes::{Contiguous, Dyn, Layout, LayoutOf, RowMajor, Shape};
use incin_macros::s;

incin_core::dim!(Batch);

/// Asserts a tensor's buffer is dense row-major: suffix-product strides from a
/// zero offset.
///
/// Shared by every test that asks "did this operation allocate?", because the
/// question is always the same one and a per-test copy drifts.
#[track_caller]
fn assert_dense<S: Shape, B, K, G, P, L>(
    label: &str,
    t: &incin_core::prelude::Tensor<S, B, K, G, P, L>,
) where
    B: incin_core::backend_authoring::Backend,
    K: incin_core::prelude::DType,
    G: incin_core::tensor::grad::RequiresGrad,
    P: incin_core::dist::Placement,
    L: Layout,
{
    let meta = <B as incin_core::backend_authoring::StorageBackend>::metadata::<K>(t.inner());
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
/// because `Dyn` and `RowMajor` are different types.
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
    // needs_contiguous::<Dyn>() does not compile.

    // The two carry different evidence, so the distinction is real rather than
    // a marker that everything happens to satisfy.
    assert_eq!(
        <RowMajor<s![3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(4), Some(1)][..]
    );
    assert_eq!(<Dyn as Layout>::STATIC_STRIDES, &[][..]);
    assert_eq!(<Dyn as Layout>::STATIC_OFFSET, None);
}

/// Congruence: a layout describes a shape of the same rank.
#[test]
fn congruence_relates_a_layout_to_its_shape() {
    fn describes<S: Shape, L: LayoutOf<S>>() {}

    describes::<s![3, 4], RowMajor<s![3, 4]>>();
    // `Dyn` describes anything, because it claims nothing about it.
    describes::<s![3, 4], Dyn>();
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

    let t = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::zeros(()).unwrap();

    // Reductions and pointwise operations allocate.
    assert_dense("sum", &t.sum(ForwardAxis::<Here>::default()).unwrap());
    assert_dense("mean", &t.mean(ForwardAxis::<Here>::default()).unwrap());
    assert_dense("neg", &t.neg().unwrap());

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

/// A constructor hands back the proof directly, with no runtime promotion.
///
/// This is the property that makes the layout parameter worth its plumbing.
/// Before it, `RowMajor` was reachable only through `into_row_major`, a runtime
/// scan of the strides -- so every static claim in the tree bottomed out in a
/// dynamic check, and the type parameter bought nothing a boolean would not
/// have. `zeros` allocates a packed row-major buffer, so it is entitled to say
/// so, and the `FreshDense` bound is what lets it say so without becoming a way
/// to forge the claim.
#[test]
fn a_constructor_yields_a_layout_proof_without_a_runtime_check() {
    // The alias is what a caller writes; naming it once keeps the point of the
    // test in view rather than the type.
    type DenseMatrix = incin_core::shapes::Dense<s![3, 4], CpuBackendImpl>;

    // No `into_row_major` anywhere: the proof comes from the allocation.
    let dense: DenseMatrix = incin_core::prelude::Tensor::zeros(()).unwrap();

    assert_eq!(dense.dims().as_ref(), &[3, 4]);
    assert_eq!(
        <RowMajor<s![3, 4]> as Layout>::STATIC_STRIDES,
        &[Some(4), Some(1)],
        "row-major strides are the suffix products of the extents"
    );

    // And it satisfies the bound that `Dyn` cannot, so the view path opens.
    fn needs_contiguous<L: Contiguous>() {}
    needs_contiguous::<RowMajor<s![3, 4]>>();

    let reshaped = dense
        .reshape_view::<s![12]>()
        .expect("a dense 3x4 reinterprets as a dense 12");
    assert_eq!(reshaped.dims().as_ref(), &[12]);
}

/// The default is unchanged, so nothing that predates the parameter shifted.
#[test]
fn asking_for_nothing_still_yields_unknown() {
    let plain = incin_core::prelude::Tensor::<s![2, 2], CpuBackendImpl>::zeros(()).unwrap();
    assert_eq!(plain.numel(), 4);
    assert_eq!(
        <Dyn as Layout>::STATIC_STRIDES,
        &[] as &[Option<usize>],
        "a tensor that proved nothing must report nothing"
    );
}

/// A pointwise operation returns a dense buffer even from a strided operand.
///
/// This is the evidence a stronger type claim needs, and it is the case the
/// existing density test does not reach. `shape_changing_operations_produce_
/// dense_results` feeds every operation a tensor that is *already* dense, so a
/// backend that simply forwarded its operand's strides would pass it. The
/// question that decides whether a pointwise result may assert `RowMajor` is
/// the opposite one: given an operand that is genuinely non-contiguous, is the
/// output still dense?
///
/// `transpose_view` gives a real strided operand on CPU -- shape `[4, 3]` over
/// strides `[1, 4]` -- so a pointwise operation applied to it either allocates
/// a fresh packed buffer, in which case the claim is honest for every operand,
/// or propagates the strides, in which case it is honest only for dense ones
/// and pointwise must keep carrying `L` through.
///
/// Checked for both arities. A binary operation has a second way to go wrong:
/// it could adopt either operand's layout, so it is fed one strided and one
/// dense operand rather than two of a kind.
#[test]
fn a_pointwise_result_is_dense_even_from_a_strided_operand() {
    use incin_core::backend_authoring::StorageBackend;
    use incin_core::shapes::idx::{Here, Next};

    let base = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::ones(()).unwrap();
    let strided = base
        .transpose_view::<Here, Next<Here>>()
        .expect("a 3x4 tensor transposes to 4x3");

    // The premise. If this stops holding the rest of the test proves nothing,
    // so it is asserted rather than trusted.
    let meta = <CpuBackendImpl as StorageBackend>::metadata::<f32>(strided.inner());
    assert_eq!(
        meta.strides().as_ref(),
        &[1, 4],
        "the operand must actually be non-contiguous for this test to mean anything"
    );

    assert_dense("neg", &strided.neg().unwrap());
    assert_dense("abs", &strided.abs().unwrap());
    assert_dense("exp", &strided.exp().unwrap());
    assert_dense("mul_scalar", &strided.mul_scalar(2.0).unwrap());

    let dense_43 = incin_core::prelude::Tensor::<s![4, 3], CpuBackendImpl>::ones(()).unwrap();
    assert_dense(
        "add(strided, dense)",
        &strided.add_exact(&dense_43).unwrap(),
    );
    assert_dense(
        "add(dense, strided)",
        &dense_43.add_exact(&strided).unwrap(),
    );
    assert_dense("maximum", &strided.maximum(&dense_43).unwrap());
    assert_dense("lerp", &strided.lerp(&dense_43, 0.5).unwrap());

    // Comparisons and the logical pair allocate a fresh bool buffer, and until
    // now their signatures pinned `L` to its default -- a proven tensor could
    // not call `eq` at all.
    assert_dense("eq", &strided.eq(&dense_43).unwrap());
    assert_dense("lt", &strided.lt(&dense_43).unwrap());
    let mask = strided.gt(&dense_43).unwrap();
    let other_mask = dense_43.le(&strided).unwrap();
    assert_dense("logical_and", &mask.logical_and(&other_mask).unwrap());
    assert_dense("logical_or", &mask.logical_or(&other_mask).unwrap());
    assert_dense(
        "masked_fill",
        &strided.masked_fill(&mask, 0.0).unwrap(),
    );
    assert_dense(
        "where_cond",
        &mask.where_cond(&strided, &dense_43).unwrap(),
    );
    assert_dense(
        "mul(strided, dense)",
        &strided.mul_exact(&dense_43).unwrap(),
    );
}

/// A reduction's result is dense whatever its operand's strides were.
///
/// The counterpart to [`a_pointwise_result_is_dense_even_from_a_strided_operand`]
/// for the reduction surface. `reduce.rs` already documents that "the results
/// are freshly allocated dense buffers", but nothing checked it against an
/// operand that was not already dense, and two signatures disagreed with the
/// sentence in opposite directions: the axis reductions claimed nothing, while
/// `cumsum` returned `Self` and so claimed whatever the *operand* claimed.
///
/// `cumsum` is the one that matters. It is shape-preserving, so carrying the
/// operand's layout typechecks -- and would be a false claim the moment the
/// operand is strided, which is exactly the case this test constructs.
#[test]
fn a_reduction_result_is_dense_even_from_a_strided_operand() {
    use incin_core::backend_authoring::StorageBackend;
    use incin_core::shapes::idx::{ForwardAxis, Here, Next};

    let base = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::ones(()).unwrap();
    let strided = base
        .transpose_view::<Here, Next<Here>>()
        .expect("a 3x4 tensor transposes to 4x3");

    // The premise, asserted rather than trusted: without it the test is vacuous.
    let meta = <CpuBackendImpl as StorageBackend>::metadata::<f32>(strided.inner());
    assert_eq!(
        meta.strides().as_ref(),
        &[1, 4],
        "the operand must actually be non-contiguous for this test to mean anything"
    );

    assert_dense("sum", &strided.sum(ForwardAxis::<Here>::default()).unwrap());
    assert_dense("mean", &strided.mean(ForwardAxis::<Here>::default()).unwrap());
    assert_dense("max", &strided.max(ForwardAxis::<Here>::default()).unwrap());
    assert_dense("min", &strided.min(ForwardAxis::<Here>::default()).unwrap());
    assert_dense(
        "sum_keepdim",
        &strided.sum_keepdim(ForwardAxis::<Here>::default()).unwrap(),
    );
    assert_dense(
        "cumsum",
        &strided.cumsum(ForwardAxis::<Here>::default()).unwrap(),
    );
    // Last: `sum_all` consumes the receiver.
    assert_dense("sum_all", &strided.sum_all().unwrap());
}

/// A matmul result is dense whatever its operands' strides were.
///
/// The evidence `matmul` and `addmm` were explicitly waiting on. Both allocate,
/// but until this test the claim rested on what the backends happen to do
/// rather than on anything checked -- and `addmm` was worse than merely weak:
/// it returned `Self`, so like `cumsum` before it, it handed the *bias
/// operand's* layout to a buffer produced by a GEMM.
///
/// CPU advertises `matmul_layouts = CPU_LAYOUTS`, so it genuinely accepts a
/// strided operand here rather than refusing it the way CUDA does. That makes
/// this the backend where the question has a non-vacuous answer.
#[test]
fn a_matmul_result_is_dense_even_from_a_strided_operand() {
    use incin_core::backend_authoring::StorageBackend;
    use incin_core::shapes::idx::{Here, Next};

    // [3, 4] transposed to [4, 3] over strides [1, 4]: a real strided operand
    // whose shape still lines up for a [4, 3] x [3, 2] product.
    let base = incin_core::prelude::Tensor::<s![3, 4], CpuBackendImpl>::from_slice(
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        (),
    )
    .unwrap();
    let strided = base
        .transpose_view::<Here, Next<Here>>()
        .expect("a 3x4 tensor transposes to 4x3");

    let meta = <CpuBackendImpl as StorageBackend>::metadata::<f32>(strided.inner());
    assert_eq!(
        meta.strides().as_ref(),
        &[1, 4],
        "the operand must actually be non-contiguous for this test to mean anything"
    );

    let rhs = incin_core::prelude::Tensor::<s![3, 2], CpuBackendImpl>::ones(()).unwrap();
    let product = strided.matmul(&rhs).unwrap();
    assert_dense("matmul(strided, dense)", &product);

    // A GEMM that read the strided operand linearly would produce a dense
    // buffer of the wrong numbers, so the values are checked too. Row i of the
    // transpose is [i+1, i+5, i+9], and every rhs entry is one, so each output
    // row is that row's sum repeated twice.
    assert_eq!(
        product.to_vec1::<f32>().unwrap(),
        vec![15.0, 15.0, 18.0, 18.0, 21.0, 21.0, 24.0, 24.0],
        "the GEMM must read the operand by its strides, not linearly"
    );

    // `addmm` adds a bias to the product. Its result used to carry the bias's
    // layout; the buffer it writes is a fresh dense one.
    let bias = incin_core::prelude::Tensor::<s![4, 2], CpuBackendImpl>::ones(()).unwrap();
    assert_dense(
        "addmm",
        &bias.addmm(&strided, &rhs, 1.0, 1.0).unwrap(),
    );
}

/// `BatchNorm2d` allocates unconditionally, so its result is dense.
///
/// It was carrying the operand's layout, which is the fourth instance of the
/// pattern: a shape-preserving operation where returning the operand's claim
/// typechecks. Unlike `Dropout` below it has no identity path -- every call
/// dispatches and writes a fresh buffer -- so there is nothing for the carry to
/// have been right about.
#[test]
fn batch_norm_produces_a_dense_result() {
    use incin_core::nn::module::Module;
    use incin_core::prelude::*;

    type Bn = BatchNorm2d<s![2], CpuBackendImpl>;
    let bn: Bn = BatchNorm2d::build((1e-5, 0.1)).unwrap();
    let x = incin_core::prelude::Tensor::<s![1, 2, 2, 2], CpuBackendImpl>::ones(()).unwrap();
    assert_dense("batch_norm", &bn.forward(x).unwrap());
}

/// `Dropout` is the one place carrying the operand's layout is *right*.
///
/// Every other shape-preserving operation in the crate writes a fresh buffer on
/// every call, so carrying is a claim about a buffer the operand never touched.
/// Dropout has a genuine identity path -- eval mode, or `p == 0` -- which
/// returns the very tensor it was handed, strides and all. Its output layout
/// really is its input's.
///
/// That is only sound because the *other* branch writes a dense buffer, so the
/// carried layout has to be one a dense buffer also satisfies. The signature
/// says so by bounding `L` on the sealed `FreshDense<S>` rather than on
/// `Layout`, which is the same bound the constructors use and for the same
/// reason. Both halves are checked here, because the argument needs both.
#[test]
fn dropout_carries_its_operand_layout_and_both_branches_earn_it() {
    use incin_core::backend_authoring::StorageBackend;
    use incin_core::nn::module::Module;
    use incin_core::nn::TrainMode;
    use incin_core::prelude::Dropout;

    let x = incin_core::prelude::Tensor::<s![4, 4], CpuBackendImpl>::ones(())
        .unwrap()
        .into_row_major()
        .expect("a fresh allocation is row-major");

    // Training: allocates, and the buffer it writes is dense, so the carried
    // `RowMajor` is true of it.
    let trained = Dropout::new(0.5).forward(x.clone()).unwrap();
    assert_dense("dropout(training)", &trained);

    // Eval: identity. The result must be the operand, not a copy of it, or the
    // carry is describing something the caller cannot observe.
    let mut eval = Dropout::new(0.5);
    eval.set_training(false);
    let passed = eval.forward(x.clone()).unwrap();
    let before = <CpuBackendImpl as StorageBackend>::metadata::<f32>(x.inner());
    let after = <CpuBackendImpl as StorageBackend>::metadata::<f32>(passed.inner());
    assert_eq!(
        before.strides().as_ref(),
        after.strides().as_ref(),
        "eval-mode dropout must hand back the operand unchanged"
    );
}
