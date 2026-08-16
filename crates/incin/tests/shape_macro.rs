//! `shape!`: the value-level counterpart of `s!`.
//!
//! The assertions that matter here are the type annotations, not the `dims()`
//! comparisons. A macro that produced the right extents with the wrong shape
//! type would pass every runtime check and lose exactly the static information
//! it exists to preserve, so each case pins the resulting `Tensor` type.

#![cfg(feature = "cpu")]

use incin::prelude::*;
use incin::types::{DimCons, Nil};

dim!(Batch, Features);

#[test]
fn all_literal_axes_produce_a_fully_static_shape() {
    let t: Tensor<s![2, 3], _, f32, NoGrad> = Cpu.zeros(shape![2, 3]).unwrap();
    assert_eq!(t.dims(), [2, 3]);
}

#[test]
fn named_shape_axes_share_the_s_macro_type_grammar() {
    let batch = 3usize;
    let runtime: Tensor<s![Batch = dyn, Features = 64], _, f32, NoGrad> =
        Cpu.zeros(shape![Batch = batch, Features = 64]).unwrap();
    assert_eq!(runtime.dims().as_ref(), &[3, 64]);

    let static_shape: Tensor<s![Batch = 25, Features = 64], _, f32, NoGrad> =
        Cpu.zeros(shape![Batch = 25, Features = 64]).unwrap();
    assert_eq!(static_shape.dims().as_ref(), &[25, 64]);
}

#[test]
fn an_expression_axis_becomes_a_runtime_axis_and_the_literals_stay_static() {
    let batch = 5usize;
    let t: Tensor<DimCons<usize, DimCons<typenum::U784, Nil>>, _, f32, NoGrad> =
        Cpu.zeros(shape![batch, 784]).unwrap();
    assert_eq!(t.dims().as_ref(), &[5, 784]);
}

/// `shape!` produces canonical structural shapes, while array form is a
/// runtime-rank adapter.
#[test]
fn all_runtime_axes_still_keep_the_rank() {
    let rows = 3usize;
    let cols = 4usize;

    let kept: Tensor<RuntimeShape2, _, f32, NoGrad> = Cpu.zeros(shape![rows, cols]).unwrap();
    assert_eq!(kept.dims().as_ref(), &[3, 4]);

    // Arrays are constructor adapters and resolve to the canonical Dyn shape.
    let array_ranked: Tensor<Dyn, _, f32, NoGrad> = Cpu.zeros([rows, cols]).unwrap();
    assert_eq!(array_ranked.dims(), [3, 4]);

    // Vec form is dynamic rank Dyn.
    let erased: Tensor<Dyn, _, f32, NoGrad> = Cpu.zeros(vec![rows, cols]).unwrap();
    assert_eq!(erased.dims(), vec![3, 4]);
}

const N_CONST: usize = 32;
const M_CONST: usize = 64;

struct ModelDimsTest;
impl ModelDimsTest {
    const WIDTH: usize = 128;
}

// An arbitrary const path cannot be evaluated by a proc macro.  It therefore
// uses Incin's explicit ConstDim adapter rather than typenum's finite literal
// aliases; raw literals above continue to use recursive binary typenum types.
type NConstShape = DimCons<ConstDim<N_CONST>, Nil>;
type NMConstShape = DimCons<ConstDim<N_CONST>, DimCons<ConstDim<M_CONST>, Nil>>;
type RuntimeMConstShape = DimCons<usize, DimCons<ConstDim<M_CONST>, Nil>>;
type WidthConstShape = DimCons<ConstDim<{ ModelDimsTest::WIDTH }>, Nil>;
type RuntimeShape1 = DimCons<usize, Nil>;
type RuntimeShape2 = DimCons<usize, DimCons<usize, Nil>>;
type RuntimeShape4 =
    DimCons<usize, DimCons<typenum::U3, DimCons<typenum::U28, DimCons<usize, Nil>>>>;

#[test]
fn const_path_axes_produce_static_typenum_dimensions() {
    let t1: Tensor<NConstShape, _, f32, NoGrad> = Cpu.zeros(shape![const N_CONST]).unwrap();
    assert_eq!(t1.dims(), [32]);

    let t2: Tensor<NMConstShape, _, f32, NoGrad> =
        Cpu.zeros(shape![const N_CONST, const M_CONST]).unwrap();
    assert_eq!(t2.dims(), [32, 64]);

    let runtime_n = 4usize;
    let t3: Tensor<RuntimeMConstShape, _, f32, NoGrad> =
        Cpu.zeros(shape![runtime_n, const M_CONST]).unwrap();
    assert_eq!(t3.dims(), [4, 64]);

    let t4: Tensor<WidthConstShape, _, f32, NoGrad> =
        Cpu.zeros(shape![const ModelDimsTest::WIDTH]).unwrap();
    assert_eq!(t4.dims(), [128]);

    // An un-prefixed identifier is treated as a runtime axis
    let t5: Tensor<RuntimeShape1, _, f32, NoGrad> = Cpu.zeros(shape![N_CONST]).unwrap();
    assert_eq!(t5.dims().as_ref(), &[32]);
}

#[test]
fn named_shape_types_accept_explicit_const_dimensions() {
    type NamedConstShape = s![Batch = const N_CONST, Features = const M_CONST];
    let tensor: Tensor<NamedConstShape, _, f32, NoGrad> = Cpu
        .zeros(shape![Batch = const N_CONST, Features = const M_CONST])
        .unwrap();
    assert_eq!(tensor.dims(), [32, 64]);
}

#[test]
fn arbitrary_expressions_are_accepted_as_runtime_axes() {
    let batch = 2usize;
    fn features() -> usize {
        7
    }
    let t: Tensor<RuntimeShape2, _, f32, NoGrad> =
        Cpu.zeros(shape![batch * 3, features()]).unwrap();
    assert_eq!(t.dims().as_ref(), &[6, 7]);
}

#[test]
fn rank_one_is_supported_static_and_runtime() {
    let t: Tensor<s![8], _, f32, NoGrad> = Cpu.zeros(shape![8]).unwrap();
    assert_eq!(t.dims(), [8]);

    let n = 4usize;
    let d: Tensor<RuntimeShape1, _, f32, NoGrad> = Cpu.zeros(shape![n]).unwrap();
    assert_eq!(d.dims().as_ref(), &[4]);
}

/// `shape![]` is rank 0, matching `s![]`. A scalar has one element, not zero.
#[test]
fn an_empty_axis_list_is_a_rank_zero_shape() {
    let t = Cpu.zeros(shape![]).unwrap();
    assert_eq!(t.dims().as_ref(), &[] as &[usize]);
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![0.0_f32]);
}

#[test]
fn higher_rank_mixes_static_and_runtime_axes_positionally() {
    let batch = 2usize;
    let width = 5usize;
    // Channels stay static; batch and width do not.
    let t: Tensor<RuntimeShape4, _, f32, NoGrad> = Cpu.zeros(shape![batch, 3, 28, width]).unwrap();
    assert_eq!(t.dims().as_ref(), &[2, 3, 28, 5]);
}

#[test]
fn a_usize_suffixed_literal_is_still_a_static_axis() {
    let t: Tensor<s![2, 3], _, f32, NoGrad> = Cpu.zeros(shape![2usize, 3]).unwrap();
    assert_eq!(t.dims(), [2, 3]);
}

/// `shape!` carries geometry only. Generated tensors take the *target's* float
/// dtype, so the same shape produces different element types from different
/// targets. The shape argument has no say in it.
#[test]
fn dtype_comes_from_the_target_not_the_shape() {
    let as_f32: Tensor<s![2, 2], _, f32, NoGrad> = Cpu.zeros(shape![2, 2]).unwrap();
    assert_eq!(as_f32.to_vec1::<f32>().unwrap(), vec![0.0_f32; 4]);

    let fp64 = Cpu.dtype::<f64>().unwrap();
    let as_f64: Tensor<s![2, 2], _, f64, NoGrad> = fp64.zeros(shape![2, 2]).unwrap();
    assert_eq!(as_f64.to_vec1::<f64>().unwrap(), vec![0.0_f64; 4]);
}

#[test]
fn shape_works_with_the_canonical_lowering_path_too() {
    let batch = 3usize;
    let t: Tensor<DimCons<usize, DimCons<typenum::U4, Nil>>, _, f32, NoGrad> =
        Cpu.zeros(shape![batch, 4]).unwrap();
    assert_eq!(t.dims().as_ref(), &[3, 4]);
}

#[test]
fn shape_macro_compile_fail_diagnostics() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/shape_compile_fail/*.rs");
}
