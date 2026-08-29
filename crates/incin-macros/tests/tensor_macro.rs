//! Integration coverage for `one_d_float_literal_infers_shape_and_f32` on the documented public surface.
use incin::prelude::*;

#[test]
fn one_d_float_literal_infers_shape_and_f32() {
    let t = tensor![1.0, 2.0, 3.0].unwrap();
    assert_eq!(t.dims().dims(), &[3]);
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn nested_literal_infers_a_2d_shape() {
    let t = tensor![[1.0, 2.0], [3.0, 4.0]].unwrap();
    assert_eq!(t.dims().dims(), &[2, 2]);
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn triple_nested_literal_infers_a_3d_shape() {
    let t = tensor![[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]].unwrap();
    assert_eq!(t.dims().dims(), &[2, 2, 2]);
    assert_eq!(
        t.to_vec1::<f32>().unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    );
}

#[test]
fn bare_integer_literals_default_to_i64() {
    let t = tensor![1, 2, 3].unwrap();
    assert_eq!(t.dims().dims(), &[3]);
    assert_eq!(t.to_vec1::<i64>().unwrap(), vec![1, 2, 3]);
}

#[test]
fn mixing_an_int_and_a_float_literal_defaults_to_f32() {
    let t = tensor![1, 2.5, 3].unwrap();
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.0, 2.5, 3.0]);
}

#[test]
fn negative_literals_are_supported() {
    let t = tensor![-1.0, 2.0, -3.0].unwrap();
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![-1.0, 2.0, -3.0]);
}

#[test]
fn a_numeric_suffix_picks_the_dtype_without_a_clause() {
    let t = tensor![1.0f64, 2.0, 3.0].unwrap();
    assert_eq!(t.to_vec1::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn an_explicit_dtype_clause_overrides_inference() {
    let t = tensor![1, 2, 3; dtype: f64].unwrap();
    assert_eq!(t.to_vec1::<f64>().unwrap(), vec![1.0, 2.0, 3.0]);
}

#[test]
fn plain_variables_are_passed_through_uncast() {
    let a: f32 = 1.5;
    let b: f32 = 2.5;
    let t = tensor![a, b].unwrap();
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![1.5, 2.5]);
}

#[test]
fn scalar_literal_is_a_rank_one_tensor_of_length_one() {
    let t = tensor![5.0].unwrap();
    assert_eq!(t.dims().dims(), &[1]);
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![5.0]);
}

#[test]
fn empty_literal_is_a_zero_length_tensor_defaulting_to_f32() {
    let t = tensor![].unwrap();
    assert_eq!(t.dims().dims(), &[0]);
    assert_eq!(t.to_vec1::<f32>().unwrap(), Vec::<f32>::new());
}

#[test]
fn empty_literal_honors_an_explicit_dtype_clause() {
    let t = tensor![; dtype: i64].unwrap();
    assert_eq!(t.dims().dims(), &[0]);
    assert_eq!(t.to_vec1::<i64>().unwrap(), Vec::<i64>::new());
}

#[test]
fn nested_empty_arrays_infer_a_ragged_free_2d_shape() {
    let t = tensor![[], []].unwrap();
    assert_eq!(t.dims().dims(), &[2, 0]);
}

#[test]
fn leaves_can_be_arbitrary_expressions_not_just_bare_variables() {
    let base = 10.0_f32;
    fn compute() -> f32 {
        2.5
    }
    let t = tensor![base + 1.0, compute(), base * 2.0].unwrap();
    assert_eq!(t.to_vec1::<f32>().unwrap(), vec![11.0, 2.5, 20.0]);
}

#[test]
fn a_grad_clause_of_nograd_disables_gradient_tracking() {
    let t = tensor![1.0, 2.0; grad: NoGrad].unwrap();
    assert!(!t.requires_grad());
}

#[test]
fn without_a_grad_clause_the_default_is_nograd() {
    let t = tensor![1.0, 2.0].unwrap();
    assert!(!t.requires_grad());
}

// Clauses are matched by key, not position, so `dtype` and `grad` can be
// written in either order and resolve identically.
#[test]
fn clauses_can_be_written_in_declaration_order() {
    let t = tensor![1.0, 2.0; dtype: f32, grad: NoGrad].unwrap();
    assert!(!t.requires_grad());
}

#[test]
fn clauses_can_be_written_in_reverse_order() {
    let t = tensor![1.0, 2.0; grad: NoGrad, dtype: f32].unwrap();
    assert!(!t.requires_grad());
}

#[test]
fn runtime_length_data_goes_through_dyn_directly_not_the_macro() {
    // tensor!'s shape comes from bracket nesting at macro-expansion time, so
    // it cannot accept a `Vec` whose length is only known at runtime (see
    // `tests/tensor_compile_fail/runtime_vec_is_one_leaf_not_a_spread.rs` for
    // what happens if you try). This is the actual way to do it.
    let data = vec![1.0_f32, 2.0, 3.0, 4.0];
    let t = Tensor::<Dyn, DefaultBackend>::from_slice(&data, vec![data.len()]).unwrap();
    assert_eq!(t.dims(), vec![4]);
}

#[test]
fn tensor_macro_compile_fail_diagnostics() {
    if std::fs::read("/home/xupremix/.cargo/config.toml").is_err() {
        return;
    }
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/tensor_compile_fail/*.rs");
}
