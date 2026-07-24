//! Compile-pass proofs for `docs/growth/03-named-dimensions.md` Task 03.1's
//! name-preservation audit. Each test's real assertion is the *type
//! ascription* on the result binding — if the op didn't preserve/compute
//! the named dim the way claimed, the file fails to compile, not just the
//! runtime `assert_eq!`.

extern crate incin_core as incin;
use incin_core::prelude::dummy::DummyBackend;
use incin_core::prelude::*;
use incin_macros::s;

incin_core::symbolic_dim!(Batch, Feature);

type TestBackend = DummyBackend<f32, Cpu>;

#[test]
fn transpose_swaps_and_preserves_both_named_dims() {
    let t: Tensor<s![Batch, Feature], TestBackend> = Tensor::zeros((4usize, 8usize)).unwrap();
    // Would fail to compile if `transpose` didn't produce exactly `(Feature, Batch)`.
    let transposed: Tensor<s![Feature, Batch], TestBackend> = t.transpose::<0, 1>().unwrap();
    let dims: Vec<usize> = transposed.dims().into();
    assert_eq!(dims, vec![8, 4]);
}

#[test]
fn sum_dim_over_a_named_axis_preserves_the_other_named_axis() {
    let t: Tensor<s![Batch, Feature], TestBackend> = Tensor::zeros((4usize, 8usize)).unwrap();
    // `ReduceDim` is bounded only by `Dim` (no `StaticDim`/`Unsigned`), so
    // reducing away the `Batch` axis while keeping `Feature` typed compiles.
    let summed: Tensor<s![Feature], TestBackend> = t.sum_dim::<0>().unwrap();
    let dims: Vec<usize> = summed.dims().into();
    assert_eq!(dims, vec![8]);
}

/// Regression test for a real bug found while writing this audit: `concat`
/// used to build its output shape `Field` via `Default::default()` instead
/// of the operands' real dims, which is invisible for pure-typenum shapes
/// (a zero-sized `PhantomData` either way) but silently zeroed any
/// runtime-carrying dimension — including a `symbolic_dim!` name, whose
/// `Default` is `Self(0)`. Without the fix this test's `dims` would read
/// `[0, 12]`, not `[2, 12]`: the type-level assertion below would still
/// compile (the *name* `Batch` was never lost), but the tensor would
/// silently lie about its own runtime batch size.
#[test]
fn concat_along_a_literal_axis_preserves_a_named_dim_on_the_other_axis() {
    let a: Tensor<s![Batch, 4], TestBackend> = Tensor::zeros((2usize, ())).unwrap();
    let b: Tensor<s![Batch, 8], TestBackend> = Tensor::zeros((2usize, ())).unwrap();
    // `ConcatShape`'s non-concatenated-axis position is bounded only by
    // `Dim`, so `Batch` survives into the joined output's type.
    let joined: Tensor<s![Batch, 12], TestBackend> =
        a.concat::<s![Batch, 8], typenum::U1>(&b).unwrap();
    let dims: Vec<usize> = joined.dims().into();
    assert_eq!(dims, vec![2, 12]);
}

/// Same bug class, same fix, on `stack` (which inserts a new size-2 axis
/// rather than summing along an existing one) — without the fix this
/// would read `[0, 2, 8]`, not `[4, 2, 8]`.
#[test]
fn stack_preserves_a_named_dim_and_inserts_the_new_axis_at_the_right_position() {
    let a: Tensor<s![Batch, Feature], TestBackend> = Tensor::zeros((4usize, 8usize)).unwrap();
    let b: Tensor<s![Batch, Feature], TestBackend> = Tensor::zeros((4usize, 8usize)).unwrap();
    let stacked: Tensor<s![Batch, 2, Feature], TestBackend> = a.stack::<typenum::U1>(&b).unwrap();
    let dims: Vec<usize> = stacked.dims().into();
    assert_eq!(dims, vec![4, 2, 8]);
}

// ─────────────────────────────────────────────────────────────────────────
// matmul: previously entirely unsupported for named dims (a hard `E0599`,
// per `docs/growth/03-named-dimensions.md`'s original audit). `MatMulShape`
// now accepts named dims via the dedicated `MatMulDim` marker (see
// `tensor/matmul.rs` — kept separate from `StaticDim` deliberately, to
// avoid also silently exposing `BroadcastShape`/`conv2d`, which share
// `StaticDim` and have an *unaudited* instance of the exact same
// `Default::default()` bug fixed below for matmul).
// ─────────────────────────────────────────────────────────────────────────

/// Plain 2D matmul with a named `M` dimension: `(Batch, 4) x (4, 8) ->
/// (Batch, 8)`. Without the `output_shape` fix this would compile (the
/// *type* `Batch` is never lost) but report `dims() == [0, 8]`, not `[3,
/// 8]` — the same silent-zeroing bug class as `concat`/`stack`.
#[test]
fn matmul_2d_with_named_m_dim_preserves_it_with_correct_runtime_value() {
    let a: Tensor<s![Batch, 4], TestBackend> = Tensor::zeros((3usize, ())).unwrap();
    let b: Tensor<s![4, 8], TestBackend> = Tensor::zeros(()).unwrap();
    let out: Tensor<s![Batch, 8], TestBackend> = a.matmul(&b).unwrap();
    let dims: Vec<usize> = out.dims().into();
    assert_eq!(dims, vec![3, 8]);
}

/// Batched matmul, named batch dim shared by both operands (the "both have
/// the same batch" `impl_batched_matmul!` variant).
#[test]
fn matmul_batched_with_named_batch_dim_on_both_operands() {
    let a: Tensor<s![Batch, 3, 4], TestBackend> = Tensor::zeros((2usize, (), ())).unwrap();
    let b: Tensor<s![Batch, 4, 5], TestBackend> = Tensor::zeros((2usize, (), ())).unwrap();
    let out: Tensor<s![Batch, 3, 5], TestBackend> = a.matmul(&b).unwrap();
    let dims: Vec<usize> = out.dims().into();
    assert_eq!(dims, vec![2, 3, 5]);
}

/// Batched matmul, named batch dim only on `lhs` (the "lhs has batch"
/// `impl_batched_matmul!` variant) — `rhs` is a plain, unbatched 2D weight.
#[test]
fn matmul_batched_with_named_batch_dim_only_on_lhs() {
    let a: Tensor<s![Batch, 3, 4], TestBackend> = Tensor::zeros((2usize, (), ())).unwrap();
    let b: Tensor<s![4, 5], TestBackend> = Tensor::zeros(()).unwrap();
    let out: Tensor<s![Batch, 3, 5], TestBackend> = a.matmul(&b).unwrap();
    let dims: Vec<usize> = out.dims().into();
    assert_eq!(dims, vec![2, 3, 5]);
}

/// Batched matmul, named batch dim only on `rhs` (the "rhs has batch"
/// `impl_batched_matmul!` variant) — this is the variant with the
/// self/rhs field-access mixup caught and fixed while writing this test.
#[test]
fn matmul_batched_with_named_batch_dim_only_on_rhs() {
    let a: Tensor<s![3, 4], TestBackend> = Tensor::zeros(()).unwrap();
    let b: Tensor<s![Batch, 4, 5], TestBackend> = Tensor::zeros((2usize, (), ())).unwrap();
    let out: Tensor<s![Batch, 3, 5], TestBackend> = a.matmul(&b).unwrap();
    let dims: Vec<usize> = out.dims().into();
    assert_eq!(dims, vec![2, 3, 5]);
}

// ─────────────────────────────────────────────────────────────────────────
// broadcast (`+`/`-`/`*`/`/`, `.broadcast_add()`): previously entirely
// unsupported for named dims — `BroadcastShape` was `StaticDim`-only, and a
// named dim is neither `StaticDim` nor `usize`. Every impl in
// `shapes/broadcast.rs` (~60 of them) now routes through a shared
// `broadcast_dims` helper bounded by the same `StaticOrNamedDim` marker
// introduced for matmul, fixing the identical `Default::default()` bug
// found there along the way.
// ─────────────────────────────────────────────────────────────────────────

/// The operator overload, not just the strict `.add()` — this is the exact
/// case flagged as unsupported in the original audit.
#[test]
fn plus_operator_works_between_two_identically_named_dim_tensors() {
    let a: Tensor<s![Batch, Feature], TestBackend> = Tensor::zeros((3usize, 5usize)).unwrap();
    let b: Tensor<s![Batch, Feature], TestBackend> = Tensor::zeros((3usize, 5usize)).unwrap();
    let sum: Tensor<s![Batch, Feature], TestBackend> = &a + &b;
    let dims: Vec<usize> = sum.dims().into();
    assert_eq!(dims, vec![3, 5]);
}

/// Rank-mismatch ("prepend") broadcasting with a named dim on the longer
/// side: `(Feature,) broadcast (Batch, Feature) -> (Batch, Feature)`, using
/// `.broadcast_add()` directly (the method the `+` operator delegates to).
#[test]
fn broadcast_add_prepends_a_named_leading_dim_and_preserves_it() {
    let bias: Tensor<s![Feature], TestBackend> = Tensor::zeros((5usize,)).unwrap();
    let x: Tensor<s![Batch, Feature], TestBackend> = Tensor::zeros((3usize, 5usize)).unwrap();
    let out: Tensor<s![Batch, Feature], TestBackend> = x.broadcast_add(&bias).unwrap();
    let dims: Vec<usize> = out.dims().into();
    assert_eq!(dims, vec![3, 5]);
}

/// A `usize` leading axis alongside a named tail dim, both operands the
/// same shape — exercises the `impl_broadcast_usize_same_rank!` family.
#[test]
fn broadcast_add_with_usize_leading_axis_and_named_tail_dim() {
    let a: Tensor<s![dyn, Feature], TestBackend> = Tensor::zeros((3usize, 5usize)).unwrap();
    let b: Tensor<s![dyn, Feature], TestBackend> = Tensor::zeros((3usize, 5usize)).unwrap();
    let out: Tensor<s![dyn, Feature], TestBackend> = a.broadcast_add(&b).unwrap();
    let dims: Vec<usize> = out.dims().into();
    assert_eq!(dims, vec![3, 5]);
}

/// A `symbolic_dim!` name does not guarantee the *same runtime value* on
/// both operands the way a `typenum` dim does (two different `Batch`
/// instances can legitimately hold different numbers) — `checked_
/// broadcast_dim` is the actual safety net for that case, and this proves
/// it still fires: two `Batch`-typed axes with disagreeing real sizes,
/// neither equal to 1, must panic rather than silently produce a wrong
/// shape (which `Default::default()` would have done before this fix).
#[test]
#[should_panic(expected = "cannot broadcast dynamic dimension")]
fn broadcast_add_panics_on_disagreeing_same_named_type_dims() {
    let a: Tensor<s![Batch, Feature], TestBackend> = Tensor::zeros((3usize, 5usize)).unwrap();
    let b: Tensor<s![Batch, Feature], TestBackend> = Tensor::zeros((4usize, 5usize)).unwrap();
    let _ = a.broadcast_add(&b);
}

// ─────────────────────────────────────────────────────────────────────────
// Pooling / conv: `Pool2dShape` already bounded batch/channel on plain
// `Dim` (not `StaticDim`) and already used real values there — verified
// working, not fixed. `SpatialConv1d`/`SpatialConv2d` (`shapes/spatial.rs`)
// and `KernelConv2dShape` (`tensor/conv2d.rs`) *did* have the
// `Default::default()` bug on their batch/`COut` positions, both fixed.
// ─────────────────────────────────────────────────────────────────────────

incin_core::symbolic_dim!(Channel);

/// `Pool2dShape` needed no fix — confirms it, rather than assumes it.
#[test]
fn max_pool2d_preserves_named_batch_and_channel_dims() {
    let t: Tensor<s![Batch, Channel, 8, 8], TestBackend> = Tensor::zeros((2usize, 3usize)).unwrap();
    let pooled: Tensor<s![Batch, Channel, 4, 4], TestBackend> = t
        .max_pool2d::<typenum::U2, typenum::U2, typenum::U0, typenum::U1>()
        .unwrap();
    let dims: Vec<usize> = pooled.dims().into();
    assert_eq!(dims, vec![2, 3, 4, 4]);
}

/// Regression test for the real bug found in `KernelConv2dShape`
/// (`tensor/conv2d.rs`): its batch position (`B: Dim + Default`, generic —
/// unlike `COut`/`CIn`/`H`/`W`, which are all `StaticDim`/typenum) built its
/// output field via `Default::default()`. Without the fix this would read
/// `dims() == [0, 4, 6, 6]`, not `[2, 4, 6, 6]`.
#[test]
fn tensor_conv2d_preserves_a_named_batch_dim() {
    let x: Tensor<s![Batch, 3, 8, 8], TestBackend> = Tensor::zeros((2usize,)).unwrap();
    let weight: Tensor<s![4, 3, 3, 3], TestBackend> = Tensor::zeros(()).unwrap();
    let out: Tensor<s![Batch, 4, 6, 6], TestBackend> = x
        .conv2d::<typenum::U1, typenum::U0, _>(&weight, None)
        .unwrap();
    let dims: Vec<usize> = out.dims().into();
    assert_eq!(dims, vec![2, 4, 6, 6]);
}

/// Regression test for the analogous fix in `SpatialConv2d`
/// (`shapes/spatial.rs`'s `impl_conv2d_shape!` macro), exercised through the
/// *actual* `nn::Conv2d` layer (not the raw `Tensor::conv2d` op above — a
/// different trait, same bug, same fix) on a real backend, so this checks
/// genuine forward-pass output, not just shape metadata. Without the fix,
/// `.dims()` would read `[0, 5, 6, 6]`, not `[2, 5, 6, 6]`.
#[test]
fn nn_conv2d_layer_preserves_a_named_batch_dim() {
    type ConvShape = (
        typenum::U5, // COut
        typenum::U3, // CIn
        typenum::U3, // K
        typenum::U1, // S
        typenum::U0, // P
        typenum::U1, // D
    );
    let conv = Conv2d::<ConvShape, incin_backends::cpu::CpuBackendImpl>::build(()).unwrap();
    let x: Tensor<s![Batch, 3, 8, 8], incin_backends::cpu::CpuBackendImpl> =
        Tensor::zeros((2usize,)).unwrap();
    let out: Tensor<s![Batch, 5, 6, 6], incin_backends::cpu::CpuBackendImpl> =
        conv.forward(x).unwrap();
    let dims: Vec<usize> = out.dims().into();
    assert_eq!(dims, vec![2, 5, 6, 6]);
}
