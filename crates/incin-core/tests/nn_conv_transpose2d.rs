//! ConvTranspose2d module tests (backend-exercising).
//!
//! These live in integration tests rather than in-crate unit tests:
//! naming a downstream backend impl (e.g. `CpuBackendImpl<Cpu>`) inside
//! `src/` resolves the test build of `incin_core` against the normal
//! build the backend was compiled against, so the trait never resolves.
//! As a separate crate target both builds coincide and the impl applies.

use incin_backends::cpu::CpuBackendImpl;
use incin_core::dist::Local;
use incin_core::exec::catalog::ConvTranspose2dAttributes;
use incin_core::exec::catalog::op;
use incin_core::exec::context::ExecutionContext;
use incin_core::exec::request::TensorHandle;
use incin_core::nn::conv_transpose2d::ConvTranspose2dShape;
use incin_core::nn::{ConvTranspose2d, Module, Param, Trainable};
use incin_core::nn::{
    StatePath, collect_state,
    optional::{False, True},
};
use incin_core::shapes::{DimCons, Nil};
use incin_core::shapes::{ShapeBuf, ShapeValue};
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::Cpu;
use typenum::consts::{U0, U1, U2};

type B = CpuBackendImpl<Cpu>;
type N1222 = DimCons<U1, DimCons<U2, DimCons<U2, DimCons<U2, Nil>>>>;
type N1122 = DimCons<U1, DimCons<U1, DimCons<U2, DimCons<U2, Nil>>>>;
type N1111 = DimCons<U1, DimCons<U1, DimCons<U1, DimCons<U1, Nil>>>>;
type N222 = DimCons<U2, DimCons<U2, DimCons<U2, Nil>>>;
type N11112 = DimCons<U1, DimCons<U1, DimCons<U1, DimCons<U1, DimCons<U2, Nil>>>>>;

/// (InC=2, OutC=1, K=1, S=1, P=0, OP=0, D=1).
type T11 =
    DimCons<U2, DimCons<U1, DimCons<U1, DimCons<U1, DimCons<U0, DimCons<U0, DimCons<U1, Nil>>>>>>>;
/// (InC=1, OutC=1, K=1, S=2, P=0, OP=0, D=1).
type TS2 =
    DimCons<U1, DimCons<U1, DimCons<U1, DimCons<U2, DimCons<U0, DimCons<U0, DimCons<U1, Nil>>>>>>>;
/// (InC=1, OutC=1, K=1, S=2, P=0, OP=1, D=1).
type TOP1 =
    DimCons<U1, DimCons<U1, DimCons<U1, DimCons<U2, DimCons<U0, DimCons<U1, DimCons<U1, Nil>>>>>>>;

type W11 = <T11 as ConvTranspose2dShape>::WeightShape;
type B11 = <T11 as ConvTranspose2dShape>::BiasShape;

fn const_weight() -> Param<W11, B, f32, Trainable> {
    Param::<W11, B, f32, Trainable>::new_init((), incin_core::nn::init::constant(1.0)).unwrap()
}

fn const_bias(v: f64) -> Param<B11, B, f32, Trainable> {
    Param::<B11, B, f32, Trainable>::new_init((), incin_core::nn::init::constant(v)).unwrap()
}

#[track_caller]
fn assert_close(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() <= tol,
            "element {i}: got {g:.6}, expected {e:.6}"
        );
    }
}

#[test]
fn kernel_one_stride_one_sums_channels() {
    // K=1, S=1: out[n, 0, h, w] = x[n, 0, h, w] + x[n, 1, h, w].
    let m = ConvTranspose2d::<T11, B, True>::from_raw_parts(const_weight(), None, 1).unwrap();
    let x =
        Tensor::<N1222, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], ()).unwrap();
    let out = m.forward(x).unwrap();
    assert_eq!(out.dims().as_ref(), &[1, 1, 2, 2]);
    assert_close(
        &out.to_vec1::<f32>().unwrap(),
        &[6.0, 8.0, 10.0, 12.0],
        1e-5,
    );
}

#[test]
fn bias_adds_per_output_channel() {
    let m =
        ConvTranspose2d::<T11, B, True>::from_raw_parts(const_weight(), Some(const_bias(0.5)), 1)
            .unwrap();
    let x =
        Tensor::<N1222, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], ()).unwrap();
    let out = m.forward(x).unwrap();
    assert_close(
        &out.to_vec1::<f32>().unwrap(),
        &[6.5, 8.5, 10.5, 12.5],
        1e-5,
    );
}

#[test]
fn stride_two_upsamples_with_zeros() {
    type W = <TS2 as ConvTranspose2dShape>::WeightShape;
    let weight =
        Param::<W, B, f32, Trainable>::new_init((), incin_core::nn::init::constant(1.0)).unwrap();
    let m = ConvTranspose2d::<TS2, B, True>::from_raw_parts(weight, None, 1).unwrap();
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    let out = m.forward(x).unwrap();
    // (2 - 1) * 2 + 1 = 3: stride-2 with a K=1 kernel spreads each input
    // to every second position, it does not double the extent.
    assert_eq!(out.dims().as_ref(), &[1, 1, 3, 3]);
    assert_close(
        &out.to_vec1::<f32>().unwrap(),
        &[
            1.0, 0.0, 2.0, //
            0.0, 0.0, 0.0, //
            3.0, 0.0, 4.0,
        ],
        1e-5,
    );
}

#[test]
fn output_padding_appends_trailing_zeros() {
    type W = <TOP1 as ConvTranspose2dShape>::WeightShape;
    let weight =
        Param::<W, B, f32, Trainable>::new_init((), incin_core::nn::init::constant(1.0)).unwrap();
    let m = ConvTranspose2d::<TOP1, B, True>::from_raw_parts(weight, None, 1).unwrap();
    let x = Tensor::<N1111, B>::from_slice(&[2.0f32], ()).unwrap();
    let out = m.forward(x).unwrap();
    assert_eq!(out.dims().as_ref(), &[1, 1, 2, 2]);
    assert_close(&out.to_vec1::<f32>().unwrap(), &[2.0, 0.0, 0.0, 0.0], 1e-5);
}

#[test]
fn unbatched_rank_three_input() {
    let m = ConvTranspose2d::<T11, B, True>::from_raw_parts(const_weight(), None, 1).unwrap();
    let x =
        Tensor::<N222, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], ()).unwrap();
    let out = m.forward(x).unwrap();
    assert_eq!(out.dims().as_ref(), &[1, 2, 2]);
    assert_close(
        &out.to_vec1::<f32>().unwrap(),
        &[6.0, 8.0, 10.0, 12.0],
        1e-5,
    );
}

#[test]
fn parity_with_direct_op_dispatch() {
    let m = ConvTranspose2d::<TS2, B, True>::from_raw_parts(
        Param::<<TS2 as ConvTranspose2dShape>::WeightShape, B, f32, Trainable>::new_init(
            (),
            incin_core::nn::init::constant(1.0),
        )
        .unwrap(),
        None,
        1,
    )
    .unwrap();
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    let got = m.forward(x).unwrap().to_vec1::<f32>().unwrap();

    // The same op invoked by hand with identical attributes.
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    let w = m.weight.as_tensor().unwrap().into_dyn();
    let shape = ShapeValue::<incin_core::shapes::Dyn>::try_new(ShapeBuf::from_slice(&[1, 1, 3, 3]))
        .unwrap();
    let inputs = [
        TensorHandle::from_storage::<B, f32, Local>(x.inner()),
        TensorHandle::from_storage::<B, f32, Local>(w.inner()),
    ];
    let context = ExecutionContext::from_scope(B::default());
    let out = incin_core::exec::dispatch::execute_shaped::<
        op::ConvTranspose2d,
        B,
        incin_core::shapes::Dyn,
    >(
        &context,
        ConvTranspose2dAttributes {
            stride: [2, 2],
            padding: [0, 0],
            output_padding: [0, 0],
            dilation: [1, 1],
            groups: 1,
            has_bias: false,
        },
        &inputs,
        &shape,
    )
    .unwrap();
    // Read back elementwise through the public storage reader: rebuilding
    // a Tensor around the raw output would need the crate-private
    // shape-value constructor.
    let dims = out.shape.dims().to_vec();
    let total: usize = dims.iter().product();
    let mut idx = vec![0usize; dims.len()];
    let mut expected = Vec::with_capacity(total);
    for _ in 0..total {
        expected.push(out.get(&idx) as f32);
        for (i, extent) in idx.iter_mut().zip(dims.iter()).rev() {
            *i += 1;
            if *i < *extent {
                break;
            }
            *i = 0;
        }
    }
    assert_close(&got, &expected, 1e-6);
}

#[test]
fn unsupported_configs_are_typed_refusals() {
    // Grouped transposed convolution is unimplemented: refused at build.
    assert!(ConvTranspose2d::<T11, B, True>::build(2usize).is_err());
    // ... and at raw-part construction.
    assert!(ConvTranspose2d::<T11, B, True>::from_raw_parts(const_weight(), None, 2).is_err());
    // Mismatched input channels are refused in forward.
    let m = ConvTranspose2d::<T11, B, True>::from_raw_parts(const_weight(), None, 1).unwrap();
    let x = Tensor::<N1122, B>::from_slice(&[1.0f32, 2.0, 3.0, 4.0], ()).unwrap();
    assert!(m.forward(x).is_err());
    // Ranks outside 3..=4 are refused in forward.
    let x = Tensor::<N11112, B>::from_slice(&[1.0f32, 2.0], ()).unwrap();
    assert!(m.forward(x).is_err());
}

#[test]
fn state_dict_registers_weight_and_bias() {
    let m =
        ConvTranspose2d::<T11, B, True>::from_raw_parts(const_weight(), Some(const_bias(0.0)), 1)
            .unwrap();
    let snap = collect_state::<B, _>(&m).unwrap();
    assert!(
        snap.get(&StatePath::root().try_child("weight").unwrap())
            .is_some()
    );
    assert!(
        snap.get(&StatePath::root().try_child("bias").unwrap())
            .is_some()
    );

    let m = ConvTranspose2d::<T11, B, False>::from_raw_parts(const_weight(), None, 1).unwrap();
    let snap = collect_state::<B, _>(&m).unwrap();
    assert!(
        snap.get(&StatePath::root().try_child("weight").unwrap())
            .is_some()
    );
    assert!(
        snap.get(&StatePath::root().try_child("bias").unwrap())
            .is_none()
    );
}
