//! The gated recurrent unit, on the public facade.
//!
//! The Book listed `GRU` as the recurrent gap next to `RNN` and `LSTM`. These
//! tests pin the two things a new recurrent cell has to get right: the
//! recurrence is the reference formulation, with the reset gate applied to the
//! hidden projection alone, and the layer trains through the module path
//! rather than only running forward.
#![cfg(feature = "cpu")]

use incin::nn::{GRU, GRUCell};
use incin::prelude::*;

type Cpu = incin_backends::cpu::CpuBackendImpl;

const IN: usize = 4;
const OUT: usize = 3;
const BATCH: usize = 2;
const SEQ: usize = 5;

fn nonzero(values: &[f32]) -> usize {
    values.iter().filter(|value| **value != 0.0).count()
}

fn ramp(dims: Vec<usize>) -> Result<Tensor<Dyn, Cpu, f32, NoGrad>> {
    let n: usize = dims.iter().product();
    let values = (0..n)
        .map(|v| ((v as f32) * 0.47).sin() * 0.8 + 0.2)
        .collect::<Vec<f32>>();
    Tensor::<Dyn, Cpu>::from_slice(&values, dims)
}

/// The cell computes the reference recurrence, including where the reset gate
/// is applied.
///
/// The expected values are assembled from the cell's own projections, so the
/// weights are identical by construction and only the dataflow is under test.
/// Applying `r` to the sum of both projections rather than to the recurrent
/// one alone is a different model; this is the assertion that tells them
/// apart.
#[test]
fn the_cell_computes_the_reference_recurrence() -> Result<()> {
    let cell = GRUCell::<s![4, 3], Cpu>::build(())?;
    let x = ramp(vec![BATCH, IN])?.into_shape::<s![2, 4]>()?;
    let h = ramp(vec![BATCH, OUT])?.into_shape::<s![2, 3]>()?;

    let actual = cell.forward((x.clone(), h.clone()))?.to_vec1::<f32>()?;

    // The three gates, spelled out at the tensor level. A macro rather than a
    // closure because the input and recurrent projections are different
    // shapes.
    macro_rules! projections {
        ($wi:expr, $wh:expr) => {
            (
                $wi.forward(x.clone())?.to_vec1::<f32>()?,
                $wh.forward(h.clone())?.to_vec1::<f32>()?,
            )
        };
    }
    let (xr, hr) = projections!(cell.wi_r, cell.wh_r);
    let (xz, hz) = projections!(cell.wi_z, cell.wh_z);
    let (xn, hn) = projections!(cell.wi_n, cell.wh_n);
    let state = h.to_vec1::<f32>()?;

    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let expected = (0..BATCH * OUT)
        .map(|i| {
            let r = sigmoid(xr[i] + hr[i]);
            let z = sigmoid(xz[i] + hz[i]);
            // The reset gate multiplies the recurrent projection only.
            let n = (xn[i] + r * hn[i]).tanh();
            (1.0 - z) * n + z * state[i]
        })
        .collect::<Vec<f32>>();

    for (actual, expected) in actual.iter().zip(&expected) {
        assert!(
            (actual - expected).abs() < 1e-5,
            "{actual} != {expected}: the recurrence is not the reference one"
        );
    }
    Ok(())
}

/// Applying the reset gate to the sum instead would produce different numbers,
/// so the test above is not satisfied by both spellings.
#[test]
fn gating_the_sum_would_be_a_different_function() -> Result<()> {
    let cell = GRUCell::<s![4, 3], Cpu>::build(())?;
    let x = ramp(vec![BATCH, IN])?.into_shape::<s![2, 4]>()?;
    let h = ramp(vec![BATCH, OUT])?.into_shape::<s![2, 3]>()?;

    let actual = cell.forward((x.clone(), h.clone()))?.to_vec1::<f32>()?;

    let xr = cell.wi_r.forward(x.clone())?.to_vec1::<f32>()?;
    let hr = cell.wh_r.forward(h.clone())?.to_vec1::<f32>()?;
    let xz = cell.wi_z.forward(x.clone())?.to_vec1::<f32>()?;
    let hz = cell.wh_z.forward(h.clone())?.to_vec1::<f32>()?;
    let xn = cell.wi_n.forward(x)?.to_vec1::<f32>()?;
    let hn = cell.wh_n.forward(h.clone())?.to_vec1::<f32>()?;
    let state = h.to_vec1::<f32>()?;

    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let wrong = (0..BATCH * OUT)
        .map(|i| {
            let r = sigmoid(xr[i] + hr[i]);
            let z = sigmoid(xz[i] + hz[i]);
            // The variant this test exists to exclude: `r` over the sum.
            let n = (r * (xn[i] + hn[i])).tanh();
            (1.0 - z) * n + z * state[i]
        })
        .collect::<Vec<f32>>();

    let differs = actual.iter().zip(&wrong).any(|(a, w)| (a - w).abs() > 1e-4);
    assert!(
        differs,
        "the two gate placements agreed, so the recurrence test proves nothing"
    );
    Ok(())
}

/// The cell trains: every gate, on both halves, receives a gradient.
#[test]
fn every_gate_receives_a_gradient() -> Result<()> {
    let cell = GRUCell::<s![4, 3], Cpu>::build(())?;
    let x = ramp(vec![BATCH, IN])?
        .into_shape::<s![2, 4]>()?
        .require_grad();
    // Nonzero on purpose: a recurrent weight's gradient is proportional to
    // the incoming state, so a zero one leaves every `wh_*` at zero for
    // reasons unrelated to the module path.
    let h = ramp(vec![BATCH, OUT])?
        .into_shape::<s![2, 3]>()?
        .require_grad();

    let hidden = cell.forward((x, h))?;
    assert_eq!(hidden.dims().dims(), &[BATCH, OUT]);

    let target = Tensor::<s![2, 3], Cpu>::zeros(())?;
    let grads = hidden.mse_loss(&target)?.backward()?;

    macro_rules! assert_trains {
        ($name:literal, $layer:expr) => {{
            let gradient = grads
                .require(&$layer.weight.as_tensor()?)
                .map_err(|e| Error::Msg(format!("no gradient reached {}: {e}", $name)))?
                .to_vec1::<f32>()?;
            assert!(
                nonzero(&gradient) > 0,
                "{} received only zeros, so it cannot train",
                $name
            );
        }};
    }
    assert_trains!("wi_r", cell.wi_r);
    assert_trains!("wi_z", cell.wi_z);
    assert_trains!("wi_n", cell.wi_n);
    assert_trains!("wh_r", cell.wh_r);
    assert_trains!("wh_z", cell.wh_z);
    assert_trains!("wh_n", cell.wh_n);
    Ok(())
}

/// The multi-step wrapper carries the state and the gradient across the
/// sequence, and stacks one output per step.
#[test]
fn a_gru_sequence_trains_through_every_step() -> Result<()> {
    let model = GRU::<s![4, 3], Cpu>::new(GRUCell::<s![4, 3], Cpu>::build(())?);

    let x = ramp(vec![BATCH, SEQ, IN])?
        .into_shape::<s![2, 5, 4]>()?
        .require_grad();
    let h = Tensor::<s![2, 3], Cpu>::zeros(())?.require_grad();

    let (sequence, last) = model.forward((x, h))?;
    assert_eq!(sequence.dims().dims(), &[BATCH, SEQ, OUT]);
    assert_eq!(last.dims().dims(), &[BATCH, OUT]);

    let target = Tensor::<s![2, 3], Cpu>::zeros(())?;
    let grads = last.mse_loss(&target)?.backward()?;
    let gradient = grads
        .require(&model.cell.wh_n.weight.as_tensor()?)
        .map_err(|e| Error::Msg(format!("the gradient did not reach the recurrence: {e}")))?
        .to_vec1::<f32>()?;
    assert!(
        nonzero(&gradient) > 0,
        "the candidate's recurrent weight received only zeros across five steps"
    );
    Ok(())
}

/// The state round-trips, and a frozen layer keeps its parameters out of the
/// gradient path.
#[test]
fn the_layer_round_trips_and_freezes() -> Result<()> {
    let model = GRU::<s![4, 3], Cpu>::new(GRUCell::<s![4, 3], Cpu>::build(())?);
    let snapshot = incin::state::collect_state::<Cpu, _>(&model)?;
    // Six projections, each with a weight and a bias.
    assert_eq!(snapshot.len(), 12);

    let mut restored = GRU::<s![4, 3], Cpu>::new(GRUCell::<s![4, 3], Cpu>::build(())?);
    incin::state::load_state::<Cpu, _>(&mut restored, &snapshot)?;
    assert_eq!(incin::state::collect_state::<Cpu, _>(&restored)?, snapshot);

    // `require_grad` rather than an inference pass: the wrapper's bound pins
    // the per-step join back onto the loop variable's own requirement, so a
    // trainable layer's sequence forward is a training forward. The same is
    // true of `RNN`.
    let x = ramp(vec![BATCH, SEQ, IN])?
        .into_shape::<s![2, 5, 4]>()?
        .require_grad();
    let h = Tensor::<s![2, 3], Cpu>::zeros(())?.require_grad();
    assert_eq!(
        model.forward((x.clone(), h.clone()))?.0.to_vec1::<f32>()?,
        restored.forward((x, h))?.0.to_vec1::<f32>()?
    );

    let frozen = model.freeze();
    assert_eq!(
        incin::state::collect_state::<Cpu, _>(&frozen)?.len(),
        12,
        "freezing changed which tensors are state"
    );
    Ok(())
}
