//! An LSTM trains when used as a module.
//!
//! `LSTMCell` and `LSTM` implemented `Module` only for a `NoGrad` operand, so
//! neither could be fed a tensor that requires a gradient and neither could be
//! trained through the module path at all. `RNNCell`/`RNN` next to them were
//! already generic over the operand's requirement, so the gap was in the
//! gated cell alone and nothing in the tree said so.
//!
//! Both impls were also written against the static `(In, Out)` pair rather
//! than against the `LstmShape` trait the struct is parameterized by, so they
//! were narrower than the type they belonged to and narrower than `RNN`'s.
//! That is a consistency fix, not a new shape: the operand is still a
//! `[Batch, In]` pair, because the impl is written over `D2<Batch, S::In>`
//! and not over `Dyn`.
//!
//! The tests pin the halves separately: that the chain compiles and carries a
//! gradient, that one arrives at the recurrent weights, and that the values
//! did not move when the gradient type widened.
#![cfg(feature = "cpu")]

use incin::prelude::*;

type Cpu = incin_backends::cpu::CpuBackendImpl;

const IN: usize = 4;
const OUT: usize = 3;
const BATCH: usize = 2;

fn nonzero(values: &[f32]) -> usize {
    values.iter().filter(|value| **value != 0.0).count()
}

/// A ramp rather than zeros: the gates saturate at zero input and several of
/// the recurrent gradients would be zero for reasons unrelated to the fix.
fn ramp(dims: Vec<usize>) -> Result<Tensor<Dyn, Cpu, f32, NoGrad>> {
    let n: usize = dims.iter().product();
    let values = (0..n)
        .map(|v| ((v as f32) * 0.53).sin() * 0.8 + 0.3)
        .collect::<Vec<f32>>();
    Tensor::<Dyn, Cpu>::from_slice(&values, dims)
}

#[test]
fn an_lstm_cell_accepts_a_gradient_carrying_operand_and_trains() -> Result<()> {
    let cell = LSTMCell::<s![4, 3], Cpu>::build(())?;

    // The call that did not compile before.
    let x = ramp(vec![BATCH, IN])?
        .into_shape::<s![2, 4]>()?
        .require_grad();
    // A nonzero initial state on purpose: the gradient of a recurrent weight
    // is proportional to `h_prev`, so starting from zeros would leave every
    // `wh_*` gradient at zero for reasons that have nothing to do with the
    // module path.
    let h = ramp(vec![BATCH, OUT])?
        .into_shape::<s![2, 3]>()?
        .require_grad();
    let c = ramp(vec![BATCH, OUT])?
        .into_shape::<s![2, 3]>()?
        .require_grad();
    let (hidden, cell_state) = cell.forward((x, (h, c)))?;
    assert_eq!(hidden.dims().dims(), &[BATCH, OUT]);
    assert_eq!(cell_state.dims().dims(), &[BATCH, OUT]);

    let target = Tensor::<s![2, 3], Cpu>::zeros(())?;
    let grads = hidden.mse_loss(&target)?.backward()?;

    // The input-to-hidden and hidden-to-hidden weights are different shapes,
    // so they cannot share a loop; one gate from each half is checked.
    let input_gate = grads
        .require(&cell.wi_i.weight.as_tensor()?)
        .map_err(|e| Error::Msg(format!("no gradient reached wi_i.weight: {e}")))?
        .to_vec1::<f32>()?;
    assert!(
        nonzero(&input_gate) > 0,
        "wi_i.weight received only zeros, so it cannot train"
    );
    let recurrent_gate = grads
        .require(&cell.wh_o.weight.as_tensor()?)
        .map_err(|e| Error::Msg(format!("no gradient reached wh_o.weight: {e}")))?
        .to_vec1::<f32>()?;
    assert!(
        nonzero(&recurrent_gate) > 0,
        "wh_o.weight received only zeros, so the recurrence cannot train"
    );
    Ok(())
}

/// The multi-step wrapper carries the requirement across the sequence.
#[test]
fn an_lstm_sequence_trains_through_every_step() -> Result<()> {
    let cell = LSTMCell::<s![4, 3], Cpu>::build(())?;
    let model = LSTM::<s![4, 3], Cpu>::new(cell);

    let x = ramp(vec![BATCH, 5, IN])?
        .into_shape::<s![2, 5, 4]>()?
        .require_grad();
    let h = Tensor::<s![2, 3], Cpu>::zeros(())?.require_grad();
    let c = Tensor::<s![2, 3], Cpu>::zeros(())?.require_grad();

    let (sequence, (last_hidden, _)) = model.forward((x, (h, c)))?;
    assert_eq!(sequence.dims().dims(), &[BATCH, 5, OUT]);

    let target = Tensor::<s![2, 3], Cpu>::zeros(())?;
    let grads = last_hidden.mse_loss(&target)?.backward()?;
    let gradient = grads
        .require(&model.cell.wh_f.weight.as_tensor()?)
        .map_err(|e| Error::Msg(format!("the gradient did not reach the recurrence: {e}")))?
        .to_vec1::<f32>()?;
    assert!(
        nonzero(&gradient) > 0,
        "the forget gate's recurrent weight received only zeros across five steps"
    );
    Ok(())
}

/// Widening the gradient type moved no values: an inference-only pass still
/// computes the textbook recurrence.
#[test]
fn an_inference_pass_computes_the_same_gates() -> Result<()> {
    let cell = LSTMCell::<s![4, 3], Cpu>::build(())?;
    let x = ramp(vec![BATCH, IN])?.into_shape::<s![2, 4]>()?;
    let h = Tensor::<s![2, 3], Cpu>::zeros(())?;
    let c = Tensor::<s![2, 3], Cpu>::zeros(())?;

    let (hidden, cell_state) = cell.forward((x.clone(), (h.clone(), c.clone())))?;

    // The same arithmetic, spelled out at the tensor level. Written as a
    // macro rather than a closure because the two projections have different
    // shapes, so a closure would need both spelled out in its signature.
    macro_rules! gate {
        ($wi:expr, $wh:expr) => {
            $wi.forward(x.clone())?
                .add_exact(&$wh.forward(h.clone())?)?
                .to_vec1::<f32>()?
        };
    }
    let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
    let i: Vec<f32> = gate!(cell.wi_i, cell.wh_i);
    let g: Vec<f32> = gate!(cell.wi_g, cell.wh_g);
    let o: Vec<f32> = gate!(cell.wi_o, cell.wh_o);

    let expected_c = i
        .iter()
        .zip(&g)
        .map(|(i, g)| sigmoid(*i) * g.tanh())
        .collect::<Vec<f32>>();
    let expected_h = o
        .iter()
        .zip(&expected_c)
        .map(|(o, c)| sigmoid(*o) * c.tanh())
        .collect::<Vec<f32>>();

    for (actual, expected) in cell_state.to_vec1::<f32>()?.iter().zip(&expected_c) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    for (actual, expected) in hidden.to_vec1::<f32>()?.iter().zip(&expected_h) {
        assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
    }
    Ok(())
}
