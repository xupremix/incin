//! An embedding table trains when used as a module.
//!
//! `Embedding` implemented `Module` with a `NoGrad` output regardless of its
//! train state, so the lookup result carried no gradient requirement and the
//! table's rows received nothing. The CPU kernel had recorded a scatter-add
//! backward closure all along (`cpu/ops/embedding.rs`), and nothing could
//! reach it: a model whose first layer was an embedding could not learn its
//! vocabulary, which is most of what an embedding is for.
//!
//! This is the same shape of defect the normalization layers had, so the
//! tests are the same two halves: that a chain through the table compiles and
//! carries a gradient, and that a nonzero gradient actually arrives at the
//! rows that were looked up and nowhere else.
#![cfg(feature = "cpu")]

use incin::prelude::*;
use incin_core::dist::placement::Local;

type Cpu = incin_backends::cpu::CpuBackendImpl;

const VOCAB: usize = 6;
const WIDTH: usize = 4;

fn ids(values: &[i64]) -> Result<Tensor<Dyn, Cpu, i64, NoGrad>> {
    Tensor::<Dyn, Cpu, i64>::from_slice(values, vec![values.len()])
}

#[test]
fn an_embedding_lookup_carries_a_gradient_and_reaches_the_table() -> Result<()> {
    let table = Embedding::<Dyn, Cpu>::build((VOCAB, WIDTH))?;
    let tokens = ids(&[1, 3, 3])?;

    // The chain that could not train before: a table feeding a trainable
    // layer, with the loss taken at the end.
    let projection = Linear::<Dyn, Cpu>::build((WIDTH, WIDTH))?;
    let embedded = table.forward(tokens)?;
    let output = projection.forward(embedded.forget_layout())?;
    assert_eq!(output.dims().dims(), &[3, WIDTH]);

    let target = Tensor::<Dyn, Cpu>::zeros(vec![3, WIDTH])?;
    let grads = output.mse_loss(&target)?.backward()?;

    let weight = table.weight.as_tensor()?;
    let gradient = grads
        .require(&weight)
        .map_err(|e| Error::Msg(format!("no gradient reached the embedding table: {e}")))?
        .to_vec1::<f32>()?;
    assert_eq!(gradient.len(), VOCAB * WIDTH);

    // Only the rows that were looked up may move, and row 3 was looked up
    // twice, so its gradient is the accumulation rather than the last write.
    let row = |index: usize| &gradient[index * WIDTH..(index + 1) * WIDTH];
    let touched = |index: usize| row(index).iter().any(|value| *value != 0.0);
    assert!(touched(1), "row 1 was looked up and received nothing");
    assert!(touched(3), "row 3 was looked up and received nothing");
    for untouched in [0usize, 2, 4, 5] {
        assert!(
            !touched(untouched),
            "row {untouched} was never looked up but received a gradient"
        );
    }
    Ok(())
}

/// A frozen table still runs, and its result still requires no gradient.
///
/// The fix widens the output's requirement to the parameter's, so a `Frozen`
/// table has to keep behaving exactly as every table did before it. The
/// assertion is the signature of `only_no_grad`, which the frozen output must
/// satisfy: there is no runtime check to make here, because the absence of a
/// gradient is the absence of a `backward` method on the loss.
#[test]
fn a_frozen_table_still_requires_no_gradient() -> Result<()> {
    fn only_no_grad<S: Shape, L: Layout<S>>(_: &Tensor<S, Cpu, f32, NoGrad, Local, L>) {}

    let table = Embedding::<Dyn, Cpu>::build((VOCAB, WIDTH))?.freeze();
    let embedded = table.forward(ids(&[0, 5])?)?;
    assert_eq!(embedded.dims().dims(), &[2, WIDTH]);
    only_no_grad(&embedded);
    Ok(())
}

/// Widening the gradient type moved no values.
#[test]
fn the_looked_up_rows_are_the_table_rows() -> Result<()> {
    let table = Embedding::<Dyn, Cpu>::build((VOCAB, WIDTH))?;
    let rows = table.weight.as_tensor()?.to_vec1::<f32>()?;
    let embedded = table.forward(ids(&[4, 0])?)?.to_vec1::<f32>()?;

    assert_eq!(&embedded[..WIDTH], &rows[4 * WIDTH..5 * WIDTH]);
    assert_eq!(&embedded[WIDTH..], &rows[..WIDTH]);
    Ok(())
}
