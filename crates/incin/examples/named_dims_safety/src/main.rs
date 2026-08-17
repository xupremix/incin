//! Named dimensions make a tensor axis's *identity* part of its type, not
//! just its size: `Tensor<s![Batch, Feature]>` and `Tensor<s![Batch, Seq]>`
//! are different types even when both axes happen to be the same runtime
//! size. Mixing them up is a compile error here. In PyTorch (whose named
//! tensors are runtime, opt-in, and easy to bypass) the same mistake trains
//! for a while and silently produces wrong numbers instead.
//!
//! Run with `cargo run -p named_dims_safety`.
#![allow(clippy::type_complexity)]

use incin::prelude::*;

dim!(Batch, Seq, Feature);

/// A function that only accepts a `(Batch, Feature)`-shaped tensor, for example
/// the output of some per-token feature projection, ready to feed into a
/// classifier head.
fn classify(x: &Tensor<s![Batch, Feature]>) -> Tensor<s![Batch]> {
    x.sum(axis!(1)).unwrap()
}

fn main() -> incin::Result<()> {
    let batch = 4usize;
    let seq = 6usize;
    let feature = 8usize;

    let projected: Tensor<s![Batch, Feature]> = Tensor::zeros((batch, feature)).unwrap();
    let attention_scores: Tensor<s![Batch, Seq]> = Tensor::zeros((batch, seq)).unwrap();

    // Compiles: `projected` really is (Batch, Feature).
    let logits = classify(&projected);
    println!("logits shape: {:?}", logits.dims());

    // Does NOT compile. Uncomment to see it for yourself:
    //
    //     let logits = classify(&attention_scores);
    //
    // error[E0308]: mismatched types
    //   expected `&Tensor<(Batch, Feature), ...>`
    //      found `&Tensor<(Batch, Seq), ...>`
    //
    // `attention_scores` is legitimately (Batch, Seq)-shaped. Seq and
    // Feature happen to both just be plain integers in every mainstream
    // framework, so this exact mix-up (feeding attention scores where
    // per-token features were expected) is a real, silent PyTorch bug this
    // type system makes impossible to compile.
    let _ = &attention_scores;

    // Named dims are checked and PRESERVED through real ops.
    // this isn't a toy that only works for one hand-picked function.

    // Transpose swaps both dims' *types*, not just their runtime values:
    let transposed = projected.transpose(axis!(0), axis!(1)).unwrap();
    println!("transposed shape: {:?}", transposed.dims());

    // Concatenating two (Batch, 4)-shaped tensors along the literal axis
    // still knows the result is Batch-identified. The named dim survives.
    // an op it never even participates in the arithmetic of:
    let more: Tensor<s![Batch, 4]> = Tensor::zeros((batch, ())).unwrap();
    let half: Tensor<s![Batch, 4]> = Tensor::zeros((batch, ())).unwrap();
    let joined: Tensor<s![Batch, 8]> = half.concat(&more, axis!(1)).unwrap();
    println!("joined shape: {:?}", joined.dims());

    // matmul carries `Batch` straight through too. This is batched matrix
    // multiply with a named batch dimension, exactly like a real
    // per-sample linear layer applied across a batch:
    let weights: Tensor<s![4, 5]> = Tensor::zeros(()).unwrap();
    let projected_again: Tensor<s![Batch, 5]> = half.matmul(&weights).unwrap();
    println!("matmul output shape: {:?}", projected_again.dims());

    // The ordinary `+` operator, not just `.add()`, works between two
    // identically-shaped named-dim tensors, as in PyTorch with static checks
    // by the compiler instead of at runtime:
    let bias: Tensor<s![Batch, Feature]> = Tensor::zeros((batch, feature)).unwrap();
    let biased = (&projected + &bias)?;
    println!("biased shape: {:?}", biased.dims());

    println!("Compiled successfully. Every shape above was checked before this program ran.");
    Ok(())
}
