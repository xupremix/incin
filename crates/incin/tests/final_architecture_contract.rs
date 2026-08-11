//! Public Framework Architecture Contract Test.
//! Verifies that standard user-facing architecture APIs (shapes, targets,
//! linear, embedding, frozen layers, recurrent modules) compile and execute.

#![cfg(feature = "target-api")]

use incin::nn;
use incin::prelude::*;

const IN: usize = 8;
const OUT: usize = 4;
const VOCAB: usize = 32;

#[test]
fn test_final_architecture_contract() -> Result<()> {
    let runtime_batch = 2usize;

    // Shape semantics.
    let _static_shape = shape![const IN, const OUT];
    let _mixed_shape = shape![runtime_batch, const IN];

    // Bare device = Native shorthand.
    let x = Cpu.zeros(shape![runtime_batch, const IN])?;

    // Explicit engine target.
    let target = Native::on(Cpu);

    // Explicit tensor dtype view.
    let ids = target.dtype::<i64>()?.zeros([runtime_batch])?;

    // Linear.
    let linear = nn::linear(shape![const IN, const OUT]).init(&target)?;

    // Embedding.
    let embedding = nn::embedding(shape![const VOCAB, const OUT]).init(&target)?;

    // Frozen layer.
    let frozen = nn::linear(shape![const IN, const OUT])
        .frozen()
        .init(&target)?;

    // Recurrent builders.
    let rnn = nn::rnn(shape![const IN, const OUT]).init(&target)?;

    let lstm = nn::lstm(shape![const IN, const OUT])
        .no_hidden_bias()
        .init(&target)?;

    let _ = (x, ids, linear, embedding, frozen, rnn, lstm);

    Ok(())
}
