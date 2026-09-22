//! A decoder-only model end to end, on the public facade.
//!
//! This is the executable form of the Book's transformer chapter: token ids
//! in, logits over the vocabulary out, trained on CPU. It replaces
//! `transformer_block.rs`, which hand-assembles one single-head block out of
//! `matmul`, `transpose` and `softmax` and records in its own doc comment that
//! it "deliberately leaves masking, normalization, dropout, and multi-head
//! packing to a later composition layer".
//!
//! # Why the blocks are named fields rather than an array
//!
//! `Sequential` is a *pair* combinator, `Sequential<L1, L2>`, so a depth-`N`
//! stack is either a right-nested tower of pairs or a struct with one field
//! per block. There is no `Sequential<[Layer; N]>`, and a `Vec<Layer>` field
//! would not traverse either: the state and parameter visitors are
//! implemented for `Option<L>` and for module types, not for collections. Two
//! named fields is what the tree actually supports today, and it keeps the
//! state paths (`block0.attention.query.weight`, ...) readable in a
//! checkpoint.
#![cfg(feature = "cpu")]

use incin::nn::{AttentionConfig, FeedForwardKind, TransformerConfig, TransformerDecoderLayer};
use incin::prelude::*;

type Cpu = incin_backends::cpu::CpuBackendImpl;

const VOCAB: usize = 16;
const D_MODEL: usize = 8;
const D_FF: usize = 16;
const HEADS: usize = 2;
const KV_HEADS: usize = 1;
const SEQ: usize = 4;

/// A GPT-style decoder-only model: embed, two causal blocks, norm, unembed.
///
/// `no_shape_info` because `Embedding` implements no `ShapeInfo`, which is the
/// same reason the CNN example carries it; the transformer layers do implement
/// it, and report their direction, norm placement and feed-forward kind.
#[module(no_stats, no_shape_info)]
struct TinyGpt {
    tokens: Embedding<Dyn, Cpu>,
    block0: TransformerDecoderLayer<D_MODEL, HEADS, KV_HEADS, D_FF, Cpu>,
    block1: TransformerDecoderLayer<D_MODEL, HEADS, KV_HEADS, D_FF, Cpu>,
    norm: LayerNorm<Dyn, Cpu>,
    head: Linear<Dyn, Cpu>,
}

impl TinyGpt {
    fn build() -> Result<Self> {
        // Rotary positions rather than a learned position table: the rotation
        // lives inside attention, so the model needs no second embedding and
        // the tables are Buffers that save without being trained.
        let config = TransformerConfig::default()
            .with_feed_forward(FeedForwardKind::SwiGlu)
            .with_attention(AttentionConfig::default().with_rotary(10_000.0, 64));
        Ok(Self {
            tokens: Embedding::<Dyn, Cpu>::build((VOCAB, D_MODEL))?,
            block0: TransformerDecoderLayer::<D_MODEL, HEADS, KV_HEADS, D_FF, Cpu>::build(
                config,
                (),
                (),
            )?,
            block1: TransformerDecoderLayer::<D_MODEL, HEADS, KV_HEADS, D_FF, Cpu>::build(
                config,
                (),
                (),
            )?,
            norm: LayerNorm::<Dyn, Cpu>::build((D_MODEL, 1e-5f32))?,
            head: Linear::<Dyn, Cpu>::build((D_MODEL, VOCAB))?,
        })
    }
}

impl Module<Tensor<Dyn, Cpu, i64, NoGrad>> for TinyGpt {
    type Output = Tensor<Dyn, Cpu, f32, Grad>;
    type Error = Error;

    fn forward(&self, tokens: Tensor<Dyn, Cpu, i64, NoGrad>) -> Result<Self::Output> {
        let hidden = self.tokens.forward(tokens)?.forget_layout();
        let hidden = self.block1.forward(self.block0.forward(hidden)?)?;
        let logits = self.head.forward(self.norm.forward(hidden)?)?;
        Ok(logits.forget_layout())
    }
}

/// `[batch, seq]` token ids, cycling through the vocabulary.
fn tokens(batch: usize) -> Result<Tensor<Dyn, Cpu, i64, NoGrad>> {
    let ids = (0..batch * SEQ)
        .map(|position| (position % VOCAB) as i64)
        .collect::<Vec<i64>>();
    Tensor::<Dyn, Cpu, i64>::from_slice(&ids, vec![batch, SEQ])
}

/// The model produces one logit per vocabulary entry per position.
#[test]
fn the_model_maps_token_ids_to_logits() -> Result<()> {
    let model = TinyGpt::build()?;
    let logits = model.forward(tokens(2)?)?;
    assert_eq!(logits.dims().dims(), &[2, SEQ, VOCAB]);
    assert!(logits.to_vec1::<f32>()?.iter().all(|v| v.is_finite()));
    Ok(())
}

/// Every parameter receives a gradient, the embedding table included, and the
/// rotary tables receive none.
#[test]
fn the_gradient_reaches_every_parameter_and_no_buffer() -> Result<()> {
    let model = TinyGpt::build()?;
    let logits = model.forward(tokens(1)?)?;
    let target = Tensor::<Dyn, Cpu>::zeros(vec![1, SEQ, VOCAB])?;
    let grads = logits.mse_loss(&target)?.backward()?;

    let mut checked = 0usize;
    for (name, parameter) in [
        ("tokens.weight", model.tokens.weight.as_tensor()?),
        (
            "block0.attention.query.weight",
            model.block0.attention.query.weight.as_tensor()?,
        ),
        (
            "block0.feed_forward.gate.weight",
            model
                .block0
                .feed_forward
                .gate
                .as_ref()
                .expect("SwiGLU builds a gate")
                .weight
                .as_tensor()?,
        ),
        (
            "block1.attention_norm.weight",
            model.block1.attention_norm.weight.as_tensor()?,
        ),
        ("norm.weight", model.norm.weight.as_tensor()?),
        ("head.weight", model.head.weight.as_tensor()?),
    ] {
        let gradient = grads
            .require(&parameter)
            .map_err(|e| Error::Msg(format!("no gradient reached {name}: {e}")))?
            .to_vec1::<f32>()?;
        assert!(
            gradient.iter().all(|value| value.is_finite()),
            "{name} received a non-finite gradient"
        );
        checked += 1;
    }
    assert_eq!(checked, 6);

    let table = model
        .block0
        .attention
        .rotary_cos
        .as_ref()
        .expect("rotary was configured")
        .as_tensor()?;
    assert!(
        grads.require(&table).is_err(),
        "a gradient reached a rotary table, which is a Buffer"
    );
    Ok(())
}

/// The model trains: the acceptance criterion for a decoder-only stack on CPU.
#[test]
fn the_model_trains_on_cpu() -> Result<()> {
    // Not `mut`: AdamW::from_module holds the parameters' variable slots, so
    // `step` writes through them rather than through the binding.
    let model = TinyGpt::build()?;
    let input = tokens(1)?;
    // Predict the next token: a one-position shift of the input, one-hot over
    // the vocabulary, which is the smallest honest language-modelling target.
    let mut rows = vec![0.0f32; SEQ * VOCAB];
    for position in 0..SEQ {
        let next = (position + 1) % VOCAB;
        rows[position * VOCAB + next] = 1.0;
    }
    let target = Tensor::<Dyn, Cpu>::from_slice(&rows, vec![1, SEQ, VOCAB])?;

    let loss_now = |model: &TinyGpt| -> Result<f32> {
        Ok(model
            .forward(input.clone())?
            .mse_loss(&target)?
            .to_vec1::<f32>()?[0])
    };

    let start = loss_now(&model)?;
    let mut optimizer = AdamW::<Cpu>::from_module(&model, 5e-2)?;
    for _ in 0..60 {
        let loss = model.forward(input.clone())?.mse_loss(&target)?;
        let grads = loss.backward()?;
        optimizer.step(&grads)?;
    }
    let end = loss_now(&model)?;

    assert!(
        end < start * 0.9,
        "the model did not learn: loss went from {start} to {end}"
    );
    Ok(())
}

/// The whole model round-trips through state collection, rotary tables and
/// all, and the restored model computes the same logits.
#[test]
fn the_model_round_trips_through_its_state() -> Result<()> {
    let model = TinyGpt::build()?;
    let snapshot = incin::state::collect_state::<Cpu, _>(&model)?;

    let mut restored = TinyGpt::build()?;
    incin::state::load_state::<Cpu, _>(&mut restored, &snapshot)?;
    assert_eq!(
        incin::state::collect_state::<Cpu, _>(&restored)?,
        snapshot,
        "the restored model's state differs from the snapshot it was loaded from"
    );

    let input = tokens(1)?;
    assert_eq!(
        model.forward(input.clone())?.to_vec1::<f32>()?,
        restored.forward(input)?.to_vec1::<f32>()?,
        "the restored model computes different logits"
    );
    Ok(())
}
