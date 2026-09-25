//! Model-level generate loop + sampler (roadmap item S1.1).
//!
//! The inference-engine memo
//! (`docs/plan/research/0.2.0/inference-engine-substrate.md`, Stage 1, item
//! S1.1) identifies the model-level generate loop as the highest-leverage
//! first item: `forward_with_cache` exists at the attention layer, but no
//! sampler, no per-step loop, and no EOS handling exist anywhere, so even a
//! 1-request chat completion is impossible. This example closes exactly that
//! gap at TinyGpt scale on CPU.
//!
//! What this file is:
//! 1. Model: a TinyGpt-style decoder (embed, two causal blocks, norm,
//!    unembed) built inline from real `nn` modules. It is NOT shared with
//!    `crates/incin/tests/gpt_decoder_model.rs`: test targets are not
//!    importable from examples, so the struct is re-declared here with the
//!    same composition (2 layers, tiny `d_model`, vocab 64) and the same
//!    `#[module(no_stats, no_shape_info)]` shape (Embedding has no
//!    `ShapeInfo`, hence the flag, exactly as in the test).
//! 2. Sampler: greedy (argmax, the default, fully reproducible) plus
//!    temperature sampling over a host-side softmax with a deterministic
//!    LCG seed. Both are documented below; greedy is the default.
//! 3. Loop: prompt ids -> per-step full forward -> sample -> append ->
//!    stop at EOS or `MAX_NEW_TOKENS`. Per-step timing and total tok/s are
//!    printed.
//!
//! HONEST LIMITATIONS (each with its queued upgrade from the memo):
//! - FULL RECOMPUTE, no KV-cache: `forward_with_cache` / `KvCache`
//!   (`crates/incin-core/src/nn/attention.rs`, `nn/kv_cache.rs`) exist only
//!   at the `MultiHeadAttention` layer. `TransformerLayer` (and hence this
//!   model) has no `forward_with_cache`, so each step re-runs the whole
//!   prefix. Wiring per-layer caches through the model is the queued
//!   upgrade (memo G1/G2: generate loop -> paged-KV spike S1.5).
//! - NO TOKENIZER in-tree: the prompt is synthetic fixed ids and the output
//!   is printed as token IDS. Detokenization needs a Hub tokenizer; the
//!   integration point is the GGUF import work (memo G4/S1.3), which is
//!   export/inspect-only today.
//! - NO BATCHING / scheduler / paged KV / quantized decode: single request,
//!   batch 1, fp32. Those are memo stages S1.4, S1.5, S2.x, all anchored on
//!   this loop.
//!
//! Run with: `cargo run -p incin --example generate_gpt --no-default-features --features incin-backends/cpu,incin/cpu`

#![cfg(feature = "cpu")]
#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use incin::nn::{AttentionConfig, FeedForwardKind, TransformerConfig, TransformerDecoderLayer};
use incin::prelude::*;
use incin_core::tensor::ops::index::IndexSpec;

type Backend = DefaultBackend;

const VOCAB: usize = 64;
const D_MODEL: usize = 16;
const D_FF: usize = 32;
const HEADS: usize = 4;
const KV_HEADS: usize = 2;
/// The stop id for this synthetic vocabulary (no tokenizer exists, so EOS
/// is a convention of this example, stated here rather than hidden).
const EOS: i64 = (VOCAB - 1) as i64;
const PROMPT: [i64; 4] = [1, 5, 9, 13];
const MAX_NEW_TOKENS: usize = 16;

/// A GPT-style decoder-only model: embed, two causal blocks, norm, unembed.
///
/// Same composition as `TinyGpt` in `crates/incin/tests/gpt_decoder_model.rs`,
/// re-declared because examples cannot import from test targets.
#[module(no_stats, no_shape_info)]
struct TinyGpt {
    tokens: Embedding<Dyn, Backend>,
    block0: TransformerDecoderLayer<D_MODEL, HEADS, KV_HEADS, D_FF, Backend>,
    block1: TransformerDecoderLayer<D_MODEL, HEADS, KV_HEADS, D_FF, Backend>,
    norm: LayerNorm<Dyn, Backend>,
    head: Linear<Dyn, Backend>,
}

impl TinyGpt {
    fn build() -> incin::Result<Self> {
        // Rotary positions rather than a learned position table: the rotation
        // lives inside attention, so the model needs no second embedding.
        // max_seq_len 64 comfortably covers prompt + MAX_NEW_TOKENS.
        let config = TransformerConfig::default()
            .with_feed_forward(FeedForwardKind::SwiGlu)
            .with_attention(AttentionConfig::default().with_rotary(10_000.0, 64));
        Ok(Self {
            tokens: Embedding::<Dyn, Backend>::build((VOCAB, D_MODEL))?,
            block0: TransformerDecoderLayer::<D_MODEL, HEADS, KV_HEADS, D_FF, Backend>::build(
                config,
                (),
                (),
            )?,
            block1: TransformerDecoderLayer::<D_MODEL, HEADS, KV_HEADS, D_FF, Backend>::build(
                config,
                (),
                (),
            )?,
            norm: LayerNorm::<Dyn, Backend>::build((D_MODEL, 1e-5f32))?,
            head: Linear::<Dyn, Backend>::build((D_MODEL, VOCAB))?,
        })
    }
}

impl Module<Tensor<Dyn, Backend, i64, NoGrad>> for TinyGpt {
    type Output = Tensor<Dyn, Backend, f32, Grad>;
    type Error = Error;

    fn forward(&self, tokens: Tensor<Dyn, Backend, i64, NoGrad>) -> Result<Self::Output> {
        let hidden = self.tokens.forward(tokens)?.forget_layout();
        let hidden = self.block1.forward(self.block0.forward(hidden)?)?;
        let logits = self.head.forward(self.norm.forward(hidden)?)?;
        Ok(logits.forget_layout())
    }
}

/// Greedy sampler: argmax over the vocabulary. Deterministic by
/// construction; the default for reproducibility.
fn greedy(logits: &[f32]) -> i64 {
    let mut best = 0usize;
    for (index, value) in logits.iter().enumerate().skip(1) {
        if *value > logits[best] {
            best = index;
        }
    }
    best as i64
}

/// Deterministic LCG step (same constants as `native_training_demo`'s host
/// RNG). The seed makes temperature sampling reproducible; it is NOT
/// cryptographic randomness and makes no such claim.
fn next_uniform(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / ((1u64 << 33) as f32)
}

/// Temperature sampler: host-side softmax of `logits / temperature`, then a
/// categorical draw from `next_uniform`. `temperature <= 0` falls back to
/// greedy. A fixed seed gives a byte-identical sequence across runs.
fn sample_temperature(logits: &[f32], temperature: f32, rng: &mut u64) -> i64 {
    if temperature <= 0.0 {
        return greedy(logits);
    }
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, b| a.max(*b));
    let mut weights: Vec<f32> = logits
        .iter()
        .map(|v| ((v - max) / temperature).exp())
        .collect();
    let total: f32 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    let mut draw = next_uniform(rng);
    // Guard the `draw == 0` edge so index 0 keeps its mass.
    if draw <= 0.0 {
        draw = f32::MIN_POSITIVE;
    }
    let mut cumulative = 0.0f32;
    for (index, weight) in weights.iter().enumerate() {
        cumulative += weight;
        if draw <= cumulative {
            return index as i64;
        }
    }
    (weights.len() - 1) as i64
}

/// One autoregressive step's logits for the last position: forward the whole
/// id sequence (FULL RECOMPUTE -- see the header), then slice out the final
/// position's row. Returns exactly VOCAB finite values.
fn last_position_logits(model: &TinyGpt, ids: &[i64]) -> incin::Result<Vec<f32>> {
    let tokens = Tensor::<Dyn, Backend, i64>::from_slice(ids, vec![1, ids.len()])?;
    let logits = model.forward(tokens)?;
    assert_eq!(
        logits.dims().dims(),
        &[1, ids.len(), VOCAB],
        "logits must be one row per position over the vocabulary"
    );
    // Slice [batch=0, last position, all vocab]; Index removes its axis, so
    // the result is [1, VOCAB].
    let last = logits.get(vec![
        IndexSpec::All,
        IndexSpec::Index((ids.len() - 1) as isize),
        IndexSpec::All,
    ])?;
    let values = last.to_vec1::<f32>()?;
    assert_eq!(values.len(), VOCAB, "last-position row must span the vocab");
    assert!(
        values.iter().all(|v| v.is_finite()),
        "logits must be finite"
    );
    Ok(values)
}

/// Autoregressive generate: starting from `prompt`, sample one id per step
/// until EOS or `max_new` tokens. Returns the generated ids (excluding the
/// prompt) and the per-step wall times in milliseconds.
fn generate(
    model: &TinyGpt,
    prompt: &[i64],
    max_new: usize,
    sampler: &mut dyn FnMut(&[f32]) -> i64,
) -> incin::Result<(Vec<i64>, Vec<f64>)> {
    let mut ids: Vec<i64> = prompt.to_vec();
    let mut generated = Vec::with_capacity(max_new);
    let mut step_ms = Vec::with_capacity(max_new);
    for _ in 0..max_new {
        let start = std::time::Instant::now();
        let logits = last_position_logits(model, &ids)?;
        let next = sampler(&logits);
        step_ms.push(start.elapsed().as_secs_f64() * 1000.0);
        assert!(
            (0..VOCAB as i64).contains(&next),
            "sampled id {next} is outside the vocabulary"
        );
        ids.push(next);
        generated.push(next);
        if next == EOS {
            break;
        }
    }
    Ok((generated, step_ms))
}

fn report(name: &str, prompt: &[i64], generated: &[i64], step_ms: &[f64]) {
    let total_ms: f64 = step_ms.iter().sum();
    let tok_per_s = if total_ms > 0.0 {
        generated.len() as f64 / (total_ms / 1000.0)
    } else {
        0.0
    };
    println!("{name}");
    println!("{}", "-".repeat(name.len()));
    println!("  prompt ids    : {prompt:?}");
    println!("  generated ids : {generated:?}");
    println!(
        "  stopped at    : {}",
        if generated.last() == Some(&EOS) {
            "EOS"
        } else {
            "max tokens"
        }
    );
    for (step, ms) in step_ms.iter().enumerate() {
        println!(
            "  step {:>2}: id {:>2}   {ms:>8.2} ms",
            step + 1,
            generated[step]
        );
    }
    println!(
        "  total: {} tokens in {total_ms:.2} ms = {tok_per_s:.2} tok/s (CPU, full recompute)",
        generated.len()
    );
}

fn main() -> incin::Result<()> {
    println!("generate_gpt (S1.1): TinyGpt 2x{D_MODEL}d / {HEADS}h, vocab {VOCAB}, CPU");
    println!("prompt is synthetic ids {PROMPT:?}; EOS = {EOS}; no tokenizer in-tree.");

    section("1. Greedy generate (default, reproducible)");
    let model = TinyGpt::build()?;
    let (greedy_ids, greedy_ms) = generate(&model, &PROMPT, MAX_NEW_TOKENS, &mut greedy)?;
    report("greedy run", &PROMPT, &greedy_ids, &greedy_ms);

    section("2. Temperature generate (seeded, deterministic under fixed seed)");
    let mut rng: u64 = 0x1234_5678_9abc_def0;
    let (temp_ids, temp_ms) = generate(&model, &PROMPT, MAX_NEW_TOKENS, &mut |logits| {
        sample_temperature(logits, 0.8, &mut rng)
    })?;
    report(
        "temperature run (T=0.8, seed 0x123456789abcdef0)",
        &PROMPT,
        &temp_ids,
        &temp_ms,
    );

    section("3. Asserts that hold");
    // Shape: one logit row per position over the vocab (checked inside
    // last_position_logits on every step; spot-check the full forward).
    let probe = Tensor::<Dyn, Backend, i64>::from_slice(&PROMPT, vec![1, PROMPT.len()])?;
    assert_eq!(
        model.forward(probe)?.dims().dims(),
        &[1, PROMPT.len(), VOCAB]
    );
    println!("  shapes: [1, seq, {VOCAB}] on the prompt forward");
    // Termination: EOS or exactly max tokens, never more.
    assert!(greedy_ids.len() <= MAX_NEW_TOKENS);
    assert!(
        greedy_ids.last() == Some(&EOS) || greedy_ids.len() == MAX_NEW_TOKENS,
        "generation must stop at EOS or at the token budget"
    );
    println!(
        "  termination: {} token(s), stopped honestly",
        greedy_ids.len()
    );
    // Determinism under fixed seed: greedy twice, temperature twice.
    let (greedy_again, _) = generate(&model, &PROMPT, MAX_NEW_TOKENS, &mut greedy)?;
    assert_eq!(greedy_ids, greedy_again, "greedy must be byte-identical");
    let run_temp = |seed: u64| -> incin::Result<Vec<i64>> {
        let mut rng = seed;
        Ok(generate(&model, &PROMPT, MAX_NEW_TOKENS, &mut |logits| {
            sample_temperature(logits, 0.8, &mut rng)
        })?
        .0)
    };
    assert_eq!(
        temp_ids,
        run_temp(0x1234_5678_9abc_def0)?,
        "same temperature seed must give identical ids"
    );
    println!("  determinism: greedy and seeded-temperature runs are byte-identical");

    println!(
        "\nQueued upgrades (memo): KV-cache wiring (G1/G2), Hub tokenizer + GGUF import (G4/S1.3)."
    );
    Ok(())
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
}
