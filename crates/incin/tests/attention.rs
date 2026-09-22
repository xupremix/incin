//! `MultiHeadAttention` checked against attention composed by hand.
//!
//! The module is only worth having if it computes what the hand-written block
//! computed, so most of what follows re-derives the same numbers from the
//! module's own weights and compares. "Gradients are finite and nonzero" is
//! also asserted, but it is not the bar: a wrong implementation passes that.
#![cfg(feature = "cpu")]

use incin::nn::{AttentionConfig, MultiHeadAttention, PositionEncoding};
use incin::prelude::*;
use incin::state::{collect_state, load_state};

type Cpu = incin_backends::cpu::CpuBackendImpl;
type Plain = Tensor<Dyn, Cpu, f32, NoGrad>;

/// Deterministic, non-symmetric input. Symmetric data hides axis mistakes:
/// a transposed score matrix compares equal to itself.
fn ramp(dims: Vec<usize>) -> Result<Plain> {
    let n: usize = dims.iter().product();
    let values = (0..n)
        .map(|v| ((v as f32) * 0.37).sin() * 0.9 + (v as f32) * 0.01)
        .collect::<Vec<f32>>();
    Tensor::<Dyn, Cpu>::from_slice(&values, dims)
}

/// `x @ wᵀ + b`, the same arithmetic `Linear` performs.
fn linear_apply(x: &Plain, w: &Plain, b: &Plain) -> Result<Plain> {
    let weight_t = w.clone().transpose(0isize, 1isize)?.forget_layout();
    Ok(x.matmul(&weight_t)?
        .forget_layout()
        .broadcast_add(b)?
        .forget_layout())
}

/// Single-head attention over `[1, seq, width]`, written out.
fn attention_by_hand(q: &Plain, k: &Plain, v: &Plain, width: usize) -> Result<Plain> {
    let scores = q
        .matmul(&k.clone().transpose(1isize, 2isize)?.forget_layout())?
        .forget_layout()
        .mul_scalar(1.0_f64 / (width as f64).sqrt())?
        .forget_layout();
    Ok(scores
        .softmax(2)?
        .forget_layout()
        .matmul(v)?
        .forget_layout())
}

fn weights(layer: &Linear<Dyn, Cpu>) -> Result<(Plain, Plain)> {
    Ok((
        layer.weight.as_tensor()?.detach().forget_layout(),
        layer
            .bias
            .as_ref()
            .expect("bias")
            .as_tensor()?
            .detach()
            .forget_layout(),
    ))
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "compared tensors differ in length");
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max)
}

/// The test issue #101 asks for: the module against the composition it replaces.
#[test]
fn a_single_head_module_reproduces_attention_composed_by_hand() -> Result<()> {
    let width = 8;
    let attention = MultiHeadAttention::<8, 1, 1, Cpu>::build(AttentionConfig::default(), (), ())?;
    let x = ramp(vec![1, 4, width])?;

    let from_module = attention.forward(x.clone())?;

    let (wq, bq) = weights(&attention.query)?;
    let (wk, bk) = weights(&attention.key)?;
    let (wv, bv) = weights(&attention.value)?;
    let (wo, bo) = weights(&attention.output)?;
    let q = linear_apply(&x, &wq, &bq)?;
    let k = linear_apply(&x, &wk, &bk)?;
    let v = linear_apply(&x, &wv, &bv)?;
    let attended = attention_by_hand(&q, &k, &v, width)?;
    let by_hand = linear_apply(&attended, &wo, &bo)?;

    assert_eq!(from_module.dims().dims(), &[1, 4, width]);
    let diff = max_abs_diff(&from_module.to_vec1::<f32>()?, &by_hand.to_vec1::<f32>()?);
    assert!(
        diff < 1e-6,
        "module and hand-composed attention disagree by {diff:e}"
    );
    Ok(())
}

/// Pins the head split: head `h` reads columns `h*head_dim .. (h+1)*head_dim`
/// of each projection, and the heads are re-joined in the same order.
///
/// A module that split the width the other way -- strided rather than
/// blocked -- still produces finite output of the right shape, and still
/// trains. Only comparing against an explicit per-head computation catches it.
#[test]
fn every_head_reads_its_own_block_of_the_projection() -> Result<()> {
    let width = 8;
    let heads = 2;
    let head_dim = width / heads;
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(AttentionConfig::default(), (), ())?;
    let x = ramp(vec![1, 5, width])?;

    let from_module = attention.forward(x.clone())?;

    let (wq, bq) = weights(&attention.query)?;
    let (wk, bk) = weights(&attention.key)?;
    let (wv, bv) = weights(&attention.value)?;
    let (wo, bo) = weights(&attention.output)?;
    let q = linear_apply(&x, &wq, &bq)?;
    let k = linear_apply(&x, &wk, &bk)?;
    let v = linear_apply(&x, &wv, &bv)?;

    let mut joined: Option<Plain> = None;
    for head in 0..heads {
        let start = head * head_dim;
        let q_h = q.clone().try_narrow(2, start, head_dim)?.forget_layout();
        let k_h = k.clone().try_narrow(2, start, head_dim)?.forget_layout();
        let v_h = v.clone().try_narrow(2, start, head_dim)?.forget_layout();
        let out_h = attention_by_hand(&q_h, &k_h, &v_h, head_dim)?;
        joined = Some(match joined {
            None => out_h,
            Some(prev) => prev.concat(&out_h, 2)?.forget_layout(),
        });
    }
    let by_hand = linear_apply(&joined.expect("at least one head"), &wo, &bo)?;

    let diff = max_abs_diff(&from_module.to_vec1::<f32>()?, &by_hand.to_vec1::<f32>()?);
    assert!(diff < 1e-6, "per-head composition disagrees by {diff:e}");
    Ok(())
}

/// Grouped-query attention: four query heads over two key/value heads, with
/// the pairing stated explicitly rather than inferred from the output shape.
///
/// Query head `h` must read key/value head `h / (n_heads / n_kv_heads)`. The
/// other plausible convention -- `h % n_kv_heads` -- produces a tensor of
/// exactly the same shape, so this is the assertion that distinguishes them.
#[test]
fn grouped_query_attention_pairs_each_head_with_its_group() -> Result<()> {
    let width = 8;
    let heads = 4;
    let kv_heads = 2;
    let head_dim = width / heads;
    let group = heads / kv_heads;
    let attention = MultiHeadAttention::<8, 4, 2, Cpu>::build(AttentionConfig::default(), (), ())?;
    assert_eq!(attention.heads_per_group(), group);

    let x = ramp(vec![1, 5, width])?;
    let from_module = attention.forward(x.clone())?;

    let (wq, bq) = weights(&attention.query)?;
    let (wk, bk) = weights(&attention.key)?;
    let (wv, bv) = weights(&attention.value)?;
    let (wo, bo) = weights(&attention.output)?;
    let q = linear_apply(&x, &wq, &bq)?;
    let k = linear_apply(&x, &wk, &bk)?;
    let v = linear_apply(&x, &wv, &bv)?;
    // Key and value are narrower than the model width, which is the point of
    // grouped-query attention.
    assert_eq!(k.dims().dims(), &[1, 5, kv_heads * head_dim]);

    let mut joined: Option<Plain> = None;
    for head in 0..heads {
        let q_start = head * head_dim;
        let kv_start = (head / group) * head_dim;
        let q_h = q.clone().try_narrow(2, q_start, head_dim)?.forget_layout();
        let k_h = k.clone().try_narrow(2, kv_start, head_dim)?.forget_layout();
        let v_h = v.clone().try_narrow(2, kv_start, head_dim)?.forget_layout();
        let out_h = attention_by_hand(&q_h, &k_h, &v_h, head_dim)?;
        joined = Some(match joined {
            None => out_h,
            Some(prev) => prev.concat(&out_h, 2)?.forget_layout(),
        });
    }
    let by_hand = linear_apply(&joined.expect("at least one head"), &wo, &bo)?;

    let diff = max_abs_diff(&from_module.to_vec1::<f32>()?, &by_hand.to_vec1::<f32>()?);
    assert!(
        diff < 1e-6,
        "grouped-query pairing disagrees by {diff:e}; the head-to-group map is wrong"
    );
    Ok(())
}

/// Multi-query attention is the same module with one key/value head.
#[test]
fn one_key_value_head_is_multi_query_attention() -> Result<()> {
    let attention = MultiHeadAttention::<8, 4, 1, Cpu>::build(AttentionConfig::default(), (), ())?;
    assert_eq!(attention.heads_per_group(), 4);
    let out = attention.forward(ramp(vec![2, 3, 8])?)?;
    assert_eq!(out.dims().dims(), &[2, 3, 8]);
    Ok(())
}

/// Causal masking, asserted as the property it exists for: changing a token
/// must not change any output before it.
///
/// Checking that the mask tensor holds negative infinity would only restate
/// how the mask is built. This checks what it is for.
#[test]
fn a_causal_module_cannot_see_a_later_token() -> Result<()> {
    let width = 8;
    let seq = 5;
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(AttentionConfig::causal(), (), ())?;

    let base = ramp(vec![1, seq, width])?;
    let mut disturbed_values = base.to_vec1::<f32>()?;
    for value in disturbed_values.iter_mut().skip((seq - 1) * width) {
        *value += 3.5;
    }
    let disturbed = Tensor::<Dyn, Cpu>::from_slice(&disturbed_values, vec![1, seq, width])?;

    let before = attention.forward(base)?.to_vec1::<f32>()?;
    let after = attention.forward(disturbed)?.to_vec1::<f32>()?;

    let prefix = (seq - 1) * width;
    let prefix_diff = max_abs_diff(&before[..prefix], &after[..prefix]);
    assert!(
        prefix_diff < 1e-6,
        "editing the last token moved earlier outputs by {prefix_diff:e}; the mask leaks"
    );
    let last_diff = max_abs_diff(&before[prefix..], &after[prefix..]);
    assert!(
        last_diff > 1e-4,
        "the last output ignored its own token; the mask masks too much"
    );
    Ok(())
}

/// Without the causal flag the same edit must reach every position, so the
/// test above is measuring the mask rather than an input that changed nothing.
#[test]
fn a_non_causal_module_does_see_a_later_token() -> Result<()> {
    let width = 8;
    let seq = 5;
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(AttentionConfig::default(), (), ())?;

    let base = ramp(vec![1, seq, width])?;
    let mut disturbed_values = base.to_vec1::<f32>()?;
    for value in disturbed_values.iter_mut().skip((seq - 1) * width) {
        *value += 3.5;
    }
    let disturbed = Tensor::<Dyn, Cpu>::from_slice(&disturbed_values, vec![1, seq, width])?;

    let before = attention.forward(base)?.to_vec1::<f32>()?;
    let after = attention.forward(disturbed)?.to_vec1::<f32>()?;
    let prefix = (seq - 1) * width;
    assert!(
        max_abs_diff(&before[..prefix], &after[..prefix]) > 1e-4,
        "a non-causal module ignored a later token"
    );
    Ok(())
}

/// Rotary positions, asserted through the property that motivates them:
/// the score between two positions depends only on their distance.
///
/// The input is constant across time, so every position projects to the same
/// query and key vector. Any variation left in the score matrix is the
/// rotation, and it must be constant along each diagonal.
#[test]
fn rotary_positions_make_scores_depend_only_on_distance() -> Result<()> {
    let width = 8;
    let seq = 6;
    let config = AttentionConfig::default().with_rotary(10_000.0, 64);
    let attention = MultiHeadAttention::<8, 1, 1, Cpu>::build(config, (), ())?;
    assert!(attention.rotary_cos.is_some(), "tables were not built");
    assert!(attention.rotary_sin.is_some(), "tables were not built");

    // One repeated token, so position is the only thing that differs.
    let row = (0..width)
        .map(|i| 0.3 + i as f32 * 0.11)
        .collect::<Vec<f32>>();
    let values = row.repeat(seq);
    let x = Tensor::<Dyn, Cpu>::from_slice(&values, vec![1, seq, width])?;

    let (wq, bq) = weights(&attention.query)?;
    let (wk, bk) = weights(&attention.key)?;
    let q = linear_apply(&x, &wq, &bq)?;
    let k = linear_apply(&x, &wk, &bk)?;

    // Rotate q and k the way the module does, then score them.
    let cos = attention
        .rotary_cos
        .as_ref()
        .expect("cos table")
        .as_tensor()?
        .try_narrow(0, 0, seq)?
        .forget_layout();
    let sin = attention
        .rotary_sin
        .as_ref()
        .expect("sin table")
        .as_tensor()?
        .try_narrow(0, 0, seq)?
        .forget_layout();
    let rotate = |t: &Plain| -> Result<Plain> {
        let half = width / 2;
        let first = t.clone().try_narrow(2, 0, half)?.forget_layout();
        let second = t.clone().try_narrow(2, half, half)?.forget_layout();
        let swapped = second
            .neg()?
            .forget_layout()
            .concat(&first, 2)?
            .forget_layout();
        let direct = t.broadcast_mul(&cos)?.forget_layout();
        let turned = swapped.broadcast_mul(&sin)?.forget_layout();
        Ok(direct.broadcast_add(&turned)?.forget_layout())
    };
    let scores = rotate(&q)?
        .matmul(&rotate(&k)?.transpose(1isize, 2isize)?.forget_layout())?
        .forget_layout()
        .to_vec1::<f32>()?;

    let at = |i: usize, j: usize| scores[i * seq + j];
    for distance in 0..seq {
        let reference = at(distance, 0);
        for start in 0..(seq - distance) {
            let value = at(start + distance, start);
            assert!(
                (value - reference).abs() < 1e-4,
                "score at distance {distance} varies with absolute position: \
                 {value} at ({}, {start}) vs {reference} at ({distance}, 0)",
                start + distance
            );
        }
    }
    // And the rotation actually did something: distance 0 and 1 must differ.
    assert!(
        (at(0, 0) - at(1, 0)).abs() > 1e-4,
        "rotary made no difference between distance 0 and 1"
    );
    Ok(())
}

/// A sequence longer than the cached tables is refused, not silently rotated
/// with the wrong angles.
#[test]
fn a_sequence_longer_than_the_rotary_tables_is_refused() -> Result<()> {
    let config = AttentionConfig::default().with_rotary(10_000.0, 4);
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(config, (), ())?;
    let error = attention
        .forward(ramp(vec![1, 9, 8])?)
        .expect_err("a sequence past the table extent must be refused");
    let text = error.to_string();
    assert!(
        text.contains('9') && text.contains('4'),
        "the refusal should name both lengths, said: {text}"
    );
    Ok(())
}

/// The one head invariant that cannot be const-proven: rotary needs an even
/// head width (the rotation pairs dimensions), and the table extent depends on
/// the runtime config.
///
/// The two divisibility invariants are compile-time now (issue #101): a
/// mismatched `D_MODEL % N_HEADS` or `N_HEADS % N_KV_HEADS` never compiles,
/// covered by the compile-fail fixtures `attention_d_model_head_mismatch` and
/// `attention_head_kv_mismatch` rather than a runtime case here.
#[test]
fn an_odd_rotary_head_dim_is_refused_when_the_module_is_built() -> Result<()> {
    let odd_head_dim = MultiHeadAttention::<12, 4, 4, Cpu>::build(
        AttentionConfig::default().with_rotary(10_000.0, 16),
        (),
        (),
    )
    .err()
    .expect("rotary needs an even head_dim");
    assert!(
        odd_head_dim.to_string().contains('3'),
        "the error should name the odd head_dim, said: {odd_head_dim}"
    );
    Ok(())
}

#[test]
fn a_rank_two_input_is_refused_with_its_rank_named() -> Result<()> {
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(AttentionConfig::default(), (), ())?;
    let error = attention
        .forward(ramp(vec![4, 8])?)
        .expect_err("attention takes [batch, seq, d_model]");
    assert!(
        error.to_string().contains("rank 2"),
        "the refusal should name the rank it got, said: {error}"
    );
    Ok(())
}

/// Trains, and the optimizer moves every projection.
#[test]
fn attention_trains_and_every_projection_receives_a_gradient() -> Result<()> {
    let width = 8;
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(
        AttentionConfig::causal().with_rotary(10_000.0, 32),
        (),
        (),
    )?;
    let x = ramp(vec![2, 4, width])?.require_grad();
    let target = Tensor::<Dyn, Cpu>::zeros(vec![2, 4, width])?;

    let output = attention.forward(x)?;
    assert!(
        output.to_vec1::<f32>()?.iter().all(|v| v.is_finite()),
        "attention produced a non-finite value"
    );
    let loss = output.mse_loss(&target)?;
    let grads = loss.backward()?;

    let mut moved = 0usize;
    for (name, parameter) in [
        ("query", &attention.query),
        ("key", &attention.key),
        ("value", &attention.value),
        ("output", &attention.output),
    ] {
        let weight = parameter.weight.as_tensor()?;
        let gradient = grads
            .require(&weight)
            .map_err(|e| Error::Msg(format!("no gradient reached {name}.weight: {e}")))?;
        let values = gradient.to_vec1::<f32>()?;
        assert!(
            values.iter().all(|v| v.is_finite()),
            "{name}.weight received a non-finite gradient"
        );
        moved += values.iter().filter(|v| **v != 0.0).count();
    }
    assert!(moved > 0, "every projection gradient was exactly zero");

    let mut optimizer = incin::AdamW::<Cpu>::from_module(&attention, 1e-2)?;
    optimizer.step(&grads)?;
    assert_eq!(optimizer.step_count(), 1);
    Ok(())
}

/// The rotary tables are buffers, so they travel with the checkpoint and come
/// back identical -- and the restored module computes the same thing.
#[test]
fn state_round_trips_including_the_rotary_tables() -> Result<()> {
    let width = 8;
    let config = AttentionConfig::causal().with_rotary(10_000.0, 32);
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(config, (), ())?;
    let snapshot = collect_state::<Cpu, _>(&attention)?;

    // Four weights, four biases, two rotary tables.
    assert_eq!(
        snapshot.len(),
        10,
        "expected eight projection tensors and two rotary tables, got {}",
        snapshot.len()
    );

    let mut restored = MultiHeadAttention::<8, 2, 2, Cpu>::build(config, (), ())?;
    load_state::<Cpu, _>(&mut restored, &snapshot)?;
    assert_eq!(collect_state::<Cpu, _>(&restored)?, snapshot);

    let x = ramp(vec![1, 6, width])?;
    let expected = attention.forward(x.clone())?.to_vec1::<f32>()?;
    let actual = restored.forward(x)?.to_vec1::<f32>()?;
    assert!(
        max_abs_diff(&expected, &actual) < 1e-7,
        "the restored module computes something else"
    );
    Ok(())
}

/// Dropout is active in training and gone in evaluation, and `eval` reaches it
/// even though the derived traversal skips the field.
#[test]
fn dropout_is_applied_in_training_and_not_in_evaluation() -> Result<()> {
    let width = 8;
    let config = AttentionConfig::default().with_dropout(0.9);
    let mut attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(config, (), ())?;

    let x = ramp(vec![1, 6, width])?;
    incin::nn::TrainMode::set_training(&mut attention, true);
    assert!(attention.dropout.is_training);
    let trained = attention.forward(x.clone())?.to_vec1::<f32>()?;

    incin::nn::TrainMode::set_training(&mut attention, false);
    assert!(
        !attention.dropout.is_training,
        "eval mode did not reach the dropout field"
    );
    let evaluated = attention.forward(x.clone())?.to_vec1::<f32>()?;
    let evaluated_again = attention.forward(x)?.to_vec1::<f32>()?;

    assert!(
        max_abs_diff(&evaluated, &evaluated_again) < 1e-7,
        "evaluation is not deterministic, so dropout is still active"
    );
    assert!(
        max_abs_diff(&trained, &evaluated) > 1e-5,
        "a 0.9 dropout changed nothing in training mode"
    );
    Ok(())
}

/// Freezing removes the projections from the gradient path and leaves the
/// module's arithmetic unchanged.
#[test]
fn a_frozen_module_computes_the_same_values() -> Result<()> {
    let width = 8;
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(AttentionConfig::default(), (), ())?;
    let x = ramp(vec![1, 4, width])?;
    let before = attention.clone().forward(x.clone())?.to_vec1::<f32>()?;
    let frozen = attention.freeze();
    let after = frozen.forward(x)?.to_vec1::<f32>()?;
    assert!(
        max_abs_diff(&before, &after) < 1e-7,
        "freezing changed the computation"
    );
    Ok(())
}

/// A batch is independent: row one must not depend on row zero.
#[test]
fn batch_rows_are_independent() -> Result<()> {
    let width = 8;
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(AttentionConfig::causal(), (), ())?;

    let first = ramp(vec![1, 4, width])?;
    let second = ramp(vec![1, 4, width])?
        .mul_scalar(-0.5_f64)?
        .forget_layout();
    let mut batched_values = first.to_vec1::<f32>()?;
    batched_values.extend(second.to_vec1::<f32>()?);
    let batched = Tensor::<Dyn, Cpu>::from_slice(&batched_values, vec![2, 4, width])?;

    let alone = attention.forward(first)?.to_vec1::<f32>()?;
    let together = attention.forward(batched)?.to_vec1::<f32>()?;
    assert!(
        max_abs_diff(&alone, &together[..alone.len()]) < 1e-6,
        "a batched row differs from the same row run alone"
    );
    Ok(())
}

/// `PositionEncoding::None` builds no tables, so nothing needless is saved.
#[test]
fn without_rotary_no_tables_are_allocated() -> Result<()> {
    let attention = MultiHeadAttention::<8, 2, 2, Cpu>::build(AttentionConfig::default(), (), ())?;
    assert_eq!(attention.config.position, PositionEncoding::None);
    assert!(attention.rotary_cos.is_none());
    assert!(attention.rotary_sin.is_none());
    assert_eq!(collect_state::<Cpu, _>(&attention)?.len(), 8);
    Ok(())
}
