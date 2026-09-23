//! `MultiHeadAttention` and `CrossAttention` checked against attention
//! composed by hand.
//!
//! The modules are only worth having if they compute what the hand-written
//! block computed, so most of what follows re-derives the same numbers from
//! the modules' own weights and compares. "Gradients are finite and nonzero"
//! is also asserted, but it is not the bar: a wrong implementation passes that.
#![cfg(feature = "cpu")]

use incin::nn::{AttentionConfig, CrossAttention, KvCache, MultiHeadAttention, PositionEncoding};
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

/// The flat slice of time step `i` in a `[1, seq, width]` tensor's values.
fn step(values: &[f32], i: usize, width: usize) -> &[f32] {
    &values[i * width..(i + 1) * width]
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

/// Cross-attention over two sequences of different lengths, composed by hand
/// from the module's own weights: queries feed only the query projection,
/// memory feeds only key and value (issue #101).
#[test]
fn cross_attention_matches_a_hand_composition_over_two_sequences() -> Result<()> {
    let width = 8;
    let attention = CrossAttention::<8, 1, 1, Cpu>::build(AttentionConfig::default(), (), ())?;
    let query = ramp(vec![1, 3, width])?;
    let memory = ramp(vec![1, 5, width])?
        .mul_scalar(-0.6_f64)?
        .forget_layout();

    let from_module = attention.forward((query.clone(), memory.clone()))?;
    assert_eq!(
        from_module.dims().dims(),
        &[1, 3, width],
        "the output follows the query length, not the memory length"
    );

    let (wq, bq) = weights(&attention.query)?;
    let (wk, bk) = weights(&attention.key)?;
    let (wv, bv) = weights(&attention.value)?;
    let (wo, bo) = weights(&attention.output)?;
    let q = linear_apply(&query, &wq, &bq)?;
    let k = linear_apply(&memory, &wk, &bk)?;
    let v = linear_apply(&memory, &wv, &bv)?;
    let attended = attention_by_hand(&q, &k, &v, width)?;
    let by_hand = linear_apply(&attended, &wo, &bo)?;

    let diff = max_abs_diff(&from_module.to_vec1::<f32>()?, &by_hand.to_vec1::<f32>()?);
    assert!(
        diff < 1e-6,
        "module and hand-composed cross-attention disagree by {diff:e}"
    );
    Ok(())
}

/// Queries and memory are not interchangeable: a query row's output depends
/// only on that row of the query stream, while -- without a causal mask --
/// every row sees the whole memory.
#[test]
fn query_rows_depend_only_on_themselves_and_memory_reaches_every_row() -> Result<()> {
    let width = 8;
    let seq_q = 3;
    let seq_m = 5;
    let attention = CrossAttention::<8, 2, 2, Cpu>::build(AttentionConfig::default(), (), ())?;

    let query = ramp(vec![1, seq_q, width])?;
    let memory = ramp(vec![1, seq_m, width])?;
    let before = attention
        .forward((query.clone(), memory.clone()))?
        .to_vec1::<f32>()?;

    // Disturb query row 0: rows 1 and 2 must not move.
    let mut disturbed_query = query.to_vec1::<f32>()?;
    for value in disturbed_query.iter_mut().take(width) {
        *value += 3.5;
    }
    let disturbed_query = Tensor::<Dyn, Cpu>::from_slice(&disturbed_query, vec![1, seq_q, width])?;
    let after = attention
        .forward((disturbed_query, memory.clone()))?
        .to_vec1::<f32>()?;
    assert!(
        max_abs_diff(step(&before, 0, width), step(&after, 0, width)) > 1e-4,
        "the disturbed query row did not change its own output"
    );
    for i in 1..seq_q {
        let diff = max_abs_diff(step(&before, i, width), step(&after, i, width));
        assert!(
            diff < 1e-6,
            "query row {i} moved by {diff:e} when only row 0 changed; \
             queries are leaking into each other"
        );
    }

    // Disturb memory position 4 -- past the causal reach asserted below --
    // and a non-causal module must move every query row.
    let mut disturbed_memory = memory.to_vec1::<f32>()?;
    for value in disturbed_memory.iter_mut().skip(4 * width).take(width) {
        *value += 3.5;
    }
    let disturbed_memory =
        Tensor::<Dyn, Cpu>::from_slice(&disturbed_memory, vec![1, seq_m, width])?;
    let after = attention
        .forward((query, disturbed_memory))?
        .to_vec1::<f32>()?;
    for i in 0..seq_q {
        let diff = max_abs_diff(step(&before, i, width), step(&after, i, width));
        assert!(
            diff > 1e-4,
            "query row {i} ignored memory position 4 in a non-causal module"
        );
    }
    Ok(())
}

/// Causal cross-attention masks memory, not queries: query row `i` must not
/// see memory position `j > i`. Asserted by disturbing one memory position
/// at a time and checking exactly which query rows move -- the rectangular
/// mask leaks or over-masks in ways a shape check would never catch.
#[test]
fn a_causal_query_cannot_see_a_later_memory_position() -> Result<()> {
    let width = 8;
    let seq_q = 3;
    let seq_m = 5;
    let attention = CrossAttention::<8, 2, 2, Cpu>::build(AttentionConfig::causal(), (), ())?;

    let query = ramp(vec![1, seq_q, width])?;
    let memory = ramp(vec![1, seq_m, width])?;
    let before = attention
        .forward((query.clone(), memory.clone()))?
        .to_vec1::<f32>()?;
    let disturb = |position: usize| -> Result<Plain> {
        let mut values = memory.to_vec1::<f32>()?;
        for value in values.iter_mut().skip(position * width).take(width) {
            *value += 3.5;
        }
        Tensor::<Dyn, Cpu>::from_slice(&values, vec![1, seq_m, width])
    };

    // Position 0: every query row may see it.
    let after = attention
        .forward((query.clone(), disturb(0)?))?
        .to_vec1::<f32>()?;
    for i in 0..seq_q {
        assert!(
            max_abs_diff(step(&before, i, width), step(&after, i, width)) > 1e-4,
            "query row {i} ignored memory position 0"
        );
    }

    // Position 1: rows 1 and 2 may see it; row 0 must not move.
    let after = attention
        .forward((query.clone(), disturb(1)?))?
        .to_vec1::<f32>()?;
    assert!(
        max_abs_diff(step(&before, 0, width), step(&after, 0, width)) < 1e-6,
        "query row 0 saw memory position 1; the rectangular mask leaks"
    );
    for i in 1..seq_q {
        assert!(
            max_abs_diff(step(&before, i, width), step(&after, i, width)) > 1e-4,
            "query row {i} ignored memory position 1; the mask masks too much"
        );
    }

    // Position 4: past every query row (the longest query is 2), so nothing
    // may move -- and the previous test proved the same edit reaches every
    // row without the mask.
    let after = attention.forward((query, disturb(4)?))?.to_vec1::<f32>()?;
    for i in 0..seq_q {
        assert!(
            max_abs_diff(step(&before, i, width), step(&after, i, width)) < 1e-6,
            "query row {i} saw memory position 4; the rectangular mask leaks"
        );
    }
    Ok(())
}

/// Trains: the forward pass records a tape, every projection and **both
/// inputs** receive finite nonzero gradients -- with different sequence
/// lengths, so a gradient routed back through the wrong stream has the wrong
/// shape and fails here.
#[test]
fn cross_attention_trains_and_both_sequences_receive_gradients() -> Result<()> {
    let width = 8;
    let attention = CrossAttention::<8, 2, 2, Cpu>::build(
        AttentionConfig::causal().with_rotary(10_000.0, 32),
        (),
        (),
    )?;
    let query = ramp(vec![2, 3, width])?.require_grad();
    let memory = ramp(vec![2, 5, width])?.require_grad();
    let target = Tensor::<Dyn, Cpu>::zeros(vec![2, 3, width])?;

    let before = incin_backends::cpu::tape_depth();
    let output = attention.forward((query.clone(), memory.clone()))?;
    let recorded = incin_backends::cpu::tape_depth().saturating_sub(before);
    assert!(recorded > 0, "the forward pass recorded no tape nodes");
    assert!(
        output.to_vec1::<f32>()?.iter().all(|v| v.is_finite()),
        "cross attention produced a non-finite value"
    );

    let grads = output.mse_loss(&target)?.backward()?;

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

    let query_grad = grads
        .require(&query)
        .map_err(|e| Error::Msg(format!("no gradient reached the query input: {e}")))?;
    assert_eq!(
        query_grad.dims().dims(),
        &[2, 3, width],
        "the query gradient came back with the wrong shape"
    );
    let values = query_grad.to_vec1::<f32>()?;
    assert!(
        values.iter().all(|v| v.is_finite()),
        "the query input received a non-finite gradient"
    );
    assert!(
        values.iter().any(|v| *v != 0.0),
        "the query input received only zeros"
    );

    let memory_grad = grads
        .require(&memory)
        .map_err(|e| Error::Msg(format!("no gradient reached the memory input: {e}")))?;
    assert_eq!(
        memory_grad.dims().dims(),
        &[2, 5, width],
        "the memory gradient came back with the wrong shape"
    );
    let values = memory_grad.to_vec1::<f32>()?;
    assert!(
        values.iter().all(|v| v.is_finite()),
        "the memory input received a non-finite gradient"
    );
    assert!(
        values.iter().any(|v| *v != 0.0),
        "the memory input received only zeros"
    );
    Ok(())
}

/// Memory prefilled once, then decoded against: one read-only step over the
/// full query stream must equal a single plain forward of the same inputs.
#[test]
fn prefilled_memory_decode_matches_plain_forward() -> Result<()> {
    let attention = CrossAttention::<8, 2, 2, Cpu>::build(AttentionConfig::default(), (), ())?;
    let query = ramp(vec![1, 4, 8])?;
    let memory = ramp(vec![1, 6, 8])?.mul_scalar(-0.5_f64)?.forget_layout();

    let plain = attention.forward((query.clone(), memory.clone()))?;

    let mut cache = KvCache::<s![1, 2, 8, 4], Cpu, f32>::new(())?;
    attention.prefill_memory(memory, &mut cache)?;
    assert_eq!(
        cache.len(),
        6,
        "prefill did not store every memory position"
    );
    let cached = attention.forward_with_cache(query, 0, &cache)?;

    assert_eq!(cached.dims().dims(), &[1, 4, 8]);
    assert!(
        !cached.requires_grad(),
        "a decode step must hand back a NoGrad tensor"
    );
    let diff = max_abs_diff(&plain.to_vec1::<f32>()?, &cached.to_vec1::<f32>()?);
    assert!(
        diff < 1e-6,
        "decode against prefilled memory disagrees with the plain forward by {diff:e}"
    );
    assert_eq!(
        cache.len(),
        6,
        "forward_with_cache must not append to the cache"
    );
    Ok(())
}

/// Rotary positions on the query stream are absolute: two half-steps decoded
/// at `query_pos` 0 and 2 must equal one full forward -- chunk boundaries
/// must not restart the rotation at position 0.
#[test]
fn a_chunked_rotary_decode_matches_one_full_forward() -> Result<()> {
    let config = AttentionConfig::default().with_rotary(10_000.0, 32);
    let attention = CrossAttention::<8, 2, 2, Cpu>::build(config, (), ())?;
    let query = ramp(vec![1, 4, 8])?;
    let memory = ramp(vec![1, 6, 8])?;

    let plain = attention.forward((query.clone(), memory.clone()))?;

    let mut cache = KvCache::<s![1, 2, 8, 4], Cpu, f32>::new(())?;
    attention.prefill_memory(memory, &mut cache)?;

    let first = attention.forward_with_cache(
        query.clone().try_narrow(1, 0, 2)?.forget_layout(),
        0,
        &cache,
    )?;
    let second = attention.forward_with_cache(
        query.clone().try_narrow(1, 2, 2)?.forget_layout(),
        2,
        &cache,
    )?;
    let chunked = first.concat(&second, 1)?.forget_layout();

    let diff = max_abs_diff(&plain.to_vec1::<f32>()?, &chunked.to_vec1::<f32>()?);
    assert!(
        diff < 1e-5,
        "chunked rotary decode disagrees with the full forward by {diff:e}"
    );
    Ok(())
}

/// Decoding against an empty cache, and prefilling a full one twice, are
/// typed failures that say what to do -- not a panic and not silence.
#[test]
fn an_empty_cache_and_a_second_prefill_are_refused() -> Result<()> {
    let attention = CrossAttention::<8, 2, 2, Cpu>::build(AttentionConfig::default(), (), ())?;
    let mut cache = KvCache::<s![1, 2, 8, 4], Cpu, f32>::new(())?;

    let error = attention
        .forward_with_cache(ramp(vec![1, 2, 8])?, 0, &cache)
        .expect_err("decoding before prefill must be refused");
    let text = error.to_string();
    assert!(
        text.contains("prefill"),
        "the refusal should say to prefill, said: {text}"
    );

    attention.prefill_memory(ramp(vec![1, 6, 8])?, &mut cache)?;
    let error = attention
        .prefill_memory(ramp(vec![1, 6, 8])?, &mut cache)
        .expect_err("a second prefill must be refused");
    let text = error.to_string();
    assert!(
        text.contains('6'),
        "the refusal should name the stored length, said: {text}"
    );
    Ok(())
}

/// A rank-two query and a batch mismatch are refused with the offending
/// geometry named -- the tuple input doubles the surface to validate.
#[test]
fn a_bad_query_rank_or_batch_mismatch_is_refused() -> Result<()> {
    let attention = CrossAttention::<8, 2, 2, Cpu>::build(AttentionConfig::default(), (), ())?;

    let error = attention
        .forward((ramp(vec![4, 8])?, ramp(vec![1, 5, 8])?))
        .expect_err("attention takes rank-3 [batch, seq, d_model] inputs");
    assert!(
        error.to_string().contains("rank 2"),
        "the refusal should name the rank it got, said: {error}"
    );

    let error = attention
        .forward((ramp(vec![2, 3, 8])?, ramp(vec![3, 5, 8])?))
        .expect_err("the two inputs must share a batch");
    let text = error.to_string();
    assert!(
        text.contains('2') && text.contains('3'),
        "the refusal should name both batch sizes, said: {text}"
    );
    Ok(())
}
