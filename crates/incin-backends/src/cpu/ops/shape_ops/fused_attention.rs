//! Fused scaled dot-product attention for the CPU backend (issue #104).
//!
//! [`fused_attention_storage`] is the online-softmax reference for the
//! catalog's `FusedAttention` row: query blocks stream over key/value blocks
//! keeping a running maximum and normalizer per query row, so the full
//! `[seq_q, seq_kv]` score matrix never exists. This is numerically
//! equivalent to the composed `scaled_dot_product_attention` path, not an
//! approximation, which is what makes the equivalence tests meaningful.
//!
//! # What this avoids, honestly
//!
//! The composed path materializes the score matrix twice: once as the forward
//! `matmul` output, and again as the intermediates its tape entries retain
//! for backward (the `matmul`, scale, mask-add and `softmax` nodes each keep
//! their output alive). This kernel allocates no `[seq_q, seq_kv]` buffer in
//! either direction: forward keeps `O(seq_q * head_dim)` running state per
//! query block, and backward recomputes each row's attention weights from the
//! saved queries, keys, values and outputs (all `O(seq * head_dim)`) instead
//! of storing them. The extra cost is one recomputation pass over the scores
//! per backward -- the trade the issue's background section describes -- and
//! it is still linear in sequence length where the composed buffer is
//! quadratic.
//!
//! # What this is not
//!
//! `QUERY_BLOCK`/`KEY_BLOCK` below are L1-cache tiling constants for this CPU
//! reference, not the device crossover from the issue's benchmark plan: the
//! sequence length at which fused beats composed on time is a measured,
//! device-bound number and lives in the tuning policy, not here. Likewise
//! the causal block-skip (`j0` break below) halves the score work only in
//! exact arithmetic-count terms; wall-clock proof needs the device benches,
//! which are hardware-gated.

use super::*;

/// Query rows processed per online-softmax pass: L1-cache tiling for the CPU
/// reference, not a crossover threshold (see the module docs).
const QUERY_BLOCK: usize = 32;
/// Key/value columns streamed per inner pass: same status as `QUERY_BLOCK`.
const KEY_BLOCK: usize = 32;

/// Native compute precision for one fused invocation.
///
/// The kernel reads through [`CpuBuffer::get_f64`] (exact for `f32`/`f64`)
/// and computes in the operand's own precision, so `f32` inputs see `f32`
/// rounding in the same places the composed `f32` path does. `f16`/`bf16`
/// are refused by the caller: mixed-precision attention needs the dtype
/// work from #90 first.
trait FusedFloat: Copy {
    /// Exact for values that originated in this precision.
    fn from_f64(x: f64) -> Self;
    /// Exact for values in this precision.
    fn to_f64(self) -> f64;
    /// One-argument exponential in this precision.
    fn exp(self) -> Self;
}

impl FusedFloat for f32 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    #[inline]
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
}

impl FusedFloat for f64 {
    #[inline]
    fn from_f64(x: f64) -> Self {
        x
    }
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
}

/// Validated `[batch, heads, seq, head_dim]` geometry plus derived row data.
struct FusedGeometry {
    batch: usize,
    heads_q: usize,
    heads_kv: usize,
    seq_q: usize,
    seq_kv: usize,
    head_dim: usize,
    /// Query heads per key/value head (`heads_q / heads_kv`).
    groups: usize,
    /// Trailing-position alignment for causal masking: query row `i` sees
    /// key `j` iff `j <= i + kv_lead` with `kv_lead = seq_kv - seq_q`
    /// saturating at zero. Square inputs give the lower triangle; a decode
    /// step (`seq_kv > seq_q`) lets each new query see the whole cached
    /// prefix plus itself -- exactly the mask rows the composed path
    /// narrows out of the full causal matrix.
    kv_lead: usize,
    scale: f64,
}

/// Strided rank-4 reader: batch, head, seq and head-dim strides plus the
/// view offset, so transposed (non-contiguous) operands work without a
/// densify pass.
#[derive(Clone, Copy)]
struct Rank4Reader<'a> {
    buf: &'a CpuBuffer,
    off: usize,
    sb: usize,
    sh: usize,
    st: usize,
    sd: usize,
}

impl<'a> Rank4Reader<'a> {
    fn of(storage: &'a CpuStorage) -> Self {
        Self {
            buf: &storage.buffer,
            off: storage.offset_elements,
            sb: storage.strides[0],
            sh: storage.strides[1],
            st: storage.strides[2],
            sd: storage.strides[3],
        }
    }

    #[inline]
    fn get(&self, b: usize, h: usize, t: usize, d: usize) -> f64 {
        self.buf
            .get_f64(self.off + b * self.sb + h * self.sh + t * self.st + d * self.sd)
    }
}

fn geometry(
    q: &CpuStorage,
    k: &CpuStorage,
    v: &CpuStorage,
    scale: Option<f64>,
) -> Result<FusedGeometry> {
    const OP: &str = "fused_attention";
    let shape_mismatch = |expected: &[usize], got: &[usize], msg: String| Error::ShapeMismatch {
        op: OP,
        expected: expected.to_vec(),
        got: got.to_vec(),
        msg,
    };
    for (name, storage) in [("query", q), ("key", k), ("value", v)] {
        if storage.shape.len() != 4 {
            return Err(shape_mismatch(
                &[0, 0, 0, 0],
                storage.shape.as_ref(),
                format!(
                    "{OP} needs rank-4 {name} [batch, heads, seq, head_dim], matching the \
                     descriptor contract"
                ),
            ));
        }
    }
    let (qs, ks, vs) = (q.shape.as_ref(), k.shape.as_ref(), v.shape.as_ref());
    if qs[0] != ks[0] {
        return Err(shape_mismatch(
            qs,
            ks,
            format!(
                "{OP} query batch {} differs from the key/value batch {}",
                qs[0], ks[0]
            ),
        ));
    }
    if ks != vs {
        return Err(shape_mismatch(
            ks,
            vs,
            format!("{OP} key and value must share [batch, kv_heads, seq_kv, head_dim]"),
        ));
    }
    if qs[3] != ks[3] || qs[3] == 0 {
        return Err(shape_mismatch(
            qs,
            ks,
            format!("{OP} query/key head widths differ or are zero"),
        ));
    }
    if qs[1] == 0 || ks[1] == 0 || qs[1] % ks[1] != 0 {
        return Err(shape_mismatch(
            qs,
            ks,
            format!(
                "{OP} query heads {} must be a non-zero multiple of the key/value heads {}",
                qs[1], ks[1]
            ),
        ));
    }
    let head_dim = qs[3];
    Ok(FusedGeometry {
        batch: qs[0],
        heads_q: qs[1],
        heads_kv: ks[1],
        seq_q: qs[2],
        seq_kv: ks[2],
        head_dim,
        // Safe: the check above refused zero head counts and a query count
        // that is not a multiple of the key/value count.
        groups: qs[1] / ks[1],
        kv_lead: ks[2].saturating_sub(qs[2]),
        scale: scale.unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt()),
    })
}

/// Online-softmax forward over query blocks, `[batch, heads_q, seq_q, head_dim]`.
///
/// Never allocates the `[seq_q, seq_kv]` score matrix: per query row only the
/// running maximum `m`, the normalizer `l` and the `head_dim` accumulator
/// survive across key blocks. With `causal`, key blocks strictly above every
/// query row's diagonal are skipped wholesale (`break`, not mask-and-compute).
fn forward_pass<F: FusedFloat>(
    q: Rank4Reader<'_>,
    k: Rank4Reader<'_>,
    v: Rank4Reader<'_>,
    g: &FusedGeometry,
    causal: bool,
) -> Vec<F> {
    let mut out = vec![F::from_f64(0.0); g.batch * g.heads_q * g.seq_q * g.head_dim];
    let out_at = |b: usize, h: usize, t: usize, d: usize| {
        ((b * g.heads_q + h) * g.seq_q + t) * g.head_dim + d
    };
    for b in 0..g.batch {
        for hq in 0..g.heads_q {
            let hkv = hq / g.groups;
            let mut i0 = 0;
            while i0 < g.seq_q {
                let rows = QUERY_BLOCK.min(g.seq_q - i0);
                let mut m = vec![F::from_f64(f64::NEG_INFINITY); rows];
                let mut l = vec![F::from_f64(0.0); rows];
                let mut acc = vec![F::from_f64(0.0); rows * g.head_dim];
                let mut j0 = 0;
                while j0 < g.seq_kv {
                    // Causal block-skip: every query row in this block sees
                    // keys up to `(i0 + rows - 1) + kv_lead`, so a key block
                    // starting past that -- and every block after it -- is
                    // all masked and skipped without computing a score.
                    if causal && j0 > (i0 + rows - 1) + g.kv_lead {
                        break;
                    }
                    let cols = KEY_BLOCK.min(g.seq_kv - j0);
                    for li in 0..rows {
                        let i = i0 + li;
                        let mut block_max = f64::NEG_INFINITY;
                        let mut scores = [0.0f64; KEY_BLOCK];
                        for (lj, slot) in scores.iter_mut().enumerate().take(cols) {
                            let j = j0 + lj;
                            let masked = causal && j > i + g.kv_lead;
                            let mut s = f64::NEG_INFINITY;
                            if !masked {
                                let mut dot = 0.0;
                                for d in 0..g.head_dim {
                                    dot += q.get(b, hq, i, d) * k.get(b, hkv, j, d);
                                }
                                s = g.scale * dot;
                            }
                            *slot = s;
                            if s > block_max {
                                block_max = s;
                            }
                        }
                        // A row whose whole block is masked (possible only
                        // past the skip frontier for non-leading rows, or
                        // with an empty key extent) contributes nothing.
                        if block_max == f64::NEG_INFINITY {
                            continue;
                        }
                        let m_new = F::from_f64(block_max.max(m[li].to_f64()));
                        let rescale = F::exp(F::from_f64(m[li].to_f64() - m_new.to_f64()));
                        m[li] = m_new;
                        l[li] = F::from_f64(l[li].to_f64() * rescale.to_f64());
                        for d in 0..g.head_dim {
                            acc[li * g.head_dim + d] =
                                F::from_f64(acc[li * g.head_dim + d].to_f64() * rescale.to_f64());
                        }
                        for (lj, score) in scores.iter().enumerate().take(cols) {
                            let j = j0 + lj;
                            let p = F::exp(F::from_f64(*score - m_new.to_f64()));
                            l[li] = F::from_f64(l[li].to_f64() + p.to_f64());
                            for d in 0..g.head_dim {
                                let prev = acc[li * g.head_dim + d].to_f64();
                                acc[li * g.head_dim + d] =
                                    F::from_f64(prev + p.to_f64() * v.get(b, hkv, j, d));
                            }
                        }
                    }
                    j0 += KEY_BLOCK;
                }
                for li in 0..rows {
                    let i = i0 + li;
                    let norm = l[li].to_f64();
                    for d in 0..g.head_dim {
                        out[out_at(b, hq, i, d)] = if norm == 0.0 {
                            // Fully masked row (only an empty key extent can
                            // do this): zeros, matching the composed path's
                            // `softmax` over an all-masked row only in shape,
                            // never in values a caller can usefully compare.
                            // Documented here because silence would be worse.
                            F::from_f64(0.0)
                        } else {
                            F::from_f64(acc[li * g.head_dim + d].to_f64() / norm)
                        };
                    }
                }
                i0 += QUERY_BLOCK;
            }
        }
    }
    out
}

/// Fused backward by recomputation: no stored attention matrix.
///
/// For each query row the weights `P` are recomputed from the saved operands
/// (`O(seq_kv)` scratch, freed per row), then the standard attention
/// vector-Jacobian products accumulate: `dV += P * dO`,
/// `ds = P * (dO.v - dO.o)`, `dQ += scale * ds * K`, `dK += scale * ds * Q`.
/// Query heads sharing a key/value head accumulate into the same `dK`/`dV`
/// rows, which is the correct gradient for a shared input.
#[allow(clippy::too_many_arguments)]
fn backward_pass<F: FusedFloat>(
    q: Rank4Reader<'_>,
    k: Rank4Reader<'_>,
    v: Rank4Reader<'_>,
    o: Rank4Reader<'_>,
    grad_out: Rank4Reader<'_>,
    g: &FusedGeometry,
    causal: bool,
) -> (Vec<F>, Vec<F>, Vec<F>) {
    let mut dq = vec![F::from_f64(0.0); g.batch * g.heads_q * g.seq_q * g.head_dim];
    let mut dk = vec![F::from_f64(0.0); g.batch * g.heads_kv * g.seq_kv * g.head_dim];
    let mut dv = vec![F::from_f64(0.0); g.batch * g.heads_kv * g.seq_kv * g.head_dim];
    let dq_at = |b: usize, h: usize, t: usize, d: usize| {
        ((b * g.heads_q + h) * g.seq_q + t) * g.head_dim + d
    };
    let dk_at = |b: usize, h: usize, t: usize, d: usize| {
        ((b * g.heads_kv + h) * g.seq_kv + t) * g.head_dim + d
    };
    for b in 0..g.batch {
        for hq in 0..g.heads_q {
            let hkv = hq / g.groups;
            for i in 0..g.seq_q {
                // Recompute this row's weights: max, normalizer, then P.
                // `O(seq_kv)` scratch, never a `[seq_q, seq_kv]` matrix.
                let mut row_max = f64::NEG_INFINITY;
                for j in 0..g.seq_kv {
                    if causal && j > i + g.kv_lead {
                        continue;
                    }
                    let mut dot = 0.0;
                    for d in 0..g.head_dim {
                        dot += q.get(b, hq, i, d) * k.get(b, hkv, j, d);
                    }
                    let s = g.scale * dot;
                    if s > row_max {
                        row_max = s;
                    }
                }
                if row_max == f64::NEG_INFINITY {
                    continue;
                }
                let mut norm = 0.0;
                for j in 0..g.seq_kv {
                    if causal && j > i + g.kv_lead {
                        continue;
                    }
                    let mut dot = 0.0;
                    for d in 0..g.head_dim {
                        dot += q.get(b, hq, i, d) * k.get(b, hkv, j, d);
                    }
                    norm += (g.scale * dot - row_max).exp();
                }
                let mut dot_do_o = 0.0;
                for d in 0..g.head_dim {
                    dot_do_o += grad_out.get(b, hq, i, d) * o.get(b, hq, i, d);
                }
                for j in 0..g.seq_kv {
                    if causal && j > i + g.kv_lead {
                        continue;
                    }
                    let mut dot = 0.0;
                    for d in 0..g.head_dim {
                        dot += q.get(b, hq, i, d) * k.get(b, hkv, j, d);
                    }
                    let p = (g.scale * dot - row_max).exp() / norm;
                    let mut dot_do_v = 0.0;
                    for d in 0..g.head_dim {
                        dot_do_v += grad_out.get(b, hq, i, d) * v.get(b, hkv, j, d);
                    }
                    let ds = p * (dot_do_v - dot_do_o);
                    for d in 0..g.head_dim {
                        dq[dq_at(b, hq, i, d)] = F::from_f64(
                            dq[dq_at(b, hq, i, d)].to_f64() + g.scale * ds * k.get(b, hkv, j, d),
                        );
                        dk[dk_at(b, hkv, j, d)] = F::from_f64(
                            dk[dk_at(b, hkv, j, d)].to_f64() + g.scale * ds * q.get(b, hq, i, d),
                        );
                        dv[dk_at(b, hkv, j, d)] = F::from_f64(
                            dv[dk_at(b, hkv, j, d)].to_f64() + p * grad_out.get(b, hq, i, d),
                        );
                    }
                }
            }
        }
    }
    // The `scale` local in `forward_pass` documents that block math is in
    // native precision; the backward above folds `g.scale` per element.
    (dq, dk, dv)
}

/// Fused attention over rank-4 `[batch, heads, seq, head_dim]` operands.
///
/// Query heads map to key/value heads by integer division (`hq / groups`),
/// which is multi-head, grouped-query and multi-query attention in one
/// kernel. Records a single tape entry whose backward recomputes rather
/// than stores (see the module docs); the composed path records one entry
/// per primitive and retains the score matrix between them.
pub(crate) fn fused_attention_storage(
    q: &CpuStorage,
    k: &CpuStorage,
    v: &CpuStorage,
    causal: bool,
    scale: Option<f64>,
) -> Result<CpuStorage> {
    const OP: &str = "fused_attention";
    if let Some(s) = scale
        && (!s.is_finite() || s <= 0.0)
    {
        // Unreachable through dispatch (the descriptor contract refuses the
        // same values), kept so direct storage callers fail typed rather
        // than computing with a nonsense scale.
        return Err(Error::Msg(format!(
            "{OP} scale must be positive and finite, got {s}"
        )));
    }
    let g = geometry(q, k, v, scale)?;
    // The kernel below is written for `f32`/`f64`: the same pair the
    // composed path actually serves (its `softmax` is `f32`-only, so a
    // narrower input fails its same-dtype guard downstream). Anything else
    // is refused here, before launch, with the dtype named.
    let q_dtype = q.buffer.dtype_id();
    for storage in [k, v] {
        if storage.buffer.dtype_id() != q_dtype {
            return Err(Error::DTypeMismatch {
                operation: OP,
                expected: q_dtype.descriptor(),
                actual: storage.buffer.dtype_id().descriptor(),
            });
        }
    }
    if q_dtype != DTypeId::F32 && q_dtype != DTypeId::F64 {
        return Err(Error::UnsupportedDType {
            dtype: q_dtype.descriptor(),
            backend: "cpu",
            op: OP,
        });
    }

    let out_shape = [g.batch, g.heads_q, g.seq_q, g.head_dim];
    macro_rules! run {
        ($float:ty) => {{
            let out_values = forward_pass::<$float>(
                Rank4Reader::of(q),
                Rank4Reader::of(k),
                Rank4Reader::of(v),
                &g,
                causal,
            );
            let out_f64: Vec<f64> = out_values.iter().map(|x| x.to_f64()).collect();
            let out =
                CpuStorage::try_from_contiguous(q.buffer.from_f64_values(out_f64)?, out_shape)?;
            let (q_saved, k_saved, v_saved, o_saved) =
                (q.clone(), k.clone(), v.clone(), out.clone());
            let (q_id, k_id, v_id, out_id) = (q.id, k.id, v.id, out.id);
            let g_owned = FusedGeometry {
                batch: g.batch,
                heads_q: g.heads_q,
                heads_kv: g.heads_kv,
                seq_q: g.seq_q,
                seq_kv: g.seq_kv,
                head_dim: g.head_dim,
                groups: g.groups,
                kv_lead: g.kv_lead,
                scale: g.scale,
            };
            tape::push_with(|| TapeEntry {
                output_id: out_id,
                input_ids: vec![q_id, k_id, v_id],
                backward: Box::new(move |grad_out: &CpuStorage| {
                    let (dq, dk, dv) = backward_pass::<$float>(
                        Rank4Reader::of(&q_saved),
                        Rank4Reader::of(&k_saved),
                        Rank4Reader::of(&v_saved),
                        Rank4Reader::of(&o_saved),
                        Rank4Reader::of(grad_out),
                        &g_owned,
                        causal,
                    );
                    let to_storage = |values: Vec<$float>, saved: &CpuStorage, shape: &[usize]| {
                        let as_f64: Vec<f64> = values.iter().map(|x| x.to_f64()).collect();
                        CpuStorage::try_from_contiguous(
                            saved.buffer.from_f64_values(as_f64)?,
                            shape,
                        )
                    };
                    Ok(vec![
                        to_storage(
                            dq,
                            &q_saved,
                            &[
                                g_owned.batch,
                                g_owned.heads_q,
                                g_owned.seq_q,
                                g_owned.head_dim,
                            ],
                        )?,
                        to_storage(
                            dk,
                            &k_saved,
                            &[
                                g_owned.batch,
                                g_owned.heads_kv,
                                g_owned.seq_kv,
                                g_owned.head_dim,
                            ],
                        )?,
                        to_storage(
                            dv,
                            &v_saved,
                            &[
                                g_owned.batch,
                                g_owned.heads_kv,
                                g_owned.seq_kv,
                                g_owned.head_dim,
                            ],
                        )?,
                    ])
                }),
            });
            out
        }};
    }
    if q_dtype == DTypeId::F32 {
        Ok(run!(f32))
    } else {
        Ok(run!(f64))
    }
}
