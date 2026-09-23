//! Matmul-family compositions on Metal: `linear` and
//! `scaled_dot_product_attention` (#92's attention-block linalg gaps).
//!
//! Both rewrite into taped primitives — `transpose`, `matmul`, `add`,
//! `mul_scalar_float` and `softmax` — in the same order CPU's
//! `ops/shape_ops/linalg.rs` and WGPU's `backend/{shape_ops,nn}.rs` use, so
//! the backward is their chain rather than hand-derived math. The module
//! compiles and its tests run on any host under `--features metal`.

use incin_core::backend_authoring::*;
use incin_core::error::Result;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::DType;

use super::backend::MetalBackendImpl;

impl<D: Device> MetalBackendImpl<D> {
    /// `linear(input, weight, bias?)`: promote a rank-one input to a single
    /// row, `input @ weight^T`, optionally add the bias (Metal's `add`
    /// broadcasts the per-column vector), then drop the promoted row again —
    /// CPU's and WGPU's recipe, with every step a taped primitive.
    pub(crate) fn linear<K: DType>(
        input: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let in_dims = input.metadata().shape().dims();
        let unbatched = in_dims.len() == 1;
        let promoted;
        let rows = if unbatched {
            promoted = Self::reshape::<K>(input, &[1, in_dims[0]])?;
            &promoted
        } else {
            input
        };
        let transposed = Self::transpose::<K>(weight, 0, 1)?;
        let product = Self::matmul::<K>(rows, &transposed)?;
        let projected = match bias {
            None => product,
            Some(bias) => Self::add::<K>(&product, bias)?,
        };
        if unbatched {
            let out_dims = projected.metadata().shape().dims();
            Self::reshape::<K>(&projected, &out_dims[1..])
        } else {
            Ok(projected)
        }
    }

    /// `scaled_dot_product_attention(q, k, v, mask?, scale?)`: CPU's and
    /// WGPU's composition — transpose `k`, matmul, scale (default
    /// `1/sqrt(d_k)`), optional additive mask, softmax on the last axis,
    /// matmul with `v`. Every step is taped, so attention trains through
    /// the whole chain.
    pub(crate) fn scaled_dot_product_attention<K: DType>(
        q: &<Self as StorageBackend>::Storage<K>,
        k: &<Self as StorageBackend>::Storage<K>,
        v: &<Self as StorageBackend>::Storage<K>,
        mask: Option<&<Self as StorageBackend>::Storage<K>>,
        scale: Option<f64>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let k_dims = k.metadata().shape().dims();
        let k_t = if k_dims.len() >= 2 {
            let rank = k_dims.len();
            Self::transpose::<K>(k, rank - 2, rank - 1)?
        } else {
            k.clone()
        };
        let scores = Self::matmul::<K>(q, &k_t)?;
        let q_dims = q.metadata().shape().dims();
        let d_k = *q_dims.last().unwrap_or(&1) as f64;
        let scaled =
            Self::mul_scalar_float::<K>(&scores, scale.unwrap_or_else(|| 1.0 / d_k.sqrt()))?;
        let masked = match mask {
            Some(mask) => Self::add::<K>(&scaled, mask)?,
            None => scaled,
        };
        let axis = scores.metadata().shape().dims().len().saturating_sub(1);
        let attention = Self::softmax::<K>(&masked, axis)?;
        Self::matmul::<K>(&attention, v)
    }
}

#[cfg(test)]
/// Host-side forward/parity tests for the matmul-family compositions.
/// Pure `Vec<f32>` math, so they run without a Metal device.
mod tests {
    use super::*;
    use incin_core::exec::GradMode;
    use incin_core::shapes::ShapeBuf;
    use incin_core::tensor::device::{DeviceId, Metal};
    use incin_core::tensor::dtype::DTypeId;

    use crate::metal::storage::{MetalStorage, MetalStorageMode};
    use crate::metal::tape::MetalGrads;

    type B = MetalBackendImpl<Metal>;

    fn storage(values: &[f32], shape: &[usize]) -> MetalStorage {
        let bytes: Vec<u8> = bytemuck::cast_slice(values).to_vec();
        let meta = incin_core::exec::TensorMeta::contiguous(
            ShapeBuf::from_slice(shape),
            DTypeId::F32.into(),
            DeviceId::metal(0),
            MetalStorage::alignment(),
            values.len(),
        )
        .expect("contiguous metadata for test storage");
        MetalStorage::from_bytes(bytes, meta, MetalStorageMode::Shared, 0)
            .expect("bytes cover the metadata span")
    }

    fn read(s: &MetalStorage) -> Vec<f32> {
        bytemuck::cast_slice(s.as_bytes().expect("shared-mode storage is host-readable")).to_vec()
    }

    fn assert_close(got: &[f32], want: &[f32], eps: f32) {
        assert_eq!(
            got.len(),
            want.len(),
            "length mismatch: {got:?} vs {want:?}"
        );
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            if *g == *w {
                continue;
            }
            let tol = eps * w.abs().max(1.0);
            assert!(
                (g - w).abs() <= tol,
                "index {i}: got {g}, want {w} (eps {eps})"
            );
        }
    }

    /// Run `f` under recording mode, walk back with a ones seed, return
    /// `(output, grads)`.
    fn recorded<F>(f: F) -> (MetalStorage, MetalGrads)
    where
        F: FnOnce() -> MetalStorage,
    {
        let out = GradMode::Enabled.scope(f);
        let grads = crate::metal::tape::backward(&out).expect("backward walk succeeds");
        (out, grads)
    }

    // ── linear ─────────────────────────────────────────────────────────────

    #[test]
    fn linear_without_bias_is_the_matrix_product_and_trains() {
        let input = storage(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let weight = storage(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let (out, grads) = recorded(|| B::linear::<f32>(&input, &weight, None).unwrap());
        // identity weight: input @ I^T = input.
        assert_close(&read(&out), &[1.0, 2.0, 3.0, 4.0], 1e-5);
        assert_eq!(
            read(
                grads
                    .get(input.id())
                    .expect("linear records grad for input")
            ),
            vec![1.0; 4],
            "ones seed through an identity weight is ones on the input"
        );
    }

    #[test]
    fn linear_with_bias_adds_the_per_column_shift() {
        let input = storage(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let weight = storage(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let bias = storage(&[0.5, -0.5], &[2]);
        let out = B::linear::<f32>(&input, &weight, Some(&bias)).unwrap();
        assert_close(&read(&out), &[1.5, 1.5, 3.5, 3.5], 1e-5);
    }

    #[test]
    fn linear_promotes_a_rank_one_input_and_demotes_the_result() {
        let input = storage(&[1.0, 2.0], &[2]);
        // Weight is [out_features, in_features] = [3, 2]: row 0 picks
        // feature 0, row 1 picks feature 1, row 2 sums both — so
        // `weight^T` reads (1, 0, 1) and (0, 1, 1) as its columns.
        let weight = storage(&[1.0, 0.0, 0.0, 1.0, 1.0, 1.0], &[3, 2]);
        let (out, grads) = recorded(|| B::linear::<f32>(&input, &weight, None).unwrap());
        assert_eq!(
            out.metadata().shape().dims(),
            &[3],
            "a rank-one input comes back rank one"
        );
        // [1,2] @ W^T = [1*1+2*0, 1*0+2*1, 1*1+2*1] = [1, 2, 3].
        assert_close(&read(&out), &[1.0, 2.0, 3.0], 1e-5);
        // Ones seed [1,1,1] through `matmul`'s `dA = dY @ W` is
        // (1+0+1, 0+1+1) = (2, 2), then the demoting reshape restores
        // the rank-one shape.
        assert_eq!(
            read(
                grads
                    .get(input.id())
                    .expect("linear records grad for input")
            ),
            &[2.0, 2.0],
            "the backward unwinds the promote/transpose chain onto rank one"
        );
    }

    // ── scaled_dot_product_attention ───────────────────────────────────────

    /// Stable f64 softmax over the last axis of a row-major `[rows, cols]`.
    fn softmax_row_major(values: &[f32], rows: usize, cols: usize) -> Vec<f64> {
        let mut out = vec![0.0f64; values.len()];
        for r in 0..rows {
            let row = &values[r * cols..(r + 1) * cols];
            let max = row
                .iter()
                .map(|&v| f64::from(v))
                .fold(f64::NEG_INFINITY, f64::max);
            let total: f64 = row.iter().map(|&v| (f64::from(v) - max).exp()).sum();
            for (c, &v) in row.iter().enumerate() {
                out[r * cols + c] = (f64::from(v) - max).exp() / total;
            }
        }
        out
    }

    #[test]
    fn attention_without_mask_matches_the_host_reference_and_trains() {
        // Single head, two keys: q/k identity, v a plain [1, 2, 2] block —
        // the same fixture WGPU's dispatch test pins.
        let q = storage(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let k = storage(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let v = storage(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let (out, grads) =
            recorded(|| B::scaled_dot_product_attention::<f32>(&q, &k, &v, None, None).unwrap());
        assert_eq!(out.metadata().shape().dims(), &[1, 2, 2]);

        // Host chain: scores = q @ k^T / sqrt(d_k) = I / sqrt(2); softmax
        // over the last axis, then @ v.
        let s = 1.0f64 / 2.0f64.sqrt();
        let scores = [s as f32, 0.0, 0.0, s as f32];
        let probs = softmax_row_major(&scores, 2, 2);
        // row0 = p_keep * [1,2] + p_other * [3,4]; row1 swaps the weights.
        let (p_keep, p_other) = (probs[0], probs[1]);
        let want = [
            (p_keep * 1.0 + p_other * 3.0) as f32,
            (p_keep * 2.0 + p_other * 4.0) as f32,
            (p_other * 1.0 + p_keep * 3.0) as f32,
            (p_other * 2.0 + p_keep * 4.0) as f32,
        ];
        assert_close(&read(&out), &want, 1e-4);
        assert!(
            grads.get(q.id()).is_some(),
            "attention advertises training: q must receive a gradient"
        );
        assert!(grads.get(v.id()).is_some(), "v must receive a gradient");
    }

    #[test]
    fn attention_with_an_explicit_scale_and_a_zero_mask_equals_the_plain_chain() {
        let q = storage(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let k = storage(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let v = storage(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let zero_mask = storage(&[0.0; 4], &[1, 2, 2]);

        let plain = B::scaled_dot_product_attention::<f32>(&q, &k, &v, None, None).unwrap();
        // scale = 1/sqrt(2) is exactly the default, and a zero additive
        // mask changes nothing: both must land on the plain result.
        let explicit = B::scaled_dot_product_attention::<f32>(
            &q,
            &k,
            &v,
            Some(&zero_mask),
            Some(1.0 / 2.0f64.sqrt()),
        )
        .unwrap();
        assert_close(&read(&explicit), &read(&plain), 1e-5);
    }

    #[test]
    fn nograd_records_nothing() {
        let q = storage(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let k = storage(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let v = storage(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let input = storage(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let weight = storage(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let before = crate::metal::tape::depth();
        let _ = GradMode::Disabled
            .scope(|| B::scaled_dot_product_attention::<f32>(&q, &k, &v, None, None).unwrap());
        let _ = GradMode::Disabled.scope(|| B::linear::<f32>(&input, &weight, None).unwrap());
        assert_eq!(
            crate::metal::tape::depth(),
            before,
            "NoGrad must record nothing"
        );
    }
}
