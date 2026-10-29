use crate::cpu::storage::{BlockMXFP4, BlockNVFP4, BlockQ8_0, CpuBuffer, CpuStorage};
use incin_core::error::{Error, Result};
use incin_core::shapes::{OperationKind, ShapeBuf};
#[cfg(test)]
use incin_core::tensor::dtype::DTypeId;
use incin_core::tensor::dtype::fp4;

extern crate alloc;
use alloc::vec::Vec;

pub(crate) fn quantize_storage(t: &CpuStorage) -> Result<CpuStorage> {
    let f32_data = match &*t.buffer {
        CpuBuffer::F32(v) => v,
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Cpu (expected F32 buffer)",
            });
        }
    };
    let n = f32_data.len();
    if n % 32 != 0 {
        return Err(Error::Msg(alloc::format!(
            "quantize Q8_0 requires buffer length multiple of 32, got {}",
            n
        )));
    }
    let mut blocks = Vec::with_capacity(n / 32);
    for chunk in f32_data.chunks_exact(32) {
        let max_abs = chunk.iter().map(|value| value.abs()).fold(0.0f32, f32::max);
        let d = max_abs / 127.0;
        let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };
        let mut qs = [0i8; 32];
        for (index, value) in chunk.iter().enumerate() {
            qs[index] = (*value * inv_d).round() as i8;
        }
        blocks.push(BlockQ8_0 {
            d: half::f16::from_f32(d),
            qs,
        });
    }
    Ok(CpuStorage::from_contiguous(
        CpuBuffer::Q8_0(blocks),
        &t.shape,
    ))
}

pub(crate) fn dequantize_storage(t: &CpuStorage) -> Result<CpuStorage> {
    match &*t.buffer {
        CpuBuffer::Q8_0(q8_data) => {
            let mut f32_data = Vec::with_capacity(q8_data.len() * 32);
            for block in q8_data {
                let d = block.d.to_f32();
                for quantized in block.qs {
                    f32_data.push(quantized as f32 * d);
                }
            }
            Ok(CpuStorage::from_contiguous(
                CpuBuffer::F32(f32_data),
                &t.shape,
            ))
        }
        // Issue #95: decode routes by buffer variant (the executor's
        // `Dequantize` arm only narrows the float *output* dtype; the input
        // variant selects the codec). Same straight-through status as Q8_0.
        CpuBuffer::NVFP4(blocks) => {
            let mut f32_data = Vec::with_capacity(blocks.len() * 16);
            for block in blocks {
                f32_data.extend_from_slice(&fp4::decode_nvfp4_block(block.scale, &block.data));
            }
            Ok(CpuStorage::from_contiguous(
                CpuBuffer::F32(f32_data),
                &t.shape,
            ))
        }
        CpuBuffer::MXFP4(blocks) => {
            let mut f32_data = Vec::with_capacity(blocks.len() * 32);
            for block in blocks {
                f32_data.extend_from_slice(&fp4::decode_mxfp4_block(block.scale, &block.data));
            }
            Ok(CpuStorage::from_contiguous(
                CpuBuffer::F32(f32_data),
                &t.shape,
            ))
        }
        _ => Err(Error::UnsupportedBackendOperation {
            op: "dequantize",
            backend: "Cpu (expected Q8_0, NVFP4, or MXFP4 buffer)",
        }),
    }
}

/// Quantizes an `F32` buffer into NVFP4 blocks (16 values + E4M3 scale).
///
/// The TE recipe at `s_global = 1.0` (see
/// [`fp4`](incin_core::tensor::dtype::fp4)): per block, `s = E4M3(amax/6)`,
/// elements `RNE(x/s)`. Error bound: `|x - x_hat| <= 0.25 * block_amax`
/// (proved in the `fp4` module docs).
pub(crate) fn quantize_nvfp4_storage(t: &CpuStorage) -> Result<CpuStorage> {
    let f32_data = match &*t.buffer {
        CpuBuffer::F32(v) => v,
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Cpu (expected F32 buffer)",
            });
        }
    };
    let n = f32_data.len();
    if n % 16 != 0 {
        return Err(Error::Msg(alloc::format!(
            "quantize NVFP4 requires buffer length multiple of 16, got {}",
            n
        )));
    }
    let mut blocks = Vec::with_capacity(n / 16);
    for chunk in f32_data.chunks_exact(16) {
        let values: [f32; 16] = chunk.try_into().expect("chunks_exact(16) yields 16");
        let (scale, data) = fp4::encode_nvfp4_block(&values);
        blocks.push(BlockNVFP4 { scale, data });
    }
    Ok(CpuStorage::from_contiguous(
        CpuBuffer::NVFP4(blocks),
        &t.shape,
    ))
}

/// Quantizes an `F32` buffer into MXFP4 blocks (32 values + E8M0 scale).
///
/// Per block, `s = min E8M0 power of two >= amax/6`, elements `RNE(x/s)`.
/// Error bound: `|x - x_hat| <= 0.5 * block_amax`.
pub(crate) fn quantize_mxfp4_storage(t: &CpuStorage) -> Result<CpuStorage> {
    let f32_data = match &*t.buffer {
        CpuBuffer::F32(v) => v,
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Cpu (expected F32 buffer)",
            });
        }
    };
    let n = f32_data.len();
    if n % 32 != 0 {
        return Err(Error::Msg(alloc::format!(
            "quantize MXFP4 requires buffer length multiple of 32, got {}",
            n
        )));
    }
    let mut blocks = Vec::with_capacity(n / 32);
    for chunk in f32_data.chunks_exact(32) {
        let values: [f32; 32] = chunk.try_into().expect("chunks_exact(32) yields 32");
        let (scale, data) = fp4::encode_mxfp4_block(&values);
        blocks.push(BlockMXFP4 { scale, data });
    }
    Ok(CpuStorage::from_contiguous(
        CpuBuffer::MXFP4(blocks),
        &t.shape,
    ))
}

pub(crate) fn quantized_matmul_storage(lhs: &CpuStorage, rhs: &CpuStorage) -> Result<CpuStorage> {
    let lhs_data = match &*lhs.buffer {
        CpuBuffer::Q8_0(v) => v,
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantized_matmul",
                backend: "Cpu (lhs expected Q8_0 buffer)",
            });
        }
    };
    let rhs_data = match &*rhs.buffer {
        CpuBuffer::Q8_0(v) => v,
        _ => {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantized_matmul",
                backend: "Cpu (rhs expected Q8_0 buffer)",
            });
        }
    };
    let lhs_shape = &lhs.shape;
    let rhs_shape = &rhs.shape;
    if lhs_shape.len() < 2 {
        return Err(Error::Msg(
            "quantized_matmul lhs requires at least 2D shapes".into(),
        ));
    }
    if rhs_shape.len() != 2 {
        return Err(Error::Msg("quantized_matmul rhs must be 2D [N, K]".into()));
    }
    let n = rhs_shape[0];
    let k2 = rhs_shape[1];
    let k = lhs_shape[lhs_shape.len() - 1];
    let m: usize = crate::cpu::stride::checked_numel(&lhs_shape[..lhs_shape.len() - 1])?;
    if k != k2 {
        return Err(Error::Msg(alloc::format!(
            "quantized_matmul K mismatch: {} != {}",
            k,
            k2
        )));
    }
    if !k.is_multiple_of(32) {
        return Err(Error::Msg(alloc::format!(
            "quantized_matmul K must be multiple of 32, got {}",
            k
        )));
    }
    let mut out_shape = lhs_shape.to_vec();
    let out_len = out_shape.len();
    out_shape[out_len - 1] = n;
    let out_total = ShapeBuf::from_slice(&[m, n]).checked_numel(OperationKind::MatMul)?;
    let mut out_data = alloc::vec![0.0f32; out_total];
    let blocks_per_row = k / 32;
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    let use_avx2 = crate::simd::avx2_detected();
    #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
    let use_avx2 = false;
    for i in 0..m {
        for j in 0..n {
            let lhs_row_start = i * blocks_per_row;
            let rhs_row_start = j * blocks_per_row;
            let sum = if use_avx2 {
                #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                {
                    // SAFETY: feature detection above proves AVX2 is
                    // available, and the block count and row offsets were
                    // derived from the validated Q8_0 matrix dimensions.
                    unsafe {
                        vec_dot_q8_0_avx2(
                            blocks_per_row,
                            lhs_data,
                            lhs_row_start,
                            rhs_data,
                            rhs_row_start,
                        )
                    }
                }
                #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
                {
                    0.0
                }
            } else {
                vec_dot_q8_0_scalar(
                    blocks_per_row,
                    lhs_data,
                    lhs_row_start,
                    rhs_data,
                    rhs_row_start,
                )
            };
            out_data[i * n + j] = sum;
        }
    }
    Ok(CpuStorage::from_contiguous(
        CpuBuffer::F32(out_data),
        out_shape,
    ))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
/// Computes one Q8_0 dot product with AVX2.
///
/// # Safety
/// The caller must run this only when AVX2 is available. The row offsets and
/// block count must keep every load within the two Q8_0 slices.
unsafe fn vec_dot_q8_0_avx2(
    n: usize,
    lhs: &[crate::cpu::storage::BlockQ8_0],
    lhs_row_start: usize,
    rhs: &[crate::cpu::storage::BlockQ8_0],
    rhs_row_start: usize,
) -> f32 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    let mut sumf = 0.0f32;
    for b in 0..n {
        let lhs_block = &lhs[lhs_row_start + b];
        let rhs_block = &rhs[rhs_row_start + b];

        // SAFETY: the caller's block-count and row-offset contract puts this
        // complete Q8_0 block within both slices; AVX2 is required here.
        let (l, r) = unsafe {
            (
                _mm256_loadu_si256(lhs_block.qs.as_ptr() as *const __m256i),
                _mm256_loadu_si256(rhs_block.qs.as_ptr() as *const __m256i),
            )
        };

        let block_sum = {
            let l_low = _mm256_castsi256_si128(l);
            let l_high = _mm256_extracti128_si256(l, 1);
            let r_low = _mm256_castsi256_si128(r);
            let r_high = _mm256_extracti128_si256(r, 1);

            let l0 = _mm256_cvtepi8_epi16(l_low);
            let l1 = _mm256_cvtepi8_epi16(l_high);
            let r0 = _mm256_cvtepi8_epi16(r_low);
            let r1 = _mm256_cvtepi8_epi16(r_high);

            let p0 = _mm256_madd_epi16(l0, r0);
            let p1 = _mm256_madd_epi16(l1, r1);

            let p = _mm256_add_epi32(p0, p1);

            let x = _mm256_extracti128_si256(p, 1);
            let y = _mm_add_epi32(_mm256_castsi256_si128(p), x);
            let z = _mm_hadd_epi32(y, y);
            let w = _mm_hadd_epi32(z, z);
            _mm_cvtsi128_si32(w)
        };

        sumf += (block_sum as f32) * lhs_block.d.to_f32() * rhs_block.d.to_f32();
    }
    sumf
}

#[inline]
/// `vec_dot_q8_0_scalar`.
fn vec_dot_q8_0_scalar(
    n: usize,
    lhs: &[crate::cpu::storage::BlockQ8_0],
    lhs_row_start: usize,
    rhs: &[crate::cpu::storage::BlockQ8_0],
    rhs_row_start: usize,
) -> f32 {
    let mut sum = 0.0f32;
    for b in 0..n {
        let lhs_block = &lhs[lhs_row_start + b];
        let rhs_block = &rhs[rhs_row_start + b];

        let mut block_sum = 0i32;
        for q in 0..32 {
            block_sum += (lhs_block.qs[q] as i32) * (rhs_block.qs[q] as i32);
        }

        sum += (block_sum as f32) * lhs_block.d.to_f32() * rhs_block.d.to_f32();
    }
    sum
}

#[cfg(test)]
/// `tests`.
mod tests {
    use super::*;

    #[test]
    /// `test_quantize_dequantize_fidelity`.
    fn test_quantize_dequantize_fidelity() {
        let mut data = vec![0.0f32; 64];
        for (i, d) in data.iter_mut().enumerate() {
            *d = (i as f32 - 32.0) * 0.1; // ranging -3.2 to +3.1
        }

        let storage = CpuStorage::from_contiguous(CpuBuffer::F32(data.clone()), vec![2, 32]);

        let q_storage = quantize_storage(&storage).unwrap();
        let deq_storage = dequantize_storage(&q_storage).unwrap();

        let deq_data = match &*deq_storage.buffer {
            CpuBuffer::F32(v) => v,
            _ => panic!("Expected F32"),
        };

        for (orig, deq) in data.iter().zip(deq_data.iter()) {
            let diff = (orig - deq).abs();
            assert!(diff < 0.05, "Diff too large: {} vs {}", orig, deq);
        }
    }

    #[test]
    /// `test_quantized_matmul`.
    fn test_quantized_matmul() {
        // LHS: 2x32
        let mut lhs_data = vec![0.0f32; 64];
        for (i, d) in lhs_data.iter_mut().enumerate() {
            *d = (i as f32 % 5.0) - 2.0;
        }
        let lhs_f32 = CpuStorage::from_contiguous(CpuBuffer::F32(lhs_data.clone()), vec![2, 32]);
        let lhs_q8 = quantize_storage(&lhs_f32).unwrap();

        // RHS: 3x32
        let mut rhs_data = vec![0.0f32; 96];
        for (i, d) in rhs_data.iter_mut().enumerate() {
            *d = (i as f32 % 4.0) - 1.5;
        }
        let rhs_f32 = CpuStorage::from_contiguous(CpuBuffer::F32(rhs_data.clone()), vec![3, 32]);
        let rhs_q8 = quantize_storage(&rhs_f32).unwrap();

        let out_storage = quantized_matmul_storage(&lhs_q8, &rhs_q8).unwrap();

        assert_eq!(out_storage.shape, vec![2, 3]);

        // Just check that it computes something non-zero and doesn't crash.
        // A more rigorous test would compare it precisely with f32 matmul.
        let out_data = match &*out_storage.buffer {
            CpuBuffer::F32(v) => v,
            _ => panic!("Expected F32"),
        };

        assert_eq!(out_data.len(), 6);
        for &val in out_data {
            assert!(val.abs() > 0.0);
        }
    }

    #[test]
    fn unsupported_quantized_elementwise_arithmetic_is_typed() {
        let source = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0; 32]), vec![32]);
        let quantized = quantize_storage(&source).unwrap();

        let error = crate::cpu::ops::elementwise::add_storage(&quantized, &quantized).unwrap_err();
        if let Error::UnsupportedDType { dtype, backend, op } = error {
            assert_eq!(dtype, DTypeId::Q8_0.descriptor());
            assert_eq!(backend, "cpu");
            assert_eq!(op, "construct arithmetic result");
        } else {
            panic!("expected Error::UnsupportedDType, got {:?}", error);
        }
    }

    /// Issue #95: NVFP4 roundtrip meets the documented bound
    /// (`|err| <= 0.25 * block_amax`), per block, over adversarial
    /// magnitudes — tiny (scale-subnormal), grid-exact, mid-tie, and large.
    #[test]
    fn nvfp4_roundtrip_meets_quarter_amax_bound() {
        // 4 blocks of 16: exact-grid, ties, small, large.
        let mut data = vec![0.0f32; 64];
        let grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        for (i, v) in data[0..16].iter_mut().enumerate() {
            *v = grid[i % 8] * if i % 2 == 0 { 1.0 } else { -1.0 };
        }
        for (i, v) in data[16..32].iter_mut().enumerate() {
            *v = 0.25 + i as f32 * 0.31; // tie-straddling sweep
        }
        for (i, v) in data[32..48].iter_mut().enumerate() {
            *v = (i as f32 - 8.0) * 0.05; // amax 0.4: E4M3-normal scale
        }
        for (i, v) in data[48..64].iter_mut().enumerate() {
            *v = (i as f32 - 8.0) * 17.3; // amax ~138, E4M3-normal scale
        }
        let storage = CpuStorage::from_contiguous(CpuBuffer::F32(data.clone()), vec![4, 16]);
        let q = quantize_nvfp4_storage(&storage).unwrap();
        assert_eq!(q.buffer.dtype_id(), DTypeId::NVFP4);
        let back = dequantize_storage(&q).unwrap();
        let out = match &*back.buffer {
            CpuBuffer::F32(v) => v,
            _ => panic!("expected F32"),
        };
        for block in 0..4 {
            let amax = data[block * 16..(block + 1) * 16]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            // Zero-ish blocks decode bit-exactly; otherwise the bound.
            for i in 0..16 {
                let err = (data[block * 16 + i] - out[block * 16 + i]).abs();
                assert!(
                    err <= 0.25 * amax,
                    "NVFP4 block {block} elem {i}: {} vs {} (amax {amax})",
                    data[block * 16 + i],
                    out[block * 16 + i]
                );
            }
        }
        // The exact-grid block (block 0, amax 6 → scale E4M3(1)=1) is exact.
        for i in 0..16 {
            assert_eq!(out[i], data[i], "grid value {i} must round-trip exactly");
        }
    }

    /// Issue #95: an NVFP4 block whose scale underflows E4M3 (block_amax/6
    /// below 2^-9) decodes as zeros — a documented flush, with error bounded
    /// by the block amax itself rather than the quarter-amax bound.
    #[test]
    fn nvfp4_subnormal_scale_block_flushes_to_zero() {
        let data: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 1e-4).collect();
        let storage = CpuStorage::from_contiguous(CpuBuffer::F32(data.clone()), vec![16]);
        let q = quantize_nvfp4_storage(&storage).unwrap();
        let back = dequantize_storage(&q).unwrap();
        let out = match &*back.buffer {
            CpuBuffer::F32(v) => v,
            _ => panic!("expected F32"),
        };
        let amax = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        assert!(amax < 0.0117, "test premise: scale must underflow E4M3");
        for (x, y) in data.iter().zip(out.iter()) {
            assert_eq!(*y, 0.0);
            assert!((x - y).abs() <= amax);
        }
    }

    /// Issue #95: MXFP4 roundtrip meets `|err| <= 0.5 * block_amax`.
    #[test]
    fn mxfp4_roundtrip_meets_half_amax_bound() {
        let mut data = vec![0.0f32; 64];
        for (i, v) in data.iter_mut().enumerate() {
            // Mixed magnitudes: sub-unit, unit, tens — amax ~30.
            *v = ((i * 37) % 61) as f32 * 0.53 - 15.0;
        }
        data[0] = 0.0;
        let storage = CpuStorage::from_contiguous(CpuBuffer::F32(data.clone()), vec![2, 32]);
        let q = quantize_mxfp4_storage(&storage).unwrap();
        assert_eq!(q.buffer.dtype_id(), DTypeId::MXFP4);
        let back = dequantize_storage(&q).unwrap();
        let out = match &*back.buffer {
            CpuBuffer::F32(v) => v,
            _ => panic!("expected F32"),
        };
        for block in 0..2 {
            let amax = data[block * 32..(block + 1) * 32]
                .iter()
                .map(|v| v.abs())
                .fold(0.0f32, f32::max);
            for i in 0..32 {
                let err = (data[block * 32 + i] - out[block * 32 + i]).abs();
                assert!(
                    err <= 0.5 * amax,
                    "MXFP4 block {block} elem {i}: {} vs {} (amax {amax})",
                    data[block * 32 + i],
                    out[block * 32 + i]
                );
            }
        }
    }

    /// Issue #95: zero blocks are bit-exact zeros; misaligned lengths are
    /// refused with the block size named.
    #[test]
    fn fp4_zero_blocks_are_exact_and_misaligned_lengths_refused() {
        let zeros = CpuStorage::from_contiguous(CpuBuffer::F32(vec![0.0; 32]), vec![32]);
        for (quantize, block_len) in [
            (
                quantize_nvfp4_storage as fn(&CpuStorage) -> Result<CpuStorage>,
                16,
            ),
            (
                quantize_mxfp4_storage as fn(&CpuStorage) -> Result<CpuStorage>,
                32,
            ),
        ] {
            let q = quantize(&zeros).unwrap();
            let back = dequantize_storage(&q).unwrap();
            let out = match &*back.buffer {
                CpuBuffer::F32(v) => v,
                _ => panic!("expected F32"),
            };
            assert!(out.iter().all(|&v| v == 0.0));
            let odd = CpuStorage::from_contiguous(CpuBuffer::F32(vec![1.0; 20]), vec![20]);
            let err = quantize(&odd).unwrap_err().to_string();
            assert!(err.contains(&block_len.to_string()), "{err}");
        }
        // Non-F32 input is refused, like Q8_0's quantize.
        let i64buf = CpuStorage::from_contiguous(CpuBuffer::I64(vec![1; 16]), vec![16]);
        assert!(quantize_nvfp4_storage(&i64buf).is_err());
    }

    /// Issue #95, checkpoint-path proof at storage level: quantized FP4
    /// blocks survive the `to_bytes`/`from_bytes` wire (scale-first
    /// interleaved order) and decode back to the same values.
    #[test]
    fn fp4_blocks_survive_the_storage_wire() {
        use crate::cpu::CpuBackendImpl;
        use incin_core::backend_authoring::HostInterop;
        use incin_core::tensor::device::{Cpu, DeviceId};
        use incin_core::tensor::dtype::{MXFP4, NVFP4};

        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.77).collect();
        let f32buf = CpuStorage::from_contiguous(CpuBuffer::F32(data.clone()), vec![2, 16]);

        // NVFP4: [32] = two 16-blocks = 18 bytes on the wire.
        let q = quantize_nvfp4_storage(&f32buf).unwrap();
        let wire = CpuBackendImpl::<Cpu>::to_bytes::<NVFP4>(&q).unwrap();
        assert_eq!(wire.len(), 18);
        let back = CpuBackendImpl::<Cpu>::from_bytes::<NVFP4>(
            &wire,
            &[2, 16],
            DTypeId::NVFP4.descriptor(),
            &DeviceId::cpu(),
        )
        .unwrap();
        assert_eq!(back.buffer.dtype_id(), DTypeId::NVFP4);
        let rt = dequantize_storage(&back).unwrap();
        let out = match &*rt.buffer {
            CpuBuffer::F32(v) => v,
            _ => panic!("expected F32"),
        };
        let amax = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        for (x, y) in data.iter().zip(out.iter()) {
            assert!((x - y).abs() <= 0.25 * amax, "{x} vs {y}");
        }

        // MXFP4: [32] = one 32-block = 17 bytes on the wire.
        let q = quantize_mxfp4_storage(&f32buf).unwrap();
        let wire = CpuBackendImpl::<Cpu>::to_bytes::<MXFP4>(&q).unwrap();
        assert_eq!(wire.len(), 17);
        let back = CpuBackendImpl::<Cpu>::from_bytes::<MXFP4>(
            &wire,
            &[2, 16],
            DTypeId::MXFP4.descriptor(),
            &DeviceId::cpu(),
        )
        .unwrap();
        assert_eq!(back.buffer.dtype_id(), DTypeId::MXFP4);
        assert_eq!(
            CpuBackendImpl::<Cpu>::to_bytes::<MXFP4>(&back).unwrap(),
            wire,
            "wire bytes must round-trip bit-identically"
        );
    }
}
