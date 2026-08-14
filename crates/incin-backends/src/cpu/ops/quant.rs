use crate::cpu::CpuBackendImpl;
use crate::cpu::storage::{BlockQ8_0, CpuBuffer, CpuStorage};
use incin_core::prelude::{
    DType, Device, Error, FloatDType, OperationKind, Q8_0, QuantDType, Result, ShapeBuf,
    StorageBackend,
};
use incin_core::__backend_compat::legacy::QuantizedOps;

extern crate alloc;
use alloc::vec::Vec;

impl<D: Device> QuantizedOps<Self> for CpuBackendImpl<D> {
    /// `quantize`.
    fn quantize<K: FloatDType, Q: QuantDType>(
        _t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<Q>> {
        if core::any::TypeId::of::<Q>() != core::any::TypeId::of::<Q8_0>()
            || core::any::TypeId::of::<K>() != core::any::TypeId::of::<f32>()
        {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantize",
                backend: "Cpu (only F32 to Q8_0 supported)",
            });
        }

        let f32_data = match &*_t.buffer {
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

        let blocks_count = n / 32;
        let mut blocks = Vec::with_capacity(blocks_count);
        for chunk in f32_data.chunks_exact(32) {
            let mut max_abs = 0.0f32;
            for &val in chunk {
                let abs = val.abs();
                if abs > max_abs {
                    max_abs = abs;
                }
            }
            let d = max_abs / 127.0;
            let inv_d = if d == 0.0 { 0.0 } else { 1.0 / d };

            let mut qs = [0i8; 32];
            for i in 0..32 {
                let q = (chunk[i] * inv_d).round() as i8;
                qs[i] = q;
            }
            blocks.push(BlockQ8_0 {
                d: half::f16::from_f32(d),
                qs,
            });
        }

        Ok(CpuStorage::from_contiguous(
            CpuBuffer::Q8_0(blocks),
            _t.shape.to_vec(),
        ))
    }

    /// `dequantize`.
    fn dequantize<Q: QuantDType, K: FloatDType>(
        _t: &<Self as StorageBackend>::Storage<Q>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        if core::any::TypeId::of::<Q>() != core::any::TypeId::of::<Q8_0>()
            || core::any::TypeId::of::<K>() != core::any::TypeId::of::<f32>()
        {
            return Err(Error::UnsupportedBackendOperation {
                op: "dequantize",
                backend: "Cpu (only Q8_0 to F32 supported)",
            });
        }

        let q8_data = match &*_t.buffer {
            CpuBuffer::Q8_0(v) => v,
            _ => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "dequantize",
                    backend: "Cpu (expected Q8_0 buffer)",
                });
            }
        };

        let n = q8_data.len() * 32;
        let mut f32_data = Vec::with_capacity(n);
        for block in q8_data {
            let d = block.d.to_f32();
            for i in 0..32 {
                f32_data.push(block.qs[i] as f32 * d);
            }
        }

        Ok(CpuStorage::from_contiguous(
            CpuBuffer::F32(f32_data),
            _t.shape.to_vec(),
        ))
    }

    /// `quantized_matmul`.
    fn quantized_matmul<Q: QuantDType>(
        _lhs: &<Self as StorageBackend>::Storage<Q>,
        _rhs: &<Self as StorageBackend>::Storage<Q>,
    ) -> Result<<Self as StorageBackend>::Storage<f32>> {
        if core::any::TypeId::of::<Q>() != core::any::TypeId::of::<Q8_0>() {
            return Err(Error::UnsupportedBackendOperation {
                op: "quantized_matmul",
                backend: "Cpu (only Q8_0 supported)",
            });
        }

        let lhs_data = match &*_lhs.buffer {
            CpuBuffer::Q8_0(v) => v,
            _ => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "quantized_matmul",
                    backend: "Cpu (lhs expected Q8_0 buffer)",
                });
            }
        };

        let rhs_data = match &*_rhs.buffer {
            CpuBuffer::Q8_0(v) => v,
            _ => {
                return Err(Error::UnsupportedBackendOperation {
                    op: "quantized_matmul",
                    backend: "Cpu (rhs expected Q8_0 buffer)",
                });
            }
        };

        let lhs_shape = &_lhs.shape;
        let rhs_shape = &_rhs.shape;
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
        let m: usize = crate::cpu::stride::checked_numel(&(lhs_shape[..lhs_shape.len() - 1]))?;

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
        let use_avx2 = is_x86_feature_detected!("avx2");
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        let use_avx2 = false;

        for i in 0..m {
            for j in 0..n {
                let lhs_row_start = i * blocks_per_row;
                let rhs_row_start = j * blocks_per_row;

                let sum = if use_avx2 {
                    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
                    {
                        // SAFETY: we just checked if the CPU supports AVX2.
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
            out_shape.to_vec(),
        ))
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
#[inline]
/// `vec_dot_q8_0_avx2`.
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
    use incin_core::__backend_compat::legacy::NumericOps;

    /// `TestBackend`.
    type TestBackend = CpuBackendImpl<incin_core::prelude::Cpu>;

    #[test]
    /// `test_quantize_dequantize_fidelity`.
    fn test_quantize_dequantize_fidelity() {
        let mut data = vec![0.0f32; 64];
        for (i, d) in data.iter_mut().enumerate() {
            *d = (i as f32 - 32.0) * 0.1; // ranging -3.2 to +3.1
        }

        let storage = CpuStorage::from_contiguous(CpuBuffer::F32(data.clone()), vec![2, 32]);

        let q_storage = TestBackend::quantize::<f32, Q8_0>(&storage).unwrap();
        let deq_storage = TestBackend::dequantize::<Q8_0, f32>(&q_storage).unwrap();

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
        let lhs_q8 = TestBackend::quantize::<f32, Q8_0>(&lhs_f32).unwrap();

        // RHS: 3x32
        let mut rhs_data = vec![0.0f32; 96];
        for (i, d) in rhs_data.iter_mut().enumerate() {
            *d = (i as f32 % 4.0) - 1.5;
        }
        let rhs_f32 = CpuStorage::from_contiguous(CpuBuffer::F32(rhs_data.clone()), vec![3, 32]);
        let rhs_q8 = TestBackend::quantize::<f32, Q8_0>(&rhs_f32).unwrap();

        let out_storage = TestBackend::quantized_matmul::<Q8_0>(&lhs_q8, &rhs_q8).unwrap();

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
        let quantized = TestBackend::quantize::<f32, Q8_0>(&source).unwrap();

        let error = TestBackend::add::<Q8_0>(&quantized, &quantized).unwrap_err();
        if let Error::UnsupportedDType { dtype, backend, op } = error {
            assert_eq!(dtype, DTypeId::Q8_0.descriptor());
            assert_eq!(backend, "cpu");
            assert_eq!(op, "construct arithmetic result");
        } else {
            panic!("expected Error::UnsupportedDType, got {:?}", error);
        }
    }
}
