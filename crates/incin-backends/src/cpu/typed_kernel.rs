//! Type-specialized CPU execution kernels eliminating f64 widening (PRF-004).

use half::{bf16, f16};
use rayon::prelude::*;

/// Parallelization threshold for contiguous slice mapping.
pub const PARALLEL_GRAIN: usize = 1024;

/// Trait for types supporting native (non-widening) elementwise CPU operations.
pub trait TypedKernel: Copy + Send + Sync + Default + 'static {
    /// Applies a unary function elementwise across `input`, writing into `output`.
    fn map_unary(input: &[Self], output: &mut [Self], f: impl Fn(Self) -> Self + Send + Sync);

    /// Applies a binary function elementwise across `lhs` and `rhs`, writing into `output`.
    fn map_binary(
        lhs: &[Self],
        rhs: &[Self],
        output: &mut [Self],
        f: impl Fn(Self, Self) -> Self + Send + Sync,
    );
}

macro_rules! impl_typed_kernel {
    ($($t:ty),*) => {
        $(
            impl TypedKernel for $t {
                fn map_unary(input: &[Self], output: &mut [Self], f: impl Fn(Self) -> Self + Send + Sync) {
                    assert_eq!(input.len(), output.len());
                    if input.len() < PARALLEL_GRAIN {
                        for (out, &in_val) in output.iter_mut().zip(input.iter()) {
                            *out = f(in_val);
                        }
                    } else {
                        output
                            .par_chunks_mut(PARALLEL_GRAIN)
                            .zip(input.par_chunks(PARALLEL_GRAIN))
                            .for_each(|(out_chunk, in_chunk)| {
                                for (out, &in_val) in out_chunk.iter_mut().zip(in_chunk.iter()) {
                                    *out = f(in_val);
                                }
                            });
                    }
                }

                fn map_binary(
                    lhs: &[Self],
                    rhs: &[Self],
                    output: &mut [Self],
                    f: impl Fn(Self, Self) -> Self + Send + Sync,
                ) {
                    assert_eq!(lhs.len(), output.len());
                    assert_eq!(rhs.len(), output.len());
                    if lhs.len() < PARALLEL_GRAIN {
                        for ((out, &l), &r) in output.iter_mut().zip(lhs.iter()).zip(rhs.iter()) {
                            *out = f(l, r);
                        }
                    } else {
                        output
                            .par_chunks_mut(PARALLEL_GRAIN)
                            .zip(lhs.par_chunks(PARALLEL_GRAIN))
                            .zip(rhs.par_chunks(PARALLEL_GRAIN))
                            .for_each(|((out_chunk, l_chunk), r_chunk)| {
                                for ((out, &l), &r) in out_chunk.iter_mut().zip(l_chunk.iter()).zip(r_chunk.iter()) {
                                    *out = f(l, r);
                                }
                            });
                    }
                }
            }
        )*
    };
}

impl_typed_kernel!(f32, f64, f16, bf16, u8, i8, u32, i32, i64);

/// Helper to allocate a `Vec<T>` and evaluate `TypedKernel::map_unary`.
pub fn map_unary_typed<T: TypedKernel>(input: &[T], f: impl Fn(T) -> T + Send + Sync) -> Vec<T> {
    let mut out = vec![T::default(); input.len()];
    T::map_unary(input, &mut out, f);
    out
}

/// Helper to allocate a `Vec<T>` and evaluate `TypedKernel::map_binary`.
pub fn map_binary_typed<T: TypedKernel>(
    lhs: &[T],
    rhs: &[T],
    f: impl Fn(T, T) -> T + Send + Sync,
) -> Vec<T> {
    assert_eq!(lhs.len(), rhs.len());
    let mut out = vec![T::default(); lhs.len()];
    T::map_binary(lhs, rhs, &mut out, f);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_typed_kernel_unary_f32() {
        let input: Vec<f32> = (0..2000).map(|i| i as f32).collect();
        let res = map_unary_typed(&input, |x| x * 2.0);
        assert_eq!(res.len(), 2000);
        for (i, &v) in res.iter().enumerate() {
            assert_eq!(v, (i * 2) as f32);
        }
    }

    #[test]
    fn test_typed_kernel_binary_f64() {
        let lhs: Vec<f64> = (0..2000).map(|i| i as f64).collect();
        let rhs: Vec<f64> = (0..2000).map(|i| (i * 3) as f64).collect();
        let res = map_binary_typed(&lhs, &rhs, |a, b| a + b);
        assert_eq!(res.len(), 2000);
        for (i, &v) in res.iter().enumerate() {
            assert_eq!(v, (i * 4) as f64);
        }
    }

    #[test]
    fn test_typed_kernel_f16_bf16() {
        let input: Vec<f16> = (0..500).map(|i| f16::from_f32(i as f32)).collect();
        let res = map_unary_typed(&input, |x| f16::from_f32(x.to_f32() + 1.0));
        assert_eq!(res.len(), 500);
        assert_eq!(res[10].to_f32(), 11.0);

        let input_bf: Vec<bf16> = (0..500).map(|i| bf16::from_f32(i as f32)).collect();
        let res_bf = map_unary_typed(&input_bf, |x| bf16::from_f32(x.to_f32() + 2.0));
        assert_eq!(res_bf.len(), 500);
        assert_eq!(res_bf[10].to_f32(), 12.0);
    }
}
