use crate::cpu::{CpuBackendImpl, CpuBuffer};
use incin_core::backend_authoring::{Backend, OptimizerOps};
use incin_core::prelude::{DType, Device, Result};

impl<T: DType, D: Device> OptimizerOps<Self> for CpuBackendImpl<T, D> {
    /// Applies a fused AdamW optimization step on the backend.
    ///
    /// This directly modifies the buffers (`var`, `m`, `v`) in place, in a
    /// single pass over them rather than through the dozen intermediate
    /// tensors the composed form allocates.
    ///
    /// The fused pass is written against `f32` only. Every other dtype falls
    /// back to [`OptimizerOps::adamw_step`]'s default body, which is composed
    /// from `NumericOps`/`FloatOps` and so is already dtype-generic. The
    /// fallback used to be an `UnsupportedBackendOperation` error, which made
    /// an optimizer that works refuse an `f64` or `f16` parameter for no
    /// reason other than that the fast path had not been written for it.
    fn adamw_step<K: DType>(
        _var: &mut <Self as Backend>::RawVar,
        _grad: &<Self as Backend>::Storage<K>,
        _m: &mut <Self as Backend>::Storage<K>,
        _v: &mut <Self as Backend>::Storage<K>,
        _lr: f64,
        _beta1: f64,
        _beta2: f64,
        _eps: f64,
        _weight_decay: f64,
        _step: usize,
    ) -> Result<()> {
        let var_storage = crate::cpu::var::var_as_tensor(_var)?;

        let mut v_buf_arc = var_storage.buffer.clone();
        let v_buf_mut = alloc::sync::Arc::make_mut(&mut v_buf_arc);

        let m_buffer_mut = alloc::sync::Arc::make_mut(&mut _m.buffer);
        let v2_buffer_mut = alloc::sync::Arc::make_mut(&mut _v.buffer);

        if let (
            CpuBuffer::F32(v_vec),
            CpuBuffer::F32(g_vec),
            CpuBuffer::F32(m_vec),
            CpuBuffer::F32(v2_vec),
        ) = (v_buf_mut, &*_grad.buffer, m_buffer_mut, v2_buffer_mut)
        {
            let lr_f32 = _lr as f32;
            let beta1_f32 = _beta1 as f32;
            let beta2_f32 = _beta2 as f32;
            let eps_f32 = _eps as f32;
            let wd_f32 = _weight_decay as f32;

            let bias_correction1 = 1.0 - beta1_f32.powi(_step as i32);
            let bias_correction2 = 1.0 - beta2_f32.powi(_step as i32);
            let effective_lr = lr_f32 * (bias_correction2.sqrt() / bias_correction1);

            // Assume T = f32 for CpuBackendImpl
            let v_data: &mut [f32] = v_vec.as_mut_slice();
            let g_data: &[f32] = g_vec.as_slice();
            let m_data: &mut [f32] = m_vec.as_mut_slice();
            let v2_data: &mut [f32] = v2_vec.as_mut_slice();

            for i in 0..v_data.len() {
                let grad = g_data[i];
                v_data[i] -= lr_f32 * wd_f32 * v_data[i];
                m_data[i] = beta1_f32 * m_data[i] + (1.0 - beta1_f32) * grad;
                v2_data[i] = beta2_f32 * v2_data[i] + (1.0 - beta2_f32) * grad * grad;
                v_data[i] -= effective_lr * m_data[i] / (v2_data[i].sqrt() + eps_f32);
            }

            let updated_storage = crate::cpu::storage::CpuStorage::try_from_parts(
                v_buf_arc,
                var_storage.shape.to_vec(),
                var_storage.strides.to_vec(),
                var_storage.offset_elements,
            )?;
            crate::cpu::var::assign_var(_var, &updated_storage)?;
            return Ok(());
        }

        incin_core::backend_authoring::adamw_step_composed::<Self, K>(
            _var,
            _grad,
            _m,
            _v,
            _lr,
            _beta1,
            _beta2,
            _eps,
            _weight_decay,
            _step,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::storage::CpuStorage;

    type TestBackend = CpuBackendImpl<f32, incin_core::prelude::Cpu>;

    fn storage(buffer: CpuBuffer, len: usize) -> CpuStorage {
        CpuStorage::from_contiguous(buffer, vec![len])
    }

    fn step_once(var_buf: CpuBuffer, grad_buf: CpuBuffer, len: usize) -> Result<CpuStorage> {
        let mut var = crate::cpu::var::var_from_tensor(&storage(var_buf, len))?;
        let grad = storage(grad_buf, len);
        let mut m = storage(CpuBuffer::F64(vec![0.0; len]), len);
        let mut v = storage(CpuBuffer::F64(vec![0.0; len]), len);
        if !matches!(&*grad.buffer, CpuBuffer::F64(_)) {
            m = storage(CpuBuffer::F32(vec![0.0; len]), len);
            v = storage(CpuBuffer::F32(vec![0.0; len]), len);
        }
        <TestBackend as OptimizerOps<TestBackend>>::adamw_step::<f32>(
            &mut var, &grad, &mut m, &mut v, 0.1, 0.9, 0.999, 1e-8, 0.0, 1,
        )?;
        crate::cpu::var::var_as_tensor(&var)
    }

    /// The fused pass covers `f32` only, and used to report every other dtype
    /// as an unsupported backend operation. An `f64` parameter is a valid
    /// AdamW request, so it has to update rather than refuse.
    #[test]
    fn a_non_f32_parameter_updates_through_the_composed_path() {
        let updated = step_once(
            CpuBuffer::F64(vec![1.0, 2.0]),
            CpuBuffer::F64(vec![0.5, -0.5]),
            2,
        )
        .expect("f64 adamw must not be refused");

        // The first step of AdamW moves each parameter by almost exactly the
        // learning rate, signed by the gradient, whatever the gradient's size.
        assert!(
            (updated.get(&[0]) - 0.9).abs() < 1e-3,
            "{}",
            updated.get(&[0])
        );
        assert!(
            (updated.get(&[1]) - 2.1).abs() < 1e-3,
            "{}",
            updated.get(&[1])
        );
    }

    /// The fused and composed paths are two spellings of one update rule, so
    /// they have to agree where they overlap. Without this, the `f32` path
    /// could drift from the definition every other dtype follows.
    #[test]
    fn the_fused_f32_path_agrees_with_the_composed_one() {
        let fused = step_once(
            CpuBuffer::F32(vec![1.0, 2.0]),
            CpuBuffer::F32(vec![0.5, -0.5]),
            2,
        )
        .unwrap();
        let composed = step_once(
            CpuBuffer::F64(vec![1.0, 2.0]),
            CpuBuffer::F64(vec![0.5, -0.5]),
            2,
        )
        .unwrap();

        for i in 0..2 {
            let (a, b) = (fused.get(&[i]), composed.get(&[i]));
            assert!((a - b).abs() < 1e-4, "element {i}: fused {a}, composed {b}");
        }
    }
}
