//! Backend-local compatibility helpers that have not yet moved to descriptors.
//!
//! This module is deliberately crate-private. It is not part of the backend
//! authoring contract; new implementations should use exact `Execute<O>`
//! descriptors. AdamW remains here because its mutation execution site is not
//! representable by the current `Execute` output contract.

use incin_core::__backend_compat::legacy::{FloatOps, NumericOps};
use incin_core::backend_authoring::VariableBackend;
use incin_core::prelude::{DType, Result};

pub(crate) trait OptimizerOps<B: VariableBackend + NumericOps<B> + FloatOps<B>> {
    fn adamw_step<K: DType>(
        var: &mut B::RawVar,
        grad: &B::Storage<K>,
        m: &mut B::Storage<K>,
        v: &mut B::Storage<K>,
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        step: usize,
    ) -> Result<()> {
        adamw_step_composed::<B, K>(
            var, grad, m, v, lr, beta1, beta2, eps, weight_decay, step,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn adamw_step_composed<B: VariableBackend + NumericOps<B> + FloatOps<B>, K: DType>(
    var: &mut B::RawVar,
    grad: &B::Storage<K>,
    m: &mut B::Storage<K>,
    v: &mut B::Storage<K>,
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step: usize,
) -> Result<()> {
    let mut t = B::var_as_tensor::<K>(var)?;
    let t_step = step as f64;
    let bias_correction1 = 1.0 - beta1.powf(t_step);
    let bias_correction2 = 1.0 - beta2.powf(t_step);

    if weight_decay > 0.0 {
        let decay = B::mul_scalar_float::<K>(&t, weight_decay * lr)?;
        t = B::sub::<K>(&t, &decay)?;
    }

    let term1_m = B::mul_scalar_float::<K>(m, beta1)?;
    let term2_m = B::mul_scalar_float::<K>(grad, 1.0 - beta1)?;
    let m_t = B::add::<K>(&term1_m, &term2_m)?;

    let grad_sq = B::mul::<K>(grad, grad)?;
    let term1_v = B::mul_scalar_float::<K>(v, beta2)?;
    let term2_v = B::mul_scalar_float::<K>(&grad_sq, 1.0 - beta2)?;
    let v_t = B::add::<K>(&term1_v, &term2_v)?;

    *m = m_t.clone();
    *v = v_t.clone();

    let m_hat = B::mul_scalar_float::<K>(&m_t, 1.0 / bias_correction1)?;
    let v_hat = B::mul_scalar_float::<K>(&v_t, 1.0 / bias_correction2)?;
    let denom = B::add_scalar_float::<K>(&B::sqrt::<K>(&v_hat)?, eps)?;
    let step_val = B::mul_scalar_float::<K>(&B::div::<K>(&m_hat, &denom)?, lr)?;

    let updated = B::sub::<K>(&t, &step_val)?;
    B::assign_var::<K>(var, &updated)?;
    Ok(())
}
