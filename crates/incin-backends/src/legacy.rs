//! Backend-local compatibility helpers that have not yet moved to descriptors.
//!
//! This module is deliberately crate-private. It is not part of the backend
//! authoring contract; new implementations should use exact `Execute<O>`
//! descriptors. AdamW remains here because its mutation execution site is not
//! representable by the current `Execute` output contract.

use incin_core::__backend_compat::legacy::{FloatOps, NumericOps};
use incin_core::__backend_compat::legacy::ReductionOps;
use incin_core::backend_authoring::VariableBackend;
use incin_core::prelude::{Backend, DType, Reduction, Result};

/// Backend-local composed loss helpers retained for compatibility and parity
/// tests. Stable loss operations use exact `Execute<O>` descriptors.
pub trait LossOps<B: Backend + NumericOps<B> + FloatOps<B> + ReductionOps<B>>:
    NumericOps<B> + FloatOps<B> + ReductionOps<B>
{
    fn mse_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: Reduction,
    ) -> Result<B::Storage<K>> {
        let diff = <B as NumericOps<B>>::sub::<K>(pred, target)?;
        let sq = <B as NumericOps<B>>::mul::<K>(&diff, &diff)?;
        match reduction {
            Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&sq),
            Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&sq),
            Reduction::None => Ok(sq),
        }
    }

    fn l1_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: Reduction,
    ) -> Result<B::Storage<K>> {
        let diff = <B as NumericOps<B>>::sub::<K>(pred, target)?;
        let abs_diff = <B as FloatOps<B>>::abs::<K>(&diff)?;
        match reduction {
            Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&abs_diff),
            Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&abs_diff),
            Reduction::None => Ok(abs_diff),
        }
    }

    fn bce_with_logits_loss<K: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<K>,
        reduction: Reduction,
    ) -> Result<B::Storage<K>> {
        let max_x_0 = <B as FloatOps<B>>::relu::<K>(pred)?;
        let x_times_z = <B as NumericOps<B>>::mul::<K>(pred, target)?;
        let term1 = <B as NumericOps<B>>::sub::<K>(&max_x_0, &x_times_z)?;
        let abs_x = <B as FloatOps<B>>::abs::<K>(pred)?;
        let neg_abs_x = <B as FloatOps<B>>::neg::<K>(&abs_x)?;
        let exp_neg_abs_x = <B as FloatOps<B>>::exp::<K>(&neg_abs_x)?;
        let one_plus = <B as FloatOps<B>>::add_scalar_float::<K>(&exp_neg_abs_x, 1.0)?;
        let term2 = <B as FloatOps<B>>::log::<K>(&one_plus)?;
        let loss_elem = <B as NumericOps<B>>::add::<K>(&term1, &term2)?;
        match reduction {
            Reduction::Mean => <B as ReductionOps<B>>::mean_all::<K>(&loss_elem),
            Reduction::Sum => <B as ReductionOps<B>>::sum_all::<K>(&loss_elem),
            Reduction::None => Ok(loss_elem),
        }
    }

    fn cross_entropy_loss<K: DType, KInt: DType>(
        pred: &B::Storage<K>,
        target: &B::Storage<KInt>,
        reduction: Reduction,
    ) -> Result<B::Storage<K>>;
}

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
