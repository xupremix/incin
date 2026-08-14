//! Backend-local compatibility helpers that have not yet moved to descriptors.
//!
//! This module is deliberately crate-private. It is not part of the backend
//! authoring contract; new implementations should use exact `Execute<O>`
//! descriptors. AdamW remains here because its mutation execution site is not
//! representable by the current `Execute` output contract.

use incin_core::__backend_compat::legacy::{FloatOps, NumericOps};
use incin_core::__backend_compat::legacy::ReductionOps;
use incin_core::prelude::{Backend, DType, Reduction, Result};

/// Backend-local composed loss helpers retained while optional backends finish
/// moving their implementations to exact `Execute<O>` descriptors.
pub(crate) fn mse_loss<B: Backend + NumericOps<B> + FloatOps<B> + ReductionOps<B>, K: DType>(
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

pub(crate) fn l1_loss<B: Backend + NumericOps<B> + FloatOps<B> + ReductionOps<B>, K: DType>(
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

pub(crate) fn bce_with_logits_loss<B: Backend + NumericOps<B> + FloatOps<B> + ReductionOps<B>, K: DType>(
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
