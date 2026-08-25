use super::*;

pub(crate) fn group_norm_storage(t: &CpuStorage, groups: usize, eps: f64) -> Result<CpuStorage> {
    let total = crate::cpu::stride::checked_numel(&t.shape)?;
    if groups == 0 {
        return Err(Error::Msg("group_norm: groups must be non-zero".into()));
    }
    let channels = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
    if channels % groups != 0 {
        return Err(Error::Msg(
            "group_norm: channels must be divisible by groups".into(),
        ));
    }
    let (batch, spatial) = if t.shape.len() >= 2 {
        (t.shape[0], t.shape[2..].iter().product::<usize>())
    } else {
        (1, total)
    };
    let group_size = channels / groups * spatial;

    // Normalization divides by statistics that are functions of the input,
    // so every step composes from tape-tracked primitives: a tape-silent
    // mean or variance here would cut the statistical path out of the
    // backward pass exactly as it once did for training-mode batch norm.
    // Group runs are contiguous in memory, so the regrouping is a pure
    // reshape recorded like any other view.
    let runs = batch * groups;
    let flat = reshape_storage(t, &[runs, group_size])?;
    let mean = crate::cpu::ops::reduce::mean_keepdim(&flat, 1)?;
    let centered = crate::cpu::ops::elementwise::sub_storage(&flat, &mean)?;
    let squared = crate::cpu::ops::elementwise::mul_storage(&centered, &centered)?;
    let variance = crate::cpu::ops::reduce::mean_keepdim(&squared, 1)?;
    let guarded = canonical_add_scalar(&variance, eps)?;
    let std = canonical_sqrt(&guarded)?;
    let normalized = crate::cpu::ops::elementwise::div_storage(&centered, &std)?;
    reshape_storage(&normalized, &t.shape)
}

pub(crate) fn instance_norm_storage(t: &CpuStorage, eps: f64) -> Result<CpuStorage> {
    let channels = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
    group_norm_storage(t, channels, eps)
}
