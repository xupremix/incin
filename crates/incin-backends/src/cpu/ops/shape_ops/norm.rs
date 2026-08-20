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
    let mut out = Vec::with_capacity(total);
    for run in 0..batch * groups {
        let mut sum = 0.0;
        let mut sq_sum = 0.0;
        for i in 0..group_size {
            let index = crate::cpu::ops::elementwise::flat_to_nd(run * group_size + i, &t.shape);
            let value = t.get(&index);
            sum += value;
            sq_sum += value * value;
        }
        let mean = sum / group_size as f64;
        let variance = (sq_sum / group_size as f64 - mean * mean).max(0.0);
        let inv_std = 1.0 / (variance + eps).sqrt();
        for i in 0..group_size {
            let index = crate::cpu::ops::elementwise::flat_to_nd(run * group_size + i, &t.shape);
            out.push((t.get(&index) - mean) * inv_std);
        }
    }
    Ok(CpuStorage::from_contiguous(
        t.buffer.from_f64_values(out)?,
        &t.shape,
    ))
}

pub(crate) fn instance_norm_storage(t: &CpuStorage, eps: f64) -> Result<CpuStorage> {
    let channels = if t.shape.len() >= 2 { t.shape[1] } else { 1 };
    group_norm_storage(t, channels, eps)
}
