use incin_core::exec::{LossScaling, PrecisionPolicy};
use incin_core::prelude::DTypeId;

#[test]
fn precision_policy_presets_and_construction() {
    let fp32 = PrecisionPolicy::fp32();
    assert_eq!(fp32.parameter, DTypeId::F32);
    assert_eq!(fp32.compute, DTypeId::F32);
    assert_eq!(fp32.accumulator, DTypeId::F32);
    assert_eq!(fp32.output, DTypeId::F32);
    assert_eq!(fp32.loss_scaling, LossScaling::None);

    let fp16 = PrecisionPolicy::mixed_f16();
    assert_eq!(fp16.parameter, DTypeId::F32);
    assert_eq!(fp16.compute, DTypeId::F16);
    assert_eq!(fp16.accumulator, DTypeId::F32);
    assert_eq!(fp16.output, DTypeId::F16);
    assert_eq!(fp16.loss_scaling.scale(), 65536.0);

    let bf16 = PrecisionPolicy::mixed_bf16();
    assert_eq!(bf16.parameter, DTypeId::F32);
    assert_eq!(bf16.compute, DTypeId::BF16);
    assert_eq!(bf16.accumulator, DTypeId::F32);
    assert_eq!(bf16.output, DTypeId::BF16);
    assert_eq!(bf16.loss_scaling, LossScaling::None);
}

#[test]
fn dynamic_loss_scaling_growth_and_overflow_backoff() {
    let mut scaler = LossScaling::dynamic(1024.0, 2.0, 0.5, 3);
    assert_eq!(scaler.scale(), 1024.0);

    // 2 finite steps -> scale unchanged
    scaler.update(false);
    scaler.update(false);
    assert_eq!(scaler.scale(), 1024.0);

    // 3rd finite step -> growth factor applied (1024 * 2 = 2048)
    scaler.update(false);
    assert_eq!(scaler.scale(), 2048.0);

    // Overflow detected -> backoff factor applied (2048 * 0.5 = 1024)
    scaler.update(true);
    assert_eq!(scaler.scale(), 1024.0);

    // Backoff cannot reduce scale below 1.0
    let mut min_scaler = LossScaling::dynamic(1.0, 2.0, 0.5, 3);
    min_scaler.update(true);
    assert_eq!(min_scaler.scale(), 1.0);
}
