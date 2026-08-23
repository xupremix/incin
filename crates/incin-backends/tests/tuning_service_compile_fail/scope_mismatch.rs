//! Integration coverage for `requires_collective` on the documented public surface.
use incin_backends::tuning::{
    identity::{CompilerFingerprint, DeviceFingerprint, SoftwareVersion, TuningEnvironmentFingerprint},
    service::{CollectiveTuning, KernelTuning, TuningContext},
};
use incin_core::{exec::Determinism, prelude::Cuda};
use std::time::Duration;

fn requires_collective(_: TuningContext<Cuda, CollectiveTuning>) {}

fn main() {
    let environment = TuningEnvironmentFingerprint::<Cuda>::new(
        DeviceFingerprint::new(
            "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            "sm_90",
            SoftwareVersion::new(12, 8, 0),
        )
        .unwrap(),
        CompilerFingerprint::new(
            "nvrtc",
            SoftwareVersion::new(12, 8, 0),
            "sm_90",
            &["default-math"],
        )
        .unwrap(),
    )
    .unwrap();
    let context = TuningContext::<Cuda, KernelTuning>::kernel(
        environment,
        Determinism::Permitted,
        0,
        Duration::from_secs(1),
    )
    .unwrap();
    requires_collective(context);
}
