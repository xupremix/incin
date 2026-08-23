//! Integration coverage for `requires_cuda` on the documented public surface.
use incin_backends::tuning::identity::{DeviceFingerprint, SoftwareVersion};
use incin_core::prelude::{Cpu, Cuda};

fn requires_cuda(_: DeviceFingerprint<Cuda>) {}

fn main() {
    let cpu = DeviceFingerprint::<Cpu>::new(
        "host-cpu-0",
        "x86_64-v3",
        SoftwareVersion::new(6, 12, 0),
    )
    .unwrap();
    requires_cuda(cpu);
}
