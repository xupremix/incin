use incin_backends::tuning::{
    cache::CacheKey,
    identity::{
        CompilerFingerprint, DeviceFingerprint, SoftwareVersion, TuningEnvironmentFingerprint,
    },
};
use incin_core::prelude::{Cpu, Cuda};

fn requires_cuda(_: CacheKey<Cuda>) {}

fn main() {
    let environment = TuningEnvironmentFingerprint::<Cpu>::new(
        DeviceFingerprint::new(
            "host-cpu-0",
            "x86_64-v3",
            SoftwareVersion::new(6, 12, 0),
        )
        .unwrap(),
        CompilerFingerprint::new(
            "rustc",
            SoftwareVersion::new(1, 88, 0),
            "x86_64",
            &["target-cpu=v3"],
        )
        .unwrap(),
    )
    .unwrap();
    let key = CacheKey::<Cpu>::new("kernel", &environment, "add:f32").unwrap();
    requires_cuda(key);
}
