//! Integration coverage for `main` on the documented public surface.
use incin_backends::tuning::identity::{
    ProcessLayoutFingerprint, SoftwareVersion, StaticWorld, TransportFingerprint,
    TuningTopologyFingerprint,
};
use incin_core::typenum::U0;

fn main() {
    let _ = TuningTopologyFingerprint::<StaticWorld<U0>>::new(
        vec![],
        vec![],
        TransportFingerprint::new("nccl", SoftwareVersion::new(2, 28, 3)).unwrap(),
        ProcessLayoutFingerprint::new(0, 0),
    );
}
