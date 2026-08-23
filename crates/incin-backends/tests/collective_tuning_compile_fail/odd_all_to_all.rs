//! Integration coverage for `odd_all_to_all` on the documented public surface.
use incin_backends::dist::{CollectiveTuningProblem, TuneAllToAll};
use incin_core::dist::mesh::TopologyFingerprint;
use incin_core::dist::GroupId;
use incin_core::exec::Determinism;
use incin_core::typenum::U3;

fn odd_all_to_all(topology: &TopologyFingerprint) {
    let _ = CollectiveTuningProblem::new_static::<f32, U3, TuneAllToAll>(
        GroupId::new(1, 2).unwrap(),
        topology,
        Determinism::Permitted,
        1024,
    );
}

fn main() {}
