//! Integration coverage for `q8_problem` on the documented public surface.
use incin_backends::dist::{CollectiveTuningProblem, TuneAllGather};
use incin_core::dist::mesh::TopologyFingerprint;
use incin_core::dist::GroupId;
use incin_core::exec::Determinism;
use incin_core::prelude::Q8_0;
use incin_core::typenum::U32;

fn q8_problem(topology: &TopologyFingerprint) {
    let _ = CollectiveTuningProblem::new_static::<Q8_0, U32, TuneAllGather>(
        GroupId::new(1, 2).unwrap(),
        topology,
        Determinism::Permitted,
        1024,
    );
}

fn main() {}
