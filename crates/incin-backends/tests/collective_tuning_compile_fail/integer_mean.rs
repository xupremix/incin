use incin_backends::dist::{CollectiveTuningProblem, TuneAllReduce};
use incin_core::dist::mesh::TopologyFingerprint;
use incin_core::dist::{GroupId, Mean};
use incin_core::exec::Determinism;
use incin_core::typenum::U32;

fn integer_mean(topology: &TopologyFingerprint) {
    let _ = CollectiveTuningProblem::new_static::<u32, U32, TuneAllReduce<Mean>>(
        GroupId::new(1, 2).unwrap(),
        topology,
        Determinism::Permitted,
        1024,
    );
}

fn main() {}
