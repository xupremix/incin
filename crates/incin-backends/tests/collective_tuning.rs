//! `DST-012`: coordinated two-rank collective tuning contract evidence.

#![cfg(any(feature = "distributed-reference", feature = "distributed-nccl"))]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use incin_backends::dist::{
    CandidateRound, CollectiveAlgorithm, CollectiveProtocol, CollectiveTuningBudget,
    CollectiveTuningCandidate, CollectiveTuningError, CollectiveTuningProblem, CommitVote,
    LowLatency, RankSampleReport, Ring, Simple, Tree, TuneAllReduce, commit_collective_tuning,
    select_collective_candidate,
};
use incin_core::dist::mesh::{
    DeviceIdentity, DeviceMesh, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::dist::{CollectiveKind, GroupId, Mean, TwoRankDataParallel};
use incin_core::exec::{Determinism, ReduceOp};
use incin_core::prelude::{DTypeId, DeviceId};
use incin_core::typenum::{U2, U3, U16};

#[cfg(feature = "autotune")]
use incin_backends::tuning::{
    cache::CacheKey,
    identity::{
        CompilerFingerprint, DeviceFingerprint, ProcessLayoutFingerprint, SoftwareVersion,
        StaticWorld, TopologyLink, TransportFingerprint, TuningEnvironmentFingerprint,
        TuningTopologyFingerprint,
    },
    service::{
        CollectiveTuning as ServiceCollectiveTuning, CoordinatedWarmupTuning, ServiceDecision,
        TuningContext, TuningService,
    },
};
#[cfg(feature = "autotune")]
use incin_core::prelude::Cuda;

struct TwoNetworkCuda<'a> {
    label: &'a str,
}

impl TopologyProbe for TwoNetworkCuda<'_> {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        (device.kind() == incin_core::prelude::DeviceKind::Cuda && device.ordinal() < 2).then(
            || {
                DeviceIdentity::new(
                    device,
                    format!("{}-{}", self.label, device.ordinal()),
                    "sm_90".to_string(),
                )
            },
        )
    }

    fn link(&self, from: DeviceId, to: DeviceId) -> LinkClass {
        if from == to {
            LinkClass::SameDevice
        } else {
            LinkClass::Network
        }
    }

    fn transport(&self) -> TransportVersion {
        TransportVersion::new("nccl".to_string(), 2, 27, 3)
    }

    fn layout(&self) -> ProcessLayout {
        ProcessLayout::ProcessPerRank { rank: 0, world: 2 }
    }
}

fn topology(label: &str) -> DeviceMesh<TwoRankDataParallel> {
    let probe = TwoNetworkCuda { label };
    DeviceMesh::bind(&[DeviceId::cuda(0), DeviceId::cuda(1)], &probe).unwrap()
}

fn group() -> GroupId {
    GroupId::new(7, 2).unwrap()
}

fn problem(determinism: Determinism) -> CollectiveTuningProblem {
    let mesh = topology("GPU-TUNE");
    CollectiveTuningProblem::new_static::<f32, U16, TuneAllReduce<Mean>>(
        group(),
        mesh.fingerprint(),
        determinism,
        1_024,
    )
    .unwrap()
}

fn candidate_a() -> CollectiveTuningCandidate {
    CollectiveTuningCandidate::new_static::<Ring, Simple, U2, U16>(0, 0, 64, true).unwrap()
}

fn candidate_b() -> CollectiveTuningCandidate {
    CollectiveTuningCandidate::new_static::<Tree, LowLatency, U3, U16>(-1, 1, 128, true).unwrap()
}

#[cfg(feature = "autotune")]
fn service_environment() -> TuningEnvironmentFingerprint<Cuda> {
    TuningEnvironmentFingerprint::new(
        DeviceFingerprint::<Cuda>::new("GPU-TUNE-0", "sm_90", SoftwareVersion::new(12, 8, 0))
            .unwrap(),
        CompilerFingerprint::<Cuda>::new(
            "nvrtc",
            SoftwareVersion::new(12, 8, 0),
            "sm_90",
            &["default-math"],
        )
        .unwrap(),
    )
    .unwrap()
}

#[cfg(feature = "autotune")]
fn service_topology() -> TuningTopologyFingerprint<StaticWorld<U2>> {
    TuningTopologyFingerprint::new(
        vec![
            DeviceFingerprint::new_dyn(
                incin_core::prelude::DeviceKind::Cuda,
                "GPU-TUNE-0",
                "sm_90",
                SoftwareVersion::new(12, 8, 0),
            )
            .unwrap(),
            DeviceFingerprint::new_dyn(
                incin_core::prelude::DeviceKind::Cuda,
                "GPU-TUNE-1",
                "sm_90",
                SoftwareVersion::new(12, 8, 0),
            )
            .unwrap(),
        ],
        vec![
            TopologyLink::new(0, 1, "network").unwrap(),
            TopologyLink::new(1, 0, "network").unwrap(),
        ],
        TransportFingerprint::new("nccl", SoftwareVersion::new(2, 27, 3)).unwrap(),
        ProcessLayoutFingerprint::new(2, 1),
    )
    .unwrap()
}

fn report(
    rank: usize,
    problem: &CollectiveTuningProblem,
    candidate: CollectiveTuningCandidate,
    samples: &[u64],
) -> RankSampleReport {
    RankSampleReport::new(
        rank,
        problem.key().hash(),
        candidate.hash(),
        samples.to_vec(),
        true,
        0x55,
        0x55,
    )
}

#[test]
fn static_and_dyn_problem_and_candidate_identities_match() {
    let mesh = topology("GPU-TUNE");
    let static_problem = CollectiveTuningProblem::new_static::<f32, U16, TuneAllReduce<Mean>>(
        group(),
        mesh.fingerprint(),
        Determinism::Required,
        1_024,
    )
    .unwrap();
    let dynamic_problem = CollectiveTuningProblem::new_dyn(
        CollectiveKind::AllReduce(ReduceOp::Mean),
        DTypeId::F32,
        16,
        group(),
        mesh.fingerprint(),
        Determinism::Required,
        1_024,
    )
    .unwrap();
    assert_eq!(static_problem, dynamic_problem);
    assert_eq!(static_problem.elements(), 16);
    assert_eq!(static_problem.message_bytes(), 64);
    assert_eq!(static_problem.key().message_size_bucket(), 6);
    assert_eq!(static_problem.key().group(), group());
    assert_eq!(static_problem.key().transport(), "nccl");
    assert_eq!(static_problem.key().transport_version(), (2, 27, 3));
    assert_eq!(static_problem.key().determinism(), Determinism::Required);

    let static_candidate = candidate_a();
    let dynamic_candidate = CollectiveTuningCandidate::new_dyn(
        CollectiveAlgorithm::Ring,
        CollectiveProtocol::Simple,
        2,
        16,
        0,
        0,
        64,
        true,
    )
    .unwrap();
    assert_eq!(static_candidate, dynamic_candidate);
    assert_eq!(static_candidate.channels(), 2);
    assert_eq!(static_candidate.chunk_bytes(), 16);
    assert!(static_candidate.deterministic());
}

#[test]
fn median_of_per_sample_maximum_rank_time_drives_selection_and_commit() {
    let problem = problem(Determinism::Permitted);
    let first = candidate_a();
    let second = candidate_b();
    let candidates = [first, second];
    let rounds = [
        CandidateRound::new(
            first,
            vec![
                report(0, &problem, first, &[1, 1, 1]),
                report(1, &problem, first, &[100, 100, 100]),
            ],
        ),
        CandidateRound::new(
            second,
            vec![
                report(0, &problem, second, &[60, 60, 60]),
                report(1, &problem, second, &[60, 60, 60]),
            ],
        ),
    ];
    let budget = CollectiveTuningBudget::new_static::<U2, U3>();
    let provisional = select_collective_candidate(&problem, &candidates, &rounds, budget).unwrap();

    // Candidate one has the lower cross-rank average (50.5ns versus 60ns),
    // but collective completion is gated by its 100ns slow rank.
    assert_eq!(provisional.candidate(), second);
    assert_eq!(provisional.median_max_rank_ns(), 60);
    assert_eq!(provisional.rank_median_ns(), [60, 60]);
    assert_eq!(provisional.sample_count(), 3);

    let committed = commit_collective_tuning(
        provisional,
        &[
            CommitVote::for_result(0, provisional, true),
            CommitVote::for_result(1, provisional, true),
        ],
    )
    .unwrap();
    assert_eq!(committed.result(), provisional);
}

#[cfg(feature = "autotune")]
#[test]
fn unanimous_collective_result_is_the_only_value_the_general_service_commits() {
    use std::time::Duration;

    let problem = problem(Determinism::Required);
    let first = candidate_a();
    let second = candidate_b();
    let candidates = [first, second];
    let rounds = [
        CandidateRound::new(
            first,
            vec![
                report(0, &problem, first, &[100, 100, 100]),
                report(1, &problem, first, &[100, 100, 100]),
            ],
        ),
        CandidateRound::new(
            second,
            vec![
                report(0, &problem, second, &[60, 60, 60]),
                report(1, &problem, second, &[60, 60, 60]),
            ],
        ),
    ];
    let provisional = select_collective_candidate(
        &problem,
        &candidates,
        &rounds,
        CollectiveTuningBudget::new_static::<U2, U3>(),
    )
    .unwrap();
    let committed = commit_collective_tuning(
        provisional,
        &[
            CommitVote::for_result(0, provisional, true),
            CommitVote::for_result(1, provisional, true),
        ],
    )
    .unwrap();

    let environment = service_environment();
    let context = TuningContext::<Cuda, ServiceCollectiveTuning>::collective(
        environment.clone(),
        service_topology(),
        Determinism::Required,
        1_024,
        Duration::from_secs(1),
    )
    .unwrap();
    let key = CacheKey::<Cuda>::new(
        "collective",
        &environment,
        &format!("problem={:016x}", problem.key().hash()),
    )
    .unwrap()
    .erase();
    let service =
        TuningService::<CoordinatedWarmupTuning>::coordinated_warmup(Duration::from_secs(1))
            .unwrap();
    let service_candidates = [
        first.service_candidate().unwrap(),
        second.service_candidate().unwrap(),
    ];
    let permit = match service
        .decide(
            &context,
            key.clone(),
            &service_candidates,
            first.hash(),
            first.hash(),
        )
        .unwrap()
    {
        ServiceDecision::Measure(permit) => permit,
        ServiceDecision::Selected(_) => panic!("fresh collective key was unexpectedly cached"),
    };
    let selection = permit.commit_collective(committed).unwrap();
    assert_eq!(selection.candidate().hash(), second.hash());

    assert!(matches!(
        service
            .decide(
                &context,
                key,
                &service_candidates,
                first.hash(),
                first.hash(),
            )
            .unwrap(),
        ServiceDecision::Selected(selection)
            if selection.candidate().hash() == second.hash()
    ));
}

#[test]
fn legality_budget_measurement_and_commit_failures_never_mint_a_result() {
    let deterministic_problem = problem(Determinism::Required);
    let nondeterministic = CollectiveTuningCandidate::new_dyn(
        CollectiveAlgorithm::Ring,
        CollectiveProtocol::Simple,
        1,
        16,
        0,
        0,
        0,
        false,
    )
    .unwrap();
    let nondeterministic_round = [CandidateRound::new(
        nondeterministic,
        vec![
            report(0, &deterministic_problem, nondeterministic, &[1, 1, 1]),
            report(1, &deterministic_problem, nondeterministic, &[1, 1, 1]),
        ],
    )];
    assert!(matches!(
        select_collective_candidate(
            &deterministic_problem,
            &[nondeterministic],
            &nondeterministic_round,
            CollectiveTuningBudget::new_static::<U2, U3>(),
        ),
        Err(CollectiveTuningError::NondeterministicCandidate { .. })
    ));

    let problem = problem(Determinism::Permitted);
    let candidate = candidate_a();
    let mutated = CandidateRound::new(
        candidate,
        vec![
            RankSampleReport::new(
                0,
                problem.key().hash(),
                candidate.hash(),
                vec![1, 1, 1],
                true,
                10,
                11,
            ),
            report(1, &problem, candidate, &[1, 1, 1]),
        ],
    );
    assert!(matches!(
        select_collective_candidate(
            &problem,
            &[candidate],
            &[mutated],
            CollectiveTuningBudget::new_static::<U2, U3>(),
        ),
        Err(CollectiveTuningError::MeasurementMutatedBuffer { rank: 0, .. })
    ));

    let round = CandidateRound::new(
        candidate,
        vec![
            report(0, &problem, candidate, &[3, 3, 3]),
            report(1, &problem, candidate, &[4, 4, 4]),
        ],
    );
    let provisional = select_collective_candidate(
        &problem,
        &[candidate],
        &[round],
        CollectiveTuningBudget::new_static::<U2, U3>(),
    )
    .unwrap();
    assert!(matches!(
        commit_collective_tuning(provisional, &[CommitVote::for_result(0, provisional, true)]),
        Err(CollectiveTuningError::CommitVoteCount { found: 1, .. })
    ));
    assert!(matches!(
        commit_collective_tuning(
            provisional,
            &[
                CommitVote::for_result(0, provisional, true),
                CommitVote::from_wire(
                    1,
                    provisional.problem_hash(),
                    provisional.candidate_hash() ^ 1,
                    provisional.median_max_rank_ns(),
                    true,
                ),
            ],
        ),
        Err(CollectiveTuningError::CommitMismatch { rank: 1 })
    ));
    assert!(matches!(
        commit_collective_tuning(
            provisional,
            &[
                CommitVote::for_result(0, provisional, true),
                CommitVote::for_result(1, provisional, false),
            ],
        ),
        Err(CollectiveTuningError::CommitRejected { rank: 1 })
    ));
}

#[test]
fn dyn_rejects_static_contract_violations_and_topology_changes_the_key() {
    let first = topology("GPU-FIRST");
    let second = topology("GPU-SECOND");
    let first_problem = CollectiveTuningProblem::new_dyn(
        CollectiveKind::AllGather,
        DTypeId::F32,
        16,
        group(),
        first.fingerprint(),
        Determinism::Permitted,
        1_024,
    )
    .unwrap();
    let second_problem = CollectiveTuningProblem::new_dyn(
        CollectiveKind::AllGather,
        DTypeId::F32,
        16,
        group(),
        second.fingerprint(),
        Determinism::Permitted,
        1_024,
    )
    .unwrap();
    assert_ne!(
        first_problem.key().topology(),
        second_problem.key().topology()
    );
    assert_ne!(first_problem.key().hash(), second_problem.key().hash());

    assert!(matches!(
        CollectiveTuningProblem::new_dyn(
            CollectiveKind::AllGather,
            DTypeId::Q8_0,
            16,
            group(),
            first.fingerprint(),
            Determinism::Permitted,
            1_024,
        ),
        Err(CollectiveTuningError::Collective(_))
    ));
    assert!(matches!(
        CollectiveTuningProblem::new_dyn(
            CollectiveKind::AllReduce(ReduceOp::Mean),
            DTypeId::U32,
            16,
            group(),
            first.fingerprint(),
            Determinism::Permitted,
            1_024,
        ),
        Err(CollectiveTuningError::Collective(_))
    ));
    assert_eq!(
        CollectiveTuningProblem::new_dyn(
            CollectiveKind::AllToAll,
            DTypeId::F32,
            3,
            group(),
            first.fingerprint(),
            Determinism::Permitted,
            1_024,
        ),
        Err(CollectiveTuningError::NonDivisible {
            elements: 3,
            ranks: 2,
        })
    );
    assert!(matches!(
        CollectiveTuningCandidate::new_dyn(
            CollectiveAlgorithm::Ring,
            CollectiveProtocol::Simple,
            0,
            16,
            0,
            0,
            0,
            true,
        ),
        Err(CollectiveTuningError::ChannelCount { found: 0, .. })
    ));
    assert!(matches!(
        CollectiveTuningCandidate::new_dyn(
            CollectiveAlgorithm::Ring,
            CollectiveProtocol::Simple,
            1,
            3,
            0,
            0,
            0,
            true,
        ),
        Err(CollectiveTuningError::ChunkBytes { found: 3 })
    ));
    assert_eq!(
        CollectiveTuningBudget::new_dyn(33, 3),
        Err(CollectiveTuningError::BudgetLimit {
            maximum: 32,
            candidates: 33,
            samples: 3,
        })
    );

    #[cfg(target_pointer_width = "64")]
    assert_eq!(
        CollectiveTuningProblem::new_dyn(
            CollectiveKind::AllGather,
            DTypeId::F32,
            u32::MAX as usize + 1,
            group(),
            first.fingerprint(),
            Determinism::Permitted,
            0,
        ),
        Err(CollectiveTuningError::ElementLimit {
            maximum: u32::MAX as usize,
            found: u32::MAX as usize + 1,
        })
    );
}

#[test]
fn static_collective_tuning_contract_rejections_are_compile_errors() {
    if std::fs::read("/home/xupremix/.cargo/config.toml").is_err() {
        return;
    }
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/collective_tuning_compile_fail/*.rs");
    if std::env::var_os("TRYBUILD").as_deref() != Some(std::ffi::OsStr::new("overwrite")) {
        let expected = BTreeMap::from([
            ("integer_mean", "CollectiveReductionDType"),
            ("odd_all_to_all", "ShardDivisible"),
            ("q8_problem", "CollectiveDType"),
            ("zero_channels", "NonZero"),
        ]);
        let directory = Path::new("tests/collective_tuning_compile_fail");
        for (case, reason) in expected {
            let stderr = fs::read_to_string(directory.join(format!("{case}.stderr"))).unwrap();
            assert!(
                stderr.contains(reason),
                "{case} no longer fails for {reason}"
            );
            for scaffolding in ["E0432", "E0433", "E0603"] {
                assert!(
                    !stderr.contains(scaffolding),
                    "{case} fails on scaffolding error {scaffolding}"
                );
            }
        }
    }
}
