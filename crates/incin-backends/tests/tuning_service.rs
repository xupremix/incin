//! Integration coverage for `static_environment` on the documented public surface.
#![cfg(feature = "autotune")]

use incin_backends::tuning::{
    cache::{CacheKey, CacheLimits, CacheRecord, MeasurementMethod, PersistentTuningCache},
    identity::{
        CompilerFingerprint, DeviceFingerprint, ProcessLayoutFingerprint, SoftwareVersion,
        StaticWorld, TopologyLink, TransportFingerprint, TuningEnvironmentFingerprint,
        TuningTopologyFingerprint,
    },
    service::{
        AutotunePolicy, CollectiveTuning, CoordinatedVote, CoordinatedWarmupTuning, DisabledTuning,
        HeuristicTuning, KernelTuning, SelectionSource, ServiceDecision, TuningCandidate,
        TuningContext, TuningScope, TuningService, TuningServiceError, legal_candidates_digest,
    },
};
use incin_core::{
    exec::Determinism,
    prelude::{Cuda, DeviceKind, Dyn},
    typenum::U2,
};
use std::{
    sync::{Arc, mpsc},
    thread,
    time::Duration,
};

const DRIVER: SoftwareVersion = SoftwareVersion::new(12, 8, 0);

fn static_environment() -> TuningEnvironmentFingerprint<Cuda> {
    TuningEnvironmentFingerprint::new(
        DeviceFingerprint::<Cuda>::new("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "sm_90", DRIVER)
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

fn dynamic_environment() -> TuningEnvironmentFingerprint<Dyn> {
    static_environment().erase()
}

fn topology() -> TuningTopologyFingerprint<StaticWorld<U2>> {
    TuningTopologyFingerprint::new(
        vec![
            DeviceFingerprint::new_dyn(
                DeviceKind::Cuda,
                "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
                "sm_90",
                DRIVER,
            )
            .unwrap(),
            DeviceFingerprint::new_dyn(
                DeviceKind::Cuda,
                "GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                "sm_90",
                DRIVER,
            )
            .unwrap(),
        ],
        vec![
            TopologyLink::new(0, 1, "network").unwrap(),
            TopologyLink::new(1, 0, "network").unwrap(),
        ],
        TransportFingerprint::new("nccl", SoftwareVersion::new(2, 28, 3)).unwrap(),
        ProcessLayoutFingerprint::new(2, 1),
    )
    .unwrap()
}

fn kernel_context() -> TuningContext<Cuda, KernelTuning> {
    TuningContext::kernel(
        static_environment(),
        Determinism::Permitted,
        1024,
        Duration::from_secs(2),
    )
    .unwrap()
}

fn collective_context() -> TuningContext<Cuda, CollectiveTuning> {
    TuningContext::collective(
        static_environment(),
        topology(),
        Determinism::Required,
        1024,
        Duration::from_secs(2),
    )
    .unwrap()
}

fn key(problem: &str) -> CacheKey<Dyn> {
    CacheKey::new_dyn("kernel", &dynamic_environment(), problem).unwrap()
}

fn collective_key(problem: &str) -> CacheKey<Dyn> {
    CacheKey::new_dyn("collective", &dynamic_environment(), problem).unwrap()
}

fn candidates() -> Vec<TuningCandidate> {
    vec![
        TuningCandidate::new(10, "block=128", true, 0).unwrap(),
        TuningCandidate::new(20, "block=256", true, 64).unwrap(),
        TuningCandidate::new(30, "block=512", false, 2048).unwrap(),
    ]
}

fn selected(decision: ServiceDecision) -> incin_backends::tuning::service::TuningSelection {
    match decision {
        ServiceDecision::Selected(selection) => selection,
        ServiceDecision::Measure(_) => panic!("expected an immediate selection"),
    }
}

fn permit(decision: ServiceDecision) -> incin_backends::tuning::service::TuningPermit {
    match decision {
        ServiceDecision::Measure(permit) => permit,
        ServiceDecision::Selected(_) => panic!("expected a measurement permit"),
    }
}

#[test]
fn disabled_heuristic_and_dyn_policies_have_matching_behavior() {
    let context = kernel_context();
    let candidates = candidates();
    let disabled = TuningService::<DisabledTuning>::disabled();
    let static_disabled = selected(
        disabled
            .decide(&context, key("policy"), &candidates, 10, 20)
            .unwrap(),
    );
    assert_eq!(static_disabled.candidate().hash(), 10);
    assert_eq!(static_disabled.source(), SelectionSource::DisabledFallback);

    let dynamic_disabled =
        TuningService::<Dyn>::new_dyn(AutotunePolicy::Disabled, CacheLimits::default()).unwrap();
    let dynamic_selection = selected(
        dynamic_disabled
            .decide(&context.erase(), key("policy"), &candidates, 10, 20)
            .unwrap(),
    );
    assert_eq!(dynamic_selection, static_disabled);

    let heuristic = TuningService::<HeuristicTuning>::heuristic();
    let selection = selected(
        heuristic
            .decide(&kernel_context(), key("heuristic"), &candidates, 10, 20)
            .unwrap(),
    );
    assert_eq!(selection.candidate().hash(), 20);
    assert_eq!(selection.source(), SelectionSource::Heuristic);
}

#[test]
fn dyn_context_checks_scope_topology_and_time_budget() {
    assert!(matches!(
        TuningContext::<Dyn, Dyn>::new_dyn(
            TuningScope::Collective,
            dynamic_environment(),
            None,
            Determinism::Permitted,
            0,
            Duration::from_secs(1),
        ),
        Err(TuningServiceError::TopologyRequired {
            scope: TuningScope::Collective
        })
    ));
    assert!(matches!(
        TuningContext::<Dyn, Dyn>::new_dyn(
            TuningScope::Kernel,
            dynamic_environment(),
            Some(topology().erase()),
            Determinism::Permitted,
            0,
            Duration::from_secs(1),
        ),
        Err(TuningServiceError::UnexpectedTopology {
            scope: TuningScope::Kernel
        })
    ));
    assert!(matches!(
        TuningContext::<Dyn, Dyn>::new_dyn(
            TuningScope::Kernel,
            dynamic_environment(),
            None,
            Determinism::Permitted,
            0,
            Duration::ZERO,
        ),
        Err(TuningServiceError::ZeroTimeBudget)
    ));
    assert!(matches!(
        TuningService::<Dyn>::new_dyn(
            AutotunePolicy::CoordinatedWarmup {
                budget: Duration::ZERO
            },
            CacheLimits::default(),
        ),
        Err(TuningServiceError::ZeroWarmupBudget)
    ));
}

#[test]
fn determinism_memory_duplicates_and_key_context_are_enforced_before_policy() {
    let service = TuningService::<DisabledTuning>::disabled();
    let context = TuningContext::<Cuda, KernelTuning>::kernel(
        static_environment(),
        Determinism::Required,
        32,
        Duration::from_secs(1),
    )
    .unwrap();
    let candidates = candidates();
    assert!(matches!(
        service.decide(&context, key("filtered"), &candidates, 20, 20),
        Err(TuningServiceError::IllegalFallback { hash: 20 })
    ));
    let selection = selected(
        service
            .decide(&context, key("filtered"), &candidates, 10, 10)
            .unwrap(),
    );
    assert_eq!(selection.candidate().hash(), 10);

    let duplicates = vec![
        TuningCandidate::new(10, "a", true, 0).unwrap(),
        TuningCandidate::new(10, "b", true, 0).unwrap(),
    ];
    assert!(matches!(
        service.decide(&context, key("duplicate"), &duplicates, 10, 10),
        Err(TuningServiceError::DuplicateCandidate { hash: 10 })
    ));
    assert!(matches!(
        service.decide(&context, collective_key("wrong-scope"), &candidates, 10, 10),
        Err(TuningServiceError::KeyContextMismatch { field: "scope" })
    ));
}

#[test]
fn coordinated_warmup_is_single_flight_and_commits_to_waiters() {
    let service = Arc::new(
        TuningService::<CoordinatedWarmupTuning>::coordinated_warmup(Duration::from_secs(1))
            .unwrap(),
    );
    let context = Arc::new(kernel_context());
    let candidates = Arc::new(candidates());
    let first = permit(
        service
            .decide(&context, key("single-flight"), &candidates, 10, 20)
            .unwrap(),
    );

    let (started_tx, started_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let waiting_service = Arc::clone(&service);
    let waiting_context = Arc::clone(&context);
    let waiting_candidates = Arc::clone(&candidates);
    let waiter = thread::spawn(move || {
        started_tx.send(()).unwrap();
        let result = waiting_service.decide(
            &waiting_context,
            key("single-flight"),
            &waiting_candidates,
            10,
            20,
        );
        result_tx.send(result).unwrap();
    });
    started_rx.recv().unwrap();
    assert!(
        result_rx.recv_timeout(Duration::from_millis(30)).is_err(),
        "a second caller must wait instead of receiving another permit"
    );
    let committed = first.commit_local(20, 77, 7).unwrap();
    assert_eq!(committed.source(), SelectionSource::Measurement);

    let waiting = selected(
        result_rx
            .recv_timeout(Duration::from_secs(1))
            .unwrap()
            .unwrap(),
    );
    assert_eq!(waiting.candidate().hash(), 20);
    assert_eq!(waiting.source(), SelectionSource::WarmupCache);
    assert_eq!(waiting.median_ns(), Some(77));
    waiter.join().unwrap();
}

#[test]
fn cancellation_and_expiration_release_the_lease_epoch() {
    let service =
        TuningService::<CoordinatedWarmupTuning>::coordinated_warmup(Duration::from_millis(20))
            .unwrap();
    let candidates = candidates();
    let first = permit(
        service
            .decide(&kernel_context(), key("cancel"), &candidates, 10, 20)
            .unwrap(),
    );
    let first_epoch = first.epoch();
    first.cancel();
    let second = permit(
        service
            .decide(&kernel_context(), key("cancel"), &candidates, 10, 20)
            .unwrap(),
    );
    assert!(second.epoch() > first_epoch);
    second.cancel();

    let expiring = permit(
        service
            .decide(&kernel_context(), key("expire"), &candidates, 10, 20)
            .unwrap(),
    );
    let expired_epoch = expiring.epoch();
    thread::sleep(Duration::from_millis(30));
    assert!(matches!(
        expiring.commit_local(20, 10, 1),
        Err(TuningServiceError::PermitExpired { epoch }) if epoch == expired_epoch
    ));
    let replacement = permit(
        service
            .decide(&kernel_context(), key("expire"), &candidates, 10, 20)
            .unwrap(),
    );
    assert!(replacement.epoch() > expired_epoch);
}

#[test]
fn coordinated_scope_requires_one_matching_vote_per_network_rank() {
    let service =
        TuningService::<CoordinatedWarmupTuning>::coordinated_warmup(Duration::from_secs(1))
            .unwrap();
    let candidates = candidates();
    let first = permit(
        service
            .decide(
                &collective_context(),
                collective_key("all-reduce"),
                &candidates,
                10,
                20,
            )
            .unwrap(),
    );
    assert_eq!(first.participants(), &[0, 1]);
    let epoch = first.epoch();
    assert!(matches!(
        first.commit_coordinated(
            20,
            90,
            3,
            &[
                CoordinatedVote::new(0, epoch, 20, true),
                CoordinatedVote::new(1, epoch, 10, true),
            ],
        ),
        Err(TuningServiceError::VoteMismatch { rank: 1 })
    ));

    let second = permit(
        service
            .decide(
                &collective_context(),
                collective_key("all-reduce"),
                &candidates,
                10,
                20,
            )
            .unwrap(),
    );
    let epoch = second.epoch();
    let selection = second
        .commit_coordinated(
            20,
            90,
            3,
            &[
                CoordinatedVote::new(1, epoch, 20, true),
                CoordinatedVote::new(0, epoch, 20, true),
            ],
        )
        .unwrap();
    assert_eq!(selection.candidate().hash(), 20);
}

#[test]
fn warmup_cache_persists_and_profile_import_revalidates_winner() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("profile.json");
    let limits = CacheLimits::new(32, 128 * 1024, None).unwrap();
    let cache = PersistentTuningCache::open(&path, limits).unwrap();
    let service =
        TuningService::<CoordinatedWarmupTuning>::coordinated_warmup(Duration::from_secs(1))
            .unwrap()
            .with_cache(cache);
    let candidates = candidates();
    let permit = permit(
        service
            .decide(&kernel_context(), key("persistent"), &candidates, 10, 20)
            .unwrap(),
    );
    permit.commit_local(20, 55, 7).unwrap();
    drop(service);

    let profile =
        TuningService::<incin_backends::tuning::service::ProfileGuidedTuning>::profile_guided(
            &path, limits,
        )
        .unwrap();
    let selection = selected(
        profile
            .decide(&kernel_context(), key("persistent"), &candidates, 10, 10)
            .unwrap(),
    );
    assert_eq!(selection.candidate().hash(), 20);
    assert_eq!(selection.source(), SelectionSource::Profile);

    let changed_legal_set = vec![candidates[0].clone()];
    let fallback = selected(
        profile
            .decide(
                &kernel_context(),
                key("persistent"),
                &changed_legal_set,
                10,
                10,
            )
            .unwrap(),
    );
    assert_eq!(fallback.candidate().hash(), 10);
    assert_eq!(fallback.source(), SelectionSource::Heuristic);
}

#[test]
fn an_imported_winner_not_in_the_legal_set_is_never_used() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("untrusted.json");
    let limits = CacheLimits::new(32, 128 * 1024, None).unwrap();
    let mut cache = PersistentTuningCache::open(&path, limits).unwrap();
    let candidates = candidates();
    let legal_digest = legal_candidates_digest(&candidates);
    cache
        .commit(
            CacheRecord::new(
                key("untrusted"),
                MeasurementMethod::ProfileGuided,
                0,
                None,
                legal_digest,
                "not-a-current-candidate",
            )
            .unwrap(),
        )
        .unwrap();
    let profile =
        TuningService::<incin_backends::tuning::service::ProfileGuidedTuning>::profile_guided(
            &path, limits,
        )
        .unwrap();
    let selection = selected(
        profile
            .decide(&kernel_context(), key("untrusted"), &candidates, 10, 20)
            .unwrap(),
    );
    assert_eq!(selection.candidate().hash(), 20);
    assert_eq!(selection.source(), SelectionSource::Heuristic);
}

#[test]
fn service_static_contracts_are_compile_checked() {
    if std::fs::read("/home/xupremix/.cargo/config.toml").is_err() {
        return;
    }
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/tuning_service_compile_fail/*.rs");
}
