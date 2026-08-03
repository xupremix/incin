//! `DST-015`: two-process rendezvous, launcher, timeout, and shutdown evidence.

#![cfg(feature = "distributed-nccl")]

use std::collections::BTreeMap;
use std::fs;
use std::net::{SocketAddr, TcpListener};
use std::path::Path;
use std::process::Command;
use std::thread;
use std::time::Duration;

use incin::experimental::distributed::{
    ContextError, ContextFailure, DistributedContext, DistributedContextState, DynRendezvousConfig,
    RendezvousEndpoint, RunId, StaticRendezvousConfig, TwoRankDataParallel, TwoRankLaunchPlan,
};
use incin::prelude::Dyn;
use incin::typenum::{U0, U1};

fn unused_local_address() -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    address
}

fn run_id(label: &str) -> RunId {
    RunId::new(format!("dst-015-{label}-{}", std::process::id())).unwrap()
}

#[test]
fn static_and_dyn_launch_plans_produce_the_same_two_rank_environment() {
    let address = unused_local_address();
    let run = run_id("launch");
    let timeout = Duration::from_millis(750);
    let static_plan =
        TwoRankLaunchPlan::<TwoRankDataParallel>::new_static(run.clone(), address, [0, 0], timeout)
            .unwrap();
    let dyn_plan = TwoRankLaunchPlan::<Dyn>::new_dyn(run, address, 2, vec![0, 0], timeout).unwrap();

    let static_root = static_plan.rank_static::<U0>();
    let static_peer = static_plan.rank_static::<U1>();
    assert_eq!(
        static_root.environment(),
        dyn_plan.rank_dyn(0).unwrap().environment()
    );
    assert_eq!(
        static_peer.environment(),
        dyn_plan.rank_dyn(1).unwrap().environment()
    );
    assert_eq!(static_root.local_cuda_device(), 0);
    assert_eq!(static_peer.local_cuda_device(), 0);
    assert_eq!(
        dyn_plan.rank_dyn(2).unwrap_err(),
        ContextError::RankOutOfRange { rank: 2, world: 2 }
    );
}

#[test]
fn static_ranks_rendezvous_and_shutdown_without_leaving_a_live_context() {
    let address = unused_local_address();
    let run = run_id("static");
    let timeout = Duration::from_secs(2);
    let root_run = run.clone();
    let root = thread::spawn(move || {
        let context = DistributedContext::<TwoRankDataParallel, U0>::rendezvous_static(
            StaticRendezvousConfig::root(root_run, address, 0, timeout),
        )
        .unwrap();
        assert_eq!(context.rank(), 0);
        assert_eq!(context.world_size(), 2);
        assert_eq!(context.identity().peer_cuda_device(), 0);
        context.ensure_active().unwrap();
        context.shutdown().unwrap();
        assert_eq!(context.state(), DistributedContextState::Shutdown);
        context
    });

    let peer = DistributedContext::<TwoRankDataParallel, U1>::rendezvous_static(
        StaticRendezvousConfig::peer(run, address, 0, timeout),
    )
    .unwrap();
    assert_eq!(peer.rank(), 1);
    assert_eq!(peer.world_size(), 2);
    peer.shutdown().unwrap();
    assert_eq!(peer.state(), DistributedContextState::Shutdown);
    assert!(matches!(
        peer.ensure_active(),
        Err(ContextError::ContextNotActive {
            state: DistributedContextState::Shutdown
        })
    ));
    assert_eq!(
        root.join().unwrap().state(),
        DistributedContextState::Shutdown
    );
}

#[test]
fn dyn_rejects_the_same_world_rank_role_and_timeout_errors_at_runtime() {
    let address = unused_local_address();
    let run = run_id("dyn-errors");

    let error = DistributedContext::<Dyn, Dyn>::rendezvous_dyn(DynRendezvousConfig::new(
        run.clone(),
        RendezvousEndpoint::Root { bind: address },
        0,
        3,
        0,
        Duration::from_millis(10),
    ))
    .unwrap_err();
    assert_eq!(
        error,
        ContextError::WorldSize {
            expected: 2,
            found: 3
        }
    );

    let error = DistributedContext::<Dyn, Dyn>::rendezvous_dyn(DynRendezvousConfig::new(
        run.clone(),
        RendezvousEndpoint::Peer { root: address },
        2,
        2,
        0,
        Duration::from_millis(10),
    ))
    .unwrap_err();
    assert_eq!(error, ContextError::RankOutOfRange { rank: 2, world: 2 });

    let error = DistributedContext::<Dyn, Dyn>::rendezvous_dyn(DynRendezvousConfig::new(
        run.clone(),
        RendezvousEndpoint::Peer { root: address },
        0,
        2,
        0,
        Duration::from_millis(10),
    ))
    .unwrap_err();
    assert_eq!(
        error,
        ContextError::RoleRankMismatch {
            role_rank: 1,
            rank: 0
        }
    );

    let error = DistributedContext::<Dyn, Dyn>::rendezvous_dyn(DynRendezvousConfig::new(
        run,
        RendezvousEndpoint::Root { bind: address },
        0,
        2,
        0,
        Duration::ZERO,
    ))
    .unwrap_err();
    assert_eq!(error, ContextError::InvalidTimeout);
}

#[test]
fn missing_peer_and_mismatched_run_id_fail_with_bounded_structured_errors() {
    let address = unused_local_address();
    let timeout_error = DistributedContext::<TwoRankDataParallel, U0>::rendezvous_static(
        StaticRendezvousConfig::root(run_id("missing"), address, 0, Duration::from_millis(20)),
    )
    .unwrap_err();
    assert!(matches!(
        timeout_error,
        ContextError::Network {
            phase: "accept rank one",
            ..
        }
    ));

    let address = unused_local_address();
    let timeout = Duration::from_secs(2);
    let root = thread::spawn(move || {
        DistributedContext::<TwoRankDataParallel, U0>::rendezvous_static(
            StaticRendezvousConfig::root(run_id("root-run"), address, 0, timeout),
        )
    });
    let peer = DistributedContext::<TwoRankDataParallel, U1>::rendezvous_static(
        StaticRendezvousConfig::peer(run_id("peer-run"), address, 0, timeout),
    );
    assert!(matches!(peer, Err(ContextError::PeerRejected { code: 1 })));
    assert!(matches!(
        root.join().unwrap(),
        Err(ContextError::RunIdMismatch)
    ));
}

#[test]
fn peer_abort_invalidates_the_shared_runtime_handle() {
    let address = unused_local_address();
    let run = run_id("abort");
    let timeout = Duration::from_secs(2);
    let root_run = run.clone();
    let root = thread::spawn(move || {
        let context = DistributedContext::<TwoRankDataParallel, U0>::rendezvous_static(
            StaticRendezvousConfig::root(root_run, address, 0, timeout),
        )
        .unwrap();
        let handle = context.handle();
        assert!(matches!(
            context.wait_for_peer(),
            Err(ContextError::PeerAborted {
                failure: ContextFailure::Transport
            })
        ));
        assert_eq!(context.state(), DistributedContextState::Failed);
        assert_eq!(handle.state(), DistributedContextState::Failed);
    });
    let peer = DistributedContext::<TwoRankDataParallel, U1>::rendezvous_static(
        StaticRendezvousConfig::peer(run, address, 0, timeout),
    )
    .unwrap();
    let handle = peer.handle();
    peer.abort(ContextFailure::Transport).unwrap();
    assert_eq!(peer.state(), DistributedContextState::Failed);
    assert_eq!(handle.state(), DistributedContextState::Failed);
    root.join().unwrap();
}

/// Child entrypoint used by `two_real_processes_use_from_env_and_coordinate_shutdown`.
#[test]
fn rank_process_child() {
    let Ok(expected_rank) = std::env::var("INCIN_TEST_RENDEZVOUS_CHILD") else {
        return;
    };
    let expected_rank: usize = expected_rank.parse().unwrap();
    let context = DistributedContext::<Dyn, Dyn>::from_env().unwrap();
    assert_eq!(context.rank(), expected_rank);
    assert_eq!(context.world_size(), 2);
    assert_eq!(context.local_cuda_device(), 0);
    context.shutdown().unwrap();
    assert_eq!(context.state(), DistributedContextState::Shutdown);
}

#[test]
fn two_real_processes_use_from_env_and_coordinate_shutdown() {
    let address = unused_local_address();
    let plan = TwoRankLaunchPlan::<Dyn>::new_dyn(
        run_id("processes"),
        address,
        2,
        vec![0, 0],
        Duration::from_secs(3),
    )
    .unwrap();
    let executable = std::env::current_exe().unwrap();

    let spawn = |rank: usize| {
        let mut command = Command::new(&executable);
        command
            .arg("--exact")
            .arg("rank_process_child")
            .arg("--nocapture")
            .env("INCIN_TEST_RENDEZVOUS_CHILD", rank.to_string());
        plan.rank_dyn(rank).unwrap().apply(&mut command);
        command.spawn().unwrap()
    };

    let mut root = spawn(0);
    let mut peer = spawn(1);
    let root_status = root.wait().unwrap();
    let peer_status = peer.wait().unwrap();
    assert!(root_status.success(), "rank zero exited with {root_status}");
    assert!(peer_status.success(), "rank one exited with {peer_status}");
}

#[test]
fn static_context_compile_failures_name_the_proof_they_pin() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/rendezvous_compile_fail/*.rs");

    if std::env::var_os("TRYBUILD").as_deref() == Some(std::ffi::OsStr::new("overwrite")) {
        return;
    }
    let expected = BTreeMap::from([
        (
            "invalid_static_rank",
            "is not a valid rank in a two-rank distributed context",
        ),
        (
            "rank_one_cannot_build_root",
            "no associated function or constant named `root`",
        ),
        ("wrong_static_world", "as ValidMesh>::World"),
    ]);
    let directory = Path::new("tests/rendezvous_compile_fail");
    for (stem, reason) in expected {
        let output = fs::read_to_string(directory.join(format!("{stem}.stderr"))).unwrap();
        assert!(
            output.contains(reason),
            "{stem} no longer fails for {reason:?}\n{output}"
        );
        for scaffolding in ["E0432", "E0433", "E0603", "E0412"] {
            assert!(
                !output.contains(scaffolding),
                "{stem} fails on scaffolding marker {scaffolding}"
            );
        }
    }
}
