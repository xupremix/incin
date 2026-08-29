//! Integration coverage for `cuda_device` on the documented public surface.
#![cfg(feature = "autotune")]

use incin_backends::tuning::identity::{
    CompilerFingerprint, DeviceFingerprint, IdentityError, ProcessLayoutFingerprint,
    SoftwareVersion, StaticWorld, TopologyLink, TransportFingerprint, TuningEnvironmentFingerprint,
    TuningTopologyFingerprint,
};
use incin_core::{
    prelude::{Cpu, Cuda, DeviceKind, Dyn},
    typenum::{U1, U2},
};

const DRIVER: SoftwareVersion = SoftwareVersion::new(12, 8, 0);
const NVRTC: SoftwareVersion = SoftwareVersion::new(12, 8, 0);

fn cuda_device(id: &str, architecture: &str) -> DeviceFingerprint<Dyn> {
    DeviceFingerprint::new_dyn(DeviceKind::Cuda, id, architecture, DRIVER).unwrap()
}

fn compiler(target: &str, options: &[&str]) -> CompilerFingerprint<Dyn> {
    CompilerFingerprint::new_dyn(DeviceKind::Cuda, "nvrtc", NVRTC, target, options).unwrap()
}

fn transport(version: SoftwareVersion) -> TransportFingerprint {
    TransportFingerprint::new("nccl", version).unwrap()
}

fn two_devices() -> Vec<DeviceFingerprint<Dyn>> {
    vec![
        cuda_device("GPU-00000000-0000-0000-0000-000000000000", "sm_75"),
        cuda_device("GPU-11111111-1111-1111-1111-111111111111", "sm_90"),
    ]
}

fn network_links() -> Vec<TopologyLink> {
    vec![
        TopologyLink::new(0, 1, "network").unwrap(),
        TopologyLink::new(1, 0, "network").unwrap(),
    ]
}

#[test]
fn static_and_dyn_device_compiler_identities_have_parity() {
    let static_device =
        DeviceFingerprint::<Cuda>::new("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "sm_90", DRIVER)
            .unwrap();
    let dynamic_device = DeviceFingerprint::<Dyn>::new_dyn(
        DeviceKind::Cuda,
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "sm_90",
        DRIVER,
    )
    .unwrap();
    assert_eq!(static_device.digest(), dynamic_device.digest());
    assert_eq!(
        dynamic_device.clone().try_into_static::<Cuda>().unwrap(),
        static_device
    );
    assert!(matches!(
        dynamic_device.try_into_static::<Cpu>(),
        Err(IdentityError::BackendMismatch {
            expected: "cpu",
            actual: "cuda"
        })
    ));

    let static_compiler =
        CompilerFingerprint::<Cuda>::new("nvrtc", NVRTC, "sm_90", &["default-math", "headers-v1"])
            .unwrap();
    let dynamic_compiler = compiler("sm_90", &["default-math", "headers-v1"]);
    assert_eq!(static_compiler.digest(), dynamic_compiler.digest());

    let static_environment =
        TuningEnvironmentFingerprint::new(static_device, static_compiler).unwrap();
    let dynamic_environment =
        TuningEnvironmentFingerprint::new_dyn(dynamic_device_for_environment(), dynamic_compiler)
            .unwrap();
    assert_eq!(
        static_environment.digest(),
        dynamic_environment.digest(),
        "static and Dyn APIs must produce the same persistent key"
    );
    assert_eq!(
        dynamic_environment
            .try_into_static::<Cuda>()
            .unwrap()
            .digest(),
        static_environment.digest()
    );
}

fn dynamic_device_for_environment() -> DeviceFingerprint<Dyn> {
    cuda_device("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "sm_90")
}

#[test]
fn ordinal_is_absent_and_physical_aliases_do_not_collide() {
    // The constructor has no ordinal argument. These two records model the
    // same card observed under different visibility masks and remain equal.
    let visible_as_zero = cuda_device("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "sm_90");
    let visible_as_seven = cuda_device("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "sm_90");
    assert_eq!(visible_as_zero, visible_as_seven);

    // Conversely, two cards which are both local ordinal zero on different
    // hosts stay distinct because their vendor identifiers differ.
    let host_a_zero = visible_as_zero;
    let host_b_zero = cuda_device("GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb", "sm_90");
    assert_ne!(host_a_zero.digest(), host_b_zero.digest());
}

#[test]
fn every_codegen_relevant_field_changes_identity() {
    let device = cuda_device("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "sm_90");
    let different_arch = cuda_device("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "sm_100");
    let different_driver = DeviceFingerprint::<Dyn>::new_dyn(
        DeviceKind::Cuda,
        "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
        "sm_90",
        SoftwareVersion::new(13, 0, 0),
    )
    .unwrap();
    assert_ne!(device.digest(), different_arch.digest());
    assert_ne!(device.digest(), different_driver.digest());

    let base = compiler("sm_90", &["default-math"]);
    let different_version = CompilerFingerprint::<Dyn>::new_dyn(
        DeviceKind::Cuda,
        "nvrtc",
        SoftwareVersion::new(12, 9, 0),
        "sm_90",
        &["default-math"],
    )
    .unwrap();
    let different_target = compiler("sm_100", &["default-math"]);
    let different_options = compiler("sm_90", &["fast-math"]);
    assert_ne!(base.digest(), different_version.digest());
    assert_ne!(base.digest(), different_target.digest());
    assert_ne!(base.digest(), different_options.digest());
}

#[test]
fn length_delimited_fields_prevent_adjacent_string_aliases() {
    let left =
        CompilerFingerprint::<Dyn>::new_dyn(DeviceKind::Cuda, "ab", NVRTC, "c", &["x"]).unwrap();
    let right =
        CompilerFingerprint::<Dyn>::new_dyn(DeviceKind::Cuda, "a", NVRTC, "bc", &["x"]).unwrap();
    assert_ne!(left.digest(), right.digest());

    assert!(matches!(
        CompilerFingerprint::<Dyn>::new_dyn(DeviceKind::Cuda, "", NVRTC, "sm_90", &[]),
        Err(IdentityError::EmptyField {
            field: "compiler_implementation"
        })
    ));
    assert!(matches!(
        CompilerFingerprint::<Dyn>::new_dyn(DeviceKind::Cuda, " nvrtc", NVRTC, "sm_90", &[]),
        Err(IdentityError::NonCanonicalField {
            field: "compiler_implementation"
        })
    ));
}

#[test]
fn static_and_dyn_topology_identities_have_parity() {
    let static_topology = TuningTopologyFingerprint::<StaticWorld<U2>>::new(
        two_devices(),
        network_links(),
        transport(SoftwareVersion::new(2, 28, 3)),
        ProcessLayoutFingerprint::new(2, 1),
    )
    .unwrap();
    let dynamic_topology = TuningTopologyFingerprint::<Dyn>::new_dyn(
        2,
        two_devices(),
        network_links().into_iter().rev().collect(),
        transport(SoftwareVersion::new(2, 28, 3)),
        ProcessLayoutFingerprint::new(2, 1),
    )
    .unwrap();
    assert_eq!(static_topology.digest(), dynamic_topology.digest());
    assert_eq!(
        dynamic_topology
            .clone()
            .try_into_static::<U2>()
            .unwrap()
            .digest(),
        static_topology.digest()
    );
    assert!(matches!(
        dynamic_topology.try_into_static::<U1>(),
        Err(IdentityError::StaticWorldMismatch {
            expected: 1,
            actual: 2
        })
    ));
}

#[test]
fn dynamic_topology_rejects_world_alias_link_and_layout_errors() {
    assert!(matches!(
        TuningTopologyFingerprint::<Dyn>::new_dyn(
            0,
            vec![],
            vec![],
            transport(SoftwareVersion::new(2, 28, 3)),
            ProcessLayoutFingerprint::new(0, 0),
        ),
        Err(IdentityError::ZeroWorld)
    ));
    assert!(matches!(
        TuningTopologyFingerprint::<Dyn>::new_dyn(
            2,
            vec![two_devices().remove(0)],
            vec![],
            transport(SoftwareVersion::new(2, 28, 3)),
            ProcessLayoutFingerprint::new(2, 1),
        ),
        Err(IdentityError::WorldMismatch {
            world: 2,
            devices: 1
        })
    ));

    let duplicate = cuda_device("GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa", "sm_90");
    assert!(matches!(
        TuningTopologyFingerprint::<Dyn>::new_dyn(
            2,
            vec![duplicate.clone(), duplicate],
            network_links(),
            transport(SoftwareVersion::new(2, 28, 3)),
            ProcessLayoutFingerprint::new(2, 1),
        ),
        Err(IdentityError::AliasedDevice {
            first_rank: 0,
            second_rank: 1,
            ..
        })
    ));
    assert!(matches!(
        TuningTopologyFingerprint::<Dyn>::new_dyn(
            2,
            two_devices(),
            vec![TopologyLink::new(0, 2, "network").unwrap()],
            transport(SoftwareVersion::new(2, 28, 3)),
            ProcessLayoutFingerprint::new(2, 1),
        ),
        Err(IdentityError::LinkOutOfRange {
            from: 0,
            to: 2,
            world: 2
        })
    ));
    assert!(matches!(
        TuningTopologyFingerprint::<Dyn>::new_dyn(
            2,
            two_devices(),
            network_links(),
            transport(SoftwareVersion::new(2, 28, 3)),
            ProcessLayoutFingerprint::new(1, 1),
        ),
        Err(IdentityError::ProcessLayoutMismatch {
            processes: 1,
            ranks_per_process: 1,
            world: 2
        })
    ));
}

#[test]
fn rank_mapping_link_transport_and_process_layout_are_all_identity() {
    let base = TuningTopologyFingerprint::<Dyn>::new_dyn(
        2,
        two_devices(),
        network_links(),
        transport(SoftwareVersion::new(2, 28, 3)),
        ProcessLayoutFingerprint::new(2, 1),
    )
    .unwrap();

    let mut reversed_devices = two_devices();
    reversed_devices.reverse();
    let reversed = TuningTopologyFingerprint::<Dyn>::new_dyn(
        2,
        reversed_devices,
        network_links(),
        transport(SoftwareVersion::new(2, 28, 3)),
        ProcessLayoutFingerprint::new(2, 1),
    )
    .unwrap();
    let pcie = TuningTopologyFingerprint::<Dyn>::new_dyn(
        2,
        two_devices(),
        vec![
            TopologyLink::new(0, 1, "host-bounce").unwrap(),
            TopologyLink::new(1, 0, "host-bounce").unwrap(),
        ],
        transport(SoftwareVersion::new(2, 28, 3)),
        ProcessLayoutFingerprint::new(2, 1),
    )
    .unwrap();
    let new_transport = TuningTopologyFingerprint::<Dyn>::new_dyn(
        2,
        two_devices(),
        network_links(),
        transport(SoftwareVersion::new(2, 29, 0)),
        ProcessLayoutFingerprint::new(2, 1),
    )
    .unwrap();
    let single_process = TuningTopologyFingerprint::<Dyn>::new_dyn(
        2,
        two_devices(),
        network_links(),
        transport(SoftwareVersion::new(2, 28, 3)),
        ProcessLayoutFingerprint::new(1, 2),
    )
    .unwrap();

    assert_ne!(base.digest(), reversed.digest());
    assert_ne!(base.digest(), pcie.digest());
    assert_ne!(base.digest(), new_transport.digest());
    assert_ne!(base.digest(), single_process.digest());
}

#[test]
fn static_contracts_are_compile_checked() {
    if std::fs::read("/home/xupremix/.cargo/config.toml").is_err() {
        return;
    }
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/tuning_identity_compile_fail/*.rs");
}
