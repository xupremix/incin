//! `DST-002`: the physical half of PROPOSALS.md §2.11.
//!
//! Every test here binds a mesh to a machine that does not exist. That is the
//! design, not a compromise: §2.11's physical proof is about installed devices,
//! link topology, and process layout, and a suite that could only run where
//! those happen to be right would never exercise a single rejection. The
//! machine is a [`FakeMachine`] implementing `TopologyProbe`, so the eight ways
//! binding can fail are eight ordinary test cases instead of eight
//! configurations nobody has.
//!
//! What this does *not* prove is that any real probe answers correctly.
//! Nothing in `incin-core` implements `TopologyProbe`; `DST-005` and `DST-006`
//! do, and their own evidence is where "the answers are true" gets checked.
//! This file proves the decisions made *from* the answers.
//!
//! The suite is gated on the `distributed` feature and so is its evidence
//! command. Appendix B ships a preview row behind a non-default feature, and a
//! `cfg`-gated suite run without the feature reports `ok` having run nothing.

#![cfg(feature = "distributed")]

use incin_core::dist::mesh::{
    BindError, CollectiveGroups, Data, DeviceIdentity, DeviceMesh, LinkClass, MeshAxis, MeshSpec,
    Pipeline, ProcessLayout, TensorParallel, TopologyProbe, TransportVersion, ValidMesh,
};
use incin_core::prelude::DeviceId;
use incin_core::typenum::{U1, U2, U3, U6};

/// A machine assembled by a test rather than by a vendor.
///
/// Devices are listed rather than probed, so "rank 4 names an ordinal that
/// does not exist" and "ranks 2 and 5 are secretly the same card" are two
/// lines of setup instead of a lab.
struct FakeMachine {
    /// `(ordinal, persistent id, architecture)`, in no particular order.
    devices: Vec<(DeviceId, String, String)>,
    /// Ordered pairs with no path between them. Everything not listed is
    /// [`LinkClass::HighBandwidth`], because the interesting cases here are
    /// the rejections and a machine where every link works is the baseline
    /// they are measured against.
    severed: Vec<(DeviceId, DeviceId)>,
    layout: ProcessLayout,
}

impl FakeMachine {
    /// `count` CUDA devices, ordinals `0..count`, all one architecture, all
    /// mutually reachable, one process.
    fn cuda(count: usize) -> Self {
        Self {
            devices: (0..count)
                .map(|i| (DeviceId::cuda(i), format!("GPU-{i}"), "sm_90".to_string()))
                .collect(),
            severed: Vec::new(),
            layout: ProcessLayout::SingleProcess,
        }
    }

    /// Replaces one device's identity, which is how the alias, family, and
    /// architecture rejections are set up.
    fn identify(mut self, ordinal: usize, device: DeviceId, persistent: &str, arch: &str) -> Self {
        self.devices[ordinal] = (device, persistent.to_string(), arch.to_string());
        self
    }

    fn sever(mut self, from: DeviceId, to: DeviceId) -> Self {
        self.severed.push((from, to));
        self
    }

    fn laid_out(mut self, layout: ProcessLayout) -> Self {
        self.layout = layout;
        self
    }
}

impl TopologyProbe for FakeMachine {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        self.devices
            .iter()
            .find(|(installed, _, _)| *installed == device)
            .map(|(installed, persistent, arch)| {
                DeviceIdentity::new(*installed, persistent.clone(), arch.clone())
            })
    }

    fn link(&self, from: DeviceId, to: DeviceId) -> LinkClass {
        if from == to {
            LinkClass::SameDevice
        } else if self.severed.contains(&(from, to)) {
            LinkClass::Unreachable
        } else {
            LinkClass::HighBandwidth
        }
    }

    fn transport(&self) -> TransportVersion {
        TransportVersion::new("fake".to_string(), 2, 21, 5)
    }

    fn layout(&self) -> ProcessLayout {
        self.layout.clone()
    }
}

/// The ordinals `0..count`, which is what a caller passes for the common case.
fn ordinals(count: usize) -> Vec<DeviceId> {
    (0..count).map(DeviceId::cuda).collect()
}

/// §2.11's own example: "For three GPUs, valid examples are `DP=3`, `TP=3`, or
/// `PP=3`."
///
/// All three bind against the same three devices, and they are three different
/// types. That pairing is the point of the module — the machine cannot tell
/// them apart and the compiler can.
#[test]
fn the_three_valid_three_gpu_topologies_all_bind_to_the_same_machine() {
    let machine = FakeMachine::cuda(3);
    let devices = ordinals(3);

    let data = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&devices, &machine).unwrap();
    let tensor =
        DeviceMesh::<MeshSpec<Data<U1>, TensorParallel<U3>>>::bind(&devices, &machine).unwrap();
    let pipeline = DeviceMesh::<MeshSpec<Data<U1>, TensorParallel<U1>, Pipeline<U3>>>::bind(
        &devices, &machine,
    )
    .unwrap();

    for rank in 0..3 {
        assert_eq!(data.device(rank).unwrap().device(), DeviceId::cuda(rank));
    }
    assert_eq!(data.groups().degree(MeshAxis::Data), 3);
    assert_eq!(tensor.groups().degree(MeshAxis::Tensor), 3);
    assert_eq!(pipeline.groups().degree(MeshAxis::Pipeline), 3);
}

/// The count guard runs first because every later guard indexes by rank.
#[test]
fn a_device_list_that_is_not_the_world_size_is_rejected() {
    let machine = FakeMachine::cuda(4);

    let err = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&ordinals(4), &machine).unwrap_err();

    assert_eq!(
        err,
        BindError::RankCount {
            expected: 3,
            found: 4
        }
    );
}

/// Two ranks pointed at one ordinal: the launcher misconfiguration that runs
/// fine, at half speed, double-counting a gradient.
#[test]
fn the_same_ordinal_at_two_ranks_is_rejected() {
    let machine = FakeMachine::cuda(3);
    let devices = vec![DeviceId::cuda(0), DeviceId::cuda(1), DeviceId::cuda(0)];

    let err = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&devices, &machine).unwrap_err();

    assert_eq!(
        err,
        BindError::RepeatedDevice {
            device: DeviceId::cuda(0),
            first: 0,
            second: 2,
        }
    );
}

/// An ordinal naming nothing. The probe's `None` is the only way this is
/// knowable, which is why it is a probe method and not a check.
#[test]
fn an_ordinal_the_probe_cannot_see_is_rejected() {
    let machine = FakeMachine::cuda(2);
    let devices = vec![DeviceId::cuda(0), DeviceId::cuda(1), DeviceId::cuda(7)];

    let err = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&devices, &machine).unwrap_err();

    assert_eq!(
        err,
        BindError::UnknownDevice {
            rank: 2,
            device: DeviceId::cuda(7),
        }
    );
}

/// §2.11: "Device ordinal alone is not a valid persistent identity."
///
/// This is the case that sentence is about, and the one no amount of ordinal
/// checking catches: two different numbers, one physical card, which is what a
/// visibility mask produces. `RepeatedDevice` passes here — the ordinals really
/// are distinct.
#[test]
fn two_ordinals_resolving_to_one_physical_device_are_rejected() {
    let machine = FakeMachine::cuda(3).identify(2, DeviceId::cuda(2), "GPU-0", "sm_90");

    let err = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&ordinals(3), &machine).unwrap_err();

    assert_eq!(
        err,
        BindError::AliasedDevice {
            first: 0,
            second: 2,
            persistent: "GPU-0".to_string(),
        }
    );
}

/// A mesh spanning two backend families has ranks that do not run the same
/// kernels.
#[test]
fn a_mesh_spanning_two_backend_families_is_rejected() {
    let machine = FakeMachine::cuda(3).identify(1, DeviceId::wgpu(1), "GPU-1", "sm_90");
    let devices = vec![DeviceId::cuda(0), DeviceId::wgpu(1), DeviceId::cuda(2)];

    let err = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&devices, &machine).unwrap_err();

    assert!(matches!(err, BindError::MixedBackendFamily { rank: 1, .. }));
}

/// Same family, different architecture — the case a family check alone misses.
#[test]
fn a_mesh_spanning_two_architectures_is_rejected() {
    let machine = FakeMachine::cuda(3).identify(2, DeviceId::cuda(2), "GPU-2", "sm_80");

    let err = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&ordinals(3), &machine).unwrap_err();

    assert_eq!(
        err,
        BindError::MixedArchitecture {
            expected: "sm_90".to_string(),
            found: "sm_80".to_string(),
            rank: 2,
        }
    );
}

/// §2.11 requires "agreement on rank/process/communicator identity". A
/// launcher that told this process it is one of four while the type says three
/// has already lost that agreement, and it is detectable before any
/// communicator exists.
#[test]
fn a_process_layout_that_disagrees_about_the_world_is_rejected() {
    let machine =
        FakeMachine::cuda(3).laid_out(ProcessLayout::ProcessPerRank { rank: 0, world: 4 });

    let err = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&ordinals(3), &machine).unwrap_err();

    assert!(matches!(
        err,
        BindError::UnsupportedProcessLayout { world: 3, .. }
    ));
}

/// A rank whose own tensor-parallel peer is unreachable cannot run the
/// collective that axis is made of.
#[test]
fn a_collective_group_whose_members_cannot_reach_each_other_is_rejected() {
    let machine = FakeMachine::cuda(2).sever(DeviceId::cuda(0), DeviceId::cuda(1));

    let err = DeviceMesh::<MeshSpec<Data<U1>, TensorParallel<U2>>>::bind(&ordinals(2), &machine)
        .unwrap_err();

    assert_eq!(
        err,
        BindError::UnreachableGroup {
            axis: MeshAxis::Tensor,
            from: 0,
            to: 1,
        }
    );
}

/// A severed link between ranks that never communicate is not this module's
/// business.
///
/// Two data-parallel replicas of a two-stage pipeline: rank 0 and rank 3 share
/// no group, so a missing path between them is not a reason to refuse. Without
/// this case the reachability guard could be "every pair must reach every
/// pair" and every test above would still pass.
#[test]
fn a_severed_link_between_ranks_that_share_no_group_is_not_a_rejection() {
    let machine = FakeMachine::cuda(4).sever(DeviceId::cuda(0), DeviceId::cuda(3));
    let bound = DeviceMesh::<MeshSpec<Data<U2>, TensorParallel<U1>, Pipeline<U2>>>::bind(
        &ordinals(4),
        &machine,
    )
    .unwrap();

    let groups = bound.groups();
    assert!(!groups.group(MeshAxis::Data, 0).unwrap().contains(&3));
    assert!(!groups.group(MeshAxis::Pipeline, 0).unwrap().contains(&3));
}

/// Rank and coordinates are inverses of each other at every rank.
///
/// A layout convention that is not its own inverse silently permutes a mesh:
/// every rank still gets a device, every group is still the right size, and
/// the wrong ranks are talking to each other. `2 × 3 × 2` is the smallest mesh
/// where all three degrees differ from each other and from one.
#[test]
fn every_rank_round_trips_through_its_coordinates() {
    type Mesh = MeshSpec<Data<U2>, TensorParallel<U3>, Pipeline<U2>>;
    let groups = CollectiveGroups::of::<Mesh>();

    assert_eq!(groups.world(), 12);
    assert_eq!(<Mesh as ValidMesh>::WORLD, 12);

    for rank in 0..groups.world() {
        let coordinates = groups.coordinates(rank).unwrap();
        assert_eq!(groups.rank_of(coordinates), Some(rank));
    }

    assert_eq!(groups.coordinates(12), None);
    assert_eq!(groups.rank_of([2, 0, 0]), None);
}

/// The layout puts tensor-parallel peers on consecutive ranks and
/// data-parallel replicas far apart, which is the whole reason to fix a
/// convention rather than let each axis pick one.
#[test]
fn the_layout_makes_tensor_peers_adjacent_and_data_replicas_distant() {
    type Mesh = MeshSpec<Data<U2>, TensorParallel<U3>, Pipeline<U2>>;
    let groups = CollectiveGroups::of::<Mesh>();

    assert_eq!(groups.group(MeshAxis::Tensor, 0).unwrap(), vec![0, 1, 2]);
    assert_eq!(groups.group(MeshAxis::Pipeline, 0).unwrap(), vec![0, 3]);
    assert_eq!(groups.group(MeshAxis::Data, 0).unwrap(), vec![0, 6]);

    // Membership is symmetric: every member of a group agrees on the group.
    for rank in 0..groups.world() {
        for axis in [MeshAxis::Data, MeshAxis::Pipeline, MeshAxis::Tensor] {
            let group = groups.group(axis, rank).unwrap();
            assert!(group.contains(&rank));
            for &peer in &group {
                assert_eq!(groups.group(axis, peer).unwrap(), group);
            }
        }
    }
}

/// Two processes that bound the same machine the same way compute the same id
/// without talking to each other. That is the property the digest exists for.
#[test]
fn the_same_machine_bound_the_same_way_has_the_same_id() {
    let first = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&ordinals(3), &FakeMachine::cuda(3))
        .unwrap()
        .id();
    let second = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&ordinals(3), &FakeMachine::cuda(3))
        .unwrap()
        .id();

    assert_eq!(first, second);
}

/// The degrees are part of the identity, and this is the case that proves they
/// have to be.
///
/// `DP=6` and `TP=6` over six fully-connected devices probe the same pairs in
/// the same order, so their *fingerprints* are identical — and they are
/// incompatible programs. Only the degrees tell them apart.
#[test]
fn one_machine_bound_two_ways_has_two_ids() {
    let machine = FakeMachine::cuda(6);
    let devices = ordinals(6);

    let replicated = DeviceMesh::<MeshSpec<Data<U6>>>::bind(&devices, &machine).unwrap();
    let sharded =
        DeviceMesh::<MeshSpec<Data<U1>, TensorParallel<U6>>>::bind(&devices, &machine).unwrap();

    assert_eq!(
        replicated.fingerprint().digest(),
        sharded.fingerprint().digest(),
        "the two bindings see an identical machine, which is what makes this case worth having"
    );
    assert_ne!(replicated.id(), sharded.id());
}

/// A different machine is a different mesh even at the same degrees and the
/// same ordinals — which is the half of the identity the degrees cannot carry.
#[test]
fn the_same_degrees_on_different_hardware_have_different_ids() {
    let devices = ordinals(3);
    let same = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&devices, &FakeMachine::cuda(3))
        .unwrap()
        .id();
    let older = FakeMachine::cuda(3)
        .identify(0, DeviceId::cuda(0), "GPU-0", "sm_80")
        .identify(1, DeviceId::cuda(1), "GPU-1", "sm_80")
        .identify(2, DeviceId::cuda(2), "GPU-2", "sm_80");
    let rebuilt = DeviceMesh::<MeshSpec<Data<U3>>>::bind(&devices, &older)
        .unwrap()
        .id();

    assert_ne!(same, rebuilt);
}

/// Two machines whose fields merely concatenate to the same bytes are two
/// machines.
///
/// The digest absorbs a length before every field for this reason. Without it
/// `persistent = "GPU-1", architecture = "sm_90"` and
/// `persistent = "GPU-", architecture = "1sm_90"` hash identically, and two
/// ranks on different hardware agree on a mesh id. Nothing else in this file
/// would notice.
#[test]
fn adjacent_fingerprint_fields_cannot_be_confused_for_each_other() {
    let split_late = FakeMachine::cuda(1).identify(0, DeviceId::cuda(0), "GPU-1", "sm_90");
    let split_early = FakeMachine::cuda(1).identify(0, DeviceId::cuda(0), "GPU-", "1sm_90");
    let devices = ordinals(1);

    let first = DeviceMesh::<MeshSpec<Data<U1>>>::bind(&devices, &split_late)
        .unwrap()
        .id();
    let second = DeviceMesh::<MeshSpec<Data<U1>>>::bind(&devices, &split_early)
        .unwrap()
        .id();

    assert_ne!(first, second);
}

/// The fingerprint records the links the mesh's own groups need, and no
/// others. "Relevant link classes" in §2.11 is a scope, not a hedge.
#[test]
fn the_fingerprint_records_the_links_the_groups_need() {
    let machine = FakeMachine::cuda(4);
    let bound = DeviceMesh::<MeshSpec<Data<U2>, TensorParallel<U1>, Pipeline<U2>>>::bind(
        &ordinals(4),
        &machine,
    )
    .unwrap();

    let links = bound.fingerprint().links();
    // Two data groups of two and two pipeline groups of two, each contributing
    // both ordered pairs: eight in total, and nothing between rank 0 and 3.
    assert_eq!(links.len(), 8);
    assert!(!links.iter().any(|&(from, to, _)| (from, to) == (0, 3)));
    assert!(links.iter().all(|&(_, _, class)| class.reaches()));
}
