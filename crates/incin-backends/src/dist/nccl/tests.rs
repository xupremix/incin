//! Tests for two-rank NCCL transport, bootstrap protocol, and collective preflight validation.

use core::ffi::c_char;
use std::net::{SocketAddr, TcpListener};
use std::thread;
use std::time::Duration;

use incin_core::dist::mesh::{
    DeviceIdentity, LinkClass, MeshId, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::dist::{
    CollectiveError, CollectiveKind, CollectivePlan, DataParallelError, GradientId,
    PipelineBoundaryId, PipelineError, PipelineTransfer, PlanError, PlanSummary, StreamId,
    TensorParallelCollective, TensorParallelError, TensorParallelId, validate_collective_dtype,
};
use incin_core::exec::ReduceOp;
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::DeviceId;
use incin_core::tensor::dtype::DTypeId;

use crate::cuda::backend::CudaBackendImpl;
use crate::dist::nccl::config::{TwoRankBootstrapConfig, UNIQUE_ID_BYTES, WIRE_BYTES, WORLD};
use crate::dist::nccl::error::{NcclTransportError, catch_nccl_panic};
use crate::dist::nccl::topology::NcclTopology;
use crate::dist::nccl::transport::{
    reassemble_tensor_parallel_storage, validate_gradient_launch, validate_launch,
    validate_pipeline_launch, validate_reduction, validate_tensor_parallel_launch,
    validate_tensor_parallel_shapes,
};
use crate::dist::nccl::wire::{
    TopologyWire, WireMessage, exchange_bootstrap, exchange_topology, format_cuda_uuid,
    validate_wire,
};

fn all_reduce_plan() -> CollectivePlan {
    type Mesh =
        incin_core::dist::mesh::MeshSpec<incin_core::dist::mesh::Data<incin_core::typenum::U2>>;
    type PartialSum = incin_core::dist::Partial<Mesh, incin_core::dist::Sum>;
    type Replica = incin_core::dist::Replicated<Mesh>;
    let identities = [
        DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
        DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
    ];
    let topology = NcclTopology::new(
        identities,
        0,
        TransportVersion::new("nccl".into(), 2, 30, 0),
    )
    .unwrap();
    let mesh = incin_core::dist::mesh::DeviceMesh::<Mesh>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .unwrap();
    let mut builder = incin_core::dist::CollectivePlanBuilder::new(&mesh);
    builder
        .push_static_tagged::<f32, PartialSum, Replica>(
            incin_core::dist::CollectiveTag::new(41),
            incin_core::dist::mesh::MeshAxis::Data,
            0,
            4,
            StreamId::default(),
            None,
        )
        .unwrap();
    builder.finish()
}

fn data_parallel_plan() -> CollectivePlan {
    let identities = [
        DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
        DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
    ];
    let topology = NcclTopology::new(
        identities,
        0,
        TransportVersion::new("nccl".into(), 2, 30, 0),
    )
    .unwrap();
    let mesh = incin_core::dist::mesh::DeviceMesh::<incin_core::dist::TwoRankDataParallel>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .unwrap();
    let mut builder = incin_core::dist::DataParallelPlanBuilder::new(&mesh, 0);
    builder
        .push_static::<f32>(GradientId::new(41).unwrap(), 4, StreamId::default())
        .unwrap();
    builder.finish().unwrap().into_collective_plan()
}

fn tensor_parallel_plan() -> CollectivePlan {
    let identities = [
        DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
        DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
    ];
    let topology = NcclTopology::new(
        identities,
        0,
        TransportVersion::new("nccl".into(), 2, 30, 0),
    )
    .unwrap();
    let mesh = incin_core::dist::mesh::DeviceMesh::<incin_core::dist::TwoRankTensorParallel>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .unwrap();
    let mut builder = incin_core::dist::TensorParallelPlanBuilder::new(&mesh, 0);
    builder
        .push_column_static::<f32, incin_core::typenum::U0, incin_core::typenum::U4>(
            TensorParallelId::new(51).unwrap(),
            1,
            StreamId::default(),
        )
        .unwrap();
    builder
        .push_row_static::<f32, incin_core::typenum::U4>(
            TensorParallelId::new(52).unwrap(),
            2,
            StreamId::default(),
        )
        .unwrap();
    builder.finish().unwrap().into_collective_plan()
}

fn pipeline_plan() -> CollectivePlan {
    let identities = [
        DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
        DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
    ];
    let topology = NcclTopology::new(
        identities,
        0,
        TransportVersion::new("nccl".into(), 2, 30, 0),
    )
    .unwrap();
    let mesh = incin_core::dist::mesh::DeviceMesh::<incin_core::dist::TwoRankPipeline>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .unwrap();
    incin_core::dist::PipelinePlanBuilder::build_static::<
        f32,
        incin_core::shapes::DimCons<
            incin_core::shapes::Static<incin_core::typenum::U2>,
            incin_core::shapes::Nil,
        >,
        incin_core::typenum::U2,
        incin_core::dist::GPipe,
    >(
        &mesh,
        0,
        PipelineBoundaryId::new(61).unwrap(),
        incin_core::dist::ActivationCheckpoint::Keep,
        StreamId::default(),
    )
    .unwrap()
    .into_collective_plan()
}

fn summary(mesh: u64, hash: u64, collectives: usize) -> PlanSummary {
    PlanSummary::from_parts(MeshId::from_digest(mesh), hash, collectives)
}

fn localhost_listener() -> (TcpListener, SocketAddr) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    (listener, address)
}

#[test]
fn wire_round_trip_is_fixed_size_and_preserves_identity() {
    let id = core::array::from_fn(|index| index as u8);
    let message = WireMessage::new(1, summary(7, 11, 13), id);
    let encoded = message.encode();
    assert_eq!(encoded.len(), WIRE_BYTES);
    let decoded = WireMessage::decode(encoded).unwrap();
    assert_eq!(decoded.rank, 1);
    assert_eq!(decoded.summary().unwrap(), summary(7, 11, 13));
    assert_eq!(decoded.unique_id, id);
}

#[test]
fn topology_is_networked_rank_local_and_stable_across_processes() {
    let identities = [
        DeviceIdentity::new(DeviceId::cuda(0), "GPU-a".into(), "sm_90".into()),
        DeviceIdentity::new(DeviceId::cuda(1), "GPU-b".into(), "sm_90".into()),
    ];
    let version = TransportVersion::new("nccl".into(), 2, 30, 0);
    let rank0 = NcclTopology::new(identities.clone(), 0, version.clone()).unwrap();
    let rank1 = NcclTopology::new(identities, 1, version).unwrap();
    assert_eq!(
        rank0.link(DeviceId::cuda(0), DeviceId::cuda(1)),
        LinkClass::Network
    );
    assert_eq!(rank0.transport(), rank1.transport());
    assert_eq!(
        rank0.layout(),
        ProcessLayout::ProcessPerRank { rank: 0, world: 2 }
    );
    assert_eq!(
        rank1.layout(),
        ProcessLayout::ProcessPerRank { rank: 1, world: 2 }
    );
}

#[test]
fn topology_wire_round_trip_preserves_identity_and_version() {
    let identity = DeviceIdentity::new(DeviceId::cuda(1), "GPU-abc".into(), "sm_90".into());
    let transport = TransportVersion::new("nccl".into(), 2, 30, 1);
    let wire = TopologyWire::new(1, &identity, &transport).unwrap();
    let decoded = TopologyWire::decode(wire.encode()).unwrap();
    assert_eq!(decoded.identity().unwrap(), identity);
    assert_eq!(decoded.transport().unwrap(), transport);
}

#[test]
fn two_tcp_ranks_discover_the_same_ordered_topology() {
    let (reserved, address) = localhost_listener();
    drop(reserved);
    let timeout = Duration::from_secs(2);
    let version = TransportVersion::new("nccl".into(), 2, 30, 0);
    let root_identity = DeviceIdentity::new(DeviceId::cuda(0), "GPU-root".into(), "sm_90".into());
    let peer_identity = DeviceIdentity::new(DeviceId::cuda(1), "GPU-peer".into(), "sm_90".into());
    let root_version = version.clone();
    let root = thread::spawn(move || {
        exchange_topology(
            TwoRankBootstrapConfig::root(address, timeout),
            root_identity,
            root_version,
        )
    });
    let peer = exchange_topology(
        TwoRankBootstrapConfig::peer(address, timeout),
        peer_identity,
        version,
    )
    .unwrap();
    let root = root.join().unwrap().unwrap();
    for device in [DeviceId::cuda(0), DeviceId::cuda(1)] {
        assert_eq!(root.identify(device), peer.identify(device));
    }
    assert_eq!(root.transport(), peer.transport());
}

#[test]
fn discovery_then_plan_bootstrap_reuses_one_config() {
    for _ in 0..16 {
        let (reserved, address) = localhost_listener();
        drop(reserved);
        let timeout = Duration::from_secs(2);
        let local_summary = summary(17, 19, 2);
        let root = thread::spawn(move || {
            let topology = exchange_topology(
                TwoRankBootstrapConfig::root(address, timeout),
                DeviceIdentity::new(DeviceId::cuda(0), "GPU-root".into(), "sm_90".into()),
                TransportVersion::new("nccl".into(), 2, 30, 0),
            )?;
            let bootstrap = exchange_bootstrap(
                TwoRankBootstrapConfig::root(address, timeout),
                local_summary,
                Some([42; UNIQUE_ID_BYTES]),
            )?;
            Ok::<_, NcclTransportError>((topology, bootstrap))
        });
        let peer_topology = exchange_topology(
            TwoRankBootstrapConfig::peer(address, timeout),
            DeviceIdentity::new(DeviceId::cuda(1), "GPU-peer".into(), "sm_90".into()),
            TransportVersion::new("nccl".into(), 2, 30, 0),
        )
        .unwrap();
        let peer_bootstrap = exchange_bootstrap(
            TwoRankBootstrapConfig::peer(address, timeout),
            local_summary,
            None,
        )
        .unwrap();
        let (root_topology, root_bootstrap) = root.join().unwrap().unwrap();
        assert_eq!(
            root_topology.identify(DeviceId::cuda(1)),
            peer_topology.identify(DeviceId::cuda(1))
        );
        assert_eq!(root_bootstrap.agreed, peer_bootstrap.agreed);
        assert_eq!(root_bootstrap.unique_id, peer_bootstrap.unique_id);
    }
}

#[test]
fn cuda_uuid_format_is_canonical_and_unsigned() {
    let bytes: [c_char; 16] = core::array::from_fn(|index| (0xf0 + index as u8) as c_char);
    assert_eq!(
        format_cuda_uuid(bytes),
        "f0f1f2f3-f4f5-f6f7-f8f9-fafbfcfdfeff"
    );
}

#[test]
#[ignore = "requires one CUDA device"]
fn local_cuda_identity_uses_uuid_and_compute_capability() {
    let identity = NcclTopology::probe_local_cuda_identity(0, 0).unwrap();
    assert_eq!(identity.device(), DeviceId::cuda(0));
    assert_eq!(identity.persistent().len(), 36);
    assert!(identity.architecture().starts_with("sm_"));
}

#[test]
fn corrupt_magic_and_wrong_identity_are_structured_failures() {
    let mut encoded = WireMessage::new(1, summary(7, 11, 13), [0; 128]).encode();
    encoded[0] ^= 1;
    assert!(matches!(
        WireMessage::decode(encoded),
        Err(NcclTransportError::Protocol(_))
    ));
    assert!(matches!(
        validate_wire(&WireMessage::new(0, summary(7, 11, 13), [0; 128]), 1),
        Err(NcclTransportError::RemoteRank {
            expected: 1,
            found: 0
        })
    ));
}

#[test]
fn two_tcp_ranks_exchange_one_id_and_agree_on_plan() {
    let (reserved, address) = localhost_listener();
    drop(reserved);
    let timeout = Duration::from_secs(2);
    let local = summary(17, 19, 3);
    let id = [23; UNIQUE_ID_BYTES];
    let root = thread::spawn(move || {
        exchange_bootstrap(
            TwoRankBootstrapConfig::root(address, timeout),
            local,
            Some(id),
        )
    });
    let peer =
        exchange_bootstrap(TwoRankBootstrapConfig::peer(address, timeout), local, None).unwrap();
    let root = root.join().unwrap().unwrap();
    assert_eq!(root.unique_id, id);
    assert_eq!(peer.unique_id, id);
    assert_eq!(root.agreed, peer.agreed);
    assert_eq!(root.agreed.ranks(), WORLD);
}

#[test]
fn divergent_plan_is_rejected_before_communicator_creation() {
    let (reserved, address) = localhost_listener();
    drop(reserved);
    let timeout = Duration::from_secs(2);
    let root = thread::spawn(move || {
        exchange_bootstrap(
            TwoRankBootstrapConfig::root(address, timeout),
            summary(17, 19, 3),
            Some([23; UNIQUE_ID_BYTES]),
        )
    });
    let peer = exchange_bootstrap(
        TwoRankBootstrapConfig::peer(address, timeout),
        summary(17, 20, 3),
        None,
    );
    let root = root.join().unwrap();
    assert!(matches!(
        root,
        Err(NcclTransportError::Plan(PlanError::PlanHashMismatch {
            rank: 1,
            expected: 19,
            found: 20
        }))
    ));
    assert!(peer.is_err());
}

#[test]
fn missing_peer_hits_a_bounded_accept_timeout() {
    let (reserved, address) = localhost_listener();
    drop(reserved);
    let timeout = Duration::from_millis(15);
    let error = exchange_bootstrap(
        TwoRankBootstrapConfig::root(address, timeout),
        summary(1, 2, 0),
        Some([0; UNIQUE_ID_BYTES]),
    )
    .unwrap_err();
    assert!(matches!(
        error,
        NcclTransportError::Timeout {
            phase: "accept rank one",
            ..
        }
    ));
}

#[test]
fn dyn_dtype_validation_matches_static_collective_policy() {
    assert!(validate_collective_dtype(DTypeId::F64).is_ok());
    assert_eq!(
        validate_collective_dtype(DTypeId::Q8_0).unwrap_err(),
        CollectiveError::UnsupportedDType {
            dtype: DTypeId::Q8_0
        }
    );
    assert!(matches!(
        validate_reduction(CollectiveKind::AllReduce(ReduceOp::Mean), DTypeId::U32),
        Err(NcclTransportError::Collective(
            CollectiveError::UnsupportedReduction {
                dtype: DTypeId::U32,
                op: ReduceOp::Mean
            }
        ))
    ));
}

#[test]
fn dynamic_loader_panics_become_structured_errors() {
    let error = catch_nccl_panic("load test", || panic!("libnccl not found")).unwrap_err();
    assert!(matches!(
        error,
        NcclTransportError::NcclUnavailable {
            operation: "load test",
            message
        } if message == "libnccl not found"
    ));
}

#[test]
#[ignore = "probes the optional system NCCL shared library"]
fn installed_nccl_probe_returns_instead_of_unwinding() {
    let _available_or_structured_error = NcclTopology::installed_transport_version();
}

#[test]
fn launch_preflight_rejects_order_dtype_count_and_byte_drift() {
    let plan = all_reduce_plan();
    let descriptor = &plan.descriptors()[0];
    assert!(validate_launch(descriptor, 0, DTypeId::F32, 4, 16).is_ok());
    assert!(matches!(
        validate_launch(descriptor, 1, DTypeId::F32, 4, 16),
        Err(NcclTransportError::Sequence {
            expected: 1,
            found: 0
        })
    ));
    assert!(matches!(
        validate_launch(descriptor, 0, DTypeId::F64, 4, 32),
        Err(NcclTransportError::DType {
            expected: DTypeId::F32,
            found: DTypeId::F64
        })
    ));
    assert!(matches!(
        validate_launch(descriptor, 0, DTypeId::F32, 3, 12),
        Err(NcclTransportError::Elements {
            expected: 4,
            found: 3
        })
    ));
    assert!(matches!(
        validate_launch(descriptor, 0, DTypeId::F32, 4, 12),
        Err(NcclTransportError::BufferBytes {
            expected: 16,
            found: 12
        })
    ));
    assert!(matches!(
        validate_launch(descriptor, 0, DTypeId::Q8_0, 4, 16),
        Err(NcclTransportError::Collective(
            CollectiveError::UnsupportedDType {
                dtype: DTypeId::Q8_0
            }
        ))
    ));
}

#[test]
fn gradient_preflight_requires_identity_mean_placement_and_dyn_float_dtype() {
    let plan = data_parallel_plan();
    let descriptor = &plan.descriptors()[0];
    assert!(
        validate_gradient_launch(
            descriptor,
            0,
            GradientId::new(41).unwrap(),
            DTypeId::F32,
            4,
            16,
        )
        .is_ok()
    );
    assert!(matches!(
        validate_gradient_launch(
            descriptor,
            0,
            GradientId::new(42).unwrap(),
            DTypeId::F32,
            4,
            16,
        ),
        Err(NcclTransportError::GradientIdentity {
            expected: 41,
            found: 42
        })
    ));
    assert!(matches!(
        validate_gradient_launch(
            descriptor,
            0,
            GradientId::new(41).unwrap(),
            DTypeId::U32,
            4,
            16,
        ),
        Err(NcclTransportError::DataParallel(
            DataParallelError::UnsupportedGradientDType {
                dtype: DTypeId::U32
            }
        ))
    ));

    let sum = all_reduce_plan();
    assert!(matches!(
        validate_gradient_launch(
            &sum.descriptors()[0],
            0,
            GradientId::new(41).unwrap(),
            DTypeId::F32,
            4,
            16,
        ),
        Err(NcclTransportError::NotDataParallelGradient { .. })
    ));
}

#[test]
fn tensor_parallel_preflight_requires_identity_semantics_and_dyn_float_dtype() {
    let plan = tensor_parallel_plan();
    let column = &plan.descriptors()[0];
    let column_kind = TensorParallelCollective::ColumnOutputGather { tensor_axis: 0 };
    assert!(
        validate_tensor_parallel_launch(
            column,
            0,
            TensorParallelId::new(51).unwrap(),
            column_kind,
            DTypeId::F32,
            2,
            8,
        )
        .is_ok()
    );
    assert!(matches!(
        validate_tensor_parallel_launch(
            column,
            0,
            TensorParallelId::new(52).unwrap(),
            column_kind,
            DTypeId::F32,
            2,
            8,
        ),
        Err(NcclTransportError::TensorParallelIdentity { .. })
    ));
    assert!(matches!(
        validate_tensor_parallel_launch(
            column,
            0,
            TensorParallelId::new(51).unwrap(),
            TensorParallelCollective::AttentionHeadGather { tensor_axis: 0 },
            DTypeId::F32,
            2,
            8,
        ),
        Err(NcclTransportError::TensorParallelIdentity { .. })
    ));
    assert!(matches!(
        validate_tensor_parallel_launch(
            column,
            0,
            TensorParallelId::new(51).unwrap(),
            column_kind,
            DTypeId::U32,
            2,
            8,
        ),
        Err(NcclTransportError::TensorParallel(
            TensorParallelError::UnsupportedTensorDType {
                dtype: DTypeId::U32
            }
        ))
    ));
    assert_eq!(
        validate_tensor_parallel_shapes(
            column,
            TensorParallelCollective::ColumnOutputGather { tensor_axis: 1 },
            &[2, 1],
            &[2, 2],
        )
        .unwrap(),
        vec![2, 1]
    );
    assert!(matches!(
        validate_tensor_parallel_shapes(
            column,
            TensorParallelCollective::ColumnOutputGather { tensor_axis: 1 },
            &[1, 2],
            &[2, 2],
        ),
        Err(NcclTransportError::TensorParallelShape { .. })
    ));
    assert!(matches!(
        validate_tensor_parallel_shapes(column, column_kind, &[3], &[6]),
        Err(NcclTransportError::TensorParallelOutputElements {
            expected: 4,
            found: 6
        })
    ));

    let row = &plan.descriptors()[1];
    assert!(
        validate_tensor_parallel_launch(
            row,
            1,
            TensorParallelId::new(52).unwrap(),
            TensorParallelCollective::RowOutputSum,
            DTypeId::F32,
            2,
            8,
        )
        .is_ok()
    );
    assert_eq!(
        validate_tensor_parallel_shapes(
            row,
            TensorParallelCollective::RowOutputSum,
            &[1, 2],
            &[1, 2],
        )
        .unwrap(),
        vec![1, 2]
    );
    assert!(matches!(
        validate_tensor_parallel_launch(
            row,
            1,
            TensorParallelId::new(52).unwrap(),
            column_kind,
            DTypeId::F32,
            2,
            8,
        ),
        Err(NcclTransportError::TensorParallelIdentity { .. })
    ));
}

#[test]
fn pipeline_preflight_requires_identity_direction_and_dyn_float_dtype() {
    let plan = pipeline_plan();
    let forward = &plan.descriptors()[0];
    assert!(
        validate_pipeline_launch(
            forward,
            0,
            PipelineBoundaryId::new(61).unwrap(),
            PipelineTransfer::ForwardActivation,
            0,
            DTypeId::F32,
            2,
            8,
        )
        .is_ok()
    );
    assert!(matches!(
        validate_pipeline_launch(
            forward,
            0,
            PipelineBoundaryId::new(62).unwrap(),
            PipelineTransfer::ForwardActivation,
            0,
            DTypeId::F32,
            2,
            8,
        ),
        Err(NcclTransportError::PipelineIdentity { .. })
    ));
    assert!(matches!(
        validate_pipeline_launch(
            forward,
            0,
            PipelineBoundaryId::new(61).unwrap(),
            PipelineTransfer::BackwardGradient,
            0,
            DTypeId::F32,
            2,
            8,
        ),
        Err(NcclTransportError::PipelineIdentity { .. })
    ));
    assert!(matches!(
        validate_pipeline_launch(
            forward,
            0,
            PipelineBoundaryId::new(61).unwrap(),
            PipelineTransfer::ForwardActivation,
            0,
            DTypeId::U32,
            2,
            8,
        ),
        Err(NcclTransportError::Pipeline(
            PipelineError::UnsupportedDType {
                dtype: DTypeId::U32
            }
        ))
    ));
    assert!(matches!(
        validate_pipeline_launch(
            forward,
            0,
            PipelineBoundaryId::new(61).unwrap(),
            PipelineTransfer::ForwardActivation,
            0,
            DTypeId::F32,
            3,
            12,
        ),
        Err(NcclTransportError::Elements {
            expected: 2,
            found: 3
        })
    ));
}

#[test]
#[ignore = "requires one CUDA device"]
fn tensor_parallel_reassembly_moves_rank_axis_on_cuda_for_static_and_dyn() {
    use incin_core::backend_authoring::HostInterop;

    type B = CudaBackendImpl<incin_core::tensor::device::CudaN<incin_core::typenum::U0>>;
    type D = incin_core::tensor::device::CudaN<incin_core::typenum::U0>;

    let rank_major = [
        1.0f32, 2.0, 3.0, 2.0, 3.0, 4.0, //
        4.0, 3.0, 7.0, 5.0, 5.0, 9.0,
    ];
    let expected = [
        1.0f32, 2.0, 3.0, 4.0, 3.0, 7.0, //
        2.0, 3.0, 4.0, 5.0, 5.0, 9.0,
    ];
    let collective = TensorParallelCollective::ColumnOutputGather { tensor_axis: 1 };

    let static_input =
        Tensor::<incin_core::shapes::Dyn, B>::from_slice(&rank_major, vec![12]).unwrap();
    let static_output = reassemble_tensor_parallel_storage::<D, f32>(
        static_input.inner(),
        collective,
        &[2, 3],
        &[2, 6],
    )
    .unwrap();
    let static_bytes = B::to_bytes::<f32>(&static_output).unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, f32>(&static_bytes), expected);

    let dyn_input = Tensor::<incin_core::shapes::Dyn, B, incin_core::shapes::Dyn>::from_bytes(
        bytemuck::cast_slice(&rank_major),
        (vec![12], DTypeId::F32),
    )
    .unwrap();
    let dyn_output = reassemble_tensor_parallel_storage::<D, incin_core::shapes::Dyn>(
        dyn_input.inner(),
        collective,
        &[2, 3],
        &[2, 6],
    )
    .unwrap();
    let dyn_bytes = B::to_bytes::<incin_core::shapes::Dyn>(&dyn_output).unwrap();
    assert_eq!(bytemuck::cast_slice::<u8, f32>(&dyn_bytes), expected);
}
