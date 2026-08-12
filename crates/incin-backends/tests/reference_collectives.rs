//! `DST-005`: deterministic reference collective values, failures, adjoints,
//! dynamic dtype checks, and a two-rank network topology.

#![cfg(feature = "distributed-reference")]

use incin_backends::dist::{
    CollectiveBackend, CollectiveError, CollectiveKind, GroupId, ReferenceBuffer,
    ReferenceTopology, ReferenceTransport, ReferenceValues, StreamId,
};
use incin_core::dist::mesh::{
    Data, DeviceIdentity, DeviceMesh, LinkClass, MeshSpec, ProcessLayout,
};
use incin_core::exec::ReduceOp;
use incin_core::prelude::{DTypeId, DeviceId, Dyn};
use incin_core::typenum::U2;

fn group() -> GroupId {
    GroupId::new(7, 2).unwrap()
}

fn f32_buffer(values: &[f32]) -> ReferenceBuffer<f32> {
    ReferenceBuffer::try_new(ReferenceValues::F32(values.to_vec()), Default::default()).unwrap()
}

fn values_f32<K: incin_core::prelude::DType>(buffer: &ReferenceBuffer<K>) -> &[f32] {
    let ReferenceValues::F32(values) = buffer.values() else {
        panic!("test constructed an f32 buffer")
    };
    values
}

#[test]
fn all_reduce_is_rank_ordered_deterministic_and_preserves_scaling() {
    let transport = ReferenceTransport;
    let stream = StreamId::new(3);
    let inputs = [f32_buffer(&[1.0, 4.0]), f32_buffer(&[3.0, 2.0])];

    let sum = transport
        .all_reduce(group(), &inputs, ReduceOp::Sum, stream)
        .unwrap();
    assert_eq!(values_f32(&sum.buffers()[0]), &[4.0, 6.0]);
    assert_eq!(sum.buffers()[0], sum.buffers()[1]);
    assert_eq!(sum.event().kind(), CollectiveKind::AllReduce(ReduceOp::Sum));
    assert_eq!(sum.event().stream(), stream);

    let mean = transport
        .all_reduce(group(), &inputs, ReduceOp::Mean, stream)
        .unwrap();
    assert_eq!(values_f32(&mean.buffers()[0]), &[2.0, 3.0]);
    assert_eq!(
        CollectiveKind::AllReduce(ReduceOp::Mean).adjoint(),
        CollectiveKind::AllReduce(ReduceOp::Mean)
    );
}

#[test]
fn all_gather_and_reduce_scatter_are_explicit_adjoints() {
    let transport = ReferenceTransport;
    let gathered = transport
        .all_gather(
            group(),
            &[f32_buffer(&[1.0, 2.0]), f32_buffer(&[3.0, 4.0])],
            StreamId::default(),
        )
        .unwrap();
    assert_eq!(values_f32(&gathered.buffers()[0]), &[1.0, 2.0, 3.0, 4.0]);
    assert_eq!(
        CollectiveKind::AllGather.adjoint(),
        CollectiveKind::ReduceScatter(ReduceOp::Sum)
    );

    let scattered = transport
        .reduce_scatter(
            group(),
            gathered.buffers(),
            ReduceOp::Sum,
            StreamId::default(),
        )
        .unwrap();
    assert_eq!(values_f32(&scattered.buffers()[0]), &[2.0, 4.0]);
    assert_eq!(values_f32(&scattered.buffers()[1]), &[6.0, 8.0]);
    assert_eq!(
        CollectiveKind::ReduceScatter(ReduceOp::Sum).adjoint(),
        CollectiveKind::AllGather
    );
}

#[test]
fn all_to_all_is_its_own_inverse_permutation() {
    let transport = ReferenceTransport;
    let inputs = [
        f32_buffer(&[0.0, 1.0, 2.0, 3.0]),
        f32_buffer(&[4.0, 5.0, 6.0, 7.0]),
    ];
    let exchanged = transport
        .all_to_all(group(), &inputs, StreamId::default())
        .unwrap();
    assert_eq!(values_f32(&exchanged.buffers()[0]), &[0.0, 1.0, 4.0, 5.0]);
    assert_eq!(values_f32(&exchanged.buffers()[1]), &[2.0, 3.0, 6.0, 7.0]);

    let restored = transport
        .all_to_all(group(), exchanged.buffers(), StreamId::default())
        .unwrap();
    assert_eq!(restored.buffers(), inputs.as_slice());
    assert_eq!(CollectiveKind::AllToAll.adjoint(), CollectiveKind::AllToAll);
}

#[test]
fn send_recv_is_global_and_its_adjoint_reverses_the_peers() {
    let transport = ReferenceTransport;
    let inputs = [f32_buffer(&[1.0, 2.0]), f32_buffer(&[9.0, 8.0])];
    let moved = transport
        .send_recv(group(), &inputs, 0, 1, StreamId::new(4))
        .unwrap();
    assert_eq!(values_f32(&moved.buffers()[0]), &[1.0, 2.0]);
    assert_eq!(values_f32(&moved.buffers()[1]), &[1.0, 2.0]);
    assert_eq!(
        moved.event().kind(),
        CollectiveKind::SendRecv {
            source: 0,
            destination: 1
        }
    );
    assert_eq!(
        CollectiveKind::SendRecv {
            source: 0,
            destination: 1
        }
        .adjoint(),
        CollectiveKind::SendRecv {
            source: 1,
            destination: 0
        }
    );
    assert_eq!(
        transport
            .send_recv(group(), &inputs, 0, 0, StreamId::default())
            .unwrap_err(),
        CollectiveError::SamePeer { rank: 0 }
    );
    assert_eq!(
        transport
            .send_recv(group(), &inputs, 0, 2, StreamId::default())
            .unwrap_err(),
        CollectiveError::PeerOutOfRange {
            endpoint: "destination",
            rank: 2,
            ranks: 2,
        }
    );
}

#[test]
fn dynamic_dtype_is_checked_and_executes_the_same_collective() {
    let transport = ReferenceTransport;
    let inputs = [
        ReferenceBuffer::<Dyn>::try_new(ReferenceValues::F64(vec![1.0, 2.0]), DTypeId::F64.into())
            .unwrap(),
        ReferenceBuffer::<Dyn>::try_new(ReferenceValues::F64(vec![3.0, 4.0]), DTypeId::F64.into())
            .unwrap(),
    ];
    let output = transport
        .all_reduce(group(), &inputs, ReduceOp::Sum, StreamId::default())
        .unwrap();
    assert_eq!(
        output.buffers()[0].values(),
        &ReferenceValues::F64(vec![4.0, 6.0])
    );

    assert_eq!(
        ReferenceBuffer::<Dyn>::try_new(ReferenceValues::F32(vec![1.0]), DTypeId::F64.into())
            .unwrap_err(),
        CollectiveError::BufferDType {
            values: DTypeId::F32,
            typed: DTypeId::F64,
        }
    );
}

#[test]
fn counts_divisibility_integer_mean_and_overflow_are_structured_errors() {
    let transport = ReferenceTransport;
    assert_eq!(
        transport
            .all_reduce(
                group(),
                &[f32_buffer(&[1.0])],
                ReduceOp::Sum,
                StreamId::default(),
            )
            .unwrap_err(),
        CollectiveError::InputCount {
            expected: 2,
            found: 1,
        }
    );

    assert_eq!(
        transport
            .reduce_scatter(
                group(),
                &[f32_buffer(&[1.0, 2.0, 3.0]), f32_buffer(&[1.0, 2.0, 3.0])],
                ReduceOp::Sum,
                StreamId::default(),
            )
            .unwrap_err(),
        CollectiveError::NonDivisible {
            elements: 3,
            ranks: 2,
        }
    );

    let integers = [
        ReferenceBuffer::<u8>::try_new(ReferenceValues::U8(vec![200]), Default::default()).unwrap(),
        ReferenceBuffer::<u8>::try_new(ReferenceValues::U8(vec![100]), Default::default()).unwrap(),
    ];
    assert_eq!(
        transport
            .all_reduce(group(), &integers, ReduceOp::Mean, StreamId::default())
            .unwrap_err(),
        CollectiveError::UnsupportedReduction {
            dtype: DTypeId::U8,
            op: ReduceOp::Mean,
        }
    );
    assert_eq!(
        transport
            .all_reduce(group(), &integers, ReduceOp::Sum, StreamId::default())
            .unwrap_err(),
        CollectiveError::ReductionOverflow {
            dtype: DTypeId::U8,
            op: ReduceOp::Sum,
            element: 0,
        }
    );
}

#[test]
fn two_networked_cuda_ranks_bind_and_derive_one_mesh_id_in_both_processes() {
    type Mesh = MeshSpec<Data<U2>>;
    let devices = [DeviceId::cuda(0), DeviceId::cuda(1)];
    let identities = vec![
        DeviceIdentity::new(devices[0], "gpu-a".into(), "sm_90".into()),
        DeviceIdentity::new(devices[1], "gpu-b".into(), "sm_90".into()),
    ];
    let rank0 = ReferenceTopology::new(
        identities.clone(),
        LinkClass::Network,
        ProcessLayout::ProcessPerRank { rank: 0, world: 2 },
    );
    let rank1 = ReferenceTopology::new(
        identities,
        LinkClass::Network,
        ProcessLayout::ProcessPerRank { rank: 1, world: 2 },
    );

    let mesh0 = DeviceMesh::<Mesh>::bind(&devices, &rank0).unwrap();
    let mesh1 = DeviceMesh::<Mesh>::bind(&devices, &rank1).unwrap();
    assert_eq!(mesh0.id(), mesh1.id());
    assert!(
        mesh0
            .fingerprint()
            .links()
            .iter()
            .all(|(_, _, link)| *link == LinkClass::Network)
    );
}

#[test]
fn unsupported_static_collective_dtypes_are_compile_errors() {
    let cases = trybuild::TestCases::new();
    cases.compile_fail("tests/reference_compile_fail/*.rs");
}
