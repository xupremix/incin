//! Column- and row-parallel partitioned matmul over the reference transport
//! (issue #99, in-process slice).
//!
//! A [`TensorParallelPlan`] says rank `r` owns output-column shard `r` of
//! a linear layer and the shards gather into the replicated output, or that
//! rank `r` owns contraction-row shard `r` and the partial sums reduce into
//! the replicated output. This file executes both sentences with real CPU
//! kernels: each rank's shard matmul runs on the CPU backend, the two
//! partial outputs cross the deterministic reference transport through the
//! plan's own `AllGather`/`AllReduce` descriptor, and a kernel concat (for
//! the column leg) rebuilds the full output - which must match the
//! single-device matmul within tolerance.
//!
//! Two partition axes, done honestly: column-parallel (gather) and
//! row-parallel (partial-sum all-reduce). Head-parallel attention crosses
//! the same transport in `incin-backends`' `tensor_parallel_reference`
//! tests at the host-arithmetic level; wiring it through backend kernels is
//! future work, and it is not claimed here. Everything runs on the CPU with
//! no hardware: the mesh binds two CUDA stand-ins for planning while
//! every tensor stays on the CPU backend. The two-host NCCL leg stays
//! hardware-gated (issue #82).

#![cfg(all(feature = "cpu", feature = "distributed-reference"))]

use incin::backend_authoring::HostReadback;
use incin::experimental::distributed::mesh::{
    DeviceIdentity, DeviceMesh, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin::experimental::distributed::{
    CollectiveKind, PlacementKind, StreamId, TensorParallelDimension, TensorParallelId,
    TensorParallelPlanBuilder, TwoRankTensorParallel, validate_two_way_extent,
};
use incin::prelude::*;
use incin_backends::dist::{
    CollectiveBackend, ReferenceBuffer, ReferenceTransport, ReferenceValues,
};
use incin_core::exec::ReduceOp;

type Backend = incin::DefaultBackend;

/// In-process planning probe: two reachable CUDA stand-ins, no hardware.
struct ReferenceTp2;

impl TopologyProbe for ReferenceTp2 {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        (device.kind() == DeviceKind::Cuda && device.ordinal() < 2).then(|| {
            DeviceIdentity::new(
                device,
                format!("reference-tp2-exec-{}", device.ordinal()),
                "sm_reference".to_string(),
            )
        })
    }

    fn link(&self, from: DeviceId, to: DeviceId) -> LinkClass {
        if from == to {
            LinkClass::SameDevice
        } else {
            LinkClass::Network
        }
    }

    fn transport(&self) -> TransportVersion {
        TransportVersion::new("reference".to_string(), 1, 0, 0)
    }

    fn layout(&self) -> ProcessLayout {
        ProcessLayout::SingleProcess
    }
}

fn mesh() -> DeviceMesh<TwoRankTensorParallel> {
    DeviceMesh::bind(&[DeviceId::cuda(0), DeviceId::cuda(1)], &ReferenceTp2).unwrap()
}

fn f32_buffer(values: Vec<f32>) -> ReferenceBuffer<f32> {
    ReferenceBuffer::try_new(ReferenceValues::F32(values), Default::default()).unwrap()
}

fn read_f32<B: HostReadback, K: DType>(storage: &B::Storage<K>) -> Vec<f64> {
    B::float_to_vec1::<K>(storage).expect("CPU readback works")
}

/// A column-parallel linear across two in-process ranks matches the
/// single-device matmul.
///
/// `X @ W` with `W = [W0 | W1]` becomes `X @ W0` and `X @ W1` on the CPU
/// backend, one `AllGather` through the reference transport under the
/// plan's descriptor (group, stream, and placement), and a kernel concat
/// along the output axis. The transport moves the bytes; the concat
/// restores the row-major `[batch, out]` layout the gather's
/// shard-concatenation leaves destrided.
#[test]
fn column_parallel_partitioned_matmul_matches_single_device() -> Result<()> {
    // Fixed problem: batch 4, 8 input features, 6 output features.
    let x_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let w_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.07 - 0.5).collect();
    // Column split on the host (data prep, not execution): row `r` of the
    // global weight contributes its first three entries to rank 0.
    let mut w0_data = Vec::with_capacity(24);
    let mut w1_data = Vec::with_capacity(24);
    for row in w_data.chunks_exact(6) {
        w0_data.extend_from_slice(&row[..3]);
        w1_data.extend_from_slice(&row[3..]);
    }

    let plan = {
        let bound = mesh();
        let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
        builder
            .push_column_dyn(
                TensorParallelId::new(11).unwrap(),
                &[4, 6],
                1,
                DTypeId::F32,
                StreamId::new(3),
            )
            .expect("a divisible column plan builds");
        builder.finish().expect("a non-empty TP plan finishes")
    };
    let operations = plan.operations();
    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].local_elements(), 12);
    assert_eq!(operations[0].global_elements(), 24);
    let descriptor = &plan.collective_plan().descriptors()[0];
    assert_eq!(descriptor.kind(), CollectiveKind::AllGather);
    assert_eq!(descriptor.source(), PlacementKind::Sharded { axis: 1 });
    assert_eq!(descriptor.destination(), PlacementKind::Replicated);

    let x = Tensor::<Dyn, Backend>::from_slice(&x_data, vec![4, 8])?;
    let w = Tensor::<Dyn, Backend>::from_slice(&w_data, vec![8, 6])?;
    let w0 = Tensor::<Dyn, Backend>::from_slice(&w0_data, vec![8, 3])?;
    let w1 = Tensor::<Dyn, Backend>::from_slice(&w1_data, vec![8, 3])?;

    // The single-device reference both ranks must reproduce.
    let full = x.matmul(&w)?;
    let expected = read_f32::<Backend, f32>(full.inner());

    // Each rank's local shard matmul, on the real CPU kernel.
    let y0 = x.matmul(&w0)?;
    let y1 = x.matmul(&w1)?;
    let part0 = read_f32::<Backend, f32>(y0.inner());
    let part1 = read_f32::<Backend, f32>(y1.inner());
    assert_eq!(part0.len(), 12);
    assert_eq!(part1.len(), 12);

    // The boundary collective, under the plan's own descriptor: group,
    // stream, and the f32 the plan validated.
    let gathered = ReferenceTransport
        .all_gather(
            descriptor.group(),
            &[
                f32_buffer(part0.iter().map(|v| *v as f32).collect()),
                f32_buffer(part1.iter().map(|v| *v as f32).collect()),
            ],
            descriptor.stream(),
        )
        .expect("the reference all-gather runs");
    let (buffers, _) = gathered.into_parts();
    assert_eq!(buffers.len(), 2, "one output buffer per rank");
    let mut shards = Vec::with_capacity(2);
    for buffer in &buffers {
        let ReferenceValues::F32(values) = buffer.values() else {
            panic!("the plan validated f32; the transport must return f32");
        };
        shards.push(values.clone());
    }
    // Both ranks receive the same rank-ordered concatenation.
    assert_eq!(shards[0], shards[1], "all-gather replicates to every rank");
    let flat: Vec<f32> = part0
        .iter()
        .map(|v| *v as f32)
        .chain(part1.iter().map(|v| *v as f32))
        .collect();
    assert_eq!(
        shards[0], flat,
        "the gather concatenates rank 0's shard before rank 1's"
    );

    // Row-major restore plus kernel concat: shard rows `[r0, r1]` of each
    // `[4, 3]` partial interleave into the full `[4, 6]` rows.
    let t0 = Tensor::<Dyn, Backend>::from_slice(
        &part0.iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        vec![4, 3],
    )?;
    let t1 = Tensor::<Dyn, Backend>::from_slice(
        &part1.iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        vec![4, 3],
    )?;
    let joined = t0.concat(&t1, 1isize)?;
    let found = read_f32::<Backend, f32>(joined.inner());

    assert_eq!(found.len(), expected.len());
    let mut max_rel: f64 = 0.0;
    for (got, want) in found.iter().zip(&expected) {
        max_rel = max_rel.max((got - want).abs() / (1.0 + want.abs()));
    }
    eprintln!("tp2 column matmul: max relative diff {max_rel}");
    for (index, (got, want)) in found.iter().zip(&expected).enumerate() {
        let limit = 1e-5 * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= limit,
            "output {index}: {got} vs single-device {want} (limit {limit})"
        );
    }
    Ok(())
}

/// A row-parallel linear across two in-process ranks matches the
/// single-device matmul.
///
/// `X @ W` with the contraction axis split as `X = [X0 | X1]` and
/// `W = [W0; W1]` becomes `X0 @ W0` and `X1 @ W1` on the CPU backend, one
/// `AllReduce(Sum)` through the reference transport under the plan's
/// descriptor (group, stream, and the f32 the plan validated), and the
/// replicated sum - which must match the single-device matmul within
/// tolerance. The transport sums the bytes; no layout restore is needed
/// because every rank already holds a complete `[batch, out]` partial sum.
#[test]
fn row_parallel_partitioned_matmul_matches_single_device() -> Result<()> {
    // Fixed problem: batch 4, 8 input features, 6 output features; the
    // contraction axis (8) splits into 4 + 4.
    let x_data: Vec<f32> = (0..32).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let w_data: Vec<f32> = (0..48).map(|i| (i as f32) * 0.07 - 0.5).collect();
    // Row split on the host (data prep, not execution): each `[1, 8]`
    // input row contributes its first four entries to rank 0, and the
    // row-major `[8, 6]` weight contributes its first four rows to rank 0.
    let mut x0_data = Vec::with_capacity(16);
    let mut x1_data = Vec::with_capacity(16);
    for row in x_data.chunks_exact(8) {
        x0_data.extend_from_slice(&row[..4]);
        x1_data.extend_from_slice(&row[4..]);
    }
    let w0_data = w_data[..24].to_vec();
    let w1_data = w_data[24..].to_vec();

    let plan = {
        let bound = mesh();
        let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
        builder
            .push_row_dyn(
                TensorParallelId::new(21).unwrap(),
                8,
                24,
                DTypeId::F32,
                StreamId::new(4),
            )
            .expect("a divisible row plan builds");
        builder.finish().expect("a non-empty TP plan finishes")
    };
    let operations = plan.operations();
    assert_eq!(operations.len(), 1);
    assert_eq!(operations[0].local_elements(), 24);
    assert_eq!(operations[0].global_elements(), 24);
    let descriptor = &plan.collective_plan().descriptors()[0];
    assert_eq!(descriptor.kind(), CollectiveKind::AllReduce(ReduceOp::Sum));
    assert_eq!(
        descriptor.source(),
        PlacementKind::Partial {
            reduction: ReduceOp::Sum
        }
    );
    assert_eq!(descriptor.destination(), PlacementKind::Replicated);

    let x = Tensor::<Dyn, Backend>::from_slice(&x_data, vec![4, 8])?;
    let w = Tensor::<Dyn, Backend>::from_slice(&w_data, vec![8, 6])?;
    let x0 = Tensor::<Dyn, Backend>::from_slice(&x0_data, vec![4, 4])?;
    let x1 = Tensor::<Dyn, Backend>::from_slice(&x1_data, vec![4, 4])?;
    let w0 = Tensor::<Dyn, Backend>::from_slice(&w0_data, vec![4, 6])?;
    let w1 = Tensor::<Dyn, Backend>::from_slice(&w1_data, vec![4, 6])?;

    // The single-device reference both ranks must reproduce.
    let full = x.matmul(&w)?;
    let expected = read_f32::<Backend, f32>(full.inner());

    // Each rank's local shard matmul, on the real CPU kernel: complete
    // `[4, 6]` partial sums, not shards of the output.
    let y0 = x0.matmul(&w0)?;
    let y1 = x1.matmul(&w1)?;
    let part0 = read_f32::<Backend, f32>(y0.inner());
    let part1 = read_f32::<Backend, f32>(y1.inner());
    assert_eq!(part0.len(), 24);
    assert_eq!(part1.len(), 24);

    // The boundary collective, under the plan's own descriptor: group,
    // stream, and the f32 the plan validated.
    let reduced = ReferenceTransport
        .all_reduce(
            descriptor.group(),
            &[
                f32_buffer(part0.iter().map(|v| *v as f32).collect()),
                f32_buffer(part1.iter().map(|v| *v as f32).collect()),
            ],
            ReduceOp::Sum,
            descriptor.stream(),
        )
        .expect("the reference all-reduce runs");
    let (buffers, _) = reduced.into_parts();
    assert_eq!(buffers.len(), 2, "one output buffer per rank");
    let mut sums = Vec::with_capacity(2);
    for buffer in &buffers {
        let ReferenceValues::F32(values) = buffer.values() else {
            panic!("the plan validated f32; the transport must return f32");
        };
        sums.push(values.clone());
    }
    // Both ranks receive the same element-wise sum.
    assert_eq!(sums[0], sums[1], "all-reduce replicates to every rank");
    let summed: Vec<f32> = part0
        .iter()
        .map(|v| *v as f32)
        .zip(part1.iter().map(|v| *v as f32))
        .map(|(a, b)| a + b)
        .collect();
    assert_eq!(
        sums[0], summed,
        "the reduce sums rank 0's partial with rank 1's"
    );

    let found: Vec<f64> = sums[0].iter().map(|v| *v as f64).collect();
    assert_eq!(found.len(), expected.len());
    let mut max_rel: f64 = 0.0;
    for (got, want) in found.iter().zip(&expected) {
        max_rel = max_rel.max((got - want).abs() / (1.0 + want.abs()));
    }
    eprintln!("tp2 row matmul: max relative diff {max_rel}");
    for (index, (got, want)) in found.iter().zip(&expected).enumerate() {
        let limit = 1e-5 * (1.0 + want.abs());
        assert!(
            (got - want).abs() <= limit,
            "output {index}: {got} vs single-device {want} (limit {limit})"
        );
    }
    Ok(())
}

/// A column extent that does not divide across two ranks is refused at
/// plan build - before any shard matmul runs, not as a wrong-shaped
/// gather. A non-floating dtype is refused on the same path.
#[test]
fn indivisible_or_non_float_columns_are_refused() {
    let bound = mesh();
    let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
    match builder.push_column_dyn(
        TensorParallelId::new(12).unwrap(),
        &[4, 5],
        1,
        DTypeId::F32,
        StreamId::new(3),
    ) {
        Err(incin::experimental::distributed::TensorParallelError::NonDivisible {
            dimension,
            extent,
            ranks,
        }) => {
            assert_eq!(dimension, TensorParallelDimension::OutputFeatures);
            assert_eq!((extent, ranks), (5, 2));
        }
        other => panic!("expected NonDivisible, got {other:?}"),
    }
    match builder.push_column_dyn(
        TensorParallelId::new(13).unwrap(),
        &[4, 6],
        1,
        DTypeId::U8,
        StreamId::new(3),
    ) {
        Err(incin::experimental::distributed::TensorParallelError::UnsupportedTensorDType {
            dtype,
        }) => assert_eq!(dtype, DTypeId::U8),
        other => panic!("expected UnsupportedTensorDType, got {other:?}"),
    }
    // The extent validator agrees without a plan in the loop.
    assert!(validate_two_way_extent(TensorParallelDimension::OutputFeatures, 6).is_ok());
    assert!(validate_two_way_extent(TensorParallelDimension::OutputFeatures, 5).is_err());
}
