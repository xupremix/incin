//! Example: real two-rank tensor parallelism over the in-process
//! reference transport.
//!
//! A `[2, 8]` input through an `[8, 4]` weight runs three ways: whole on
//! one device, column-parallel (one shard matmul per rank, `AllGather`
//! through the plan's descriptor, kernel concat), and row-parallel (one
//! partial matmul per rank, `AllReduce`-sum through the plan's
//! descriptor). Both distributed paths must match the single-device
//! matmul. The shard matmuls run on the real CPU kernels; only the
//! byte movement goes through the reference transport.
//!
//! Run it with:
//! `cargo run -p incin --example tensor_parallel_split --features distributed-reference`

use incin::backend_authoring::HostReadback;
use incin::experimental::distributed::mesh::{
    DeviceIdentity, DeviceMesh, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin::experimental::distributed::{
    CollectiveKind, StreamId, TensorParallelId, TensorParallelPlanBuilder, TwoRankTensorParallel,
};
use incin::prelude::*;
use incin_backends::dist::{
    CollectiveBackend, ReferenceBuffer, ReferenceTransport, ReferenceValues,
};

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

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

fn main() -> Result<()> {
    // Fixed values, no randomness.
    let x_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1 - 0.5).collect();
    let w_data: Vec<f32> = (0..32).map(|i| i as f32 * 0.07 - 0.5).collect();
    let x = Tensor::<Dyn, Backend>::from_slice(&x_data, vec![2, 8])?;
    let w = Tensor::<Dyn, Backend>::from_slice(&w_data, vec![8, 4])?;

    // 1. Whole layer on one device: the reference both ranks reproduce.
    let full = x.matmul(&w)?;
    let expected = read_f32::<Backend, f32>(full.inner());
    println!("single-device [2, 4] output: {expected:?}");

    // 2. Column-parallel: rank r owns output columns [3r, 3r+3).
    // Weight rows split on the host (data prep, not execution).
    let mut w0_data = Vec::with_capacity(16);
    let mut w1_data = Vec::with_capacity(16);
    for row in w_data.chunks_exact(4) {
        w0_data.extend_from_slice(&row[..2]);
        w1_data.extend_from_slice(&row[2..]);
    }
    let plan = {
        let bound = mesh();
        let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
        builder
            .push_column_dyn(
                TensorParallelId::new(11).unwrap(),
                &[2, 4],
                1,
                DTypeId::F32,
                StreamId::new(3),
            )
            .expect("a divisible column plan builds");
        builder.finish().expect("a non-empty TP plan finishes")
    };
    let descriptor = &plan.collective_plan().descriptors()[0];
    assert_eq!(descriptor.kind(), CollectiveKind::AllGather);

    // Each rank's shard matmul, on the real CPU kernel.
    let w0 = Tensor::<Dyn, Backend>::from_slice(&w0_data, vec![8, 2])?;
    let w1 = Tensor::<Dyn, Backend>::from_slice(&w1_data, vec![8, 2])?;
    let y0 = x.matmul(&w0)?;
    let y1 = x.matmul(&w1)?;
    let part0 = read_f32::<Backend, f32>(y0.inner());
    let part1 = read_f32::<Backend, f32>(y1.inner());

    // The boundary collective, under the plan's own descriptor.
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
    assert_eq!(shards[0], shards[1], "all-gather replicates to every rank");

    // Kernel concat rebuilds the row-major [2, 4] rows.
    let t0 = Tensor::<Dyn, Backend>::from_slice(
        &part0.iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        vec![2, 2],
    )?;
    let t1 = Tensor::<Dyn, Backend>::from_slice(
        &part1.iter().map(|v| *v as f32).collect::<Vec<f32>>(),
        vec![2, 2],
    )?;
    let joined = t0.concat(&t1, 1isize)?;
    let found = read_f32::<Backend, f32>(joined.inner());
    let drift = max_abs_diff(&found, &expected);
    println!("column-parallel (all-gather) max drift vs whole: {drift:e}");
    assert!(drift < 1e-5, "column shards must rebuild the whole output");

    // 3. Row-parallel: rank r owns input columns [4r, 4r+4), i.e. the
    // first/last four rows of the row-major [8, 4] weight. The partial
    // products add through an all-reduce sum.
    let mut xa_data = Vec::with_capacity(8);
    let mut xb_data = Vec::with_capacity(8);
    for row in x_data.chunks_exact(8) {
        xa_data.extend_from_slice(&row[..4]);
        xb_data.extend_from_slice(&row[4..]);
    }
    let wa_data = w_data[..16].to_vec();
    let wb_data = w_data[16..].to_vec();
    let row_plan = {
        let bound = mesh();
        let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
        builder
            .push_row_dyn(
                TensorParallelId::new(12).unwrap(),
                8,
                8,
                DTypeId::F32,
                StreamId::new(4),
            )
            .expect("a divisible row plan builds");
        builder.finish().expect("a non-empty TP plan finishes")
    };
    let row_descriptor = &row_plan.collective_plan().descriptors()[0];
    assert!(
        matches!(row_descriptor.kind(), CollectiveKind::AllReduce(_)),
        "row parallelism reduces partial sums"
    );

    let xa = Tensor::<Dyn, Backend>::from_slice(&xa_data, vec![2, 4])?;
    let xb = Tensor::<Dyn, Backend>::from_slice(&xb_data, vec![2, 4])?;
    let wa = Tensor::<Dyn, Backend>::from_slice(&wa_data, vec![4, 4])?;
    let wb = Tensor::<Dyn, Backend>::from_slice(&wb_data, vec![4, 4])?;
    let pa = xa.matmul(&wa)?;
    let pb = xb.matmul(&wb)?;
    let va = read_f32::<Backend, f32>(pa.inner());
    let vb = read_f32::<Backend, f32>(pb.inner());

    let reduced = ReferenceTransport
        .all_reduce(
            row_descriptor.group(),
            &[
                f32_buffer(va.iter().map(|v| *v as f32).collect()),
                f32_buffer(vb.iter().map(|v| *v as f32).collect()),
            ],
            incin_core::exec::ReduceOp::Sum,
            row_descriptor.stream(),
        )
        .expect("the reference all-reduce runs");
    let (buffers, _) = reduced.into_parts();
    assert_eq!(buffers.len(), 2, "one output buffer per rank");
    let ReferenceValues::F32(sum) = buffers[0].values() else {
        panic!("the plan validated f32; the transport must return f32");
    };
    let sum: Vec<f64> = sum.iter().map(|&v| f64::from(v)).collect();
    let drift = max_abs_diff(&sum, &expected);
    println!("row-parallel (all-reduce) max drift vs whole: {drift:e}");
    assert!(drift < 1e-4, "row partials must sum to the whole output");

    println!("PASS: column (gather) and row (reduce) ranks both match the whole layer");
    Ok(())
}
