//! CPU-runnable numerical evidence for TP=2 linear and attention semantics.

#![cfg(all(feature = "distributed-reference", feature = "cpu"))]

use incin_backends::dist::{
    CollectiveBackend, ReferenceBuffer, ReferenceTransport, ReferenceValues,
};
use incin_core::dist::mesh::{
    DeviceIdentity, DeviceMesh, LinkClass, ProcessLayout, TopologyProbe, TransportVersion,
};
use incin_core::dist::{
    StreamId, TensorParallelId, TensorParallelPlanBuilder, TwoRankTensorParallel,
};
use incin_core::exec::ReduceOp;
use incin_core::prelude::{DTypeId, DeviceId, Dyn};
use incin_core::typenum::{U0, U2, U4, U6};

struct ReferenceTp2;

impl TopologyProbe for ReferenceTp2 {
    fn identify(&self, device: DeviceId) -> Option<DeviceIdentity> {
        (device.kind() == incin_core::prelude::DeviceKind::Cuda && device.ordinal() < 2).then(
            || {
                DeviceIdentity::new(
                    device,
                    format!("reference-tp2-{}", device.ordinal()),
                    "sm_reference".to_string(),
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

#[test]
fn static_column_and_row_linear_match_one_device() {
    let bound = mesh();
    let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
    builder
        .push_column_static::<f32, U0, U6>(TensorParallelId::new(1).unwrap(), 1, StreamId::new(1))
        .unwrap();
    builder
        .push_row_static::<f32, U4>(TensorParallelId::new(2).unwrap(), 3, StreamId::new(2))
        .unwrap();
    let plan = builder.finish().unwrap();

    let input = [1.0, 2.0, 3.0, 4.0];
    let weights = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.0],
    ];
    let full_column = matvec(&weights, &input);
    let rank_zero_column = matvec(&weights[..3], &input);
    let rank_one_column = matvec(&weights[3..], &input);
    let column = &plan.collective_plan().descriptors()[0];
    let gathered = ReferenceTransport
        .all_gather(
            column.group(),
            &[f32_buffer(rank_zero_column), f32_buffer(rank_one_column)],
            column.stream(),
        )
        .unwrap();
    assert_eq!(
        gathered.buffers()[0].values(),
        &ReferenceValues::F32(full_column)
    );

    let row_weights = [
        [1.0, 1.0, 1.0, 1.0],
        [1.0, 2.0, 3.0, 4.0],
        [-1.0, 1.0, -1.0, 1.0],
    ];
    let full_row = matvec(&row_weights, &input);
    let rank_zero_row = row_partial(&row_weights, &input, 0..2);
    let rank_one_row = row_partial(&row_weights, &input, 2..4);
    let row = &plan.collective_plan().descriptors()[1];
    let reduced = ReferenceTransport
        .all_reduce(
            row.group(),
            &[f32_buffer(rank_zero_row), f32_buffer(rank_one_row)],
            ReduceOp::Sum,
            row.stream(),
        )
        .unwrap();
    assert_eq!(
        reduced.buffers()[0].values(),
        &ReferenceValues::F32(full_row)
    );
}

#[test]
fn static_head_parallel_attention_matches_one_device() {
    let bound = mesh();
    let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
    builder
        .push_attention_static::<f32, U0, U2>(
            TensorParallelId::new(3).unwrap(),
            2,
            StreamId::new(3),
        )
        .unwrap();
    let plan = builder.finish().unwrap();

    // Two heads, sequence length two, head width one. Each rank computes one
    // independent head, then all-gather restores the head-major global result.
    let heads = [
        attention_head([1.0, 2.0], [1.0, -1.0], [3.0, 5.0]),
        attention_head([0.5, -0.5], [2.0, 1.0], [7.0, 11.0]),
    ];
    let expected: Vec<f32> = heads.into_iter().flatten().collect();
    let descriptor = &plan.collective_plan().descriptors()[0];
    let gathered = ReferenceTransport
        .all_gather(
            descriptor.group(),
            &[f32_buffer(heads[0].to_vec()), f32_buffer(heads[1].to_vec())],
            descriptor.stream(),
        )
        .unwrap();
    let ReferenceValues::F32(found) = gathered.buffers()[0].values() else {
        panic!("f32 attention plan produced another dtype");
    };
    assert_close(found, &expected);
}

#[test]
fn dyn_f64_uses_the_same_three_collective_paths() {
    let bound = mesh();
    let mut builder = TensorParallelPlanBuilder::new(&bound, 0);
    builder
        .push_column_dyn(
            TensorParallelId::new(10).unwrap(),
            &[4],
            0,
            DTypeId::F64,
            StreamId::new(4),
        )
        .unwrap();
    builder
        .push_row_dyn(
            TensorParallelId::new(11).unwrap(),
            4,
            2,
            DTypeId::F64,
            StreamId::new(5),
        )
        .unwrap();
    builder
        .push_attention_dyn(
            TensorParallelId::new(12).unwrap(),
            &[2, 2],
            0,
            DTypeId::F64,
            StreamId::new(6),
        )
        .unwrap();
    let plan = builder.finish().unwrap();
    let descriptors = plan.collective_plan().descriptors();

    let dyn_buffer = |values| {
        ReferenceBuffer::<Dyn>::try_new(ReferenceValues::F64(values), DTypeId::F64).unwrap()
    };
    let column = ReferenceTransport
        .all_gather(
            descriptors[0].group(),
            &[dyn_buffer(vec![1.0, 2.0]), dyn_buffer(vec![3.0, 4.0])],
            descriptors[0].stream(),
        )
        .unwrap();
    assert_eq!(
        column.buffers()[0].values(),
        &ReferenceValues::F64(vec![1.0, 2.0, 3.0, 4.0])
    );

    let row = ReferenceTransport
        .all_reduce(
            descriptors[1].group(),
            &[dyn_buffer(vec![1.0, 10.0]), dyn_buffer(vec![2.0, 20.0])],
            ReduceOp::Sum,
            descriptors[1].stream(),
        )
        .unwrap();
    assert_eq!(
        row.buffers()[0].values(),
        &ReferenceValues::F64(vec![3.0, 30.0])
    );

    let attention = ReferenceTransport
        .all_gather(
            descriptors[2].group(),
            &[dyn_buffer(vec![5.0, 6.0]), dyn_buffer(vec![7.0, 8.0])],
            descriptors[2].stream(),
        )
        .unwrap();
    assert_eq!(
        attention.buffers()[0].values(),
        &ReferenceValues::F64(vec![5.0, 6.0, 7.0, 8.0])
    );
}

fn matvec(weights: &[[f32; 4]], input: &[f32; 4]) -> Vec<f32> {
    weights
        .iter()
        .map(|row| row.iter().zip(input).map(|(w, x)| w * x).sum())
        .collect()
}

fn row_partial(
    weights: &[[f32; 4]; 3],
    input: &[f32; 4],
    columns: core::ops::Range<usize>,
) -> Vec<f32> {
    weights
        .iter()
        .map(|row| {
            columns
                .clone()
                .map(|column| row[column] * input[column])
                .sum()
        })
        .collect()
}

fn attention_head(query: [f32; 2], key: [f32; 2], value: [f32; 2]) -> [f32; 2] {
    let mut output = [0.0; 2];
    for query_index in 0..2 {
        let logits = [query[query_index] * key[0], query[query_index] * key[1]];
        let maximum = logits[0].max(logits[1]);
        let exponentials = [(logits[0] - maximum).exp(), (logits[1] - maximum).exp()];
        let denominator = exponentials[0] + exponentials[1];
        output[query_index] =
            (exponentials[0] * value[0] + exponentials[1] * value[1]) / denominator;
    }
    output
}

fn assert_close(found: &[f32], expected: &[f32]) {
    assert_eq!(found.len(), expected.len());
    for (&found, &expected) in found.iter().zip(expected) {
        assert!(
            (found - expected).abs() <= 1e-6,
            "found {found}, expected {expected}"
        );
    }
}
