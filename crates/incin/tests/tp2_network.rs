//! Two-host hardware evidence for `DST-009`.
//!
//! The same ignored test runs in two processes. Each host computes its local
//! CUDA column-linear, row-linear, and attention-head shard, then NCCL gathers
//! or sums it according to one agreed TP=2 plan.
//!
//! ```text
//! # host 0
//! INCIN_RANK=0 INCIN_WORLD_SIZE=2 INCIN_RUN_ID=tp-example \
//! INCIN_LOCAL_CUDA_DEVICE=0 INCIN_RENDEZVOUS_ADDR=0.0.0.0:29520 \
//! INCIN_RENDEZVOUS_TIMEOUT_MS=30000 \
//!   cargo test -p incin --features distributed-nccl --test tp2_network \
//!   -- --ignored --exact tp2_static_and_dyn_match_single_device
//!
//! # host 1
//! INCIN_RANK=1 INCIN_WORLD_SIZE=2 INCIN_RUN_ID=tp-example \
//! INCIN_LOCAL_CUDA_DEVICE=0 INCIN_RENDEZVOUS_ADDR=10.0.0.10:29520 \
//! INCIN_RENDEZVOUS_TIMEOUT_MS=30000 \
//!   cargo test -p incin --features distributed-nccl --test tp2_network \
//!   -- --ignored --exact tp2_static_and_dyn_match_single_device
//! ```

#![cfg(feature = "distributed-nccl")]

use std::time::Duration;

use incin::cuda::CudaBackendImpl;
use incin::experimental::distributed::{
    DistributedContext, NcclTopology, NcclTransport, StreamId, TensorParallelCollective,
    TensorParallelId, TensorParallelPlanBuilder, TwoRankTensorParallel,
};
use incin::prelude::*;
use incin::typenum::{U0, U1, U2, U4, U6};

type CudaB = CudaBackendImpl<CudaN<U0>>;

const COLUMN_EXPECTED: [f32; 12] = [
    1.0, 2.0, 3.0, 4.0, 3.0, 7.0, //
    2.0, 3.0, 4.0, 5.0, 5.0, 9.0,
];
const ROW_EXPECTED: [f32; 3] = [10.0, 30.0, 2.0];

#[test]
#[ignore = "requires two network-accessible CUDA hosts with NCCL"]
fn tp2_static_and_dyn_match_single_device() {
    let context = DistributedContext::<Dyn, Dyn>::from_env().expect("two-rank rendezvous");
    let rank = context.rank();
    let timeout = context.timeout();

    let topology = NcclTopology::discover_context(&context).expect("discover two CUDA identities");
    let mesh = incin::experimental::distributed::mesh::DeviceMesh::<TwoRankTensorParallel>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .expect("bind TP=2 network topology");
    let mut builder = TensorParallelPlanBuilder::new(&mesh, rank);
    push_static_plan(&mut builder);
    push_dyn_plan(&mut builder);
    let plan = builder.finish().expect("non-empty TP plan");
    let mut transport = NcclTransport::connect_context(&context, plan.into_collective_plan())
        .expect("initialize TP NCCL communicator");

    run_static(rank, timeout, &mut transport);
    run_dyn(rank, timeout, &mut transport);
    assert_eq!(transport.cursor(), 6);
    drop(transport);
    context.shutdown().expect("coordinated TP shutdown");
}

#[test]
#[ignore = "requires one CUDA device"]
fn local_cuda_static_and_dyn_tp_math_match() {
    let attention_expected = attention_expected();

    let (column, row, attention) = static_local_outputs(0);
    assert_close_slice(
        &read_f32::<f32>(column.inner()),
        &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
    );
    assert_close_slice(&read_f32::<f32>(row.inner()), &[3.0, 5.0, 1.0]);
    assert_close_slice(
        &read_f32::<f32>(attention.inner()),
        &attention_expected[..2],
    );

    let (column, row, attention) = dyn_local_outputs(0);
    assert_close_slice(
        &read_f32::<Dyn>(column.inner()),
        &[1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
    );
    assert_close_slice(&read_f32::<Dyn>(row.inner()), &[3.0, 5.0, 1.0]);
    assert_close_slice(
        &read_f32::<Dyn>(attention.inner()),
        &attention_expected[..2],
    );
}

fn push_static_plan(builder: &mut TensorParallelPlanBuilder<'_>) {
    builder
        .push_column_static::<f32, U1, U6>(TensorParallelId::new(101).unwrap(), 2, StreamId::new(0))
        .unwrap();
    builder
        .push_row_static::<f32, U4>(TensorParallelId::new(102).unwrap(), 3, StreamId::new(1))
        .unwrap();
    builder
        .push_attention_static::<f32, U0, U2>(
            TensorParallelId::new(103).unwrap(),
            2,
            StreamId::new(2),
        )
        .unwrap();
}

fn push_dyn_plan(builder: &mut TensorParallelPlanBuilder<'_>) {
    builder
        .push_column_dyn(
            TensorParallelId::new(201).unwrap(),
            &[2, 6],
            1,
            DTypeId::F32,
            StreamId::new(3),
        )
        .unwrap();
    builder
        .push_row_dyn(
            TensorParallelId::new(202).unwrap(),
            4,
            3,
            DTypeId::F32,
            StreamId::new(4),
        )
        .unwrap();
    builder
        .push_attention_dyn(
            TensorParallelId::new(203).unwrap(),
            &[2, 2],
            0,
            DTypeId::F32,
            StreamId::new(5),
        )
        .unwrap();
}

fn run_static(rank: usize, timeout: Duration, transport: &mut NcclTransport) {
    let (column, row, attention) = static_local_outputs(rank);
    let (column, event) = transport
        .execute_tensor_parallel(
            TensorParallelId::new(101).unwrap(),
            TensorParallelCollective::ColumnOutputGather { tensor_axis: 1 },
            &column,
            &[2, 6],
        )
        .expect("static column output gather");
    event
        .wait_timeout(timeout)
        .expect("column gather completion");
    let column = Tensor::<Dyn, CudaB>::from_raw(column, vec![2, 6]).unwrap();
    assert_close_slice(&read_f32::<f32>(column.inner()), &COLUMN_EXPECTED);

    let (row, event) = transport
        .execute_tensor_parallel(
            TensorParallelId::new(102).unwrap(),
            TensorParallelCollective::RowOutputSum,
            &row,
            &[1, 3],
        )
        .expect("static row output sum");
    event.wait_timeout(timeout).expect("row sum completion");
    let row = Tensor::<Dyn, CudaB>::from_raw(row, vec![1, 3]).unwrap();
    assert_close_slice(&read_f32::<f32>(row.inner()), &ROW_EXPECTED);

    let (attention, event) = transport
        .execute_tensor_parallel(
            TensorParallelId::new(103).unwrap(),
            TensorParallelCollective::AttentionHeadGather { tensor_axis: 0 },
            &attention,
            &[2, 2],
        )
        .expect("static attention-head gather");
    event
        .wait_timeout(timeout)
        .expect("attention gather completion");
    let attention = Tensor::<Dyn, CudaB>::from_raw(attention, vec![2, 2]).unwrap();
    assert_close_slice(&read_f32::<f32>(attention.inner()), &attention_expected());
}

fn run_dyn(rank: usize, timeout: Duration, transport: &mut NcclTransport) {
    let (column, row, attention) = dyn_local_outputs(rank);
    let (column, event) = transport
        .execute_tensor_parallel(
            TensorParallelId::new(201).unwrap(),
            TensorParallelCollective::ColumnOutputGather { tensor_axis: 1 },
            &column,
            &[2, 6],
        )
        .expect("Dyn column output gather");
    event.wait_timeout(timeout).expect("Dyn column completion");
    let column = Tensor::<Dyn, CudaB, Dyn>::from_raw(column, (vec![2, 6], DTypeId::F32)).unwrap();
    assert_close_slice(&read_f32::<Dyn>(column.inner()), &COLUMN_EXPECTED);

    let (row, event) = transport
        .execute_tensor_parallel(
            TensorParallelId::new(202).unwrap(),
            TensorParallelCollective::RowOutputSum,
            &row,
            &[1, 3],
        )
        .expect("Dyn row output sum");
    event.wait_timeout(timeout).expect("Dyn row completion");
    let row = Tensor::<Dyn, CudaB, Dyn>::from_raw(row, (vec![1, 3], DTypeId::F32)).unwrap();
    assert_close_slice(&read_f32::<Dyn>(row.inner()), &ROW_EXPECTED);

    let (attention, event) = transport
        .execute_tensor_parallel(
            TensorParallelId::new(203).unwrap(),
            TensorParallelCollective::AttentionHeadGather { tensor_axis: 0 },
            &attention,
            &[2, 2],
        )
        .expect("Dyn attention-head gather");
    event
        .wait_timeout(timeout)
        .expect("Dyn attention completion");
    let attention =
        Tensor::<Dyn, CudaB, Dyn>::from_raw(attention, (vec![2, 2], DTypeId::F32)).unwrap();
    assert_close_slice(&read_f32::<Dyn>(attention.inner()), &attention_expected());
}

fn static_local_outputs(
    rank: usize,
) -> (Tensor<Dyn, CudaB>, Tensor<Dyn, CudaB>, Tensor<Dyn, CudaB>) {
    let input =
        Tensor::<Dyn, CudaB>::from_slice(&[1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], vec![2, 4])
            .unwrap();
    let column_weight = Tensor::<Dyn, CudaB>::from_slice(column_weight(rank), vec![3, 4]).unwrap();
    let column = input
        .matmul(&column_weight.transpose::<0, 1>().unwrap())
        .unwrap();

    let row_input = Tensor::<Dyn, CudaB>::from_slice(row_input(rank), vec![1, 2]).unwrap();
    let row_weight = Tensor::<Dyn, CudaB>::from_slice(row_weight(rank), vec![3, 2]).unwrap();
    let row = row_input
        .matmul(&row_weight.transpose::<0, 1>().unwrap())
        .unwrap();

    let (query, key, value) = attention_inputs(rank);
    let query = Tensor::<Dyn, CudaB>::from_slice(&query, vec![2, 1]).unwrap();
    let key = Tensor::<Dyn, CudaB>::from_slice(&key, vec![2, 1]).unwrap();
    let value = Tensor::<Dyn, CudaB>::from_slice(&value, vec![2, 1]).unwrap();
    let scores = query
        .matmul(&key.transpose::<0, 1>().unwrap())
        .unwrap()
        .softmax(1)
        .unwrap();
    let attention = scores
        .matmul(&value)
        .unwrap()
        .try_reshape::<Dyn>(vec![1, 2])
        .unwrap();
    (column, row, attention)
}

fn dyn_local_outputs(
    rank: usize,
) -> (
    Tensor<Dyn, CudaB, Dyn>,
    Tensor<Dyn, CudaB, Dyn>,
    Tensor<Dyn, CudaB, Dyn>,
) {
    let dyn_tensor = |values: &[f32], shape: Vec<usize>| {
        Tensor::<Dyn, CudaB, Dyn>::from_bytes(bytemuck::cast_slice(values), (shape, DTypeId::F32))
            .unwrap()
    };
    let input = dyn_tensor(&[1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0], vec![2, 4]);
    let column_weight = dyn_tensor(column_weight(rank), vec![3, 4]);
    let column = input
        .matmul(&column_weight.transpose::<0, 1>().unwrap())
        .unwrap();

    let row_input = dyn_tensor(row_input(rank), vec![1, 2]);
    let row_weight = dyn_tensor(row_weight(rank), vec![3, 2]);
    let row = row_input
        .matmul(&row_weight.transpose::<0, 1>().unwrap())
        .unwrap();

    let (query, key, value) = attention_inputs(rank);
    let query = dyn_tensor(&query, vec![2, 1]);
    let key = dyn_tensor(&key, vec![2, 1]);
    let value = dyn_tensor(&value, vec![2, 1]);
    let scores = query
        .matmul(&key.transpose::<0, 1>().unwrap())
        .unwrap()
        .softmax(1)
        .unwrap();
    let attention = scores
        .matmul(&value)
        .unwrap()
        .try_reshape::<Dyn>(vec![1, 2])
        .unwrap();
    (column, row, attention)
}

fn column_weight(rank: usize) -> &'static [f32] {
    if rank == 0 {
        &[
            1.0, 0.0, 0.0, 0.0, //
            0.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 0.0,
        ]
    } else {
        &[
            0.0, 0.0, 0.0, 1.0, //
            1.0, 1.0, 0.0, 0.0, //
            0.0, 0.0, 1.0, 1.0,
        ]
    }
}

fn row_input(rank: usize) -> &'static [f32] {
    if rank == 0 { &[1.0, 2.0] } else { &[3.0, 4.0] }
}

fn row_weight(rank: usize) -> &'static [f32] {
    if rank == 0 {
        &[
            1.0, 1.0, //
            1.0, 2.0, //
            -1.0, 1.0,
        ]
    } else {
        &[
            1.0, 1.0, //
            3.0, 4.0, //
            -1.0, 1.0,
        ]
    }
}

fn attention_inputs(rank: usize) -> ([f32; 2], [f32; 2], [f32; 2]) {
    if rank == 0 {
        ([1.0, 2.0], [1.0, -1.0], [3.0, 5.0])
    } else {
        ([0.5, -0.5], [2.0, 1.0], [7.0, 11.0])
    }
}

fn attention_expected() -> Vec<f32> {
    [attention_head(0), attention_head(1)]
        .into_iter()
        .flatten()
        .collect()
}

fn attention_head(rank: usize) -> [f32; 2] {
    let (query, key, value) = attention_inputs(rank);
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

fn read_f32<K: DType>(storage: &<CudaB as Backend>::Storage<K>) -> Vec<f32> {
    let bytes = CudaB::to_bytes::<K>(storage).unwrap();
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn assert_close_slice(found: &[f32], expected: &[f32]) {
    assert_eq!(found.len(), expected.len());
    for (&found, &expected) in found.iter().zip(expected) {
        assert!(
            (found - expected).abs() <= 1e-4,
            "found {found}, expected {expected}"
        );
    }
}
