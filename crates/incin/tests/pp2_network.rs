//! Two-host hardware evidence for `DST-010`.
//!
//! The same ignored test runs in two processes. Rank zero computes stage-zero
//! activations, rank one receives them and computes stage-one outputs, then
//! sends activation gradients back for rank-zero weight-gradient computation.
//!
//! ```text
//! # host 0
//! INCIN_RANK=0 INCIN_WORLD_SIZE=2 INCIN_RUN_ID=pp-example \
//! INCIN_LOCAL_CUDA_DEVICE=0 INCIN_RENDEZVOUS_ADDR=0.0.0.0:29530 \
//! INCIN_RENDEZVOUS_TIMEOUT_MS=30000 \
//!   cargo test -p incin --features distributed-nccl --test pp2_network \
//!   -- --ignored --exact pp2_static_and_dyn_match_single_device
//!
//! # host 1
//! INCIN_RANK=1 INCIN_WORLD_SIZE=2 INCIN_RUN_ID=pp-example \
//! INCIN_LOCAL_CUDA_DEVICE=0 INCIN_RENDEZVOUS_ADDR=10.0.0.10:29530 \
//! INCIN_RENDEZVOUS_TIMEOUT_MS=30000 \
//!   cargo test -p incin --features distributed-nccl --test pp2_network \
//!   -- --ignored --exact pp2_static_and_dyn_match_single_device
//! ```

#![cfg(feature = "distributed-nccl")]

use incin::cuda::CudaBackendImpl;
use incin::dist::{
    ActivationCheckpoint, DistributedContext, GPipe, NcclTopology, NcclTransport,
    PipelineBoundaryId, PipelinePlanBuilder, PipelineTransfer, StreamId, TwoRankPipeline,
};
use incin::prelude::*;
use incin::typenum::{U0, U2, U4};

type CudaB = CudaBackendImpl<f32, CudaN<U0>>;

const INPUTS: [[f32; 2]; 4] = [[1.0, 2.0], [3.0, 4.0], [-1.0, 2.0], [2.0, -3.0]];
const OUTPUTS: [f32; 4] = [9.0, 17.0, 11.0, -17.0];
const GRAD_W0: [f32; 4] = [10.0, 10.0, -5.0, -5.0];

#[test]
#[ignore = "requires two network-accessible CUDA hosts with NCCL"]
fn pp2_static_and_dyn_match_single_device() {
    let context = DistributedContext::<Dyn, Dyn>::from_env().expect("two-rank rendezvous");
    let rank = context.rank();
    let timeout = context.timeout();

    let topology = NcclTopology::discover_context(&context).expect("discover two CUDA identities");
    let mesh = incin::dist::mesh::DeviceMesh::<TwoRankPipeline>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .expect("bind PP=2 network topology");
    let plan = PipelinePlanBuilder::build_static::<f32, (U2,), U4, GPipe>(
        &mesh,
        rank,
        PipelineBoundaryId::new(301).unwrap(),
        ActivationCheckpoint::Keep,
        StreamId::new(0),
    )
    .expect("build PP=2 GPipe plan");
    let transfers = plan.transfers().to_vec();
    let mut transport = NcclTransport::connect_context(&context, plan.into_collective_plan())
        .expect("initialize PP NCCL communicator");

    let mut outputs = Vec::new();
    let mut grad_w0 = [0.0; 4];
    for semantic in transfers {
        let microbatch = semantic.microbatch();
        let dynamic = microbatch >= 2;
        match (semantic.transfer(), dynamic) {
            (PipelineTransfer::ForwardActivation, false) => {
                let input = static_input(microbatch);
                let payload = if rank == 0 {
                    stage_zero_static(&input)
                } else {
                    static_zeros()
                };
                let (moved, event) = transport
                    .execute_pipeline(
                        PipelineBoundaryId::new(301).unwrap(),
                        PipelineTransfer::ForwardActivation,
                        microbatch,
                        &payload,
                    )
                    .expect("static forward activation transfer");
                event
                    .wait_timeout(timeout)
                    .expect("static forward completion");
                let moved = Tensor::<Dyn, CudaB>::from_raw(moved, vec![2]).unwrap();
                assert_close_slice(&read_f32::<f32>(moved.inner()), &activation(microbatch));
                if rank == 1 {
                    outputs.push((microbatch, stage_one_static(&moved)));
                }
            }
            (PipelineTransfer::ForwardActivation, true) => {
                let input = dyn_input(microbatch);
                let payload = if rank == 0 {
                    stage_zero_dyn(&input)
                } else {
                    dyn_zeros()
                };
                let (moved, event) = transport
                    .execute_pipeline(
                        PipelineBoundaryId::new(301).unwrap(),
                        PipelineTransfer::ForwardActivation,
                        microbatch,
                        &payload,
                    )
                    .expect("Dyn forward activation transfer");
                event.wait_timeout(timeout).expect("Dyn forward completion");
                let moved =
                    Tensor::<Dyn, CudaB, Dyn>::from_raw(moved, (vec![2], DTypeId::F32)).unwrap();
                assert_close_slice(&read_f32::<Dyn>(moved.inner()), &activation(microbatch));
                if rank == 1 {
                    outputs.push((microbatch, stage_one_dyn(&moved)));
                }
            }
            (PipelineTransfer::BackwardGradient, false) => {
                let payload = if rank == 1 {
                    static_gradient()
                } else {
                    static_zeros()
                };
                let (moved, event) = transport
                    .execute_pipeline(
                        PipelineBoundaryId::new(301).unwrap(),
                        PipelineTransfer::BackwardGradient,
                        microbatch,
                        &payload,
                    )
                    .expect("static backward gradient transfer");
                event
                    .wait_timeout(timeout)
                    .expect("static backward completion");
                let moved = Tensor::<Dyn, CudaB>::from_raw(moved, vec![2]).unwrap();
                assert_close_slice(&read_f32::<f32>(moved.inner()), &[2.0, -1.0]);
                if rank == 0 {
                    accumulate(
                        &mut grad_w0,
                        &weight_gradient_static(&moved, &static_input(microbatch)),
                    );
                }
            }
            (PipelineTransfer::BackwardGradient, true) => {
                let payload = if rank == 1 {
                    dyn_gradient()
                } else {
                    dyn_zeros()
                };
                let (moved, event) = transport
                    .execute_pipeline(
                        PipelineBoundaryId::new(301).unwrap(),
                        PipelineTransfer::BackwardGradient,
                        microbatch,
                        &payload,
                    )
                    .expect("Dyn backward gradient transfer");
                event
                    .wait_timeout(timeout)
                    .expect("Dyn backward completion");
                let moved =
                    Tensor::<Dyn, CudaB, Dyn>::from_raw(moved, (vec![2], DTypeId::F32)).unwrap();
                assert_close_slice(&read_f32::<Dyn>(moved.inner()), &[2.0, -1.0]);
                if rank == 0 {
                    accumulate(
                        &mut grad_w0,
                        &weight_gradient_dyn(&moved, &dyn_input(microbatch)),
                    );
                }
            }
        }
    }

    if rank == 1 {
        outputs.sort_by_key(|(microbatch, _)| *microbatch);
        assert_close_slice(
            &outputs
                .into_iter()
                .map(|(_, output)| output)
                .collect::<Vec<_>>(),
            &OUTPUTS,
        );
    } else {
        assert_close_slice(&grad_w0, &GRAD_W0);
    }
    assert_eq!(transport.cursor(), 8);
    drop(transport);
    context.shutdown().expect("coordinated PP shutdown");
}

#[test]
#[ignore = "requires one CUDA device"]
fn local_cuda_static_and_dyn_pipeline_math_match() {
    let static_input = static_input(0);
    let static_activation = stage_zero_static(&static_input);
    assert_close_slice(&read_f32::<f32>(static_activation.inner()), &activation(0));
    assert_close(stage_one_static(&static_activation), OUTPUTS[0]);
    assert_close_slice(
        &weight_gradient_static(&static_gradient(), &static_input),
        &[2.0, 4.0, -1.0, -2.0],
    );

    let dyn_input = dyn_input(0);
    let dyn_activation = stage_zero_dyn(&dyn_input);
    assert_close_slice(&read_f32::<Dyn>(dyn_activation.inner()), &activation(0));
    assert_close(stage_one_dyn(&dyn_activation), OUTPUTS[0]);
    assert_close_slice(
        &weight_gradient_dyn(&dyn_gradient(), &dyn_input),
        &[2.0, 4.0, -1.0, -2.0],
    );
}

fn static_input(microbatch: usize) -> Tensor<Dyn, CudaB> {
    Tensor::from_slice(&INPUTS[microbatch], vec![1, 2]).unwrap()
}

fn dyn_input(microbatch: usize) -> Tensor<Dyn, CudaB, Dyn> {
    dyn_tensor(&INPUTS[microbatch], vec![1, 2])
}

fn static_zeros() -> Tensor<Dyn, CudaB> {
    Tensor::from_slice(&[0.0, 0.0], vec![1, 2]).unwrap()
}

fn dyn_zeros() -> Tensor<Dyn, CudaB, Dyn> {
    dyn_tensor(&[0.0, 0.0], vec![1, 2])
}

fn static_gradient() -> Tensor<Dyn, CudaB> {
    Tensor::from_slice(&[2.0, -1.0], vec![1, 2]).unwrap()
}

fn dyn_gradient() -> Tensor<Dyn, CudaB, Dyn> {
    dyn_tensor(&[2.0, -1.0], vec![1, 2])
}

fn stage_zero_static(input: &Tensor<Dyn, CudaB>) -> Tensor<Dyn, CudaB> {
    let weight = Tensor::<Dyn, CudaB>::from_slice(&[1.0, 2.0, 3.0, -1.0], vec![2, 2]).unwrap();
    input.matmul(&weight.transpose::<0, 1>().unwrap()).unwrap()
}

fn stage_zero_dyn(input: &Tensor<Dyn, CudaB, Dyn>) -> Tensor<Dyn, CudaB, Dyn> {
    let weight = dyn_tensor(&[1.0, 2.0, 3.0, -1.0], vec![2, 2]);
    input.matmul(&weight.transpose::<0, 1>().unwrap()).unwrap()
}

fn stage_one_static(activation: &Tensor<Dyn, CudaB>) -> f32 {
    let weight = Tensor::<Dyn, CudaB>::from_slice(&[2.0, -1.0], vec![1, 2]).unwrap();
    let output = activation
        .matmul(&weight.transpose::<0, 1>().unwrap())
        .unwrap();
    read_f32::<f32>(output.inner())[0]
}

fn stage_one_dyn(activation: &Tensor<Dyn, CudaB, Dyn>) -> f32 {
    let weight = dyn_tensor(&[2.0, -1.0], vec![1, 2]);
    let output = activation
        .matmul(&weight.transpose::<0, 1>().unwrap())
        .unwrap();
    read_f32::<Dyn>(output.inner())[0]
}

fn weight_gradient_static(gradient: &Tensor<Dyn, CudaB>, input: &Tensor<Dyn, CudaB>) -> Vec<f32> {
    let outer = gradient.transpose::<0, 1>().unwrap().matmul(input).unwrap();
    read_f32::<f32>(outer.inner())
}

fn weight_gradient_dyn(
    gradient: &Tensor<Dyn, CudaB, Dyn>,
    input: &Tensor<Dyn, CudaB, Dyn>,
) -> Vec<f32> {
    let outer = gradient.transpose::<0, 1>().unwrap().matmul(input).unwrap();
    read_f32::<Dyn>(outer.inner())
}

fn dyn_tensor(values: &[f32], shape: Vec<usize>) -> Tensor<Dyn, CudaB, Dyn> {
    Tensor::from_bytes(bytemuck::cast_slice(values), (shape, DTypeId::F32)).unwrap()
}

fn activation(microbatch: usize) -> [f32; 2] {
    let [x0, x1] = INPUTS[microbatch];
    [x0 + 2.0 * x1, 3.0 * x0 - x1]
}

fn accumulate(total: &mut [f32; 4], contribution: &[f32]) {
    for (total, contribution) in total.iter_mut().zip(contribution) {
        *total += contribution;
    }
}

fn read_f32<K: DType>(storage: &<CudaB as Backend>::Storage<K>) -> Vec<f32> {
    let bytes = CudaB::to_bytes::<K>(storage).unwrap();
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn assert_close(found: f32, expected: f32) {
    assert!(
        (found - expected).abs() <= 1e-4,
        "found {found}, expected {expected}"
    );
}

fn assert_close_slice(found: &[f32], expected: &[f32]) {
    assert_eq!(found.len(), expected.len());
    for (&found, &expected) in found.iter().zip(expected) {
        assert_close(found, expected);
    }
}
