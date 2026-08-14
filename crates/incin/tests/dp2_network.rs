//! Two-host hardware evidence for `DST-008`.
//!
//! Start the same ignored test in two processes. Each host exposes only its
//! local CUDA device; rank identity and the second device arrive over TCP.
//!
//! ```text
//! # host 0
//! INCIN_RANK=0 INCIN_WORLD_SIZE=2 INCIN_RUN_ID=dp-example \
//! INCIN_LOCAL_CUDA_DEVICE=0 INCIN_RENDEZVOUS_ADDR=0.0.0.0:29510 \
//! INCIN_RENDEZVOUS_TIMEOUT_MS=30000 \
//!   cargo test -p incin --features distributed-nccl --test dp2_network \
//!   -- --ignored --exact dp2_static_and_dyn_match_single_device
//!
//! # host 1
//! INCIN_RANK=1 INCIN_WORLD_SIZE=2 INCIN_RUN_ID=dp-example \
//! INCIN_LOCAL_CUDA_DEVICE=0 INCIN_RENDEZVOUS_ADDR=10.0.0.10:29510 \
//! INCIN_RENDEZVOUS_TIMEOUT_MS=30000 \
//!   cargo test -p incin --features distributed-nccl --test dp2_network \
//!   -- --ignored --exact dp2_static_and_dyn_match_single_device
//! ```

#![cfg(feature = "distributed-nccl")]

use std::time::Duration;

use incin::experimental::distributed::{
    DataParallelPlanBuilder, DistributedContext, GradientId, NcclTopology, NcclTransport, StreamId,
    TwoRankDataParallel,
};
use incin::prelude::*;
use incin::typenum::U0;
use incin_backends::cpu::CpuBackendImpl;
use incin_backends::cuda::CudaBackendImpl;

type CudaB = CudaBackendImpl<CudaN<U0>>;
type CpuB = CpuBackendImpl;

#[test]
#[ignore = "requires two network-accessible CUDA hosts with NCCL"]
fn dp2_static_and_dyn_match_single_device() {
    let context = DistributedContext::<Dyn, Dyn>::from_env().expect("two-rank rendezvous");
    let rank = context.rank();
    let timeout = context.timeout();

    let topology = NcclTopology::discover_context(&context).expect("discover two CUDA identities");
    let mesh = incin::experimental::distributed::mesh::DeviceMesh::<TwoRankDataParallel>::bind(
        &[DeviceId::cuda(0), DeviceId::cuda(1)],
        &topology,
    )
    .expect("bind DP=2 network topology");
    let mut builder = DataParallelPlanBuilder::new(&mesh, rank);
    builder
        .push_static::<f32>(GradientId::new(101).unwrap(), 2, StreamId::new(0))
        .expect("static f32 gradient");
    builder
        .push_dyn(
            GradientId::new(202).unwrap(),
            2,
            DTypeId::F32,
            StreamId::new(1),
        )
        .expect("Dyn f32 gradient");
    let plan = builder.finish().expect("non-empty DP plan");
    let mut transport = NcclTransport::connect_context(&context, plan.into_collective_plan())
        .expect("initialize DP NCCL communicator");

    let (reference_loss, reference_gradient, reference_update) = cpu_reference();
    run_static_step(
        rank,
        timeout,
        &mut transport,
        reference_loss[rank],
        &reference_gradient,
        &reference_update,
    );
    run_dyn_step(
        rank,
        timeout,
        &mut transport,
        reference_loss[rank],
        &reference_gradient,
        &reference_update,
    );
    assert_eq!(transport.cursor(), 2);
    drop(transport);
    context.shutdown().expect("coordinated DP shutdown");
}

#[test]
#[ignore = "requires one CUDA device"]
fn local_cuda_static_and_dyn_backward_paths_match() {
    let static_weight = Tensor::<Dyn, CudaB>::from_slice(&[2.0, -1.0], vec![2]).unwrap();
    let static_target = Tensor::<Dyn, CudaB>::from_slice(&[0.0, 1.0], vec![2]).unwrap();
    let static_loss = static_weight.mse_loss(&static_target).unwrap();
    assert_close(read_scalar::<CudaB, f32>(static_loss.inner()), 4.0);
    let static_gradients = static_loss.backward().unwrap();
    let static_gradient =
        CudaB::get_grad::<f32>(static_weight.inner(), static_gradients.as_backend())
            .unwrap()
            .unwrap();
    assert_close_slice(&read_f32::<CudaB, f32>(&static_gradient), &[2.0, -2.0]);

    let args = (vec![2], DTypeId::F32);
    let dyn_weight =
        Tensor::<Dyn, CudaB, Dyn>::from_bytes(bytemuck::cast_slice(&[2.0f32, -1.0]), args.clone())
            .unwrap();
    let dyn_target =
        Tensor::<Dyn, CudaB, Dyn>::from_bytes(bytemuck::cast_slice(&[0.0f32, 1.0]), args).unwrap();
    let dyn_loss = dyn_weight.mse_loss(&dyn_target).unwrap();
    assert_close(read_scalar::<CudaB, Dyn>(dyn_loss.inner()), 4.0);
    let dyn_gradients = dyn_loss.backward().unwrap();
    let dyn_gradient = CudaB::get_grad::<Dyn>(dyn_weight.inner(), dyn_gradients.as_backend())
        .unwrap()
        .unwrap();
    assert_close_slice(&read_f32::<CudaB, Dyn>(&dyn_gradient), &[2.0, -2.0]);
}

fn cpu_reference() -> ([f32; 2], Vec<f32>, Vec<f32>) {
    let weight = Tensor::<Dyn, CpuB>::from_slice(&[2.0, -1.0], vec![2]).unwrap();
    let target0 = Tensor::<Dyn, CpuB>::from_slice(&[0.0, 1.0], vec![2]).unwrap();
    let target1 = Tensor::<Dyn, CpuB>::from_slice(&[1.0, 2.0], vec![2]).unwrap();
    let loss0 = weight.mse_loss(&target0).unwrap();
    let loss1 = weight.mse_loss(&target1).unwrap();
    let full_loss = loss0.add(&loss1).unwrap().mul_scalar(0.5).unwrap();
    let gradients = full_loss.backward().unwrap();
    let gradient = CpuB::get_grad::<f32>(weight.inner(), gradients.as_backend())
        .unwrap()
        .unwrap();
    let update = CpuB::sub::<f32>(
        weight.inner(),
        &CpuB::mul_scalar_float::<f32>(&gradient, 0.1).unwrap(),
    )
    .unwrap();
    (
        [
            read_scalar::<CpuB, f32>(loss0.inner()),
            read_scalar::<CpuB, f32>(loss1.inner()),
        ],
        read_f32::<CpuB, f32>(&gradient),
        read_f32::<CpuB, f32>(&update),
    )
}

fn run_static_step(
    rank: usize,
    timeout: Duration,
    transport: &mut NcclTransport,
    expected_local_loss: f32,
    expected_gradient: &[f32],
    expected_update: &[f32],
) {
    let weight = Tensor::<Dyn, CudaB>::from_slice(&[2.0, -1.0], vec![2]).unwrap();
    let target_values = if rank == 0 { [0.0, 1.0] } else { [1.0, 2.0] };
    let target = Tensor::<Dyn, CudaB>::from_slice(&target_values, vec![2]).unwrap();
    let loss = weight.mse_loss(&target).unwrap();
    assert_close(read_scalar::<CudaB, f32>(loss.inner()), expected_local_loss);
    let mut gradients = loss.backward().unwrap();
    let event = transport
        .synchronize_gradient(
            GradientId::new(101).unwrap(),
            &weight,
            gradients.as_backend_mut(),
        )
        .expect("static gradient mean all-reduce");
    event.wait_timeout(timeout).expect("static DP completion");
    assert_synchronized_update::<f32>(
        &weight,
        gradients.as_backend(),
        expected_gradient,
        expected_update,
    );
}

fn run_dyn_step(
    rank: usize,
    timeout: Duration,
    transport: &mut NcclTransport,
    expected_local_loss: f32,
    expected_gradient: &[f32],
    expected_update: &[f32],
) {
    let args = (vec![2], DTypeId::F32);
    let weight =
        Tensor::<Dyn, CudaB, Dyn>::from_bytes(bytemuck::cast_slice(&[2.0f32, -1.0]), args.clone())
            .unwrap();
    let target_values = if rank == 0 { [0.0, 1.0] } else { [1.0, 2.0] };
    let target =
        Tensor::<Dyn, CudaB, Dyn>::from_bytes(bytemuck::cast_slice(&target_values), args).unwrap();
    let loss = weight.mse_loss(&target).unwrap();
    assert_close(read_scalar::<CudaB, Dyn>(loss.inner()), expected_local_loss);
    let mut gradients = loss.backward().unwrap();
    let event = transport
        .synchronize_gradient(
            GradientId::new(202).unwrap(),
            &weight,
            gradients.as_backend_mut(),
        )
        .expect("Dyn gradient mean all-reduce");
    event.wait_timeout(timeout).expect("Dyn DP completion");
    assert_synchronized_update::<Dyn>(
        &weight,
        gradients.as_backend(),
        expected_gradient,
        expected_update,
    );
}

fn assert_synchronized_update<K: DType>(
    weight: &Tensor<Dyn, CudaB, K>,
    gradients: &<CudaB as Backend>::Grads,
    expected_gradient: &[f32],
    expected_update: &[f32],
) {
    let gradient = CudaB::get_grad::<K>(weight.inner(), gradients)
        .unwrap()
        .expect("parameter gradient");
    assert_close_slice(&read_f32::<CudaB, K>(&gradient), expected_gradient);
    let update = CudaB::sub::<K>(
        weight.inner(),
        &CudaB::mul_scalar_float::<K>(&gradient, 0.1).unwrap(),
    )
    .unwrap();
    assert_close_slice(&read_f32::<CudaB, K>(&update), expected_update);
}

fn read_f32<B: Backend, K: DType>(storage: &B::Storage<K>) -> Vec<f32> {
    let bytes = B::to_bytes::<K>(storage).unwrap();
    bytemuck::cast_slice::<u8, f32>(&bytes).to_vec()
}

fn read_scalar<B: Backend, K: DType>(storage: &B::Storage<K>) -> f32 {
    let values = read_f32::<B, K>(storage);
    assert_eq!(values.len(), 1);
    values[0]
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
