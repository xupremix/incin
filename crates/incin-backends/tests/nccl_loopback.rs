//! NCCL two-GPU loopback: one process, one communicator per GPU, one
//! all-reduce across both. This is the transport half of the `DST-006`
//! evidence the two-process `nccl_two_rank` harness needs a homogeneous
//! pair for: it proves the NCCL runtime, the driver and both devices move
//! collectives correctly, without the mesh's homogeneous-architecture
//! policy (which refuses heterogeneous pairs at bind time by design).
//!
//! Needs the NCCL shared library where the dynamic loader finds it.
//! cudarc resolves `libnccl.so` (and versioned variants up to `.so.12`)
//! but not `libnccl.so.2` bare, so a pip install needs one symlink:
//!
//! ```text
//! pip install nvidia-nccl-cu12   # or: apt install libnccl2
//! ln -s <site-packages>/nvidia/nccl/lib/libnccl.so.2 <dir>/libnccl.so
//! LD_LIBRARY_PATH=<dir> cargo test -p incin-backends \
//!   --features distributed-nccl --test nccl_loopback -- --ignored
//! ```
//!
//! Without the library the test fails loudly (`NcclUnavailable` naming
//! the searched names) rather than skipping: reaching an `#[ignore]`d
//! test is an explicit request for the hardware run.

#![cfg(feature = "distributed-nccl")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaStream};
use cudarc::nccl::safe::{Comm, ReduceOp};

#[test]
#[ignore = "requires NCCL plus two CUDA GPUs in one process"]
fn two_gpu_loopback_all_reduce_is_bit_correct_on_both_ranks() {
    let ctx0 = CudaContext::new(0).expect("CUDA ordinal 0 must open");
    let ctx1 = CudaContext::new(1).expect("CUDA ordinal 1 must open");
    let s0: Arc<CudaStream> = ctx0.default_stream();
    let s1: Arc<CudaStream> = ctx1.default_stream();
    let comms =
        Comm::from_devices(vec![s0.clone(), s1.clone()]).expect("one NCCL communicator per GPU");
    assert_eq!(
        comms.iter().map(|c| c.rank()).collect::<Vec<_>>(),
        vec![0, 1],
        "communicator ranks follow device order"
    );

    // Distinct inputs per rank prove cross-device traffic: a loopback
    // that never leaves the GPU would return each rank's own values.
    let d0 = s0.clone_htod(&[1.0f32, 2.0]).expect("rank 0 upload");
    let d1 = s1.clone_htod(&[3.0f32, 4.0]).expect("rank 1 upload");
    let mut r0 = s0.alloc_zeros::<f32>(2).expect("rank 0 output");
    let mut r1 = s1.alloc_zeros::<f32>(2).expect("rank 1 output");
    comms[0]
        .all_reduce(&d0, &mut r0, &ReduceOp::Sum)
        .expect("rank 0 all-reduce");
    comms[1]
        .all_reduce(&d1, &mut r1, &ReduceOp::Sum)
        .expect("rank 1 all-reduce");
    s0.synchronize().expect("rank 0 fence");
    s1.synchronize().expect("rank 1 fence");
    let o0: Vec<f32> = s0.clone_dtoh(&r0).expect("rank 0 readback");
    let o1: Vec<f32> = s1.clone_dtoh(&r1).expect("rank 1 readback");
    assert_eq!(o0, vec![4.0, 6.0], "rank 0 sees the global sum");
    assert_eq!(o1, vec![4.0, 6.0], "rank 1 sees the global sum");
}
