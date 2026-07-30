//! Hardware evidence harness for `DST-006`.
//!
//! Run this same ignored test in two processes on two network-accessible CUDA
//! hosts. Rank zero listens; rank one connects. Each process needs only its
//! local CUDA ordinal:
//!
//! ```text
//! # host 0
//! INCIN_DIST_RANK=0 INCIN_DIST_BIND=0.0.0.0:29500 \
//!   cargo test -p incin-backends --features distributed-nccl \
//!   --test nccl_two_rank -- --ignored --exact two_networked_cuda_ranks_static_and_dyn
//!
//! # host 1 (replace the address with host 0's reachable address)
//! INCIN_DIST_RANK=1 INCIN_DIST_ROOT=10.0.0.10:29500 \
//!   cargo test -p incin-backends --features distributed-nccl \
//!   --test nccl_two_rank -- --ignored --exact two_networked_cuda_ranks_static_and_dyn
//! ```

#![cfg(feature = "distributed-nccl")]

use std::net::SocketAddr;
use std::time::Duration;

use cudarc::driver::CudaContext;
use incin_backends::dist::{
    NcclBuffer, NcclTopology, NcclTransport, NcclTransportError, TwoRankBootstrapConfig,
};
use incin_core::dist::mesh::{Data, DeviceMesh, MeshAxis, MeshSpec};
use incin_core::dist::{
    CollectivePlanBuilder, Partial, PlacementKind, PlanError, Replicated, StreamId, Sum,
};
use incin_core::prelude::{DTypeId, DeviceId, Dyn};
use incin_core::typenum::U2;

type Mesh = MeshSpec<Data<U2>>;
type PartialSum = Partial<Mesh, Sum>;
type Replica = Replicated<Mesh>;

#[test]
#[ignore = "requires two network-accessible CUDA hosts with NCCL"]
fn two_networked_cuda_ranks_static_and_dyn() {
    let rank = required_env("INCIN_DIST_RANK")
        .parse::<usize>()
        .expect("INCIN_DIST_RANK must be 0 or 1");
    assert!(rank < 2, "INCIN_DIST_RANK must be 0 or 1");
    let device_ordinal = std::env::var("INCIN_DIST_DEVICE")
        .unwrap_or_else(|_| "0".into())
        .parse::<usize>()
        .expect("INCIN_DIST_DEVICE must be a CUDA ordinal");
    let timeout = Duration::from_secs(30);
    let config = if rank == 0 {
        let bind = std::env::var("INCIN_DIST_BIND").unwrap_or_else(|_| "0.0.0.0:29500".into());
        TwoRankBootstrapConfig::root(parse_address("INCIN_DIST_BIND", &bind), timeout)
    } else {
        let root = required_env("INCIN_DIST_ROOT");
        TwoRankBootstrapConfig::peer(parse_address("INCIN_DIST_ROOT", &root), timeout)
    };

    let topology =
        NcclTopology::discover(config, device_ordinal).expect("discover two CUDA identities");
    let devices = [DeviceId::cuda(0), DeviceId::cuda(1)];
    let mesh = DeviceMesh::<Mesh>::bind(&devices, &topology).expect("bind discovered topology");
    let mut builder = CollectivePlanBuilder::new(&mesh);
    let first = builder
        .push_static::<f32, PartialSum, Replica>(MeshAxis::Data, rank, 2, StreamId::new(0), None)
        .expect("static f32 all-reduce descriptor");
    builder
        .push_dyn(
            MeshAxis::Data,
            rank,
            2,
            DTypeId::F64,
            PlacementKind::Partial {
                reduction: incin_core::exec::ReduceOp::Sum,
            },
            PlacementKind::Replicated,
            StreamId::new(1),
            Some(first),
        )
        .expect("Dyn f64 all-reduce descriptor");
    let plan = builder.finish();

    let mut transport =
        NcclTransport::connect(config, plan, device_ordinal).expect("initialize NCCL communicator");
    let context = CudaContext::new(device_ordinal).expect("local CUDA context");
    let stream = context.default_stream();

    let f32_values = if rank == 0 { [1.0, 2.0] } else { [3.0, 4.0] };
    let f32_bytes = encode_f32(&f32_values);
    let f32_device = stream.clone_htod(&f32_bytes).expect("copy f32 input");
    let f32_input =
        NcclBuffer::<f32>::try_from_device_bytes(f32_device, 2, Default::default()).unwrap();
    let (f32_output, f32_event) = transport.execute(&f32_input).expect("static all-reduce");
    f32_event.wait_timeout(timeout).expect("static completion");
    let f32_host = stream
        .clone_dtoh(f32_output.device_bytes())
        .expect("copy f32 output");
    assert_eq!(decode_f32(&f32_host), vec![4.0, 6.0]);

    let f64_values = if rank == 0 {
        [10.0, 20.0]
    } else {
        [11.0, 21.0]
    };
    let f64_bytes = encode_f64(&f64_values);
    let f64_device = stream.clone_htod(&f64_bytes).expect("copy f64 input");
    let f64_input = NcclBuffer::<Dyn>::try_from_device_bytes(f64_device, 2, DTypeId::F64).unwrap();
    let (f64_output, f64_event) = transport.execute(&f64_input).expect("Dyn all-reduce");
    f64_event.wait_timeout(timeout).expect("Dyn completion");
    let f64_host = stream
        .clone_dtoh(f64_output.device_bytes())
        .expect("copy f64 output");
    assert_eq!(decode_f64(&f64_host), vec![21.0, 41.0]);
    assert_eq!(transport.cursor(), 2);

    // Deliberately disagree on the next plan's message count. Rank zero must
    // reject it before another communicator is created; rank one observes the
    // bounded bootstrap connection close with an error.
    let mut divergent = CollectivePlanBuilder::new(&mesh);
    divergent
        .push_static::<f32, PartialSum, Replica>(
            MeshAxis::Data,
            rank,
            if rank == 0 { 2 } else { 3 },
            StreamId::default(),
            None,
        )
        .unwrap();
    let rejected = NcclTransport::connect(config, divergent.finish(), device_ordinal);
    if rank == 0 {
        assert!(matches!(
            rejected,
            Err(NcclTransportError::Plan(PlanError::PlanHashMismatch {
                rank: 1,
                ..
            }))
        ));
    } else {
        assert!(rejected.is_err());
    }
}

fn required_env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("{name} is required"))
}

fn parse_address(name: &str, value: &str) -> SocketAddr {
    value
        .parse()
        .unwrap_or_else(|_| panic!("{name} must be an IP socket address"))
}

fn encode_f32(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|bytes| f32::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}

fn encode_f64(values: &[f64]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn decode_f64(bytes: &[u8]) -> Vec<f64> {
    bytes
        .chunks_exact(8)
        .map(|bytes| f64::from_ne_bytes(bytes.try_into().unwrap()))
        .collect()
}
