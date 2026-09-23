//! Two-rank data-parallel gradient evidence (`DST-008` / issue #97).
//!
//! Two layers of coverage, split by what they need to run:
//!
//! - **CPU-runnable protocol evidence** (`train` + `cpu`): the synchronizer
//!   interface, element-wise two-rank mean against a scripted peer, and the
//!   equal-shard identity - mean of two shard gradients equals the
//!   full-batch gradient. These prove the read-reduce-write contract without
//!   network hardware.
//! - **Two-host NCCL connectivity** (`distributed-nccl` + `hardware-tests`,
//!   ignored): plan construction and communicator setup against real CUDA
//!   hosts. Hardware-gated; never part of a default `cargo test`.

// ============================================================================
// Hardware-gated: two-host NCCL (DST-008)
// ============================================================================

#[cfg(all(feature = "distributed-nccl", feature = "hardware-tests"))]
mod nccl {
    use incin::experimental::distributed::{
        DataParallelPlanBuilder, DistributedContext, GradientId, NcclTopology, NcclTransport,
        StreamId, TwoRankDataParallel,
    };
    use incin::prelude::*;

    #[test]
    #[ignore = "requires two network-accessible CUDA hosts with NCCL"]
    fn dp2_plan_and_communicator_initialize() {
        let context = DistributedContext::<Dyn, Dyn>::from_env().expect("two-rank rendezvous");
        let rank = context.rank();
        let topology = NcclTopology::discover_context(&context).expect("discover CUDA identities");
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
        let transport = NcclTransport::connect_context(&context, plan.into_collective_plan())
            .expect("initialize DP NCCL communicator");
        assert_eq!(transport.cursor(), 0);
        drop(transport);
        context.shutdown().expect("coordinated DP shutdown");
    }
}

// ============================================================================
// CPU-runnable protocol evidence
// ============================================================================

#[cfg(all(feature = "train", feature = "cpu"))]
mod protocol {
    use incin::backend_authoring::HostReadback;
    use incin::experimental::training::{
        GradientSynchronizer, SyncError, all_reduce_model_gradients,
    };
    use incin::nn::{ParameterVisitor, TrainState, VisitParameters};
    use incin::prelude::*;
    use incin::state::{collect_state, load_state};
    use std::sync::Mutex;

    type Backend = incin::DefaultBackend;
    type Model = SeqTy!(Linear<Dyn, Backend>, ReLU, Linear<Dyn, Backend>);

    fn model() -> Result<Model> {
        Ok(seq![
            Linear::<Dyn, Backend>::build((4, 8))?,
            ReLU,
            Linear::<Dyn, Backend>::build((8, 2))?
        ])
    }

    /// Collects every parameter's gradient as per-tensor `f64` chunks, in
    /// `VisitParameters` order - the same order `all_reduce_model_gradients`
    /// issues its collectives in. Parameters without a gradient are skipped
    /// so peer buffers line up with the synchronizer's call sequence.
    struct GradFlatten<'a> {
        grads: &'a Gradients<Backend>,
        chunks: Vec<Vec<f64>>,
    }

    impl ParameterVisitor<Backend> for GradFlatten<'_> {
        fn visit_param<S, K, Train>(
            &mut self,
            _path: &StatePath,
            param: &incin::nn::param::Param<S, Backend, K, Train>,
        ) -> Result<()>
        where
            S: Shape,
            K: DType,
            Train: TrainState,
        {
            let tensor = param.as_tensor()?;
            if tensor.dtype() != <f32 as ConstDType>::DESCRIPTOR {
                return Ok(());
            }
            if let Some(grad) = self.grads.get(&tensor)? {
                let values = Backend::float_to_vec1::<f32>(grad.inner())?;
                self.chunks.push(values);
            }
            Ok(())
        }
    }

    fn flatten(model: &Model, grads: &Gradients<Backend>) -> Result<Vec<Vec<f64>>> {
        let mut visitor = GradFlatten {
            grads,
            chunks: Vec::new(),
        };
        model.visit_parameters(&StatePath::root(), &mut visitor)?;
        Ok(visitor.chunks)
    }

    /// A scripted second rank: pops one peer buffer per collective call and
    /// replaces this rank's values with the element-wise mean.
    #[derive(Debug)]
    struct TwoRankPeer {
        /// Buffers the "other rank" contributes, FIFO per collective call.
        peers: Mutex<Vec<Vec<f64>>>,
        /// What this rank observed entering each collective, for assertions.
        seen: Mutex<Vec<Vec<f64>>>,
    }

    impl TwoRankPeer {
        fn new(peers: Vec<Vec<f64>>) -> Self {
            Self {
                peers: Mutex::new(peers),
                seen: Mutex::new(Vec::new()),
            }
        }
    }

    impl GradientSynchronizer for TwoRankPeer {
        fn world_size(&self) -> usize {
            2
        }
        fn rank(&self) -> usize {
            0
        }
        fn all_reduce_mean(&self, values: &mut [f64]) -> std::result::Result<(), SyncError> {
            self.seen.lock().expect("seen lock").push(values.to_vec());
            let peer = {
                let mut peers = self.peers.lock().expect("peers lock");
                if peers.is_empty() {
                    return Err(SyncError::Synchronizer {
                        message: "peer buffer exhausted".to_string(),
                    });
                }
                peers.remove(0)
            };
            if peer.len() != values.len() {
                return Err(SyncError::Synchronizer {
                    message: format!(
                        "peer buffer length {} != local {}",
                        peer.len(),
                        values.len()
                    ),
                });
            }
            for (value, peer_value) in values.iter_mut().zip(&peer) {
                *value = (*value + *peer_value) / 2.0;
            }
            Ok(())
        }
    }

    fn assert_close(actual: &[f64], expected: &[f64], tolerance: f64, what: &str) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "{what}: length {} != {}",
            actual.len(),
            expected.len()
        );
        for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
            let limit = tolerance * (1.0 + e.abs());
            assert!(
                (a - e).abs() <= limit,
                "{what}[{index}]: {a} vs {e} (limit {limit})"
            );
        }
    }

    /// The reduction itself: every element must become the mean of the local
    /// value and the scripted peer's value, and the synchronizer must have
    /// seen every tensor this model carries, in traversal order.
    #[test]
    fn an_elementwise_two_rank_mean_matches_the_scripted_peer() -> Result<()> {
        let model = model()?;
        let input = Tensor::<Dyn, Backend>::ones(vec![4, 4])?;
        let target = Tensor::<Dyn, Backend>::zeros(vec![4, 2])?.forget_layout();
        let loss = model.forward(input)?.mse_loss(&target)?;
        let mut grads = loss.backward()?;

        let local = flatten(&model, &grads)?;
        assert!(!local.is_empty(), "backward produced no gradients");
        let peer: Vec<Vec<f64>> = local
            .iter()
            .map(|chunk| chunk.iter().map(|value| value + 100.0).collect())
            .collect();
        let expected: Vec<f64> = local
            .iter()
            .zip(&peer)
            .flat_map(|(chunk, peer_chunk)| {
                chunk
                    .iter()
                    .zip(peer_chunk)
                    .map(|(l, p)| (l + p) / 2.0)
                    .collect::<Vec<f64>>()
            })
            .collect();

        let script = TwoRankPeer::new(peer.clone());
        all_reduce_model_gradients(&model, &mut grads, &script)
            .map_err(|error| incin::Error::Msg(error.to_string()))?;

        let after = flatten(&model, &grads)?;
        let after_flat: Vec<f64> = after.iter().flatten().copied().collect();
        assert_close(&after_flat, &expected, 1e-5, "element-wise two-rank mean");

        let seen = script.seen.lock().expect("seen lock");
        assert_eq!(
            seen.len(),
            local.len(),
            "one collective per gradient-bearing parameter"
        );
        for (index, (observed, chunk)) in seen.iter().zip(&local).enumerate() {
            assert_eq!(
                observed, chunk,
                "collective {index} must receive the local values unchanged"
            );
        }
        Ok(())
    }

    /// The data-parallel identity this whole path exists for: two equal-size
    /// shards, mean-reduced, must produce the full-batch gradient. This is
    /// the arithmetic NCCL would carry - proven here on the host so the
    /// hardware-gated transport only has to reproduce it.
    #[test]
    fn the_mean_of_two_equal_shards_gradients_equals_the_full_batch_gradient() -> Result<()> {
        // Three models with identical weights: full batch, shard A, shard B.
        let model_full = model()?;
        let initial = collect_state::<Backend, _>(&model_full)?;
        let mut model_a = model()?;
        load_state::<Backend, _>(&mut model_a, &initial)?;
        let mut model_b = model()?;
        load_state::<Backend, _>(&mut model_b, &initial)?;

        let x_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect();
        let y_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.05 - 0.2).collect();
        let full_x = Tensor::<Dyn, Backend>::from_slice(&x_data, vec![4, 4])?;
        let full_y = Tensor::<Dyn, Backend>::from_slice(&y_data, vec![4, 2])?;

        let shard_a_x = full_x.clone().try_narrow(0, 0, 2)?.forget_layout();
        let shard_a_y = full_y.clone().try_narrow(0, 0, 2)?.forget_layout();
        let shard_b_x = full_x.clone().try_narrow(0, 2, 2)?.forget_layout();
        let shard_b_y = full_y.clone().try_narrow(0, 2, 2)?.forget_layout();

        let loss_full = model_full.forward(full_x)?.mse_loss(&full_y)?;
        let grads_full = loss_full.backward()?;

        let loss_a = model_a.forward(shard_a_x)?.mse_loss(&shard_a_y)?;
        let mut grads_a = loss_a.backward()?;

        let loss_b = model_b.forward(shard_b_x)?.mse_loss(&shard_b_y)?;
        let grads_b = loss_b.backward()?;

        let expected = flatten(&model_full, &grads_full)?;
        let peer = flatten(&model_b, &grads_b)?;
        assert_eq!(
            expected.len(),
            peer.len(),
            "identical models must produce gradient buffers of the same structure"
        );

        // Rank 0 (shard A) reduces against rank 1 (shard B)'s gradients.
        let script = TwoRankPeer::new(peer);
        all_reduce_model_gradients(&model_a, &mut grads_a, &script)
            .map_err(|error| incin::Error::Msg(error.to_string()))?;

        let reduced = flatten(&model_a, &grads_a)?;
        assert_eq!(reduced.len(), expected.len());
        for (index, (got, want)) in reduced.iter().zip(&expected).enumerate() {
            assert_close(
                got,
                want,
                1e-4,
                &format!("shard mean vs full batch, tensor {index}"),
            );
        }
        Ok(())
    }
}
