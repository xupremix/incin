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

// ============================================================================
// In-process two-rank execution over the reference transport (issue #97)
// ============================================================================

/// Two-rank data-parallel runs with no hardware: two [`Trainer`]s on two
/// threads pair through [`ReferenceDataParallel`], each over its own data
/// shard, reducing bucketed gradient means through the deterministic
/// reference transport. The NCCL leg stays hardware-gated (above); this
/// module is the milestone-floor proof that the plan → bucket → transport
/// path trains.
#[cfg(all(feature = "train", feature = "cpu", feature = "distributed-reference"))]
mod reference_run {
    use incin::experimental::distributed::BucketPolicy;
    use incin::experimental::training::{Machine, ReferenceDataParallel, TrainError, Trainer};
    use incin::prelude::*;
    use incin::state::{StateSnapshot, collect_state, load_state};
    use std::time::{Duration, Instant};

    type Backend = incin::DefaultBackend;
    type Model = SeqTy!(Linear<Dyn, Backend>, ReLU, Linear<Dyn, Backend>);
    /// One input/target pair in the rank-local batch lists.
    type Batch = (Tensor<Dyn, Backend>, Tensor<Dyn, Backend>);

    /// Two CUDA devices that do not exist: the plan needs two devices for
    /// the refusal to lift, while every tensor stays on the CPU backend -
    /// the same split the `trainer` multi-device tests use.
    struct TwoCuda;

    impl Machine for TwoCuda {
        fn compiled_in(&self, kind: DeviceKind) -> bool {
            kind == DeviceKind::Cuda
        }
        fn has_device(&self, device: DeviceId) -> bool {
            device.kind() == DeviceKind::Cuda && device.ordinal() < 2
        }
    }

    struct CpuOnly;

    impl Machine for CpuOnly {
        fn compiled_in(&self, kind: DeviceKind) -> bool {
            kind == DeviceKind::Cpu
        }
        fn has_device(&self, device: DeviceId) -> bool {
            device == DeviceId::cpu()
        }
    }

    fn model() -> Result<Model> {
        Ok(seq![
            Linear::<Dyn, Backend>::build((4, 8))?,
            ReLU,
            Linear::<Dyn, Backend>::build((8, 2))?
        ])
    }

    /// Four global batches of a small deterministic regression problem.
    fn global_batches() -> Result<Vec<Batch>> {
        let mut batches = Vec::new();
        for batch in 0..4 {
            let shift = batch as f32 * 0.3;
            let inputs: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7 + shift).collect();
            let targets: Vec<f32> = (0..8)
                .map(|i| (i as f32) * 0.05 - 0.2 + shift * 0.5)
                .collect();
            batches.push((
                Tensor::<Dyn, Backend>::from_slice(&inputs, vec![4, 4])?,
                Tensor::<Dyn, Backend>::from_slice(&targets, vec![4, 2])?,
            ));
        }
        Ok(batches)
    }

    /// Rank `rank`'s half of every global batch: rows `[rank*2, rank*2+2)`.
    fn shard_batches(global: &[Batch], rank: usize) -> Result<Vec<Batch>> {
        global
            .iter()
            .map(|(input, target)| {
                Ok((
                    input.clone().try_narrow(0, rank * 2, 2)?.forget_layout(),
                    target.clone().try_narrow(0, rank * 2, 2)?.forget_layout(),
                ))
            })
            .collect()
    }

    fn probe(model: &Model, input: &Tensor<Dyn, Backend>, target: &Tensor<Dyn, Backend>) -> f32 {
        model
            .forward(input.clone())
            .and_then(|out| out.mse_loss(target))
            .and_then(|loss| loss.to_scalar::<f32>())
            .expect("the probe batch evaluates")
    }

    /// Maximum absolute element difference across two snapshots, decoded
    /// as little-endian `f32` (the test model is all-`f32`).
    fn max_snapshot_diff(left: &StateSnapshot, right: &StateSnapshot) -> f32 {
        assert_eq!(left.len(), right.len(), "same parameters on both sides");
        let mut max = 0.0f32;
        for (path, a) in left.iter() {
            let b = right.get(path).expect("same paths on both sides");
            assert_eq!(a.shape(), b.shape(), "shape of {path:?}");
            assert_eq!(a.dtype(), b.dtype(), "dtype of {path:?}");
            let (a_bytes, b_bytes) = (a.bytes(), b.bytes());
            assert_eq!(a_bytes.len(), b_bytes.len(), "bytes of {path:?}");
            for (x, y) in a_bytes.chunks_exact(4).zip(b_bytes.chunks_exact(4)) {
                let diff = (f32::from_le_bytes(x.try_into().expect("4 bytes"))
                    - f32::from_le_bytes(y.try_into().expect("4 bytes")))
                .abs();
                max = max.max(diff);
            }
        }
        max
    }

    #[derive(Debug)]
    struct RankOutcome {
        epochs: usize,
        batches: usize,
        snapshot: StateSnapshot,
        probe_after: f32,
    }

    /// Runs one rank to completion on its shard and returns what the main
    /// thread compares. Built to move into a thread: everything the rank
    /// touches is owned here.
    fn run_rank(
        _rank: usize,
        initial: StateSnapshot,
        shards: Vec<Batch>,
        probe_input: Tensor<Dyn, Backend>,
        probe_target: Tensor<Dyn, Backend>,
        synchronizer: incin::experimental::training::ReferenceRankSynchronizer,
        epochs: usize,
    ) -> Result<RankOutcome> {
        let mut model = model().expect("the model builds");
        load_state::<Backend, _>(&mut model, &initial)?;
        let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
        let trainer = Trainer::new(
            Trainer::plan()
                .devices(DeviceSet::cuda(0..2).expect("two"))
                .epochs(epochs)
                .build_on(&TwoCuda)
                .expect("two cuda devices"),
        )
        .with_synchronizer(synchronizer);
        let outcome = trainer
            .fit(
                &mut model,
                &mut optimizer,
                &shards,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .map_err(|error| incin::Error::Msg(error.to_string()))?;
        Ok(RankOutcome {
            epochs: outcome.epochs,
            batches: outcome.batches,
            snapshot: collect_state::<Backend, _>(&model)?,
            probe_after: probe(&model, &probe_input, &probe_target),
        })
    }

    /// A two-rank bucketed run over per-rank shards reaches the same
    /// trajectory as the single-device full-batch run: identical initial
    /// weights, mean-reduced shard gradients, identical optimizer.
    ///
    /// Random inits occasionally land the hidden ReLU on its flat side for
    /// every batch, and then nothing trains - the same hazard the
    /// `trainer` model comment names. A run that trained nothing proves no
    /// trajectory claim, so degenerate inits retry with fresh randomness
    /// (bounded); a genuine divergence fails the bounds below on every
    /// init and cannot hide behind the retry.
    #[test]
    fn two_rank_reference_run_matches_the_single_device_trajectory() -> Result<()> {
        let mut trained_runs = 0;
        for _ in 0..5 {
            let numbers = trajectory_attempt()?;
            if !numbers.trained {
                continue;
            }
            trained_runs += 1;
            eprintln!(
                "dp2 trajectory: reference probe {}, rank probe {}, probe diff {}, \
                 max param diff {}",
                numbers.reference_probe, numbers.rank_probe, numbers.loss_diff, numbers.param_diff
            );
            assert!(
                numbers.param_diff < 1e-5,
                "the 2-rank trajectory must track the single-device trajectory"
            );
            assert!(
                numbers.loss_diff < 1e-5,
                "the 2-rank probe loss must track the single-device probe loss"
            );
            break;
        }
        assert!(
            trained_runs > 0,
            "five inits in a row trained nothing; the test cannot prove a trajectory claim"
        );
        Ok(())
    }

    struct TrajectoryNumbers {
        reference_probe: f32,
        rank_probe: f32,
        loss_diff: f32,
        param_diff: f32,
        trained: bool,
    }

    /// One full comparison from a fresh random init. Reports whether the
    /// reference run trained at all; divergence bounds are asserted by the
    /// caller so only degenerate inits retry.
    fn trajectory_attempt() -> Result<TrajectoryNumbers> {
        let global = global_batches()?;
        let (probe_input, probe_target) = global[0].clone();

        let mut reference = model().expect("the model builds");
        let initial = collect_state::<Backend, _>(&reference)?;
        let probe_before = probe(&reference, &probe_input, &probe_target);
        let mut ref_optimizer = SGD::<Backend>::from_module(&reference, 0.01)?;
        let reference_outcome = Trainer::new(
            Trainer::plan()
                .epochs(2)
                .build_on(&CpuOnly)
                .expect("the CPU is there"),
        )
        .fit(
            &mut reference,
            &mut ref_optimizer,
            &global,
            |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
        )
        .expect("the single-device run completes");
        let reference_probe = probe(&reference, &probe_input, &probe_target);
        assert_eq!(reference_outcome.epochs, 2);

        // Two tensors per bucket: the model carries four gradient tensors,
        // so every step issues two reference-transport collectives.
        let (_controller, rank0_sync, rank1_sync) =
            ReferenceDataParallel::pair(BucketPolicy::Count { max_tensors: 2 })
                .expect("a nonzero bucket budget pairs");
        let shards0 = shard_batches(&global, 0)?;
        let shards1 = shard_batches(&global, 1)?;
        let initial0 = initial.clone();
        let (probe0, target0) = (probe_input.clone(), probe_target.clone());
        let (probe1, target1) = (probe_input.clone(), probe_target.clone());
        let (outcome0, outcome1) = std::thread::scope(|scope| {
            let rank0 =
                scope.spawn(move || run_rank(0, initial0, shards0, probe0, target0, rank0_sync, 2));
            let outcome1 = run_rank(1, initial, shards1, probe1, target1, rank1_sync, 2)?;
            let outcome0 = rank0.join().expect("rank 0 finishes")?;
            Ok::<_, incin::Error>((outcome0, outcome1))
        })?;

        assert_eq!(
            outcome0.snapshot, outcome1.snapshot,
            "identical init plus identical averaged gradients keeps both ranks bit-identical"
        );
        assert_eq!((outcome0.epochs, outcome0.batches), (2, 8));
        assert_eq!((outcome1.epochs, outcome1.batches), (2, 8));

        let reference_snapshot = collect_state::<Backend, _>(&reference)?;
        // Shard means averaged in `f64` equal the full-batch gradient
        // up to summation order, so the trajectories agree to a few
        // ULPs - not bit-identically on every init.
        let param_diff = max_snapshot_diff(&outcome0.snapshot, &reference_snapshot);
        let loss_diff = (outcome0.probe_after - reference_probe).abs();
        Ok(TrajectoryNumbers {
            reference_probe,
            rank_probe: outcome0.probe_after,
            loss_diff,
            param_diff,
            trained: reference_probe < probe_before,
        })
    }

    /// Bucketing and launch order, measured rather than asserted:
    /// with two tensors per bucket the run issues one collective per
    /// bucket (not per tensor), the step's buckets launch back-to-back in
    /// traversal order, and every tensor launches exactly once per step.
    #[test]
    fn buckets_launch_incrementally_in_traversal_order() -> Result<()> {
        let global = global_batches()?;
        let (probe_input, probe_target) = global[0].clone();
        let initial = collect_state::<Backend, _>(&model()?)?;

        // One batch, one epoch, one collective per tensor: counts the
        // model's gradient tensors.
        let (probe_pair, probe0, probe1) =
            ReferenceDataParallel::pair(BucketPolicy::SingleTensor).expect("pairs");
        let shards0 = shard_batches(&global[..1], 0)?;
        let shards1 = shard_batches(&global[..1], 1)?;
        std::thread::scope(|scope| {
            let p0 = probe0;
            let p1 = probe1;
            let i = initial.clone();
            let (s0, t0) = (probe_input.clone(), probe_target.clone());
            let (s1, t1) = (probe_input.clone(), probe_target.clone());
            let r0 = scope.spawn(move || run_rank(0, i, shards0, s0, t0, p0, 1));
            let _ = run_rank(1, initial.clone(), shards1, s1, t1, p1, 1)?;
            let _ = r0.join().expect("rank 0 finishes")?;
            Ok::<_, incin::Error>(())
        })?;
        let tensors_per_step = probe_pair.transport_calls();
        assert!(
            tensors_per_step >= 2,
            "the model must carry gradient tensors"
        );

        // Same step bucketed two tensors at a time.
        let (controller, rank0_sync, rank1_sync) =
            ReferenceDataParallel::pair(BucketPolicy::Count { max_tensors: 2 }).expect("pairs");
        let shards0 = shard_batches(&global[..1], 0)?;
        let shards1 = shard_batches(&global[..1], 1)?;
        let (probe0, target0) = (probe_input.clone(), probe_target.clone());
        let (probe1, target1) = (probe_input.clone(), probe_target.clone());
        std::thread::scope(|scope| {
            let initial0 = initial.clone();
            let r0 =
                scope.spawn(move || run_rank(0, initial0, shards0, probe0, target0, rank0_sync, 1));
            let _ = run_rank(1, initial, shards1, probe1, target1, rank1_sync, 1)?;
            let _ = r0.join().expect("rank 0 finishes")?;
            Ok::<_, incin::Error>(())
        })?;

        let calls = controller.transport_calls();
        let expected = tensors_per_step.div_ceil(2);
        assert_eq!(
            calls, expected,
            "two tensors per bucket issues ceil(tensors/2) collectives, not one per tensor"
        );
        assert!(
            calls < tensors_per_step,
            "bucketing must save collectives on a multi-tensor model"
        );
        let launches = controller.launches();
        assert_eq!(launches.len(), calls);
        let mut previous_bucket = None;
        let mut launched_tensors = 0;
        for launch in &launches {
            if let Some(previous) = previous_bucket {
                assert_eq!(
                    launch.bucket(),
                    previous + 1,
                    "one step's buckets launch back-to-back in traversal order"
                );
            }
            previous_bucket = Some(launch.bucket());
            launched_tensors += launch.tensors();
            // The step deposits as one batch, so every launch of the step
            // sees both ranks' full batches already deposited.
            assert_eq!(launch.deposits_seen(), 2 * tensors_per_step as u64);
        }
        assert_eq!(
            launched_tensors, tensors_per_step,
            "every gradient tensor launches exactly once per step"
        );
        Ok(())
    }

    /// A rank that drops its handle fail-stops its peer at once: the run
    /// ends as a typed step error on the first batch, without running out
    /// the peer-wait timeout.
    #[test]
    fn a_dead_rank_fail_stops_its_peer_within_a_bound() -> Result<()> {
        let global = global_batches()?;
        let initial = collect_state::<Backend, _>(&model()?)?;

        let (_controller, rank0_sync, rank1_sync) = ReferenceDataParallel::pair_with_timeout(
            BucketPolicy::Count { max_tensors: 2 },
            Duration::from_secs(10),
        )
        .expect("pairs");
        drop(rank1_sync);

        let mut model = model().expect("the model builds");
        load_state::<Backend, _>(&mut model, &initial)?;
        let mut optimizer = SGD::<Backend>::from_module(&model, 0.01)?;
        let trainer = Trainer::new(
            Trainer::plan()
                .devices(DeviceSet::cuda(0..2).expect("two"))
                .epochs(1)
                .build_on(&TwoCuda)
                .expect("two cuda devices"),
        )
        .with_synchronizer(rank0_sync);
        let shards = shard_batches(&global, 0)?;

        let started = Instant::now();
        match trainer
            .fit(
                &mut model,
                &mut optimizer,
                &shards,
                |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
            )
            .expect_err("a dead peer must stop the run")
        {
            TrainError::Step {
                epoch,
                batch,
                ref message,
            } => {
                assert_eq!((epoch, batch), (0, 0), "the very first collective");
                assert!(
                    message.contains("is gone"),
                    "the refusal names the dead peer: {message}"
                );
            }
            other => panic!("expected a step fail-stop, got {other:?}"),
        }
        assert!(
            started.elapsed() < Duration::from_secs(10),
            "the dead peer trips the wait at once instead of running out the timeout"
        );
        Ok(())
    }

    /// A reference-transport refusal fail-stops both ranks: dropping rank
    /// 1's payload makes the transport itself refuse with `InputCount`,
    /// and both fits end as typed step errors instead of hanging.
    #[test]
    fn a_transport_refusal_fail_stops_both_ranks() -> Result<()> {
        let global = global_batches()?;
        let (probe_input, probe_target) = global[0].clone();
        let initial = collect_state::<Backend, _>(&model()?)?;

        let (controller, rank0_sync, rank1_sync) =
            ReferenceDataParallel::pair(BucketPolicy::Count { max_tensors: 2 }).expect("pairs");
        controller.inject_transport_error_once();

        let shards0 = shard_batches(&global, 0)?;
        let shards1 = shard_batches(&global, 1)?;
        std::thread::scope(|scope| {
            let initial0 = initial.clone();
            let rank0 = scope.spawn(move || {
                let mut model = model().expect("the model builds");
                load_state::<Backend, _>(&mut model, &initial0).expect("init loads");
                let mut optimizer =
                    SGD::<Backend>::from_module(&model, 0.01).expect("optimizer builds");
                let trainer = Trainer::new(
                    Trainer::plan()
                        .devices(DeviceSet::cuda(0..2).expect("two"))
                        .epochs(1)
                        .build_on(&TwoCuda)
                        .expect("two cuda devices"),
                )
                .with_synchronizer(rank0_sync);
                trainer
                    .fit(
                        &mut model,
                        &mut optimizer,
                        &shards0,
                        |model, (input, target)| model.forward(input.clone())?.mse_loss(target),
                    )
                    .expect_err("the injected refusal must stop rank 0")
            });
            let rank1_error = run_rank(
                1,
                initial,
                shards1,
                probe_input,
                probe_target,
                rank1_sync,
                1,
            )
            .expect_err("the injected refusal must stop rank 1");
            let rank0_error = rank0.join().expect("rank 0 finishes");
            match rank0_error {
                TrainError::Step { ref message, .. } => assert!(
                    message.contains("refused"),
                    "rank 0 carries the transport refusal: {message}"
                ),
                other => panic!("rank 0: expected a step fail-stop, got {other:?}"),
            }
            let rank1_message = rank1_error.to_string();
            assert!(
                rank1_message.contains("refused"),
                "rank 1 carries the transport refusal: {rank1_message}"
            );
            Ok::<_, incin::Error>(())
        })?;
        assert_eq!(
            controller.transport_calls(),
            1,
            "exactly one collective was attempted before the fail-stop"
        );
        Ok(())
    }
}
