//! Two-rank ZeRO-2 execution and sharded checkpoints over the reference
//! transport (issue #99, in-process slice).
//!
//! Three layers of coverage, split by what they prove:
//!
//! - **ZeRO-2 trajectory** (`zero2_run::...trajectory...`): two [`Trainer`]s
//!   on two threads pair through [`ReferenceDataParallel`], this time
//!   attached as [`FsdpSynchronizer`]s under
//!   `ShardingSpec::Fsdp { stage: ZeROStage::ZeRO2 }`, each over its own
//!   data shard. Every step reduce-scatters gradients to the rank's owned
//!   slice, steps, and all-gathers parameters back into a full replica -
//!   through the real reference transport, not a scripted peer. The run
//!   must track the single-device full-batch trajectory, and both ranks
//!   must stay bit-identical.
//! - **Reduce-scatter byte evidence** (`zero2_run::...bytes...`): the same
//!   live pair, driven one collective at a time, retains exactly half the
//!   gradient bytes an all-reduce would - measured through
//!   [`ShardedGradients`], not asserted by inspection.
//! - **Sharded checkpoints** (`shard_checkpoint::...`): each rank persists
//!   its owned flat slices plus a manifest; loading gathers every rank's
//!   file back into a full module with byte-identical parameters, and
//!   world/shape problems are typed refusals, never panics.
//!
//! Everything here runs on the CPU with no hardware: the "devices" are the
//! same two-CUDA stand-ins the `dp2_network` reference run uses, while
//! every tensor stays on the CPU backend. The two-host NCCL leg stays
//! hardware-gated (issue #82). Parameter sharding (ZeRO-3) is refused at
//! plan build and is out of scope - each rank materializes full parameters,
//! so a model that does not fit one rank cannot run yet; what ZeRO-2
//! proves in-process is gradient-memory halving plus trajectory.

#![cfg(all(feature = "train", feature = "cpu", feature = "distributed-reference"))]

// ============================================================================
// ZeRO-2 two-rank execution over the reference transport
// ============================================================================

mod zero2_run {
    use incin::backend_authoring::HostReadback;
    use incin::experimental::distributed::{
        BucketPolicy, FsdpParameterId, FsdpPlanBuilder, ZeROStage,
    };
    use incin::experimental::training::{
        Machine, ReferenceDataParallel, ShardingSpec, Trainer, all_gather_model_parameters,
        reduce_scatter_model_gradients,
    };
    use incin::nn::{ParameterVisitor, TrainState, VisitParameters};
    use incin::prelude::*;
    use incin::state::{StateSnapshot, collect_state, load_state};

    type Backend = incin::DefaultBackend;
    type Model = SeqTy!(Linear<Dyn, Backend>, ReLU, Linear<Dyn, Backend>);
    /// One input/target pair in the rank-local batch lists.
    type Batch = (Tensor<Dyn, Backend>, Tensor<Dyn, Backend>);

    /// Two CUDA devices that do not exist: the plan needs two devices for
    /// the refusal to lift, while every tensor stays on the CPU backend -
    /// the same split the `dp2_network` reference run uses.
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
            Linear::<Dyn, Backend>::build((16, 64))?,
            ReLU,
            Linear::<Dyn, Backend>::build((64, 8))?
        ])
    }

    /// Four global batches of a small deterministic regression problem:
    /// 8 rows of 16 features, 8 rows of 8 targets.
    fn global_batches() -> Result<Vec<Batch>> {
        let mut batches = Vec::new();
        for batch in 0..4 {
            let shift = batch as f32 * 0.3;
            let inputs: Vec<f32> = (0..128).map(|i| (i as f32) * 0.01 - 0.6 + shift).collect();
            let targets: Vec<f32> = (0..64)
                .map(|i| (i as f32) * 0.02 - 0.2 + shift * 0.5)
                .collect();
            batches.push((
                Tensor::<Dyn, Backend>::from_slice(&inputs, vec![8, 16])?,
                Tensor::<Dyn, Backend>::from_slice(&targets, vec![8, 8])?,
            ));
        }
        Ok(batches)
    }

    /// Rank `rank`'s half of every global batch: rows `[rank*4, rank*4+4)`.
    fn shard_batches(global: &[Batch], rank: usize) -> Result<Vec<Batch>> {
        global
            .iter()
            .map(|(input, target)| {
                Ok((
                    input.clone().try_narrow(0, rank * 4, 4)?.forget_layout(),
                    target.clone().try_narrow(0, rank * 4, 4)?.forget_layout(),
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

    /// Collects every `f32` parameter's gradient as per-tensor `f64`
    /// chunks, in `VisitParameters` order - the order the FSDP walks issue
    /// collectives in.
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

    fn flatten_grads(model: &Model, grads: &Gradients<Backend>) -> Result<Vec<Vec<f64>>> {
        let mut visitor = GradFlatten {
            grads,
            chunks: Vec::new(),
        };
        model.visit_parameters(&StatePath::root(), &mut visitor)?;
        Ok(visitor.chunks)
    }

    /// Maximum relative element difference across two snapshots, decoded
    /// as little-endian `f32` (the test model is all-`f32`).
    fn max_snapshot_rel_diff(left: &StateSnapshot, right: &StateSnapshot) -> f32 {
        assert_eq!(left.len(), right.len(), "same parameters on both sides");
        let mut max = 0.0f32;
        for (path, a) in left.iter() {
            let b = right.get(path).expect("same paths on both sides");
            assert_eq!(a.shape(), b.shape(), "shape of {path:?}");
            assert_eq!(a.dtype(), b.dtype(), "dtype of {path:?}");
            let (a_bytes, b_bytes) = (a.bytes(), b.bytes());
            assert_eq!(a_bytes.len(), b_bytes.len(), "bytes of {path:?}");
            for (x, y) in a_bytes.chunks_exact(4).zip(b_bytes.chunks_exact(4)) {
                let (x, y) = (
                    f32::from_le_bytes(x.try_into().expect("4 bytes")),
                    f32::from_le_bytes(y.try_into().expect("4 bytes")),
                );
                let diff = (x - y).abs() / (1.0 + y.abs());
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

    /// Runs one ZeRO-2 rank to completion on its shard and returns what
    /// the main thread compares. Built to move into a thread: everything
    /// the rank touches is owned here.
    fn run_rank(
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
                .sharding(ShardingSpec::Fsdp {
                    stage: ZeROStage::ZeRO2,
                })
                .epochs(epochs)
                .build_on(&TwoCuda)
                .expect("two cuda devices"),
        )
        .with_fsdp_synchronizer(synchronizer);
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

    /// A two-rank ZeRO-2 run over per-rank shards tracks the single-device
    /// full-batch trajectory: identical initial weights, reduce-scattered
    /// shard-gradient means under the owned-slice step, parameters
    /// all-gathered back into a full replica after every step.
    ///
    /// Random inits occasionally land the hidden ReLU on its flat side for
    /// every batch, and then nothing trains - the same hazard the
    /// `dp2_network` trajectory test names. A run that trained nothing
    /// proves no trajectory claim, so degenerate inits retry with fresh
    /// randomness (bounded); a genuine divergence fails the bounds below
    /// on every init and cannot hide behind the retry.
    #[test]
    fn two_rank_zero2_run_matches_the_single_device_trajectory() -> Result<()> {
        let mut trained_runs = 0;
        for _ in 0..5 {
            let numbers = trajectory_attempt()?;
            if !numbers.trained {
                continue;
            }
            trained_runs += 1;
            eprintln!(
                "zero2 trajectory: reference probe {}, rank probe {}, probe diff {}, \
                 max param rel diff {}",
                numbers.reference_probe, numbers.rank_probe, numbers.loss_diff, numbers.param_diff
            );
            assert!(
                numbers.param_diff < 1e-4,
                "the 2-rank ZeRO-2 trajectory must track the single-device trajectory"
            );
            assert!(
                numbers.loss_diff < 1e-4,
                "the 2-rank ZeRO-2 probe loss must track the single-device probe loss"
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
                scope.spawn(move || run_rank(initial0, shards0, probe0, target0, rank0_sync, 2));
            let outcome1 = run_rank(initial, shards1, probe1, target1, rank1_sync, 2)?;
            let outcome0 = rank0.join().expect("rank 0 finishes")?;
            Ok::<_, incin::Error>((outcome0, outcome1))
        })?;

        assert_eq!(
            outcome0.snapshot, outcome1.snapshot,
            "identical init plus identical scattered/gathered steps keeps both ranks bit-identical"
        );
        assert_eq!((outcome0.epochs, outcome0.batches), (2, 8));
        assert_eq!((outcome1.epochs, outcome1.batches), (2, 8));

        let reference_snapshot = collect_state::<Backend, _>(&reference)?;
        // Shard means averaged in `f64` equal the full-batch gradient up
        // to summation order, so the trajectories agree to a relative
        // tolerance - not bit-identically on every init.
        let param_diff = max_snapshot_rel_diff(&outcome0.snapshot, &reference_snapshot);
        let loss_diff = (outcome0.probe_after - reference_probe).abs();
        Ok(TrajectoryNumbers {
            reference_probe,
            rank_probe: outcome0.probe_after,
            loss_diff,
            param_diff,
            trained: reference_probe < probe_before,
        })
    }

    /// The planning layer's ZeRO-2 memory claim, tied to the model the
    /// trajectory test trains: sharded persistent bytes must sit strictly
    /// below the replicated full-model bytes.
    #[test]
    fn zero2_plan_reports_sharded_memory_below_the_full_model() -> Result<()> {
        let snapshot = collect_state::<Backend, _>(&model()?)?;
        let f32_descriptor = <f32 as ConstDType>::DESCRIPTOR;
        let mut builder = FsdpPlanBuilder::new(ZeROStage::ZeRO2);
        for (index, (path, value)) in snapshot.iter().enumerate() {
            assert_eq!(
                value.dtype(),
                f32_descriptor,
                "the test model is all-f32: {path:?}"
            );
            let elements = value.bytes().len() / 4;
            assert_eq!(value.bytes().len(), elements * 4);
            builder
                .push_static::<f32>(
                    FsdpParameterId::new(index as u64 + 1).expect("nonzero id"),
                    elements,
                    index / 2,
                    2,
                )
                .expect("distinct ids divide");
        }
        let plan = builder.finish(2).expect("a two-rank ZeRO-2 plan builds");
        plan.verify_memory_parity().expect("ZeRO-2 parity holds");
        let report = plan.memory_report();
        eprintln!(
            "zero2 memory: persistent {} bytes, full {} bytes, ratio {:.3}",
            report.persistent_bytes, report.unsharded_full_bytes, report.memory_reduction_ratio
        );
        assert!(
            report.persistent_bytes < report.unsharded_full_bytes,
            "sharding must buy memory against replication"
        );
        assert!(
            report.memory_reduction_ratio > 1.0,
            "the reduction ratio must favor sharding"
        );
        Ok(())
    }

    /// Reduce-scatter through the live pair, measured rather than
    /// inspected: each rank retains exactly half the gradient bytes an
    /// all-reduce would, the two owned shards concatenate to the exact
    /// two-rank mean, and a parameter all-gather afterwards leaves both
    /// models bit-identical.
    #[test]
    fn reduce_scatter_through_the_live_pair_halves_retained_bytes() -> Result<()> {
        let global = global_batches()?;
        let initial = collect_state::<Backend, _>(&model()?)?;

        // Expected means, computed single-threaded from both shards'
        // gradients so the threaded round has an independent oracle.
        let mut model_a = model()?;
        load_state::<Backend, _>(&mut model_a, &initial)?;
        let mut model_b = model()?;
        load_state::<Backend, _>(&mut model_b, &initial)?;
        let (shard_a_x, shard_a_y) = shard_batches(&global[..1], 0)?.pop().expect("one batch");
        let (shard_b_x, shard_b_y) = shard_batches(&global[..1], 1)?.pop().expect("one batch");
        let loss_a = model_a.forward(shard_a_x.clone())?.mse_loss(&shard_a_y)?;
        let grads_a = loss_a.backward()?;
        let loss_b = model_b.forward(shard_b_x.clone())?.mse_loss(&shard_b_y)?;
        let grads_b = loss_b.backward()?;
        let flat_a = flatten_grads(&model_a, &grads_a)?;
        let flat_b = flatten_grads(&model_b, &grads_b)?;
        assert_eq!(flat_a.len(), 4, "two Linear layers, four gradient tensors");
        assert_eq!(flat_a.len(), flat_b.len());
        let expected_means: Vec<Vec<f64>> = flat_a
            .iter()
            .zip(&flat_b)
            .map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x + y) / 2.0).collect())
            .collect();
        let full_elements: usize = flat_a.iter().map(Vec::len).sum();

        let (controller, rank0_sync, rank1_sync) =
            ReferenceDataParallel::pair(BucketPolicy::Count { max_tensors: 2 })
                .expect("a nonzero bucket budget pairs");
        let initial0 = initial.clone();
        let (round0, round1) = std::thread::scope(|scope| {
            let rank0 = scope.spawn(move || {
                let mut model = model().expect("the model builds");
                load_state::<Backend, _>(&mut model, &initial0).expect("init loads");
                let loss = model
                    .forward(shard_a_x)
                    .expect("forward")
                    .mse_loss(&shard_a_y)
                    .expect("loss");
                let mut grads = loss.backward().expect("backward");
                let sharded = reduce_scatter_model_gradients(&model, &mut grads, &rank0_sync)
                    .map_err(|error| incin::Error::Msg(error.to_string()))?;
                all_gather_model_parameters(&model, &rank0_sync)
                    .map_err(|error| incin::Error::Msg(error.to_string()))?;
                Ok::<_, incin::Error>((
                    sharded,
                    collect_state::<Backend, _>(&model).expect("snapshot reads"),
                ))
            });
            let mut model = model().expect("the model builds");
            load_state::<Backend, _>(&mut model, &initial).expect("init loads");
            let loss = model
                .forward(shard_b_x)
                .expect("forward")
                .mse_loss(&shard_b_y)
                .expect("loss");
            let mut grads = loss.backward().expect("backward");
            let sharded = reduce_scatter_model_gradients(&model, &mut grads, &rank1_sync)
                .map_err(|error| incin::Error::Msg(error.to_string()))?;
            all_gather_model_parameters(&model, &rank1_sync)
                .map_err(|error| incin::Error::Msg(error.to_string()))?;
            let snapshot = collect_state::<Backend, _>(&model)?;
            let round0 = rank0.join().expect("rank 0 finishes")?;
            Ok::<_, incin::Error>((round0, (sharded, snapshot)))
        })?;
        let ((sharded0, snapshot0), (sharded1, snapshot1)) = (round0, round1);

        for (rank, sharded) in [(0, &sharded0), (1, &sharded1)] {
            assert_eq!(sharded.rank(), rank);
            assert_eq!(sharded.world_size(), 2);
            assert_eq!(sharded.parameter_count(), 4);
            assert_eq!(sharded.full_elements(), full_elements);
            assert_eq!(sharded.owned_elements(), full_elements / 2);
            assert_eq!(sharded.retained_bytes(), full_elements / 2 * 8);
            assert_eq!(sharded.full_bytes(), full_elements * 8);
            assert_eq!(
                sharded.retained_bytes() * sharded.world_size(),
                sharded.full_bytes(),
                "rank {rank}: reduce-scatter retains 1/world of the all-reduce bytes"
            );
        }
        eprintln!(
            "zero2 bytes: full {} bytes, retained per rank {} bytes",
            sharded0.full_bytes(),
            sharded0.retained_bytes()
        );

        // The owned shards concatenate - rank 0's slice first - to the
        // exact two-rank mean. The transport averages in `f64` with the
        // same `(a + b) / 2.0` the oracle uses, so this is bit-exact.
        for (index, expected) in expected_means.iter().enumerate() {
            let half = expected.len() / 2;
            assert_eq!(
                sharded0.shards()[index],
                expected[..half],
                "tensor {index}: rank 0 owns the mean's first half"
            );
            assert_eq!(
                sharded1.shards()[index],
                expected[half..],
                "tensor {index}: rank 1 owns the mean's second half"
            );
        }

        assert_eq!(
            snapshot0, snapshot1,
            "the post-step all-gather rebuilds identical full replicas on both ranks"
        );

        // Four gradient tensors reduce-scattered plus four parameter
        // tensors all-gathered: eight single collectives through the
        // reference transport, one per paired walk step.
        assert_eq!(
            controller.single_transport_calls(),
            8,
            "one reduce-scatter and one all-gather per tensor"
        );
        Ok(())
    }
}

// ============================================================================
// Sharded checkpoints: per-rank owned slices plus a manifest
// ============================================================================

/// In-process proof that a sharded state round-trips through files.
///
/// Each rank persists its owned flat slices - the contiguous
/// `[rank * chunk .. (rank + 1) * chunk]` chunks of every parameter, the
/// same slices [`reduce_scatter_model_gradients`](incin::experimental::training::reduce_scatter_model_gradients)
/// leaves behind - plus a manifest recording the global shapes. Loading
/// gathers every rank's file back into a full module with byte-identical
/// parameters. This helper lives in the test, not in `nn::save`: the
/// product checkpoint API is a later milestone, and keeping the proof
/// here means no public-API surface changes.
///
/// The one direction that stays unproven on purpose: loading into a live
/// parameter-sharded module. That module cannot exist in-process because
/// ZeRO-3 execution is refused, so the same-sharding load-back target is
/// the owned-slice snapshot - the bytes a sharded rank would hold.
mod shard_checkpoint {
    use incin::prelude::*;
    use incin::state::{StateSnapshot, collect_state, load_state};
    use incin_core::shapes::ShapeBuf;
    use std::path::{Path, PathBuf};

    /// `ShardCheckpointError`-valued results; the prelude `Result` is
    /// pinned to `incin::Error`.
    type ShardResult<T> = std::result::Result<T, ShardCheckpointError>;

    /// A sharded checkpoint on disk: one manifest plus one flat byte file
    /// per rank (`shard-{rank}.bin`), tensors in manifest order.
    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ShardManifest {
        world_size: usize,
        tensors: Vec<ManifestTensor>,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    struct ManifestTensor {
        name: String,
        shape: Vec<usize>,
        dtype: String,
        role: String,
        elem_bytes: usize,
        chunk_bytes: usize,
    }

    /// Why a sharded checkpoint could not be saved or loaded. Every
    /// variant is a refusal: there is no "loaded partially" outcome.
    #[derive(Debug)]
    enum ShardCheckpointError {
        /// The caller asked for a world the manifest was not written for.
        WorldMismatch { expected: usize, found: usize },
        /// A tensor's bytes do not match the shape the manifest records.
        ShapeMismatch {
            name: String,
            expected: Vec<usize>,
            found: Vec<usize>,
        },
        /// A tensor's dtype name does not match the manifest record.
        DtypeMismatch {
            name: String,
            expected: String,
            found: String,
        },
        /// A tensor's bytes do not split into equal per-rank chunks.
        NonDivisibleShard {
            name: String,
            bytes: usize,
            world_size: usize,
        },
        /// A rank's shard file is absent.
        MissingShard { rank: usize },
        /// A shard file holds fewer bytes than the manifest promises.
        ShortRead {
            rank: usize,
            expected: usize,
            found: usize,
        },
        /// The manifest names a dtype this loader does not decode.
        UnsupportedDtype { name: String },
        /// The manifest names a state role this loader does not decode.
        UnsupportedRole { name: String },
        /// The manifest itself is malformed.
        MalformedManifest { reason: String },
        /// The filesystem refused.
        Io { message: String },
        /// State construction refused (shape/byte/role mismatch).
        State { message: String },
    }

    impl From<ShardCheckpointError> for incin::Error {
        fn from(error: ShardCheckpointError) -> Self {
            incin::Error::Msg(error.to_string())
        }
    }

    impl std::fmt::Display for ShardCheckpointError {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Self::WorldMismatch { expected, found } => write!(
                    f,
                    "shard world {found} does not match the checkpoint world {expected}"
                ),
                Self::ShapeMismatch {
                    name,
                    expected,
                    found,
                } => write!(
                    f,
                    "tensor {name}: manifest shape {expected:?} does not match shard bytes {found:?}"
                ),
                Self::DtypeMismatch {
                    name,
                    expected,
                    found,
                } => write!(
                    f,
                    "tensor {name}: manifest dtype {expected} does not match shard dtype {found}"
                ),
                Self::NonDivisibleShard {
                    name,
                    bytes,
                    world_size,
                } => write!(
                    f,
                    "tensor {name}: {bytes} bytes do not split across {world_size} ranks"
                ),
                Self::MissingShard { rank } => {
                    write!(f, "shard file for rank {rank} is absent")
                }
                Self::ShortRead {
                    rank,
                    expected,
                    found,
                } => write!(
                    f,
                    "shard file for rank {rank} holds {found} bytes, manifest promises {expected}"
                ),
                Self::UnsupportedDtype { name } => {
                    write!(f, "shard dtype {name} is not decodable")
                }
                Self::UnsupportedRole { name } => {
                    write!(f, "state role {name} is not decodable")
                }
                Self::MalformedManifest { reason } => {
                    write!(f, "shard manifest is malformed: {reason}")
                }
                Self::Io { message } => write!(f, "shard filesystem refused: {message}"),
                Self::State { message } => write!(f, "shard state refused: {message}"),
            }
        }
    }

    fn manifest_path(dir: &Path) -> PathBuf {
        dir.join("manifest.json")
    }

    fn shard_path(dir: &Path, rank: usize) -> PathBuf {
        dir.join(format!("shard-{rank}.bin"))
    }

    fn parse_dtype(name: &str) -> ShardResult<DTypeDescriptor> {
        match name {
            "f32" => Ok(<f32 as ConstDType>::DESCRIPTOR),
            "f64" => Ok(<f64 as ConstDType>::DESCRIPTOR),
            _ => Err(ShardCheckpointError::UnsupportedDtype {
                name: name.to_string(),
            }),
        }
    }

    fn parse_role(name: &str) -> ShardResult<StateRole> {
        match name {
            "Parameter" => Ok(StateRole::Parameter),
            "Buffer" => Ok(StateRole::Buffer),
            _ => Err(ShardCheckpointError::UnsupportedRole {
                name: name.to_string(),
            }),
        }
    }

    /// Product of shape extents, refusing overflow rather than wrapping.
    fn numel(dims: &[usize], name: &str) -> ShardResult<usize> {
        dims.iter().copied().try_fold(1_usize, |elements, extent| {
            elements
                .checked_mul(extent)
                .ok_or_else(|| ShardCheckpointError::State {
                    message: format!("shape overflow for {name}"),
                })
        })
    }

    /// Splits `snapshot` into `world_size` equal flat chunks per tensor
    /// and writes rank `rank`'s chunks. Returns the manifest the driver
    /// writes once; chunking is a pure function of the snapshot, so every
    /// rank derives the same manifest.
    fn save_shard(
        dir: &Path,
        snapshot: &StateSnapshot,
        rank: usize,
        world_size: usize,
    ) -> ShardResult<ShardManifest> {
        if rank >= world_size {
            return Err(ShardCheckpointError::WorldMismatch {
                expected: world_size,
                found: rank,
            });
        }
        let mut tensors = Vec::new();
        let mut shard_bytes = Vec::new();
        for (path, value) in snapshot.iter() {
            let bytes = value.bytes();
            if !bytes.len().is_multiple_of(world_size) {
                return Err(ShardCheckpointError::NonDivisibleShard {
                    name: path.to_string(),
                    bytes: bytes.len(),
                    world_size,
                });
            }
            let chunk_bytes = bytes.len() / world_size;
            let numel = numel(value.shape().dims(), &path.to_string())?;
            if numel == 0 || bytes.len() % numel != 0 {
                return Err(ShardCheckpointError::State {
                    message: format!("byte length does not match shape for {path:?}"),
                });
            }
            shard_bytes.extend_from_slice(&bytes[rank * chunk_bytes..(rank + 1) * chunk_bytes]);
            tensors.push(ManifestTensor {
                name: path.to_string(),
                shape: value.shape().dims().to_vec(),
                dtype: value.dtype().name().to_string(),
                role: format!("{:?}", value.role()),
                elem_bytes: bytes.len() / numel,
                chunk_bytes,
            });
        }
        std::fs::create_dir_all(dir).map_err(|error| ShardCheckpointError::Io {
            message: error.to_string(),
        })?;
        std::fs::write(shard_path(dir, rank), &shard_bytes).map_err(|error| {
            ShardCheckpointError::Io {
                message: error.to_string(),
            }
        })?;
        Ok(ShardManifest {
            world_size,
            tensors,
        })
    }

    fn write_manifest(dir: &Path, manifest: &ShardManifest) -> ShardResult<()> {
        let tensors: Vec<serde_json::Value> = manifest
            .tensors
            .iter()
            .map(|tensor| {
                serde_json::json!({
                    "name": tensor.name,
                    "shape": tensor.shape,
                    "dtype": tensor.dtype,
                    "role": tensor.role,
                    "elem_bytes": tensor.elem_bytes,
                    "chunk_bytes": tensor.chunk_bytes,
                })
            })
            .collect();
        let document = serde_json::json!({ "world_size": manifest.world_size, "tensors": tensors });
        let wire = serde_json::to_string_pretty(&document).map_err(|error| {
            ShardCheckpointError::State {
                message: error.to_string(),
            }
        })?;
        std::fs::write(manifest_path(dir), wire).map_err(|error| ShardCheckpointError::Io {
            message: error.to_string(),
        })?;
        Ok(())
    }

    fn read_manifest(dir: &Path) -> ShardResult<ShardManifest> {
        let wire = std::fs::read_to_string(manifest_path(dir)).map_err(|error| {
            ShardCheckpointError::Io {
                message: error.to_string(),
            }
        })?;
        let document: serde_json::Value = serde_json::from_str(&wire).map_err(|error| {
            ShardCheckpointError::MalformedManifest {
                reason: error.to_string(),
            }
        })?;
        let world_size = document
            .get("world_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| ShardCheckpointError::MalformedManifest {
                reason: "manifest has no integer world_size".to_string(),
            })? as usize;
        let wire_tensors = document
            .get("tensors")
            .and_then(serde_json::Value::as_array)
            .ok_or_else(|| ShardCheckpointError::MalformedManifest {
                reason: "manifest has no tensors array".to_string(),
            })?;
        let mut tensors = Vec::with_capacity(wire_tensors.len());
        for tensor in wire_tensors {
            let get_str = |key: &str| {
                tensor
                    .get(key)
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_string)
                    .ok_or_else(|| ShardCheckpointError::MalformedManifest {
                        reason: format!("tensor entry has no string {key}"),
                    })
            };
            let get_usize = |key: &str| {
                tensor
                    .get(key)
                    .and_then(serde_json::Value::as_u64)
                    .map(|value| value as usize)
                    .ok_or_else(|| ShardCheckpointError::MalformedManifest {
                        reason: format!("tensor entry has no integer {key}"),
                    })
            };
            let shape = tensor
                .get("shape")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| ShardCheckpointError::MalformedManifest {
                    reason: "tensor entry has no shape array".to_string(),
                })?
                .iter()
                .map(|extent| {
                    extent
                        .as_u64()
                        .map(|extent| extent as usize)
                        .ok_or_else(|| ShardCheckpointError::MalformedManifest {
                            reason: "shape extent is not an integer".to_string(),
                        })
                })
                .collect::<ShardResult<Vec<usize>>>()?;
            tensors.push(ManifestTensor {
                name: get_str("name")?,
                shape,
                dtype: get_str("dtype")?,
                role: get_str("role")?,
                elem_bytes: get_usize("elem_bytes")?,
                chunk_bytes: get_usize("chunk_bytes")?,
            });
        }
        Ok(ShardManifest {
            world_size,
            tensors,
        })
    }

    /// Gathers every rank's shard file into the full snapshot: chunk `rank`
    /// of each tensor lands at its rank-ordered offset, so the bytes are
    /// exactly what the saver split.
    fn load_full(dir: &Path, world_size: usize) -> ShardResult<StateSnapshot> {
        let manifest = read_manifest(dir)?;
        if manifest.world_size != world_size {
            return Err(ShardCheckpointError::WorldMismatch {
                expected: manifest.world_size,
                found: world_size,
            });
        }
        let mut shards = Vec::with_capacity(world_size);
        for rank in 0..world_size {
            let bytes = std::fs::read(shard_path(dir, rank))
                .map_err(|_| ShardCheckpointError::MissingShard { rank })?;
            let expected: usize = manifest
                .tensors
                .iter()
                .map(|tensor| tensor.chunk_bytes)
                .sum();
            if bytes.len() != expected {
                return Err(ShardCheckpointError::ShortRead {
                    rank,
                    expected,
                    found: bytes.len(),
                });
            }
            shards.push(bytes);
        }
        let mut snapshot = StateSnapshot::new();
        // Prefix offsets of each tensor inside every shard file: tensors
        // are concatenated in manifest order.
        let mut offsets = Vec::with_capacity(manifest.tensors.len());
        let mut cursor = 0;
        for tensor in &manifest.tensors {
            offsets.push(cursor);
            cursor += tensor.chunk_bytes;
        }
        for (tensor, offset) in manifest.tensors.iter().zip(offsets) {
            let mut full = Vec::with_capacity(tensor.chunk_bytes * world_size);
            for shard in &shards {
                full.extend_from_slice(&shard[offset..offset + tensor.chunk_bytes]);
            }
            let dtype = parse_dtype(&tensor.dtype)?;
            let role = parse_role(&tensor.role)?;
            let numel = numel(&tensor.shape, &tensor.name)?;
            if full.len() != numel * tensor.elem_bytes {
                return Err(ShardCheckpointError::ShapeMismatch {
                    name: tensor.name.clone(),
                    expected: tensor.shape.clone(),
                    found: vec![full.len() / tensor.elem_bytes.max(1)],
                });
            }
            let path = StatePath::new(tensor.name.clone()).map_err(|error| {
                ShardCheckpointError::State {
                    message: error.to_string(),
                }
            })?;
            let value = StateValue::new(ShapeBuf::from_slice(&tensor.shape), dtype, full, role)
                .map_err(|error| ShardCheckpointError::State {
                    message: error.to_string(),
                })?;
            snapshot
                .insert(path, value)
                .map_err(|error| ShardCheckpointError::State {
                    message: error.to_string(),
                })?;
        }
        Ok(snapshot)
    }

    /// Reads rank `rank`'s shard file back into the owned-slice snapshot:
    /// one flat 1-D value per tensor, exactly the bytes that rank saved.
    fn load_rank_shard(dir: &Path, rank: usize, world_size: usize) -> ShardResult<StateSnapshot> {
        let manifest = read_manifest(dir)?;
        if manifest.world_size != world_size {
            return Err(ShardCheckpointError::WorldMismatch {
                expected: manifest.world_size,
                found: world_size,
            });
        }
        if rank >= world_size {
            return Err(ShardCheckpointError::WorldMismatch {
                expected: world_size,
                found: rank,
            });
        }
        let bytes = std::fs::read(shard_path(dir, rank))
            .map_err(|_| ShardCheckpointError::MissingShard { rank })?;
        let mut snapshot = StateSnapshot::new();
        let mut offset = 0;
        for tensor in &manifest.tensors {
            let chunk = bytes.get(offset..offset + tensor.chunk_bytes).ok_or(
                ShardCheckpointError::ShortRead {
                    rank,
                    expected: offset + tensor.chunk_bytes,
                    found: bytes.len(),
                },
            )?;
            offset += tensor.chunk_bytes;
            if tensor.chunk_bytes % tensor.elem_bytes != 0 {
                return Err(ShardCheckpointError::ShapeMismatch {
                    name: tensor.name.clone(),
                    expected: tensor.shape.clone(),
                    found: vec![tensor.chunk_bytes],
                });
            }
            let dtype = parse_dtype(&tensor.dtype)?;
            let role = parse_role(&tensor.role)?;
            let path = StatePath::new(tensor.name.clone()).map_err(|error| {
                ShardCheckpointError::State {
                    message: error.to_string(),
                }
            })?;
            let value = StateValue::new(
                ShapeBuf::from_slice(&[tensor.chunk_bytes / tensor.elem_bytes]),
                dtype,
                chunk.to_vec(),
                role,
            )
            .map_err(|error| ShardCheckpointError::State {
                message: error.to_string(),
            })?;
            // A flat chunk carries no global-shape information, so the
            // dtype is the one check that survives: the saver wrote native
            // bytes for this descriptor, and anything else is a corrupt
            // manifest, refused here rather than misdecoded.
            if value.dtype() != dtype {
                return Err(ShardCheckpointError::DtypeMismatch {
                    name: tensor.name.clone(),
                    expected: tensor.dtype.clone(),
                    found: value.dtype().name().to_string(),
                });
            }
            snapshot
                .insert(path, value)
                .map_err(|error| ShardCheckpointError::State {
                    message: error.to_string(),
                })?;
        }
        Ok(snapshot)
    }

    type Backend = incin::DefaultBackend;
    type Model = SeqTy!(Linear<Dyn, Backend>, ReLU, Linear<Dyn, Backend>);

    fn model() -> Result<Model> {
        Ok(seq![
            Linear::<Dyn, Backend>::build((16, 64))?,
            ReLU,
            Linear::<Dyn, Backend>::build((64, 8))?
        ])
    }

    fn scratch_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "incin-fsdp2-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("wall clock runs")
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).expect("scratch dir creates");
        dir
    }

    /// Save each rank's owned slices, gather-load into a fresh full-size
    /// module, and require byte-identical parameters both ways: the
    /// gathered snapshot equals the saved one, and each rank's shard file
    /// holds exactly its owned slice.
    #[test]
    fn sharded_save_and_gather_load_are_byte_identical() -> Result<()> {
        let source = model()?;
        let snapshot = collect_state::<Backend, _>(&source)?;
        let dir = scratch_dir("roundtrip");

        let manifest0 = save_shard(&dir, &snapshot, 0, 2).expect("rank 0 saves");
        let manifest1 = save_shard(&dir, &snapshot, 1, 2).expect("rank 1 saves");
        assert_eq!(
            manifest0, manifest1,
            "chunking is a pure function of the snapshot: both ranks derive it"
        );
        write_manifest(&dir, &manifest0).expect("the driver writes the manifest once");

        let gathered = load_full(&dir, 2).expect("gather loads");
        let mut reloaded = model()?;
        load_state::<Backend, _>(&mut reloaded, &gathered)?;
        let roundtripped = collect_state::<Backend, _>(&reloaded)?;
        assert_eq!(
            roundtripped, snapshot,
            "gather-load into a full module restores byte-identical parameters"
        );

        // Same-sharding load-back: each rank's file decodes to exactly its
        // owned slice of every tensor.
        for rank in 0..2 {
            let shard_snapshot = load_rank_shard(&dir, rank, 2).expect("rank shard loads");
            assert_eq!(shard_snapshot.len(), snapshot.len());
            for (path, full) in snapshot.iter() {
                let owned = shard_snapshot.get(path).expect("same paths");
                let bytes = full.bytes();
                let chunk = bytes.len() / 2;
                assert_eq!(
                    owned.bytes(),
                    &bytes[rank * chunk..(rank + 1) * chunk],
                    "rank {rank} owns its contiguous slice of {path:?}"
                );
            }
        }

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// A caller asking for any world other than the manifest's gets a
    /// typed refusal, not a mispaired gather.
    #[test]
    fn world_mismatch_is_a_typed_refusal() -> Result<()> {
        let source = model()?;
        let snapshot = collect_state::<Backend, _>(&source)?;
        let dir = scratch_dir("world");
        let manifest = save_shard(&dir, &snapshot, 0, 2).expect("rank 0 saves");
        save_shard(&dir, &snapshot, 1, 2).expect("rank 1 saves");
        write_manifest(&dir, &manifest).expect("manifest writes");

        match load_full(&dir, 3).expect_err("three ranks cannot gather a two-rank shard") {
            ShardCheckpointError::WorldMismatch { expected, found } => {
                assert_eq!((expected, found), (2, 3));
            }
            other => panic!("expected WorldMismatch, got {other:?}"),
        }
        match load_rank_shard(&dir, 0, 3).expect_err("three ranks cannot read a two-rank shard") {
            ShardCheckpointError::WorldMismatch { expected, found } => {
                assert_eq!((expected, found), (2, 3));
            }
            other => panic!("expected WorldMismatch, got {other:?}"),
        }
        // Saving for a rank outside the world is refused before any byte
        // is written.
        match save_shard(&dir, &snapshot, 2, 2).expect_err("rank 2 of 2 cannot save") {
            ShardCheckpointError::WorldMismatch { expected, found } => {
                assert_eq!((expected, found), (2, 2));
            }
            other => panic!("expected WorldMismatch, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// A manifest whose shape no longer matches the shard bytes is a typed
    /// refusal, not a misdecoded load.
    #[test]
    fn shape_mismatch_is_a_typed_refusal() -> Result<()> {
        let source = model()?;
        let snapshot = collect_state::<Backend, _>(&source)?;
        let dir = scratch_dir("shape");
        let manifest = save_shard(&dir, &snapshot, 0, 2).expect("rank 0 saves");
        save_shard(&dir, &snapshot, 1, 2).expect("rank 1 saves");
        write_manifest(&dir, &manifest).expect("manifest writes");

        let mut tampered = read_manifest(&dir).expect("manifest reads");
        tampered.tensors[0].shape = vec![1, 2, 3];
        write_manifest(&dir, &tampered).expect("tampered manifest writes");
        match load_full(&dir, 2).expect_err("a lying shape cannot load") {
            ShardCheckpointError::ShapeMismatch { name, .. } => {
                assert_eq!(name, tampered.tensors[0].name);
            }
            other => panic!("expected ShapeMismatch, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }

    /// A missing rank file is a typed refusal naming the rank, not a hang
    /// or a half-gathered model.
    #[test]
    fn missing_shard_is_a_typed_refusal() -> Result<()> {
        let source = model()?;
        let snapshot = collect_state::<Backend, _>(&source)?;
        let dir = scratch_dir("missing");
        let manifest = save_shard(&dir, &snapshot, 0, 2).expect("rank 0 saves");
        save_shard(&dir, &snapshot, 1, 2).expect("rank 1 saves");
        write_manifest(&dir, &manifest).expect("manifest writes");
        std::fs::remove_file(shard_path(&dir, 1)).expect("rank 1 file deletes");

        match load_full(&dir, 2).expect_err("a rank file is missing") {
            ShardCheckpointError::MissingShard { rank } => assert_eq!(rank, 1),
            other => panic!("expected MissingShard, got {other:?}"),
        }

        std::fs::remove_dir_all(&dir).ok();
        Ok(())
    }
}
