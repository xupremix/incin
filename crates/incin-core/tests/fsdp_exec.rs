//! FSDP execution protocol on the CPU (#99).
//!
//! The ZeRO-1/ZeRO-2 lowering `incin-core` owns: reduce-scatter arithmetic
//! against a scripted two-rank peer, the byte measurement that says
//! reduce-scatter retains `1/N` of what all-reduce retains, divisibility
//! and rank refusals issued before the offending collective, all-gather
//! write-back into the model, ZeRO-1 mask equivalence with ZeRO-2
//! reduce-scatter, and a two-rank ZeRO-2 training trajectory equal to the
//! single-device full-batch reference.
//!
//! Transport (NCCL etc.), ZeRO-3 parameter sharding, and the TP/PP tiers of
//! #99 stay hardware-gated - see the `incin_core::dist::sync` module docs.

#![cfg(feature = "std")]

use std::sync::Mutex;

use incin_backends::cpu::CpuBackendImpl;
use incin_core::autograd::Gradients;
use incin_core::backend_authoring::HostReadback;
use incin_core::dist::sync::{
    FsdpSynchronizer, GradientSynchronizer, SyncError, all_gather_model_parameters,
    all_reduce_model_gradients, mask_gradients_to_owned_shard, reduce_scatter_model_gradients,
};
use incin_core::error::{Error, Result};
use incin_core::nn::param::Param;
use incin_core::nn::{ParameterVisitor, StatePath, TrainState, VisitParameters};
use incin_core::optim::{Optimizer, SGD};
use incin_core::prelude::*;
use incin_core::tensor::dtype::ConstDType;
use incin_core::{SeqTy, seq};

type Backend = CpuBackendImpl;
type Model = SeqTy!(Linear<Dyn, Backend>, ReLU, Linear<Dyn, Backend>);

fn model() -> Result<Model> {
    Ok(seq![
        Linear::<Dyn, Backend>::build((4, 8))?,
        ReLU,
        Linear::<Dyn, Backend>::build((8, 2))?
    ])
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

/// Collects every parameter's gradient as per-tensor `f64` chunks, in
/// `VisitParameters` order - the order the FSDP walks issue collectives in.
struct GradFlatten<'a> {
    grads: &'a Gradients<Backend>,
    chunks: Vec<Vec<f64>>,
}

impl ParameterVisitor<Backend> for GradFlatten<'_> {
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &Param<S, Backend, K, Train>,
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

/// Collects every parameter's current values as per-tensor `f64` chunks, in
/// `VisitParameters` order - the sequence `all_gather_model_parameters`
/// writes back.
struct ParamFlatten {
    chunks: Vec<Vec<f64>>,
}

impl ParameterVisitor<Backend> for ParamFlatten {
    fn visit_param<S, K, Train>(
        &mut self,
        _path: &StatePath,
        param: &Param<S, Backend, K, Train>,
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
        self.chunks
            .push(Backend::float_to_vec1::<f32>(tensor.inner())?);
        Ok(())
    }
}

fn flatten_params(model: &Model) -> Result<Vec<Vec<f64>>> {
    let mut visitor = ParamFlatten { chunks: Vec::new() };
    model.visit_parameters(&StatePath::root(), &mut visitor)?;
    Ok(visitor.chunks)
}

/// A scripted second rank for the FSDP collectives.
///
/// One queue per collective kind, FIFO per call, mirroring the
/// `dp2_network` peer: `reduce` scripts carry the peer's full gradient
/// values, `gather` scripts carry the peer's owned shard, and
/// `all_reduce` scripts carry the peer's full values for the ZeRO-1 path.
/// The `seen` vectors record what this rank issued, for call-count
/// assertions. Methods are only reached by walks that pass their rank and
/// divisibility checks first; the world-1 case is identity.
#[derive(Debug)]
struct ScriptedFsdp {
    rank: usize,
    world: usize,
    reduce_scripts: Mutex<Vec<Vec<f64>>>,
    gather_scripts: Mutex<Vec<Vec<f64>>>,
    all_reduce_scripts: Mutex<Vec<Vec<f64>>>,
    reduce_seen: Mutex<Vec<Vec<f64>>>,
    gather_seen: Mutex<Vec<Vec<f64>>>,
}

impl ScriptedFsdp {
    fn rank0_of_2(reduce: Vec<Vec<f64>>, gather: Vec<Vec<f64>>) -> Self {
        Self {
            rank: 0,
            world: 2,
            reduce_scripts: Mutex::new(reduce),
            gather_scripts: Mutex::new(gather),
            all_reduce_scripts: Mutex::new(Vec::new()),
            reduce_seen: Mutex::new(Vec::new()),
            gather_seen: Mutex::new(Vec::new()),
        }
    }

    fn rank1_of_2(reduce: Vec<Vec<f64>>) -> Self {
        Self {
            rank: 1,
            world: 2,
            reduce_scripts: Mutex::new(reduce),
            gather_scripts: Mutex::new(Vec::new()),
            all_reduce_scripts: Mutex::new(Vec::new()),
            reduce_seen: Mutex::new(Vec::new()),
            gather_seen: Mutex::new(Vec::new()),
        }
    }

    fn with_world(rank: usize, world: usize) -> Self {
        Self {
            rank,
            world,
            reduce_scripts: Mutex::new(Vec::new()),
            gather_scripts: Mutex::new(Vec::new()),
            all_reduce_scripts: Mutex::new(Vec::new()),
            reduce_seen: Mutex::new(Vec::new()),
            gather_seen: Mutex::new(Vec::new()),
        }
    }

    fn pop(queue: &Mutex<Vec<Vec<f64>>>, what: &str) -> std::result::Result<Vec<f64>, SyncError> {
        let mut queue = queue.lock().expect("script queue lock");
        if queue.is_empty() {
            return Err(SyncError::Synchronizer {
                message: format!("{what} script exhausted"),
            });
        }
        Ok(queue.remove(0))
    }
}

impl GradientSynchronizer for ScriptedFsdp {
    fn world_size(&self) -> usize {
        self.world
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn all_reduce_mean(&self, values: &mut [f64]) -> std::result::Result<(), SyncError> {
        let peer = Self::pop(&self.all_reduce_scripts, "all-reduce")?;
        if peer.len() != values.len() {
            return Err(SyncError::Synchronizer {
                message: format!(
                    "all-reduce peer length {} != local {}",
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

impl FsdpSynchronizer for ScriptedFsdp {
    fn reduce_scatter_mean(&self, values: &[f64]) -> std::result::Result<Vec<f64>, SyncError> {
        self.reduce_seen
            .lock()
            .expect("reduce_seen lock")
            .push(values.to_vec());
        let peer = Self::pop(&self.reduce_scripts, "reduce-scatter")?;
        if peer.len() != values.len() {
            return Err(SyncError::Synchronizer {
                message: format!(
                    "reduce-scatter peer length {} != local {}",
                    peer.len(),
                    values.len()
                ),
            });
        }
        if self.world == 1 {
            return Ok(values.to_vec());
        }
        let chunk = values.len() / self.world;
        let start = self.rank * chunk;
        let mean: Vec<f64> = values
            .iter()
            .zip(&peer)
            .map(|(local, remote)| (local + remote) / 2.0)
            .collect();
        Ok(mean[start..start + chunk].to_vec())
    }

    fn all_gather(&self, shard: &[f64]) -> std::result::Result<Vec<f64>, SyncError> {
        self.gather_seen
            .lock()
            .expect("gather_seen lock")
            .push(shard.to_vec());
        if self.world == 1 {
            return Ok(shard.to_vec());
        }
        let peer = Self::pop(&self.gather_scripts, "all-gather")?;
        if peer.len() != shard.len() {
            return Err(SyncError::Synchronizer {
                message: format!(
                    "all-gather peer shard {} != local {}",
                    peer.len(),
                    shard.len()
                ),
            });
        }
        // Rank-ordered concatenation: each slice at its rank's slot.
        if self.rank == 0 {
            let mut full = shard.to_vec();
            full.extend_from_slice(&peer);
            Ok(full)
        } else {
            let mut full = peer;
            full.extend_from_slice(shard);
            Ok(full)
        }
    }
}

fn backward_on(model: &Model, input: Tensor<Dyn, Backend>) -> Result<Gradients<Backend>> {
    let target = Tensor::<Dyn, Backend>::zeros(vec![4, 2])?.forget_layout();
    let loss = model.forward(input)?.mse_loss(&target)?;
    loss.backward()
}

/// Reduce-scatter arithmetic and the byte measurement.
///
/// Rank 0 ends the walk owning the first half of every gradient (the mean
/// of the local value and the scripted peer's) and zeros elsewhere; the
/// `ShardedGradients` report retains `1/2` of the bytes the all-reduce
/// round would have kept on this rank.
#[test]
fn reduce_scatter_keeps_owned_mean_shards_and_one_over_world_bytes() -> Result<()> {
    let model = model()?;
    let input = Tensor::<Dyn, Backend>::ones(vec![4, 4])?;
    let mut grads = backward_on(&model, input)?;

    let local = flatten_grads(&model, &grads)?;
    assert_eq!(local.len(), 4, "two Linear layers, four gradient tensors");
    let peer: Vec<Vec<f64>> = local
        .iter()
        .map(|chunk| chunk.iter().map(|value| value + 100.0).collect())
        .collect();
    let means: Vec<Vec<f64>> = local
        .iter()
        .zip(&peer)
        .map(|(chunk, peer_chunk)| {
            chunk
                .iter()
                .zip(peer_chunk)
                .map(|(l, p)| (l + p) / 2.0)
                .collect()
        })
        .collect();

    let script = ScriptedFsdp::rank0_of_2(peer, Vec::new());
    let sharded = reduce_scatter_model_gradients(&model, &mut grads, &script)
        .map_err(|error| Error::Msg(error.to_string()))?;

    // Owned slice: the mean's first half; the second half is zero-filled.
    let after = flatten_grads(&model, &grads)?;
    for (index, (got, mean)) in after.iter().zip(&means).enumerate() {
        let chunk = mean.len() / 2;
        assert_close(&got[..chunk], &mean[..chunk], 1e-5, "owned mean slice");
        assert!(
            got[chunk..].iter().all(|value| *value == 0.0),
            "tensor {index}: non-owned half must be zero, got {:?}",
            &got[chunk..]
        );
    }

    // The report: shards are the owned slices, and the bytes this rank
    // retains are 1/world of the all-reduce bytes.
    assert_eq!(sharded.rank(), 0);
    assert_eq!(sharded.world_size(), 2);
    assert_eq!(sharded.parameter_count(), local.len());
    for (index, (shard, mean)) in sharded.shards().iter().zip(&means).enumerate() {
        assert_close(
            shard,
            &mean[..mean.len() / 2],
            1e-5,
            &format!("shard report {index}"),
        );
    }
    let full_elements: usize = local.iter().map(Vec::len).sum();
    assert_eq!(sharded.full_elements(), full_elements);
    assert_eq!(sharded.owned_elements(), full_elements / 2);
    assert_eq!(sharded.retained_bytes(), full_elements / 2 * 8);
    assert_eq!(sharded.full_bytes(), full_elements * 8);
    assert_eq!(
        sharded.retained_bytes() * sharded.world_size(),
        sharded.full_bytes(),
        "reduce-scatter retains 1/world of the gradient bytes an all-reduce would retain"
    );
    assert!(
        sharded.retained_bytes() < sharded.full_bytes(),
        "a two-rank round must retain strictly fewer bytes than all-reduce"
    );

    let seen = script.reduce_seen.lock().expect("reduce_seen lock");
    assert_eq!(
        seen.len(),
        local.len(),
        "one collective per gradient tensor"
    );
    for (observed, chunk) in seen.iter().zip(&local) {
        assert_eq!(
            observed, chunk,
            "the collective must receive the local values unchanged"
        );
    }
    Ok(())
}

/// ZeRO-1's mask produces the same buffer as ZeRO-2's reduce-scatter.
///
/// The two stages differ only in which collective they use (all-reduce
/// then mask, versus reduce-scatter); on identical inputs the resulting
/// gradients - and therefore the step - must be identical.
#[test]
fn zero1_mask_after_all_reduce_matches_zero2_reduce_scatter() -> Result<()> {
    use incin_core::nn::{collect_state, load_state};

    let base = model()?;
    let initial = collect_state::<Backend, _>(&base)?;

    let full_x = Tensor::<Dyn, Backend>::from_slice(
        &(0..16)
            .map(|i| (i as f32) * 0.1 - 0.7)
            .collect::<Vec<f32>>(),
        vec![4, 4],
    )?;
    let shard_a_x = full_x.clone().try_narrow(0, 0, 2)?.forget_layout();
    let shard_b_x = full_x.clone().try_narrow(0, 2, 2)?.forget_layout();
    let full_y = Tensor::<Dyn, Backend>::from_slice(
        &(0..8)
            .map(|i| (i as f32) * 0.05 - 0.2)
            .collect::<Vec<f32>>(),
        vec![4, 2],
    )?;
    let shard_a_y = full_y.clone().try_narrow(0, 0, 2)?.forget_layout();
    let shard_b_y = full_y.clone().try_narrow(0, 2, 2)?.forget_layout();

    // The peer's (shard B) gradients, from an identically initialized model.
    let mut peer_model = model()?;
    load_state::<Backend, _>(&mut peer_model, &initial)?;
    let peer_loss = peer_model.forward(shard_b_x)?.mse_loss(&shard_b_y)?;
    let peer_grads = peer_loss.backward()?;
    let peer_script = flatten_grads(&peer_model, &peer_grads)?;

    // ZeRO-2 path: reduce-scatter shard A against shard B.
    let mut model_rs = model()?;
    load_state::<Backend, _>(&mut model_rs, &initial)?;
    let loss_rs = model_rs.forward(shard_a_x.clone())?.mse_loss(&shard_a_y)?;
    let mut grads_rs = loss_rs.backward()?;
    let rs_script = ScriptedFsdp::rank0_of_2(peer_script.clone(), Vec::new());
    let sharded = reduce_scatter_model_gradients(&model_rs, &mut grads_rs, &rs_script)
        .map_err(|error| Error::Msg(error.to_string()))?;
    assert_eq!(sharded.parameter_count(), 4);

    // ZeRO-1 path: all-reduce shard A against shard B, then mask to the
    // same owned slice.
    let mut model_z1 = model()?;
    load_state::<Backend, _>(&mut model_z1, &initial)?;
    let loss_z1 = model_z1.forward(shard_a_x)?.mse_loss(&shard_a_y)?;
    let mut grads_z1 = loss_z1.backward()?;
    let mut ar_script = ScriptedFsdp::rank0_of_2(Vec::new(), Vec::new());
    ar_script.all_reduce_scripts = Mutex::new(peer_script);
    all_reduce_model_gradients(&model_z1, &mut grads_z1, &ar_script)
        .map_err(|error| Error::Msg(error.to_string()))?;
    mask_gradients_to_owned_shard(&model_z1, &mut grads_z1, &ar_script)
        .map_err(|error| Error::Msg(error.to_string()))?;

    let rs_flat = flatten_grads(&model_rs, &grads_rs)?;
    let z1_flat = flatten_grads(&model_z1, &grads_z1)?;
    assert_eq!(rs_flat.len(), z1_flat.len());
    for (index, (rs, z1)) in rs_flat.iter().zip(&z1_flat).enumerate() {
        assert_close(
            rs,
            z1,
            1e-6,
            &format!("ZeRO-1 mask vs ZeRO-2 scatter, tensor {index}"),
        );
    }
    Ok(())
}

/// A gradient whose length does not divide is refused before its
/// collective runs - the script queue must be untouched.
#[test]
fn a_gradient_that_does_not_divide_is_refused_before_its_collective() {
    let odd: Linear<Dyn, Backend> = Linear::build((1, 3)).expect("odd-width layer");
    let input = Tensor::<Dyn, Backend>::ones(vec![1, 1]).expect("1x1 input");
    let target = Tensor::<Dyn, Backend>::zeros(vec![1, 3])
        .expect("1x3 target")
        .forget_layout();
    let loss = odd
        .forward(input)
        .expect("forward")
        .mse_loss(&target)
        .expect("loss");
    let mut grads = loss.backward().expect("backward");
    // Both tensors are 3 elements: 3 does not divide across 2 ranks.
    let script = ScriptedFsdp::rank0_of_2(vec![vec![0.0; 3]; 4], Vec::new());

    let error = reduce_scatter_model_gradients(&odd, &mut grads, &script)
        .expect_err("a non-divisible gradient must be refused");
    match error {
        SyncError::NonDivisibleShard {
            elements,
            world_size,
        } => {
            assert_eq!(elements, 3);
            assert_eq!(world_size, 2);
        }
        other => panic!("expected NonDivisibleShard, got {other:?}"),
    }
    assert!(
        script.reduce_seen.lock().expect("lock").is_empty(),
        "the offending parameter's collective must never run"
    );
    assert_eq!(
        script.reduce_scripts.lock().expect("lock").len(),
        4,
        "no script may be consumed before the refusal"
    );
}

/// A rank outside its world is refused before the walk starts.
#[test]
fn a_rank_outside_the_world_is_refused_before_any_collective() -> Result<()> {
    let model = model()?;
    let input = Tensor::<Dyn, Backend>::ones(vec![4, 4])?;
    let mut grads = backward_on(&model, input)?;
    let script = ScriptedFsdp::with_world(2, 2);

    let error = reduce_scatter_model_gradients(&model, &mut grads, &script)
        .expect_err("rank 2 of a 2-rank world must be refused");
    assert!(
        matches!(
            error,
            SyncError::RankOutOfRange {
                rank: 2,
                world_size: 2
            }
        ),
        "got {error:?}"
    );
    assert!(script.reduce_seen.lock().expect("lock").is_empty());
    Ok(())
}

/// All-gather rebuilds each parameter from the owners' shards and writes
/// the result back through the model.
///
/// The local (rank 0) shard stays; the scripted owner's shard replaces
/// the non-owned half - proof the gather's write-back is visible in the
/// parameters the next forward pass would read.
#[test]
fn all_gather_rebuilds_parameters_from_owner_shards() -> Result<()> {
    let model = model()?;
    let original = flatten_params(&model)?;

    // Every rank owns the first half; the scripted peer contributes the
    // second half, deliberately different from the local copy so the
    // write-back is observable.
    let peer_shards: Vec<Vec<f64>> = original
        .iter()
        .map(|chunk| {
            let half = chunk.len() / 2;
            chunk[half..].iter().map(|value| value + 1000.0).collect()
        })
        .collect();
    let script = ScriptedFsdp::rank0_of_2(Vec::new(), peer_shards.clone());

    all_gather_model_parameters(&model, &script).map_err(|error| Error::Msg(error.to_string()))?;

    let after = flatten_params(&model)?;
    assert_eq!(after.len(), original.len());
    for (index, (got, (before, peer_shard))) in after
        .iter()
        .zip(original.iter().zip(&peer_shards))
        .enumerate()
    {
        let half = before.len() / 2;
        assert_close(&got[..half], &before[..half], 1e-6, "owned half stays");
        let mut expected = before[..half].to_vec();
        expected.extend_from_slice(peer_shard);
        assert_close(
            got,
            &expected,
            1e-6,
            &format!("reconstructed parameter {index}"),
        );
    }

    let seen = script.gather_seen.lock().expect("gather_seen lock");
    assert_eq!(
        seen.len(),
        original.len(),
        "one gather per parameter tensor"
    );
    Ok(())
}

/// The ZeRO-2 trajectory: two ranks, reduce-scatter + step + all-gather
/// per step, must land where a single device training on the full batch
/// lands.
///
/// Rank 1 is a symmetric peer: its own reduce-scatter against rank 0's
/// full gradients lands the *mean*'s second half on its owned slice
/// before it steps - masking its local gradient instead would step the
/// half-batch gradient and drift from the reference. Equal shard sizes
/// make the mean of the half-batch gradients the full-batch gradient, so
/// after each reconstructed step rank 0's parameters must track the
/// reference within f32 tolerance.
#[test]
fn a_two_rank_zero2_trajectory_matches_the_single_device_reference() -> Result<()> {
    use incin_core::nn::{collect_state, load_state};

    let reference = model()?;
    let initial = collect_state::<Backend, _>(&reference)?;
    let mut model_full = model()?;
    load_state::<Backend, _>(&mut model_full, &initial)?;
    let mut rank0 = model()?;
    load_state::<Backend, _>(&mut rank0, &initial)?;
    let mut rank1 = model()?;
    load_state::<Backend, _>(&mut rank1, &initial)?;

    let mut sgd_full: SGD<Backend, f32> = SGD::from_module(&model_full, 0.01)?;
    let mut sgd0: SGD<Backend, f32> = SGD::from_module(&rank0, 0.01)?;
    let mut sgd1: SGD<Backend, f32> = SGD::from_module(&rank1, 0.01)?;

    let x_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect();
    let y_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.05 - 0.2).collect();
    let full_x = Tensor::<Dyn, Backend>::from_slice(&x_data, vec![4, 4])?;
    let full_y = Tensor::<Dyn, Backend>::from_slice(&y_data, vec![4, 2])?;
    let shard_a_x = full_x.clone().try_narrow(0, 0, 2)?.forget_layout();
    let shard_a_y = full_y.clone().try_narrow(0, 0, 2)?.forget_layout();
    let shard_b_x = full_x.clone().try_narrow(0, 2, 2)?.forget_layout();
    let shard_b_y = full_y.clone().try_narrow(0, 2, 2)?.forget_layout();

    // Two steps is a trajectory, not a single round: drift would compound.
    for step in 0..2 {
        // --- Reference: full batch, one step.
        let loss_full = model_full.forward(full_x.clone())?.mse_loss(&full_y)?;
        let grads_full = loss_full.backward()?;
        sgd_full.step(&grads_full)?;

        // --- Both ranks backward their shard first: each reduce-scatter
        // overwrites its local gradient buffer in place, so the peer's
        // full values must be read before either walk runs.
        let loss_a = rank0.forward(shard_a_x.clone())?.mse_loss(&shard_a_y)?;
        let mut grads_a = loss_a.backward()?;
        let local_a = flatten_grads(&rank0, &grads_a)?;
        let loss_b = rank1.forward(shard_b_x.clone())?.mse_loss(&shard_b_y)?;
        let mut grads_b = loss_b.backward()?;
        let local_b = flatten_grads(&rank1, &grads_b)?;

        // --- Rank 1: reduce-scatter shard B against rank 0's gradients,
        // landing the mean's second half on the owned slice, then step
        // and hand its owned parameter shards to rank 0's gather queue.
        let rank1_rs = ScriptedFsdp::rank1_of_2(local_a.clone());
        let sharded1 = reduce_scatter_model_gradients(&rank1, &mut grads_b, &rank1_rs)
            .map_err(|error| Error::Msg(error.to_string()))?;
        assert_eq!(sharded1.parameter_count(), 4, "step {step}");
        sgd1.step(&grads_b)?;
        let peer_params = flatten_params(&rank1)?;

        // --- Rank 0: reduce-scatter shard A against rank 1's gradients,
        // step, all-gather against rank 1's owned shards.
        let rs_script = ScriptedFsdp::rank0_of_2(local_b.clone(), Vec::new());
        let sharded = reduce_scatter_model_gradients(&rank0, &mut grads_a, &rs_script)
            .map_err(|error| Error::Msg(error.to_string()))?;
        assert_eq!(sharded.parameter_count(), 4, "step {step}");

        // The masked buffer: owned half is the mean of the two shards'
        // gradients, other half is zero.
        let reduced = flatten_grads(&rank0, &grads_a)?;
        for (index, ((got, local), peer_chunk)) in
            reduced.iter().zip(&local_a).zip(&local_b).enumerate()
        {
            let half = got.len() / 2;
            for (element, value) in got[..half].iter().enumerate() {
                let mean = (local[element] + peer_chunk[element]) / 2.0;
                assert!(
                    (value - mean).abs() <= 1e-5 * (1.0 + mean.abs()),
                    "step {step} tensor {index} element {element}: {value} vs mean {mean}"
                );
            }
            assert!(
                got[half..].iter().all(|value| *value == 0.0),
                "step {step} tensor {index}: non-owned half must be zero"
            );
        }

        sgd0.step(&grads_a)?;

        let gather_peer: Vec<Vec<f64>> = peer_params
            .iter()
            .map(|chunk| chunk[chunk.len() / 2..].to_vec())
            .collect();
        let gather_script = ScriptedFsdp::rank0_of_2(Vec::new(), gather_peer);
        all_gather_model_parameters(&rank0, &gather_script)
            .map_err(|error| Error::Msg(error.to_string()))?;

        // Both phases consumed exactly one script per parameter tensor on
        // both ranks: the ranks' collective sequences lined up.
        assert!(
            rs_script.reduce_scripts.lock().expect("queue").is_empty(),
            "step {step}: every reduce-scatter script must be consumed"
        );
        assert!(
            rank1_rs.reduce_scripts.lock().expect("queue").is_empty(),
            "step {step}: rank 1 must consume its reduce-scatter scripts too"
        );
        assert!(
            gather_script
                .gather_scripts
                .lock()
                .expect("queue")
                .is_empty(),
            "step {step}: every all-gather script must be consumed"
        );
    }

    // The whole point: rank 0's reconstructed replica tracks the
    // single-device full-batch trajectory.
    let reference_params = flatten_params(&model_full)?;
    let rank0_params = flatten_params(&rank0)?;
    assert_eq!(rank0_params.len(), reference_params.len());
    assert!(!rank0_params.is_empty(), "the model must carry parameters");
    for (index, (got, want)) in rank0_params.iter().zip(&reference_params).enumerate() {
        assert_close(
            got,
            want,
            1e-4,
            &format!("two-rank ZeRO-2 vs full-batch reference, tensor {index}"),
        );
    }
    Ok(())
}
