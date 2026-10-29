//! `SGD`, `AdamW`, `Adam` and `RMSprop`, the four concrete optimizers. Kept
//! in one file rather than one each: all four share the same nine names from
//! `super::group`/`super::support`/`super::traits` (`ParameterGroup`,
//! `PreparedUpdate`, `Optimizer`, `OptimizerBackend`, `commit_parameter_updates`,
//! `require_gradients_reached_the_group`, `validate_learning_rate`, plus
//! `Gradients`/`ScaledOptimizer`), and `AdamW`/`Adam` additionally share
//! `validate_adam_config`/`load_adam_state`/`prepare_adam_update` - splitting
//! them apart would mean writing that same import list four times over a
//! contiguous span rather than reading it once.

use super::group::{ParameterGroup, PreparedUpdate};
use super::scheduler::LRScheduler;
use super::support::{
    commit_parameter_updates, load_adam_state, load_adam_step, load_state_buffers,
    prepare_adam_update, prepare_rmsprop_update, prepare_sgd_update,
    require_full_gradient_coverage, require_gradients_reached_the_group, resolve_param_lr,
    save_adam_step, save_state_buffers, validate_adam_config, validate_param_lr,
    validate_rmsprop_config, validate_sgd_config,
};
use super::traits::{Optimizer, OptimizerBackend, ScaledOptimizer};
use crate::autograd::Gradients;
use crate::backend_authoring::{Capabilities, Execute};
use crate::err::{Error, Result};
use crate::exec::catalog::op;
use crate::nn::VisitParameters;
use crate::shapes::{Dyn, ShapeBuf};
use crate::tensor::backend::{AutogradBackend, HostReadback, VariableBackend};
use crate::tensor::base::Tensor;
use crate::tensor::dtype::{ConstDType, DType};
use alloc::string::String;

/// Stochastic Gradient Descent (SGD) optimizer.
///
/// Applies the update rule `w ← w - lr * direction`, where `direction` is the
/// (possibly weight-decayed) gradient, smoothed by classical momentum when
/// `momentum > 0.0`:
///
/// ```text
/// d ← grad + weight_decay * w
/// v ← momentum * v + d          (only when momentum > 0; v starts at zero)
/// direction ← d + momentum * v  (nesterov) or v (plain momentum) or d
/// w ← w - lr * direction
/// ```
///
/// Weight decay here is the *classic* (coupled) form: the penalty joins the
/// gradient before the momentum buffer, so past penalties linger in the
/// velocity. [`AdamW`](AdamW) does the opposite (decoupled decay straight
/// onto the parameter); the two agree only at `weight_decay == 0.0`.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// # use incin_backends::prelude::*;
/// # use incin_core::tensor::device::Cpu;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
/// let input = Cpu.ones(shape![8, 4])?;
/// let target = Cpu.zeros(shape![8, 2])?;
///
/// // The gradients must come from a backward pass over *this* model. A step
/// // whose gradients reach none of the group's parameters is refused rather
/// // than silently committing nothing.
/// let mut optimizer = SGD::<DefaultBackend>::from_module(&model, 0.01)?;
/// optimizer.momentum = 0.9;
/// let initial = model.forward(input.clone())?.mse_loss(&target)?.to_vec1::<f32>()?[0];
/// let mut loss_value = initial;
/// for _ in 0..100 {
///     let loss = model.forward(input.clone())?.mse_loss(&target)?;
///     loss_value = loss.to_vec1::<f32>()?[0];
///     optimizer.step(&loss.backward()?)?;
/// }
/// assert!(
///     loss_value < 0.1 * initial,
///     "momentum SGD should descend the bowl: {loss_value} vs initial {initial}"
/// );
/// # Ok(()) }
/// ```
pub struct SGD<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    /// `lr`.
    pub lr: f64,
    /// Per-parameter learning-rate overrides keyed by parameter-path
    /// prefix, resolved by [`lr_for`](Self::lr_for) with longest-prefix
    /// matching. Empty by default (every parameter trains at
    /// [`lr`](Self::lr)). Overrides are absolute rates, not multipliers:
    /// [`set_lr`](Self::set_lr) and [`step_scheduler`](Self::step_scheduler)
    /// move only the base rate; pinned rates stay put. Not part of
    /// `state_dict` (like `lr` itself): re-apply after loading.
    pub lr_overrides: alloc::collections::BTreeMap<String, f64>,
    /// Classical momentum coefficient in `[0, +∞)`, `0.0` disables momentum.
    /// Typical values are `0.9` or `0.99`; takes effect on the next step.
    ///
    /// At `0.0` (the default) the update is exactly `w - lr * grad`, with no
    /// smoothing or penalty. The pin below reads one step's parameters and
    /// gradients back through the canonical visitor and checks the rule
    /// elementwise:
    ///
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # fn main() -> incin::prelude::Result<()> {
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::backend_authoring::HostReadback;
    /// use incin::nn::param::Param;
    /// use incin::nn::{ParameterVisitor, StatePath, VisitParameters};
    /// use incin::prelude::*;
    ///
    /// struct ParamChunks {
    ///     chunks: Vec<Vec<f64>>,
    /// }
    ///
    /// impl ParameterVisitor<DefaultBackend> for ParamChunks {
    ///     fn visit_param<S, K, Train>(
    ///         &mut self,
    ///         _path: &StatePath,
    ///         param: &Param<S, DefaultBackend, K, Train>,
    ///     ) -> Result<()>
    ///     where
    ///         S: Shape,
    ///         K: DType,
    ///         Train: TrainState,
    ///     {
    ///         let tensor = param.as_tensor()?;
    ///         self.chunks
    ///             .push(DefaultBackend::float_to_vec1::<f32>(tensor.inner())?);
    ///         Ok(())
    ///     }
    /// }
    ///
    /// struct GradChunks<'a> {
    ///     grads: &'a Gradients<DefaultBackend>,
    ///     chunks: Vec<Vec<f64>>,
    /// }
    ///
    /// impl ParameterVisitor<DefaultBackend> for GradChunks<'_> {
    ///     fn visit_param<S, K, Train>(
    ///         &mut self,
    ///         _path: &StatePath,
    ///         param: &Param<S, DefaultBackend, K, Train>,
    ///     ) -> Result<()>
    ///     where
    ///         S: Shape,
    ///         K: DType,
    ///         Train: TrainState,
    ///     {
    ///         let tensor = param.as_tensor()?;
    ///         if let Some(grad) = self.grads.get(&tensor)? {
    ///             self.chunks
    ///                 .push(DefaultBackend::float_to_vec1::<f32>(grad.inner())?);
    ///         }
    ///         Ok(())
    ///     }
    /// }
    ///
    /// fn read_params(model: &Linear<s![4, 2], DefaultBackend>) -> Result<Vec<Vec<f64>>> {
    ///     let mut visitor = ParamChunks { chunks: Vec::new() };
    ///     model.visit_parameters(&StatePath::root(), &mut visitor)?;
    ///     Ok(visitor.chunks)
    /// }
    ///
    /// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
    /// let input = Cpu.ones(shape![8, 4])?;
    /// let target = Cpu.zeros(shape![8, 2])?;
    /// let grads = model.forward(input.clone())?.mse_loss(&target)?.backward()?;
    /// let before = read_params(&model)?;
    /// let mut gv = GradChunks {
    ///     grads: &grads,
    ///     chunks: Vec::new(),
    /// };
    /// model.visit_parameters(&StatePath::root(), &mut gv)?;
    ///
    /// let lr = 0.05;
    /// SGD::<DefaultBackend>::from_module(&model, lr)?.step(&grads)?;
    /// let after = read_params(&model)?;
    ///
    /// for ((p0, g), p1) in before.iter().zip(&gv.chunks).zip(&after) {
    ///     for ((w, dw), got) in p0.iter().zip(g).zip(p1) {
    ///         let want = w - lr * dw;
    ///         assert!(
    ///             (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
    ///             "{got} vs plain update {want}"
    ///         );
    ///     }
    /// }
    /// # Ok(()) }
    /// ```
    pub momentum: f64,
    /// Classic (coupled) L2 penalty added to the gradient before momentum.
    /// `0.0` disables it; takes effect on the next step.
    ///
    /// One step with `weight_decay` set is exactly `w - lr * (grad +
    /// decay * w)`: the penalty is smoothed by momentum on later steps
    /// rather than applied straight to the parameter:
    ///
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # fn main() -> incin::prelude::Result<()> {
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::backend_authoring::HostReadback;
    /// use incin::nn::param::Param;
    /// use incin::nn::{ParameterVisitor, StatePath, VisitParameters};
    /// use incin::prelude::*;
    ///
    /// struct ParamChunks {
    ///     chunks: Vec<Vec<f64>>,
    /// }
    ///
    /// impl ParameterVisitor<DefaultBackend> for ParamChunks {
    ///     fn visit_param<S, K, Train>(
    ///         &mut self,
    ///         _path: &StatePath,
    ///         param: &Param<S, DefaultBackend, K, Train>,
    ///     ) -> Result<()>
    ///     where
    ///         S: Shape,
    ///         K: DType,
    ///         Train: TrainState,
    ///     {
    ///         let tensor = param.as_tensor()?;
    ///         self.chunks
    ///             .push(DefaultBackend::float_to_vec1::<f32>(tensor.inner())?);
    ///         Ok(())
    ///     }
    /// }
    ///
    /// struct GradChunks<'a> {
    ///     grads: &'a Gradients<DefaultBackend>,
    ///     chunks: Vec<Vec<f64>>,
    /// }
    ///
    /// impl ParameterVisitor<DefaultBackend> for GradChunks<'_> {
    ///     fn visit_param<S, K, Train>(
    ///         &mut self,
    ///         _path: &StatePath,
    ///         param: &Param<S, DefaultBackend, K, Train>,
    ///     ) -> Result<()>
    ///     where
    ///         S: Shape,
    ///         K: DType,
    ///         Train: TrainState,
    ///     {
    ///         let tensor = param.as_tensor()?;
    ///         if let Some(grad) = self.grads.get(&tensor)? {
    ///             self.chunks
    ///                 .push(DefaultBackend::float_to_vec1::<f32>(grad.inner())?);
    ///         }
    ///         Ok(())
    ///     }
    /// }
    ///
    /// fn read_params(model: &Linear<s![4, 2], DefaultBackend>) -> Result<Vec<Vec<f64>>> {
    ///     let mut visitor = ParamChunks { chunks: Vec::new() };
    ///     model.visit_parameters(&StatePath::root(), &mut visitor)?;
    ///     Ok(visitor.chunks)
    /// }
    ///
    /// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
    /// let input = Cpu.ones(shape![8, 4])?;
    /// let target = Cpu.zeros(shape![8, 2])?;
    /// let grads = model.forward(input.clone())?.mse_loss(&target)?.backward()?;
    /// let before = read_params(&model)?;
    /// let mut gv = GradChunks {
    ///     grads: &grads,
    ///     chunks: Vec::new(),
    /// };
    /// model.visit_parameters(&StatePath::root(), &mut gv)?;
    ///
    /// let lr = 0.05;
    /// let decay = 0.1;
    /// let mut optimizer = SGD::<DefaultBackend>::from_module(&model, lr)?;
    /// optimizer.weight_decay = decay;
    /// optimizer.step(&grads)?;
    /// let after = read_params(&model)?;
    ///
    /// for ((p0, g), p1) in before.iter().zip(&gv.chunks).zip(&after) {
    ///     for ((w, dw), got) in p0.iter().zip(g).zip(p1) {
    ///         let want = w - lr * (dw + decay * w);
    ///         assert!(
    ///             (got - want).abs() <= 1e-5 * (1.0 + want.abs()),
    ///             "{got} vs coupled update {want}"
    ///         );
    ///     }
    /// }
    /// # Ok(()) }
    /// ```
    pub weight_decay: f64,
    /// Nesterov lookahead (`d + momentum * v` instead of `v`). Only valid
    /// with a positive [`momentum`](Self::momentum); enabling it at zero
    /// momentum is refused at step time rather than silently training plain
    /// SGD.
    ///
    /// The lookahead pays off on ill-conditioned bowls, where plain momentum
    /// oscillates along the stiff axis. Below, two identically-initialized
    /// regressions with curvatures 25 and 0.04 train on the same batches:
    /// Nesterov's trajectory differs from plain momentum's and lands
    /// strictly lower:
    ///
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # fn main() -> incin::prelude::Result<()> {
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    ///
    /// let x = Tensor::<Dyn, DefaultBackend>::from_slice(
    ///     &[5.0f32, 0.0, 0.0, 0.2],
    ///     vec![2, 2],
    /// )?;
    /// let y = Tensor::<Dyn, DefaultBackend>::from_slice(&[1.0f32, 1.0], vec![2, 1])?;
    ///
    /// let plain_model = Linear::<Dyn, DefaultBackend>::build((2, 1))?;
    /// let initial = collect_state::<DefaultBackend, _>(&plain_model)?;
    /// let mut nesterov_model = Linear::<Dyn, DefaultBackend>::build((2, 1))?;
    /// load_state::<DefaultBackend, _>(&mut nesterov_model, &initial)?;
    ///
    /// let mut plain = SGD::<DefaultBackend>::from_module(&plain_model, 0.03)?;
    /// plain.momentum = 0.9;
    /// let mut nesterov = SGD::<DefaultBackend>::from_module(&nesterov_model, 0.03)?;
    /// nesterov.momentum = 0.9;
    /// nesterov.nesterov = true;
    ///
    /// let mut plain_loss = f64::INFINITY;
    /// let mut nesterov_loss = f64::INFINITY;
    /// let mut differ = false;
    /// for _ in 0..80 {
    ///     let plain_step = plain_model.forward(x.clone())?.mse_loss(&y)?;
    ///     plain_loss = plain_step.to_vec1::<f32>()?[0] as f64;
    ///     plain.step(&plain_step.backward()?)?;
    ///     let nesterov_step = nesterov_model.forward(x.clone())?.mse_loss(&y)?;
    ///     nesterov_loss = nesterov_step.to_vec1::<f32>()?[0] as f64;
    ///     nesterov.step(&nesterov_step.backward()?)?;
    ///     differ |= plain_loss != nesterov_loss;
    /// }
    /// assert!(differ, "nesterov must not retrace plain momentum exactly");
    /// assert!(
    ///     nesterov_loss < plain_loss,
    ///     "nesterov {nesterov_loss} should beat plain momentum {plain_loss}"
    /// );
    /// # Ok(()) }
    /// ```
    pub nesterov: bool,
    velocity: alloc::collections::BTreeMap<String, B::Storage<K>>,
    _marker: core::marker::PhantomData<K>,
}

impl<B: VariableBackend, K: DType> SGD<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    ///
    /// Starts with `momentum == 0.0`, `weight_decay == 0.0`, and no Nesterov
    /// lookahead, which reproduces the pre-momentum update exactly; set the
    /// public fields before stepping to opt into the new behavior.
    pub fn new(
        params: alloc::collections::BTreeMap<
            String,
            <B as crate::tensor::backend::VariableBackend>::Var<K>,
        >,
        lr: f64,
    ) -> Self {
        Self {
            params,
            lr,
            lr_overrides: alloc::collections::BTreeMap::new(),
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocity: alloc::collections::BTreeMap::new(),
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates an optimizer from the canonical module-derived parameter group.
    pub fn from_group(group: ParameterGroup<B, K>, lr: f64) -> Self
    where
        K: ConstDType,
    {
        Self::new(group.into_map(), lr)
    }

    /// Collects a module's trainable parameters and creates the optimizer.
    pub fn from_module<M>(module: &M, lr: f64) -> Result<Self>
    where
        M: VisitParameters<B>,
        K: ConstDType,
    {
        Ok(Self::from_group(ParameterGroup::from_module(module)?, lr))
    }

    /// Sets the learning rate; takes effect on the next step.
    ///
    /// The value is validated when the next step runs, like every other
    /// optimizer field, so an out-of-range rate surfaces as a typed step
    /// error rather than corrupting the assignment.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Resolves the learning rate for one parameter: longest-prefix
    /// match over [`lr_overrides`](Self::lr_overrides), else the base
    /// [`lr`](Self::lr).
    pub fn lr_for(&self, param: &str) -> f64 {
        resolve_param_lr(&self.lr_overrides, self.lr, param)
    }

    /// Pins a learning rate for a parameter subtree: every parameter whose
    /// dotted path equals `prefix` or starts with `prefix + "."` trains at
    /// `lr` instead of the base rate. Checked (finite, non-negative) at
    /// the next step, naming the parameter on refusal.
    ///
    /// Overrides are absolute, not multipliers: [`set_lr`](Self::set_lr)
    /// and [`step_scheduler`](Self::step_scheduler) move only the base
    /// rate, so a scheduler ramp never lifts a pinned group. Not part of
    /// `state_dict` (like `lr` itself): re-apply after loading.
    ///
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # fn main() -> incin::prelude::Result<()> {
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::nn::state::StateSnapshot;
    /// use incin::optim::{LinearWarmup, SGD};
    /// use incin::prelude::*;
    ///
    /// fn entry(snap: &StateSnapshot, name: &str) -> Vec<u8> {
    ///     snap.iter()
    ///         .find(|(p, _)| format!("{p:?}").contains(name))
    ///         .map(|(_, v)| v.bytes().to_vec())
    ///         .unwrap()
    /// }
    ///
    /// let model = Linear::<s![2, 2], DefaultBackend>::build(())?;
    /// let x = Cpu.ones(shape![2, 2])?.require_grad();
    /// let y = Cpu.zeros(shape![2, 2])?;
    /// // Pin the weight at lr 0: it freezes while the bias trains.
    /// let mut sgd = SGD::<DefaultBackend>::from_module(&model, 0.1)?;
    /// sgd.set_param_lr("weight", 0.0);
    /// assert_eq!(sgd.lr_for("weight"), 0.0);
    /// assert_eq!(sgd.lr_for("bias"), 0.1);
    /// let before = collect_state::<DefaultBackend, _>(&model)?;
    /// let loss = model.forward(x.clone())?.mse_loss(&y)?;
    /// sgd.step(&loss.backward()?)?;
    /// let after = collect_state::<DefaultBackend, _>(&model)?;
    /// let (w0, b0) = (entry(&before, "weight"), entry(&before, "bias"));
    /// let (w1, b1) = (entry(&after, "weight"), entry(&after, "bias"));
    /// assert_eq!(w0, w1, "pinned weight must not move");
    /// assert_ne!(b0, b1, "unpinned bias must move");
    /// // A scheduler ramp moves the base only; the pin stays put.
    /// let mut scheduler = LinearWarmup::new(0.2, 4);
    /// sgd.step_scheduler(&scheduler);
    /// assert_eq!(sgd.lr, 0.0);
    /// assert_eq!(sgd.lr_for("weight"), 0.0);
    /// sgd.clear_param_lrs();
    /// assert_eq!(sgd.lr_for("weight"), sgd.lr);
    /// # Ok(()) }
    /// ```
    pub fn set_param_lr(&mut self, prefix: impl Into<String>, lr: f64) {
        self.lr_overrides.insert(prefix.into(), lr);
    }

    /// Drops all per-parameter overrides; every parameter trains at the
    /// base rate again.
    pub fn clear_param_lrs(&mut self) {
        self.lr_overrides.clear();
    }

    /// Applies a scheduler's current learning rate to this optimizer.
    ///
    /// This copies `scheduler.get_lr()` into [`lr`](Self::lr) and nothing
    /// else: advance the schedule yourself. The two-line training-loop
    /// pattern is
    ///
    /// ```text
    /// optimizer.step_scheduler(&scheduler);
    /// scheduler.step();
    /// ```
    ///
    /// Every optimizer binds the same way. Pinning the pattern for all four
    /// against a warmup ramp:
    ///
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # fn main() -> incin::prelude::Result<()> {
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::optim::{Adam, AdamW, LinearWarmup, RMSprop};
    /// use incin::prelude::*;
    ///
    /// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
    /// let mut sgd = SGD::<DefaultBackend>::from_module(&model, 0.1)?;
    /// let mut adam = Adam::<DefaultBackend>::from_module(&model, 0.1)?;
    /// let mut adamw = AdamW::<DefaultBackend>::from_module(&model, 0.1)?;
    /// let mut rmsprop = RMSprop::<DefaultBackend>::from_module(&model, 0.1)?;
    ///
    /// for optimizer in [&mut sgd.lr, &mut adam.lr, &mut adamw.lr, &mut rmsprop.lr] {
    ///     *optimizer = 0.02;
    /// }
    /// assert_eq!((sgd.lr, adam.lr, adamw.lr, rmsprop.lr), (0.02, 0.02, 0.02, 0.02));
    ///
    /// // Warmup step 0 reads exactly 0.0; two scheduler steps later every
    /// // bound optimizer tracks the ramp, proving the copy (not a stale
    /// // field) is what they train with.
    /// let mut scheduler = LinearWarmup::new(0.2, 4);
    /// sgd.step_scheduler(&scheduler);
    /// adam.step_scheduler(&scheduler);
    /// adamw.step_scheduler(&scheduler);
    /// rmsprop.step_scheduler(&scheduler);
    /// assert_eq!((sgd.lr, adam.lr, adamw.lr, rmsprop.lr), (0.0, 0.0, 0.0, 0.0));
    /// scheduler.step();
    /// scheduler.step();
    /// sgd.step_scheduler(&scheduler);
    /// adam.step_scheduler(&scheduler);
    /// adamw.step_scheduler(&scheduler);
    /// rmsprop.step_scheduler(&scheduler);
    /// for (name, lr) in [("sgd", sgd.lr), ("adam", adam.lr), ("adamw", adamw.lr), ("rmsprop", rmsprop.lr)] {
    ///     assert!((lr - 0.1).abs() < 1e-12, "{name} should track the ramp at 0.1, got {lr}");
    /// }
    /// # Ok(()) }
    /// ```
    pub fn step_scheduler(&mut self, scheduler: &impl LRScheduler) {
        self.set_lr(scheduler.get_lr());
    }

    /// Exports the velocity buffers under `{prefix.}momentum_buffer.{name}`.
    ///
    /// Parameters without a buffer yet (momentum disabled, or no step taken)
    /// contribute no entry and restart from zero on load, exactly the state a
    /// fresh optimizer holds.
    ///
    /// A resumed run replays the uninterrupted trajectory bit for bit: the
    /// saved velocity (not just the parameters) is what makes that true.
    ///
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # fn main() -> incin::prelude::Result<()> {
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    /// use std::collections::BTreeMap;
    ///
    /// fn snapshot(model: &Linear<Dyn, DefaultBackend>) -> Result<BTreeMap<StatePath, Vec<u8>>> {
    ///     let mut flat = BTreeMap::new();
    ///     for (path, value) in collect_state::<DefaultBackend, _>(model)?.iter() {
    ///         flat.insert(path.clone(), value.bytes().to_vec());
    ///     }
    ///     Ok(flat)
    /// }
    ///
    /// let mut model = Linear::<Dyn, DefaultBackend>::build((4, 2))?;
    /// let x_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect();
    /// let y_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.05 - 0.2).collect();
    /// let x = Tensor::<Dyn, DefaultBackend>::from_slice(&x_data, vec![4, 4])?;
    /// let y = Tensor::<Dyn, DefaultBackend>::from_slice(&y_data, vec![4, 2])?;
    ///
    /// let mut first = SGD::<DefaultBackend>::from_module(&model, 0.05)?;
    /// first.momentum = 0.9;
    /// for _ in 0..3 {
    ///     let loss = model.forward(x.clone())?.mse_loss(&y)?;
    ///     first.step(&loss.backward()?)?;
    /// }
    /// let mut dict = BTreeMap::new();
    /// first.state_dict("sgd", &mut dict)?;
    /// assert!(dict.keys().any(|key| key.starts_with("sgd.momentum_buffer.")));
    /// let at_three = collect_state::<DefaultBackend, _>(&model)?;
    ///
    /// // Two more uninterrupted steps are the reference trajectory.
    /// for _ in 0..2 {
    ///     let loss = model.forward(x.clone())?.mse_loss(&y)?;
    ///     first.step(&loss.backward()?)?;
    /// }
    /// let reference = snapshot(&model)?;
    ///
    /// // The resumed run restarts from the step-3 parameters with the saved
    /// // velocity and must land on the same bytes.
    /// load_state::<DefaultBackend, _>(&mut model, &at_three)?;
    /// let mut second = SGD::<DefaultBackend>::from_module(&model, 0.05)?;
    /// second.momentum = 0.9;
    /// second.load_state_dict("sgd", &dict)?;
    /// for _ in 0..2 {
    ///     let loss = model.forward(x.clone())?.mse_loss(&y)?;
    ///     second.step(&loss.backward()?)?;
    /// }
    /// assert_eq!(snapshot(&model)?, reference);
    /// # Ok(()) }
    /// ```
    pub fn state_dict(
        &self,
        prefix: &str,
        dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        save_state_buffers("momentum_buffer", prefix, &self.velocity, dict)
    }

    /// Loads velocity buffers saved by [`state_dict`](Self::state_dict).
    ///
    /// Entries naming unknown parameters or mismatching a parameter's shape,
    /// dtype, or device are typed errors; entries under other prefixes pass
    /// through so one dictionary can carry several optimizers' states.
    pub fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        self.velocity = load_state_buffers(
            "sgd_load_state_dict",
            prefix,
            "momentum_buffer",
            &self.params,
            dict,
        )?;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> SGD<B, K> {
    /// Strict step: like [`Optimizer::step`](super::traits::Optimizer::step),
    /// but refuses a partial step in which some — but not all — parameters
    /// received gradients.
    ///
    /// The lenient `step` stays PyTorch-compatible and skips unreached
    /// parameters; this reports them as
    /// [`Error::InvalidModuleState`] instead. A step that reaches zero
    /// parameters is refused by both spellings.
    pub fn step_strict(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "sgd_step_strict";
        validate_sgd_config(
            OPERATION,
            self.lr,
            self.momentum,
            self.weight_decay,
            self.nesterov,
        )?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let lr = self.lr_for(name);
            if lr != self.lr {
                validate_param_lr(OPERATION, name, lr)?;
            }
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, velocity) = prepare_sgd_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.velocity.get(name),
                    lr,
                    self.momentum,
                    self.weight_decay,
                    self.nesterov,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: velocity,
                    second_moment: None,
                });
            }
        }
        require_full_gradient_coverage(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            if let Some(velocity) = update.first_moment {
                self.velocity.insert(update.name, velocity);
            }
        }
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> Optimizer<B> for SGD<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "sgd_step";
        validate_sgd_config(
            OPERATION,
            self.lr,
            self.momentum,
            self.weight_decay,
            self.nesterov,
        )?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let lr = self.lr_for(name);
            if lr != self.lr {
                validate_param_lr(OPERATION, name, lr)?;
            }
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, velocity) = prepare_sgd_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.velocity.get(name),
                    lr,
                    self.momentum,
                    self.weight_decay,
                    self.nesterov,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: velocity,
                    second_moment: None,
                });
            }
        }
        require_gradients_reached_the_group(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            if let Some(velocity) = update.first_moment {
                self.velocity.insert(update.name, velocity);
            }
        }
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend + crate::tensor::backend::HostReadback, K: ConstDType>
    ScaledOptimizer<B> for SGD<B, K>
{
    fn step_scaled(
        &mut self,
        grads: &mut Gradients<B>,
        scaler: &mut crate::exec::LossScaleState,
    ) -> Result<bool> {
        if !scaler.unscale_and_update_vars(self.params.values(), grads)? {
            return Ok(false);
        }
        self.step(grads)?;
        Ok(true)
    }
}

/// AdamW optimizer (Adam with decoupled weight decay).
///
/// AdamW modifies the standard Adam algorithm by decoupling the weight decay from the
/// gradient updates. This leads to better generalization performance, particularly when
/// training transformer models and deep networks.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// # use incin_backends::prelude::*;
/// # use incin_core::tensor::device::Cpu;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
///
/// // The gradients must come from a backward pass over *this* model. A step
/// // whose gradients reach none of the group's parameters is refused rather
/// // than silently committing nothing.
/// let input = Cpu.ones(shape![1, 4])?.require_grad();
/// let loss = model.forward(input)?.sum_all()?;
/// let gradients = loss.backward()?;
///
/// let mut optimizer = AdamW::<DefaultBackend>::from_module(&model, 1e-4)?;
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
pub struct AdamW<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    /// `lr`.
    pub lr: f64,
    /// Per-parameter learning-rate overrides keyed by parameter-path
    /// prefix, resolved by [`lr_for`](Self::lr_for) with longest-prefix
    /// matching. Empty by default (every parameter trains at
    /// [`lr`](Self::lr)). Overrides are absolute rates, not multipliers:
    /// [`set_lr`](Self::set_lr) and [`step_scheduler`](Self::step_scheduler)
    /// move only the base rate; pinned rates stay put. Not part of
    /// `state_dict` (like `lr` itself): re-apply after loading.
    pub lr_overrides: alloc::collections::BTreeMap<String, f64>,
    /// `beta1`.
    pub beta1: f64,
    /// `beta2`.
    pub beta2: f64,
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f64,
    /// `weight_decay`.
    pub weight_decay: f64,
    m: alloc::collections::BTreeMap<String, B::Storage<K>>,
    v: alloc::collections::BTreeMap<String, B::Storage<K>>,
    step: usize,
}

impl<B: VariableBackend, K: DType> AdamW<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    pub fn new(
        params: alloc::collections::BTreeMap<
            String,
            <B as crate::tensor::backend::VariableBackend>::Var<K>,
        >,
        lr: f64,
    ) -> Self {
        Self {
            params,
            lr,
            lr_overrides: alloc::collections::BTreeMap::new(),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
            m: alloc::collections::BTreeMap::new(),
            v: alloc::collections::BTreeMap::new(),
            step: 0,
        }
    }

    /// Creates an optimizer from the canonical module-derived parameter group.
    pub fn from_group(group: ParameterGroup<B, K>, lr: f64) -> Self
    where
        K: ConstDType,
    {
        Self::new(group.into_map(), lr)
    }

    /// Collects a module's trainable parameters and creates the optimizer.
    pub fn from_module<M>(module: &M, lr: f64) -> Result<Self>
    where
        M: VisitParameters<B>,
        K: ConstDType,
    {
        Ok(Self::from_group(ParameterGroup::from_module(module)?, lr))
    }

    /// Gets the current step counter.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Sets the current step counter value.
    pub fn set_step_count(&mut self, step: usize) {
        self.step = step;
    }

    /// Sets the learning rate; takes effect on the next step.
    ///
    /// The value is validated when the next step runs, like every other
    /// optimizer field, so an out-of-range rate surfaces as a typed step
    /// error rather than corrupting the assignment.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Resolves the learning rate for one parameter: longest-prefix
    /// match over [`lr_overrides`](Self::lr_overrides), else the base
    /// [`lr`](Self::lr).
    pub fn lr_for(&self, param: &str) -> f64 {
        resolve_param_lr(&self.lr_overrides, self.lr, param)
    }

    /// Pins a learning rate for a parameter subtree: every parameter whose
    /// dotted path equals `prefix` or starts with `prefix + "."` trains at
    /// `lr` instead of the base rate. Checked (finite, non-negative) at
    /// the next step, naming the parameter on refusal.
    pub fn set_param_lr(&mut self, prefix: impl Into<String>, lr: f64) {
        self.lr_overrides.insert(prefix.into(), lr);
    }

    /// Drops all per-parameter overrides; every parameter trains at the
    /// base rate again.
    pub fn clear_param_lrs(&mut self) {
        self.lr_overrides.clear();
    }

    /// Applies a scheduler's current learning rate to this optimizer.
    ///
    /// This copies `scheduler.get_lr()` into [`lr`](Self::lr) and nothing
    /// else: advance the schedule yourself. The two-line training-loop
    /// pattern is
    ///
    /// ```text
    /// optimizer.step_scheduler(&scheduler);
    /// scheduler.step();
    /// ```
    pub fn step_scheduler(&mut self, scheduler: &impl LRScheduler) {
        self.set_lr(scheduler.get_lr());
    }

    /// Exports optimizer state tensors (`m` and `v` momentum buffers) plus a
    /// scalar `step` counter entry, so a resumed run bias-corrects with the
    /// same `t` the moments were accumulated under. The counter inherits
    /// `K`'s precision (`f16`/`bf16` counters are exact to 2048 steps) and
    /// requires the backend's `Full` creation row for `K`; without it saving
    /// fails with a typed refusal rather than silently dropping the counter.
    pub fn state_dict(
        &self,
        prefix: &str,
        dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()>
    where
        B: Execute<op::Full> + Capabilities,
        <B as Execute<op::Full>>::Output: Into<B::Storage<K>>,
    {
        let p = if prefix.is_empty() {
            alloc::string::String::new()
        } else {
            alloc::format!("{}.", prefix)
        };
        for (name, m_val) in &self.m {
            let shape = B::shape(m_val);
            let tensor = Tensor::<Dyn, B, K>::from_parts(
                m_val.clone(),
                ShapeBuf::from_slice(&shape),
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            )?;
            dict.insert(alloc::format!("{}m.{}", p, name), tensor);
        }
        for (name, v_val) in &self.v {
            let shape = B::shape(v_val);
            let tensor = Tensor::<Dyn, B, K>::from_parts(
                v_val.clone(),
                ShapeBuf::from_slice(&shape),
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            )?;
            dict.insert(alloc::format!("{}v.{}", p, name), tensor);
        }
        save_adam_step(prefix, self.step, dict)?;
        Ok(())
    }

    /// Loads optimizer state tensors from a dictionary.
    ///
    /// Restores the `step` counter saved by [`state_dict`](Self::state_dict);
    /// dictionaries predating the counter entry restore moments only and keep
    /// the current counter (set one explicitly with `set_step_count`).
    pub fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()>
    where
        B: HostReadback,
    {
        let (next_m, next_v) =
            load_adam_state::<B, K>("adamw_load_state_dict", prefix, &self.params, dict)?;
        self.m = next_m;
        self.v = next_v;
        if let Some(step) = load_adam_step::<B, K>("adamw_load_state_dict", prefix, dict)? {
            self.step = step;
        }
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> AdamW<B, K> {
    /// Strict step: like `step`, but refuses a partial step in which some —
    /// but not all — parameters received gradients. See
    /// [`SGD::step_strict`](SGD::step_strict).
    pub fn step_strict(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "adamw_step_strict";
        validate_adam_config(
            OPERATION,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            Some(self.weight_decay),
        )?;
        let next_step = self.step.checked_add(1).ok_or(Error::ArithmeticOverflow {
            operation: OPERATION,
            expression: "optimizer step + 1",
        })?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let lr = self.lr_for(name);
            if lr != self.lr {
                validate_param_lr(OPERATION, name, lr)?;
            }
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, m_t, v_t) = prepare_adam_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.m.get(name),
                    self.v.get(name),
                    lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    next_step,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: Some(m_t),
                    second_moment: Some(v_t),
                });
            }
        }
        require_full_gradient_coverage(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            self.m.insert(
                update.name.clone(),
                update.first_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared AdamW update lost first moment",
                })?,
            );
            self.v.insert(
                update.name,
                update.second_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared AdamW update lost second moment",
                })?,
            );
        }
        self.step = next_step;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> Optimizer<B> for AdamW<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "adamw_step";
        validate_adam_config(
            OPERATION,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            Some(self.weight_decay),
        )?;
        let next_step = self.step.checked_add(1).ok_or(Error::ArithmeticOverflow {
            operation: OPERATION,
            expression: "optimizer step + 1",
        })?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let lr = self.lr_for(name);
            if lr != self.lr {
                validate_param_lr(OPERATION, name, lr)?;
            }
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, m_t, v_t) = prepare_adam_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.m.get(name),
                    self.v.get(name),
                    lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    next_step,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: Some(m_t),
                    second_moment: Some(v_t),
                });
            }
        }
        require_gradients_reached_the_group(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            self.m.insert(
                update.name.clone(),
                update.first_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared AdamW update lost first moment",
                })?,
            );
            self.v.insert(
                update.name,
                update.second_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared AdamW update lost second moment",
                })?,
            );
        }
        self.step = next_step;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend + crate::tensor::backend::HostReadback, K: ConstDType>
    ScaledOptimizer<B> for AdamW<B, K>
{
    fn step_scaled(
        &mut self,
        grads: &mut Gradients<B>,
        scaler: &mut crate::exec::LossScaleState,
    ) -> Result<bool> {
        if !scaler.unscale_and_update_vars(self.params.values(), grads)? {
            return Ok(false);
        }
        self.step(grads)?;
        Ok(true)
    }
}

/// Adam optimization algorithm.
///
/// Implements the standard Adam optimizer with momentum and variance tracking.
/// For models sensitive to weight decay (like Transformers), prefer [`AdamW`].
///
/// `weight_decay`, when non-zero, is applied in the decoupled (AdamW-style)
/// form through the shared update helper: straight onto the parameter,
/// outside the moment estimates. It defaults to `0.0`, which preserves the
/// classic Adam trajectory exactly.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// # use incin_backends::prelude::*;
/// # use incin_core::tensor::device::Cpu;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
///
/// // The gradients must come from a backward pass over *this* model. A step
/// // whose gradients reach none of the group's parameters is refused rather
/// // than silently committing nothing.
/// let input = Cpu.ones(shape![1, 4])?.require_grad();
/// let loss = model.forward(input)?.sum_all()?;
/// let gradients = loss.backward()?;
///
/// let mut optimizer = Adam::<DefaultBackend>::from_module(&model, 1e-3)?;
/// optimizer.step(&gradients)?;
/// # Ok(()) }
/// ```
pub struct Adam<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    /// `lr`.
    pub lr: f64,
    /// Per-parameter learning-rate overrides keyed by parameter-path
    /// prefix, resolved by [`lr_for`](Self::lr_for) with longest-prefix
    /// matching. Empty by default (every parameter trains at
    /// [`lr`](Self::lr)). Overrides are absolute rates, not multipliers:
    /// [`set_lr`](Self::set_lr) and [`step_scheduler`](Self::step_scheduler)
    /// move only the base rate; pinned rates stay put. Not part of
    /// `state_dict` (like `lr` itself): re-apply after loading.
    pub lr_overrides: alloc::collections::BTreeMap<String, f64>,
    /// `beta1`.
    pub beta1: f64,
    /// `beta2`.
    pub beta2: f64,
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f64,
    /// Decoupled (AdamW-style) weight decay applied straight to the
    /// parameter, outside the moment estimates. Defaults to `0.0`, which
    /// preserves the classic Adam trajectory exactly.
    ///
    /// A non-zero decay moves the update, and a negative one is refused at
    /// step time:
    ///
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # fn main() -> incin::prelude::Result<()> {
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::prelude::*;
    ///
    /// let plain_model = Linear::<s![4, 2], DefaultBackend>::build(())?;
    /// let initial = collect_state::<DefaultBackend, _>(&plain_model)?;
    /// let mut decayed_model = Linear::<s![4, 2], DefaultBackend>::build(())?;
    /// load_state::<DefaultBackend, _>(&mut decayed_model, &initial)?;
    /// let input = Cpu.ones(shape![8, 4])?;
    /// let target = Cpu.zeros(shape![8, 2])?;
    ///
    /// let mut plain = Adam::<DefaultBackend>::from_module(&plain_model, 0.01)?;
    /// assert_eq!(plain.weight_decay, 0.0);
    /// let mut decayed = Adam::<DefaultBackend>::from_module(&decayed_model, 0.01)?;
    /// decayed.weight_decay = 0.1;
    ///
    /// let plain_loss = plain_model.forward(input.clone())?.mse_loss(&target)?;
    /// plain.step(&plain_loss.backward()?)?;
    /// let decayed_loss = decayed_model.forward(input.clone())?.mse_loss(&target)?;
    /// decayed.step(&decayed_loss.backward()?)?;
    ///
    /// let before = collect_state::<DefaultBackend, _>(&plain_model)?;
    /// let after = collect_state::<DefaultBackend, _>(&decayed_model)?;
    /// let mut drift = 0.0f64;
    /// for ((_, a), (_, b)) in before.iter().zip(after.iter()) {
    ///     for (x, y) in a.bytes().iter().zip(b.bytes()) {
    ///         drift = drift.max(((*x as f64) - (*y as f64)).abs());
    ///     }
    /// }
    /// assert!(drift > 0.0, "non-zero weight decay must move the update");
    ///
    /// let mut bad = Adam::<DefaultBackend>::from_module(&plain_model, 0.01)?;
    /// bad.weight_decay = -0.5;
    /// let loss = plain_model.forward(input.clone())?.mse_loss(&target)?;
    /// assert!(matches!(
    ///     bad.step(&loss.backward()?),
    ///     Err(Error::InvalidModuleState { .. })
    /// ));
    /// # Ok(()) }
    /// ```
    pub weight_decay: f64,
    m: alloc::collections::BTreeMap<String, B::Storage<K>>,
    v: alloc::collections::BTreeMap<String, B::Storage<K>>,
    step: usize,
}

impl<B: VariableBackend, K: DType> Adam<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    ///
    /// Starts with `weight_decay == 0.0`, which preserves the classic Adam
    /// trajectory exactly; set the public field before stepping to opt into
    /// decoupled decay.
    pub fn new(
        params: alloc::collections::BTreeMap<
            String,
            <B as crate::tensor::backend::VariableBackend>::Var<K>,
        >,
        lr: f64,
    ) -> Self {
        Self {
            params,
            lr,
            lr_overrides: alloc::collections::BTreeMap::new(),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            m: alloc::collections::BTreeMap::new(),
            v: alloc::collections::BTreeMap::new(),
            step: 0,
        }
    }

    /// Creates an optimizer from the canonical module-derived parameter group.
    pub fn from_group(group: ParameterGroup<B, K>, lr: f64) -> Self
    where
        K: ConstDType,
    {
        Self::new(group.into_map(), lr)
    }

    /// Collects a module's trainable parameters and creates the optimizer.
    pub fn from_module<M>(module: &M, lr: f64) -> Result<Self>
    where
        M: VisitParameters<B>,
        K: ConstDType,
    {
        Ok(Self::from_group(ParameterGroup::from_module(module)?, lr))
    }

    /// Gets the current step counter.
    pub fn step_count(&self) -> usize {
        self.step
    }

    /// Sets the current step counter value.
    pub fn set_step_count(&mut self, step: usize) {
        self.step = step;
    }

    /// Sets the learning rate; takes effect on the next step.
    ///
    /// The value is validated when the next step runs, like every other
    /// optimizer field, so an out-of-range rate surfaces as a typed step
    /// error rather than corrupting the assignment.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Resolves the learning rate for one parameter: longest-prefix
    /// match over [`lr_overrides`](Self::lr_overrides), else the base
    /// [`lr`](Self::lr).
    pub fn lr_for(&self, param: &str) -> f64 {
        resolve_param_lr(&self.lr_overrides, self.lr, param)
    }

    /// Pins a learning rate for a parameter subtree: every parameter whose
    /// dotted path equals `prefix` or starts with `prefix + "."` trains at
    /// `lr` instead of the base rate. Checked (finite, non-negative) at
    /// the next step, naming the parameter on refusal.
    pub fn set_param_lr(&mut self, prefix: impl Into<String>, lr: f64) {
        self.lr_overrides.insert(prefix.into(), lr);
    }

    /// Drops all per-parameter overrides; every parameter trains at the
    /// base rate again.
    pub fn clear_param_lrs(&mut self) {
        self.lr_overrides.clear();
    }

    /// Applies a scheduler's current learning rate to this optimizer.
    ///
    /// This copies `scheduler.get_lr()` into [`lr`](Self::lr) and nothing
    /// else: advance the schedule yourself. The two-line training-loop
    /// pattern is
    ///
    /// ```text
    /// optimizer.step_scheduler(&scheduler);
    /// scheduler.step();
    /// ```
    pub fn step_scheduler(&mut self, scheduler: &impl LRScheduler) {
        self.set_lr(scheduler.get_lr());
    }

    /// Exports optimizer state tensors (`m` and `v` momentum buffers) plus a
    /// scalar `step` counter entry, so a resumed run bias-corrects with the
    /// same `t` the moments were accumulated under. The counter inherits
    /// `K`'s precision (`f16`/`bf16` counters are exact to 2048 steps) and
    /// requires the backend's `Full` creation row for `K`; without it saving
    /// fails with a typed refusal rather than silently dropping the counter.
    pub fn state_dict(
        &self,
        prefix: &str,
        dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()>
    where
        B: Execute<op::Full> + Capabilities,
        <B as Execute<op::Full>>::Output: Into<B::Storage<K>>,
    {
        let p = if prefix.is_empty() {
            alloc::string::String::new()
        } else {
            alloc::format!("{}.", prefix)
        };
        for (name, m_val) in &self.m {
            let shape = B::shape(m_val);
            let tensor = Tensor::<Dyn, B, K>::from_parts(
                m_val.clone(),
                ShapeBuf::from_slice(&shape),
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            )?;
            dict.insert(alloc::format!("{}m.{}", p, name), tensor);
        }
        for (name, v_val) in &self.v {
            let shape = B::shape(v_val);
            let tensor = Tensor::<Dyn, B, K>::from_parts(
                v_val.clone(),
                ShapeBuf::from_slice(&shape),
                Default::default(),
                Default::default(),
                core::marker::PhantomData,
            )?;
            dict.insert(alloc::format!("{}v.{}", p, name), tensor);
        }
        save_adam_step(prefix, self.step, dict)?;
        Ok(())
    }

    /// Loads optimizer state tensors from a dictionary.
    ///
    /// Restores the `step` counter saved by [`state_dict`](Self::state_dict);
    /// dictionaries predating the counter entry restore moments only and keep
    /// the current counter (set one explicitly with `set_step_count`).
    pub fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()>
    where
        B: HostReadback,
    {
        let (next_m, next_v) =
            load_adam_state::<B, K>("adam_load_state_dict", prefix, &self.params, dict)?;
        self.m = next_m;
        self.v = next_v;
        if let Some(step) = load_adam_step::<B, K>("adam_load_state_dict", prefix, dict)? {
            self.step = step;
        }
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> Adam<B, K> {
    /// Strict step: like `step`, but refuses a partial step in which some —
    /// but not all — parameters received gradients. See
    /// [`SGD::step_strict`](SGD::step_strict).
    pub fn step_strict(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "adam_step_strict";
        validate_adam_config(
            OPERATION,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            Some(self.weight_decay),
        )?;
        let next_step = self.step.checked_add(1).ok_or(Error::ArithmeticOverflow {
            operation: OPERATION,
            expression: "optimizer step + 1",
        })?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let lr = self.lr_for(name);
            if lr != self.lr {
                validate_param_lr(OPERATION, name, lr)?;
            }
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, m_t, v_t) = prepare_adam_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.m.get(name),
                    self.v.get(name),
                    lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    next_step,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: Some(m_t),
                    second_moment: Some(v_t),
                });
            }
        }
        require_full_gradient_coverage(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            self.m.insert(
                update.name.clone(),
                update.first_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared Adam update lost first moment",
                })?,
            );
            self.v.insert(
                update.name,
                update.second_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared Adam update lost second moment",
                })?,
            );
        }
        self.step = next_step;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> Optimizer<B> for Adam<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "adam_step";
        validate_adam_config(
            OPERATION,
            self.lr,
            self.beta1,
            self.beta2,
            self.eps,
            Some(self.weight_decay),
        )?;
        let next_step = self.step.checked_add(1).ok_or(Error::ArithmeticOverflow {
            operation: OPERATION,
            expression: "optimizer step + 1",
        })?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let lr = self.lr_for(name);
            if lr != self.lr {
                validate_param_lr(OPERATION, name, lr)?;
            }
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, m_t, v_t) = prepare_adam_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.m.get(name),
                    self.v.get(name),
                    lr,
                    self.beta1,
                    self.beta2,
                    self.eps,
                    self.weight_decay,
                    next_step,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: Some(m_t),
                    second_moment: Some(v_t),
                });
            }
        }
        require_gradients_reached_the_group(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            self.m.insert(
                update.name.clone(),
                update.first_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared Adam update lost first moment",
                })?,
            );
            self.v.insert(
                update.name,
                update.second_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared Adam update lost second moment",
                })?,
            );
        }
        self.step = next_step;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend + crate::tensor::backend::HostReadback, K: ConstDType>
    ScaledOptimizer<B> for Adam<B, K>
{
    fn step_scaled(
        &mut self,
        grads: &mut Gradients<B>,
        scaler: &mut crate::exec::LossScaleState,
    ) -> Result<bool> {
        if !scaler.unscale_and_update_vars(self.params.values(), grads)? {
            return Ok(false);
        }
        self.step(grads)?;
        Ok(true)
    }
}

/// RMSprop optimizer (non-centered form).
///
/// Maintains a per-parameter running average of squared gradients and
/// divides the gradient by its root before stepping:
///
/// ```text
/// g ← grad + weight_decay * w
/// v ← alpha * v + (1 - alpha) * g²   (v starts at zero)
/// buf ← momentum * buf + g / (sqrt(v) + eps)   (only when momentum > 0)
/// w ← w - lr * buf   (or w - lr * g / (sqrt(v) + eps) without momentum)
/// ```
///
/// Weight decay is L2 regularization folded into the gradient (the same
/// coupled choice [`SGD`](SGD) makes), not a decoupled parameter update.
/// There is no `centered` variant: it would add a third state map for a form
/// this crate's backends never needed.
///
/// ## Examples
/// ```rust
/// # extern crate incin_core as incin;
/// # fn main() -> incin::prelude::Result<()> {
/// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
/// # use incin_backends::prelude::*;
/// # use incin_core::tensor::device::Cpu;
/// use incin::optim::RMSprop;
/// use incin::prelude::*;
///
/// let model = Linear::<s![4, 2], DefaultBackend>::build(())?;
/// let input = Cpu.ones(shape![8, 4])?;
/// let target = Cpu.zeros(shape![8, 2])?;
///
/// // The gradients must come from a backward pass over *this* model. A step
/// // whose gradients reach none of the group's parameters is refused rather
/// // than silently committing nothing.
/// let mut optimizer = RMSprop::<DefaultBackend>::from_module(&model, 0.01)?;
/// let initial = model.forward(input.clone())?.mse_loss(&target)?.to_vec1::<f32>()?[0];
/// let mut loss_value = initial;
/// for _ in 0..100 {
///     let loss = model.forward(input.clone())?.mse_loss(&target)?;
///     loss_value = loss.to_vec1::<f32>()?[0];
///     optimizer.step(&loss.backward()?)?;
/// }
/// assert!(
///     loss_value < 0.1 * initial,
///     "rmsprop should descend the bowl: {loss_value} vs initial {initial}"
/// );
/// # Ok(()) }
/// ```
pub struct RMSprop<B: VariableBackend, K: DType = f32> {
    params: alloc::collections::BTreeMap<
        String,
        <B as crate::tensor::backend::VariableBackend>::Var<K>,
    >,
    /// `lr`.
    pub lr: f64,
    /// Per-parameter learning-rate overrides keyed by parameter-path
    /// prefix, resolved by [`lr_for`](Self::lr_for) with longest-prefix
    /// matching. Empty by default (every parameter trains at
    /// [`lr`](Self::lr)). Overrides are absolute rates, not multipliers:
    /// [`set_lr`](Self::set_lr) and [`step_scheduler`](Self::step_scheduler)
    /// move only the base rate; pinned rates stay put. Not part of
    /// `state_dict` (like `lr` itself): re-apply after loading.
    pub lr_overrides: alloc::collections::BTreeMap<String, f64>,
    /// Smoothing constant for the squared-gradient average, in `[0, 1)`.
    /// Typical value is `0.99`; takes effect on the next step.
    pub alpha: f64,
    /// Small epsilon added to the denominator for numerical stability.
    pub eps: f64,
    /// Momentum on the normalized gradient, `0.0` disables it. Takes effect
    /// on the next step.
    pub momentum: f64,
    /// L2 penalty folded into the gradient before the square average.
    /// `0.0` disables it; takes effect on the next step.
    pub weight_decay: f64,
    square_avg: alloc::collections::BTreeMap<String, B::Storage<K>>,
    momentum_buffer: alloc::collections::BTreeMap<String, B::Storage<K>>,
}

impl<B: VariableBackend, K: DType> RMSprop<B, K> {
    /// Creates a new instance with default (statically inferred) shape arguments.
    ///
    /// Starts with PyTorch's defaults (`alpha == 0.99`, `eps == 1e-8`, no
    /// momentum, no weight decay); set the public fields before stepping to
    /// change them.
    pub fn new(
        params: alloc::collections::BTreeMap<
            String,
            <B as crate::tensor::backend::VariableBackend>::Var<K>,
        >,
        lr: f64,
    ) -> Self {
        Self {
            params,
            lr,
            lr_overrides: alloc::collections::BTreeMap::new(),
            alpha: 0.99,
            eps: 1e-8,
            momentum: 0.0,
            weight_decay: 0.0,
            square_avg: alloc::collections::BTreeMap::new(),
            momentum_buffer: alloc::collections::BTreeMap::new(),
        }
    }

    /// Creates an optimizer from the canonical module-derived parameter group.
    pub fn from_group(group: ParameterGroup<B, K>, lr: f64) -> Self
    where
        K: ConstDType,
    {
        Self::new(group.into_map(), lr)
    }

    /// Collects a module's trainable parameters and creates the optimizer.
    pub fn from_module<M>(module: &M, lr: f64) -> Result<Self>
    where
        M: VisitParameters<B>,
        K: ConstDType,
    {
        Ok(Self::from_group(ParameterGroup::from_module(module)?, lr))
    }

    /// Sets the learning rate; takes effect on the next step.
    ///
    /// The value is validated when the next step runs, like every other
    /// optimizer field, so an out-of-range rate surfaces as a typed step
    /// error rather than corrupting the assignment.
    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Resolves the learning rate for one parameter: longest-prefix
    /// match over [`lr_overrides`](Self::lr_overrides), else the base
    /// [`lr`](Self::lr).
    pub fn lr_for(&self, param: &str) -> f64 {
        resolve_param_lr(&self.lr_overrides, self.lr, param)
    }

    /// Pins a learning rate for a parameter subtree: every parameter whose
    /// dotted path equals `prefix` or starts with `prefix + "."` trains at
    /// `lr` instead of the base rate. Checked (finite, non-negative) at
    /// the next step, naming the parameter on refusal.
    pub fn set_param_lr(&mut self, prefix: impl Into<String>, lr: f64) {
        self.lr_overrides.insert(prefix.into(), lr);
    }

    /// Drops all per-parameter overrides; every parameter trains at the
    /// base rate again.
    pub fn clear_param_lrs(&mut self) {
        self.lr_overrides.clear();
    }

    /// Applies a scheduler's current learning rate to this optimizer.
    ///
    /// This copies `scheduler.get_lr()` into [`lr`](Self::lr) and nothing
    /// else: advance the schedule yourself. The two-line training-loop
    /// pattern is
    ///
    /// ```text
    /// optimizer.step_scheduler(&scheduler);
    /// scheduler.step();
    /// ```
    pub fn step_scheduler(&mut self, scheduler: &impl LRScheduler) {
        self.set_lr(scheduler.get_lr());
    }

    /// Exports the squared-gradient averages (`square_avg`) and momentum
    /// buffers (`momentum_buffer`) under `{prefix.}square_avg.{name}` and
    /// `{prefix.}momentum_buffer.{name}`.
    ///
    /// Parameters without state yet contribute no entry and restart from zero
    /// on load, exactly the state a fresh optimizer holds.
    ///
    /// A resumed run replays the uninterrupted trajectory bit for bit, same
    /// guarantee as [`SGD::state_dict`](SGD::state_dict):
    ///
    /// ```rust
    /// # extern crate incin_core as incin;
    /// # fn main() -> incin::prelude::Result<()> {
    /// # type DefaultBackend = incin_backends::cpu::CpuBackendImpl;
    /// # use incin_backends::prelude::*;
    /// # use incin_core::tensor::device::Cpu;
    /// use incin::optim::RMSprop;
    /// use incin::prelude::*;
    /// use std::collections::BTreeMap;
    ///
    /// fn snapshot(model: &Linear<Dyn, DefaultBackend>) -> Result<BTreeMap<StatePath, Vec<u8>>> {
    ///     let mut flat = BTreeMap::new();
    ///     for (path, value) in collect_state::<DefaultBackend, _>(model)?.iter() {
    ///         flat.insert(path.clone(), value.bytes().to_vec());
    ///     }
    ///     Ok(flat)
    /// }
    ///
    /// let mut model = Linear::<Dyn, DefaultBackend>::build((4, 2))?;
    /// let x_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.7).collect();
    /// let y_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.05 - 0.2).collect();
    /// let x = Tensor::<Dyn, DefaultBackend>::from_slice(&x_data, vec![4, 4])?;
    /// let y = Tensor::<Dyn, DefaultBackend>::from_slice(&y_data, vec![4, 2])?;
    ///
    /// let mut first = RMSprop::<DefaultBackend>::from_module(&model, 0.01)?;
    /// first.momentum = 0.9;
    /// for _ in 0..3 {
    ///     let loss = model.forward(x.clone())?.mse_loss(&y)?;
    ///     first.step(&loss.backward()?)?;
    /// }
    /// let mut dict = BTreeMap::new();
    /// first.state_dict("rms", &mut dict)?;
    /// assert!(dict.keys().any(|key| key.starts_with("rms.square_avg.")));
    /// assert!(dict.keys().any(|key| key.starts_with("rms.momentum_buffer.")));
    /// let at_three = collect_state::<DefaultBackend, _>(&model)?;
    ///
    /// // Two more uninterrupted steps are the reference trajectory.
    /// for _ in 0..2 {
    ///     let loss = model.forward(x.clone())?.mse_loss(&y)?;
    ///     first.step(&loss.backward()?)?;
    /// }
    /// let reference = snapshot(&model)?;
    ///
    /// // The resumed run restarts from the step-3 parameters with the saved
    /// // square average and momentum buffer, and must land on the same bytes.
    /// load_state::<DefaultBackend, _>(&mut model, &at_three)?;
    /// let mut second = RMSprop::<DefaultBackend>::from_module(&model, 0.01)?;
    /// second.momentum = 0.9;
    /// second.load_state_dict("rms", &dict)?;
    /// for _ in 0..2 {
    ///     let loss = model.forward(x.clone())?.mse_loss(&y)?;
    ///     second.step(&loss.backward()?)?;
    /// }
    /// assert_eq!(snapshot(&model)?, reference);
    /// # Ok(()) }
    /// ```
    pub fn state_dict(
        &self,
        prefix: &str,
        dict: &mut alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        save_state_buffers("square_avg", prefix, &self.square_avg, dict)?;
        save_state_buffers("momentum_buffer", prefix, &self.momentum_buffer, dict)
    }

    /// Loads state saved by [`state_dict`](Self::state_dict).
    ///
    /// Entries naming unknown parameters or mismatching a parameter's shape,
    /// dtype, or device are typed errors; entries under other prefixes pass
    /// through so one dictionary can carry several optimizers' states.
    pub fn load_state_dict(
        &mut self,
        prefix: &str,
        dict: &alloc::collections::BTreeMap<String, Tensor<Dyn, B, K>>,
    ) -> Result<()> {
        self.square_avg = load_state_buffers(
            "rmsprop_load_state_dict",
            prefix,
            "square_avg",
            &self.params,
            dict,
        )?;
        self.momentum_buffer = load_state_buffers(
            "rmsprop_load_state_dict",
            prefix,
            "momentum_buffer",
            &self.params,
            dict,
        )?;
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> RMSprop<B, K> {
    /// Strict step: like `step`, but refuses a partial step in which some —
    /// but not all — parameters received gradients. See
    /// [`SGD::step_strict`](SGD::step_strict).
    pub fn step_strict(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "rmsprop_step_strict";
        validate_rmsprop_config(
            OPERATION,
            self.lr,
            self.alpha,
            self.eps,
            self.momentum,
            self.weight_decay,
        )?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let lr = self.lr_for(name);
            if lr != self.lr {
                validate_param_lr(OPERATION, name, lr)?;
            }
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, square_avg, momentum_buffer) = prepare_rmsprop_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.square_avg.get(name),
                    self.momentum_buffer.get(name),
                    lr,
                    self.alpha,
                    self.eps,
                    self.momentum,
                    self.weight_decay,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: Some(square_avg),
                    second_moment: momentum_buffer,
                });
            }
        }
        require_full_gradient_coverage(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            self.square_avg.insert(
                update.name.clone(),
                update.first_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared RMSprop update lost its square average",
                })?,
            );
            if let Some(buffer) = update.second_moment {
                self.momentum_buffer.insert(update.name, buffer);
            }
        }
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend, K: DType> Optimizer<B> for RMSprop<B, K> {
    /// `step`.
    fn step(&mut self, grads: &Gradients<B>) -> Result<()> {
        const OPERATION: &str = "rmsprop_step";
        validate_rmsprop_config(
            OPERATION,
            self.lr,
            self.alpha,
            self.eps,
            self.momentum,
            self.weight_decay,
        )?;
        let mut updates = alloc::vec::Vec::new();
        for (name, var) in &self.params {
            let lr = self.lr_for(name);
            if lr != self.lr {
                validate_param_lr(OPERATION, name, lr)?;
            }
            let t = B::var_as_tensor::<K>(var)?;
            if let Some(grad) = B::get_grad::<K>(&t, grads.as_backend())? {
                let (updated, square_avg, momentum_buffer) = prepare_rmsprop_update::<B, K>(
                    OPERATION,
                    &t,
                    &grad,
                    self.square_avg.get(name),
                    self.momentum_buffer.get(name),
                    lr,
                    self.alpha,
                    self.eps,
                    self.momentum,
                    self.weight_decay,
                )?;
                updates.push(PreparedUpdate {
                    name: name.clone(),
                    before: t,
                    updated,
                    first_moment: Some(square_avg),
                    second_moment: momentum_buffer,
                });
            }
        }
        require_gradients_reached_the_group(OPERATION, self.params.len(), updates.len())?;
        commit_parameter_updates::<B, K>(OPERATION, &mut self.params, &updates)?;
        for update in updates {
            self.square_avg.insert(
                update.name.clone(),
                update.first_moment.ok_or(Error::InternalInvariant {
                    operation: OPERATION,
                    reason: "prepared RMSprop update lost its square average",
                })?,
            );
            if let Some(buffer) = update.second_moment {
                self.momentum_buffer.insert(update.name, buffer);
            }
        }
        Ok(())
    }
}

impl<B: OptimizerBackend<K> + AutogradBackend + crate::tensor::backend::HostReadback, K: ConstDType>
    ScaledOptimizer<B> for RMSprop<B, K>
{
    fn step_scaled(
        &mut self,
        grads: &mut Gradients<B>,
        scaler: &mut crate::exec::LossScaleState,
    ) -> Result<bool> {
        if !scaler.unscale_and_update_vars(self.params.values(), grads)? {
            return Ok(false);
        }
        self.step(grads)?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_invalid(result: Result<()>, what: &str) {
        assert!(
            matches!(result, Err(Error::InvalidModuleState { .. })),
            "{what} must be a typed InvalidModuleState refusal"
        );
    }

    #[test]
    fn sgd_config_accepts_defaults_and_momentum_with_nesterov() {
        assert!(validate_sgd_config("test", 0.01, 0.0, 0.0, false).is_ok());
        assert!(validate_sgd_config("test", 0.01, 0.9, 0.01, false).is_ok());
        assert!(validate_sgd_config("test", 0.01, 0.9, 0.01, true).is_ok());
        assert!(validate_sgd_config("test", 0.0, 0.0, 0.0, false).is_ok());
    }

    #[test]
    fn sgd_config_refuses_nesterov_without_momentum_and_bad_values() {
        assert_invalid(
            validate_sgd_config("test", 0.01, 0.0, 0.0, true),
            "nesterov at zero momentum",
        );
        assert_invalid(
            validate_sgd_config("test", -0.01, 0.0, 0.0, false),
            "negative learning rate",
        );
        assert_invalid(
            validate_sgd_config("test", f64::NAN, 0.0, 0.0, false),
            "NaN learning rate",
        );
        assert_invalid(
            validate_sgd_config("test", 0.01, -0.5, 0.0, false),
            "negative momentum",
        );
        assert_invalid(
            validate_sgd_config("test", 0.01, f64::INFINITY, 0.0, false),
            "infinite momentum",
        );
        assert_invalid(
            validate_sgd_config("test", 0.01, 0.0, -0.1, false),
            "negative weight decay",
        );
    }

    #[test]
    fn rmsprop_config_accepts_defaults_and_refuses_bad_values() {
        assert!(validate_rmsprop_config("test", 0.01, 0.99, 1e-8, 0.0, 0.0).is_ok());
        assert!(validate_rmsprop_config("test", 0.01, 0.9, 1e-8, 0.9, 0.01).is_ok());
        assert_invalid(
            validate_rmsprop_config("test", 0.01, 1.0, 1e-8, 0.0, 0.0),
            "alpha == 1.0",
        );
        assert_invalid(
            validate_rmsprop_config("test", 0.01, -0.1, 1e-8, 0.0, 0.0),
            "negative alpha",
        );
        assert_invalid(
            validate_rmsprop_config("test", 0.01, 0.99, 0.0, 0.0, 0.0),
            "zero epsilon",
        );
        assert_invalid(
            validate_rmsprop_config("test", 0.01, 0.99, 1e-8, -0.1, 0.0),
            "negative momentum",
        );
        assert_invalid(
            validate_rmsprop_config("test", 0.01, 0.99, 1e-8, 0.0, -0.1),
            "negative weight decay",
        );
        assert_invalid(
            validate_rmsprop_config("test", -0.01, 0.99, 1e-8, 0.0, 0.0),
            "negative learning rate",
        );
    }

    #[test]
    fn adam_config_wires_weight_decay_through_validation() {
        // Adam passes `Some(weight_decay)` now: a negative decay is refused
        // at step time, and the zero default validates cleanly.
        assert!(validate_adam_config("test", 0.01, 0.9, 0.999, 1e-8, Some(0.0)).is_ok());
        assert!(validate_adam_config("test", 0.01, 0.9, 0.999, 1e-8, Some(0.1)).is_ok());
        assert_invalid(
            validate_adam_config("test", 0.01, 0.9, 0.999, 1e-8, Some(-0.5)),
            "negative Adam weight decay",
        );
    }

    fn overrides(pairs: &[(&str, f64)]) -> alloc::collections::BTreeMap<String, f64> {
        pairs
            .iter()
            .map(|(k, v)| (alloc::string::ToString::to_string(k), *v))
            .collect()
    }

    #[test]
    fn resolve_param_lr_matches_exact_prefix_and_base() {
        let map = overrides(&[("encoder", 0.01)]);
        assert_eq!(resolve_param_lr(&map, 0.1, "encoder"), 0.01);
        assert_eq!(resolve_param_lr(&map, 0.1, "encoder.layer.0"), 0.01);
        assert_eq!(resolve_param_lr(&map, 0.1, "decoder"), 0.1);
        assert_eq!(
            resolve_param_lr(&overrides(&[]), 0.1, "anything.at.all"),
            0.1
        );
    }

    #[test]
    fn resolve_param_lr_prefers_longest_segment_aware_prefix() {
        let map = overrides(&[("a", 0.01), ("a.b", 0.02)]);
        assert_eq!(resolve_param_lr(&map, 0.1, "a.b.c"), 0.02);
        assert_eq!(resolve_param_lr(&map, 0.1, "a.c"), 0.01);
        // Segment-aware: "enc" must not match "encoder".
        let map = overrides(&[("enc", 0.05)]);
        assert_eq!(resolve_param_lr(&map, 0.1, "encoder"), 0.1);
        // An empty prefix matches nothing, not everything.
        let map = overrides(&[("", 0.05)]);
        assert_eq!(resolve_param_lr(&map, 0.1, "encoder"), 0.1);
    }

    #[test]
    fn validate_param_lr_names_the_parameter() {
        assert!(validate_param_lr("test", "head", 0.01).is_ok());
        let err = validate_param_lr("test", "head", -0.5).unwrap_err();
        let msg = alloc::format!("{err:?}");
        assert!(
            msg.contains("head"),
            "refusal must name the parameter, got {msg}"
        );
    }
}
