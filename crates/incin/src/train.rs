//! The automatic `Trainer` from `PROPOSALS.md` §2 (`UX-001`).
//!
//! §2's UX principles put this at level 1 of three, "select three devices and
//! let Incin produce and explain a safe plan", and the sentence after the list
//! is what makes it a level rather than a convenience wrapper:
//!
//! > "Easy" must not mean silent CPU transfer, hidden padding, relaxed
//! > determinism, or unbounded autotuning. Every automatic decision is
//! > inspectable and reproducible.
//!
//! So the load-bearing property of this module is not the loop. It is that a
//! [`Trainer`] asked for devices it cannot have refuses to build, and that the
//! [`Plan`] it does build says what was decided and why. A trainer that quietly
//! runs on the CPU when the CUDA devices it was handed are absent is worse than
//! no trainer at all, because the failure mode is a training run that finishes.
//!
//! ```rust
//! # use incin::experimental::training::{Machine, Trainer};
//! # use incin::prelude::{DeviceKind, DeviceId, DeviceSet};
//! # struct ThreeGpus;
//! # impl Machine for ThreeGpus {
//! #     fn compiled_in(&self, kind: DeviceKind) -> bool { kind == DeviceKind::Cuda }
//! #     fn has_device(&self, device: DeviceId) -> bool {
//! #         device.kind() == DeviceKind::Cuda && device.ordinal() < 3
//! #     }
//! # }
//! let plan = Trainer::plan()
//!     .devices(DeviceSet::cuda(0..3).unwrap())
//!     .epochs(10)
//!     .build_on(&ThreeGpus)
//!     .unwrap();
//!
//! assert_eq!(plan.devices().len(), 3);
//! assert!(plan.is_multi_device());
//! ```
//!
//! # What this row does not do
//!
//! `ParallelStrategy` and the plan objective are `DST-011`'s, and `.explain()`
//! as a rendered planning report is `UX-005`'s - which depends on both this row
//! and `DST-011`. Multi-device *execution* needs `DST-005`'s collectives. This
//! module therefore plans and validates a multi-device run and refuses to
//! pretend it can execute one: [`Trainer::fit`] on a multi-device plan is an
//! explicit [`TrainError::CollectivesUnavailable`], not a silent single-GPU run.

use incin_core::backend_authoring::Backend;
use incin_core::backend_authoring::{AutogradBackend, HostInterop, VariableBackend};
use incin_core::exec::{LossScaleState, LossScaling};
use incin_core::optim::{Optimizer, ScaledOptimizer};
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::{DeviceId, DeviceKind, DevicePreference, DeviceSet};

/// The devices a [`DevicePreference::Fastest`] resolution tries, most capable
/// first.
///
/// The same order `incin_backends::detect::PREFERENCE` uses, restated here
/// rather than imported so that this module's behaviour does not change when a
/// backend crate reorders its own detection for an unrelated reason.
const FASTEST_ORDER: &[DeviceKind] = &[
    DeviceKind::Cuda,
    DeviceKind::Metal,
    DeviceKind::Wgpu,
    DeviceKind::Cpu,
];

// ============================================================================
// The machine
// ============================================================================

/// Everything the planner asks about the hardware it is planning for.
///
/// The entire impure surface, two methods, for the same reason `UX-014` put the
/// doctor's hardware questions behind a trait: this row's own deliverable is
/// that "an unchanged model runs on CPU and on three GPUs", and a test that can
/// only describe the runner it happens to be on cannot check the second half of
/// that. A three-GPU machine costs a unit struct here.
pub trait Machine {
    /// Whether this build contains the backend family at all.
    ///
    /// Independent of whether hardware is present, and asked separately because
    /// the two produce different diagnostics: a missing feature is fixed in
    /// `Cargo.toml`, a missing device is not.
    fn compiled_in(&self, kind: DeviceKind) -> bool;

    /// Whether this specific device is present and usable right now.
    fn has_device(&self, device: DeviceId) -> bool;
}

/// [`Machine`] answered by the machine this process is running on.
#[derive(Debug, Clone, Copy, Default)]
pub struct HostMachine;

impl Machine for HostMachine {
    fn compiled_in(&self, kind: DeviceKind) -> bool {
        incin_backends::detect::is_compiled_in(kind)
    }

    fn has_device(&self, device: DeviceId) -> bool {
        // `detect::probe` answers per family, not per ordinal. It reports
        // whether the family has any usable device. That is the right answer at
        // ordinal 0 and no answer at all above it, so anything higher is
        // reported absent rather than guessed present. Being wrong in this
        // direction fails a build that would have worked; being wrong in the
        // other starts a run that cannot.
        if device.ordinal() != 0 {
            return false;
        }
        incin_backends::detect::probe(device.kind()) == Some(device)
    }
}

// ============================================================================
// The plan
// ============================================================================

/// One decision the planner made, and why.
///
/// Carries a stable `code` for the same reason `cargo incin doctor`'s findings
/// do: a support workflow greps the code and a human reads the detail.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    /// Stable identifier for the kind of decision.
    pub code: &'static str,
    /// Human-readable specifics.
    pub detail: String,
}

impl Decision {
    fn new(code: &'static str, detail: impl Into<String>) -> Self {
        Self {
            code,
            detail: detail.into(),
        }
    }
}

/// What the builder decided, before anything runs.
///
/// §2: "The returned build report states the selected strategy, mesh, inserted
/// collectives, per-device memory estimate, tuning policy, and fallback
/// decisions." Of those, this row owns the device selection and the fallback
/// decisions; strategy and mesh are `DST-011`'s and are absent rather than
/// stubbed, because a field naming a strategy nothing can plan is worse than a
/// missing one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Plan {
    devices: DeviceSet,
    epochs: usize,
    loss_scaling: LossScaling,
    decisions: Vec<Decision>,
}

impl Plan {
    /// The devices this run will use.
    #[must_use]
    pub fn devices(&self) -> &DeviceSet {
        &self.devices
    }

    /// How many passes over the data [`Trainer::fit`] will make.
    #[must_use]
    pub fn epochs(&self) -> usize {
        self.epochs
    }

    /// The loss scaling policy configured for this plan.
    #[must_use]
    pub fn loss_scaling(&self) -> LossScaling {
        self.loss_scaling
    }

    /// Every decision the planner made, in the order it made them.
    #[must_use]
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }

    /// Whether this plan needs collectives to execute.
    #[must_use]
    pub fn is_multi_device(&self) -> bool {
        self.devices.is_multi_device()
    }

    /// Renders a human-readable text explanation of the plan and decisions.
    #[must_use]
    pub fn explain(&self) -> String {
        let mut out = String::new();
        out.push_str("Execution Plan:\n");
        out.push_str(&format!(
            "  • Devices: {} device(s) ({})\n",
            self.devices.len(),
            self.devices.primary().kind().name()
        ));
        out.push_str(&format!("  • Epochs: {}\n", self.epochs));
        out.push_str("  • Decisions:\n");
        for decision in &self.decisions {
            out.push_str(&format!(
                "      - [{}]: {}\n",
                decision.code, decision.detail
            ));
        }
        out
    }

    /// Renders a JSON representation of the plan.
    #[must_use]
    pub fn explain_json(&self) -> String {
        let decisions: Vec<serde_json::Value> = self
            .decisions
            .iter()
            .map(|d| serde_json::json!({ "code": d.code, "detail": d.detail }))
            .collect();
        let json = serde_json::json!({
            "devices": {
                "count": self.devices.len(),
                "primary": self.devices.primary().kind().name(),
                "is_multi_device": self.is_multi_device(),
            },
            "epochs": self.epochs,
            "decisions": decisions,
        });
        serde_json::to_string_pretty(&json).unwrap_or_default()
    }
}

// ============================================================================
// Errors
// ============================================================================

/// Why a [`Trainer`] could not be built or could not run.
///
/// Every variant is a refusal. There is deliberately no variant meaning "ran
/// somewhere other than you asked".
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum TrainError {
    /// The requested backend family is not compiled into this build.
    NotCompiledIn {
        /// The family that was asked for.
        kind: DeviceKind,
        /// The Cargo feature that would add it.
        feature: &'static str,
    },
    /// The requested backend family is compiled in, but this device is absent.
    DeviceUnavailable {
        /// The device that was asked for and is not there.
        device: DeviceId,
    },
    /// [`DevicePreference::Fastest`] found nothing at all.
    ///
    /// Reachable only in a build with no backend compiled in, since the CPU is
    /// last in the preference order and is always present when compiled.
    NoDeviceAvailable,
    /// The plan needs collectives, which `DST-005` has not built yet.
    ///
    /// A distinct variant rather than a generic "unsupported" so that the day
    /// `DST-005` lands, the thing to delete is findable.
    CollectivesUnavailable {
        /// How many devices the plan named.
        devices: usize,
    },
    /// A forward pass, backward pass, or optimizer step failed.
    Step {
        /// The epoch the failure happened in, counting from zero.
        epoch: usize,
        /// The batch within that epoch, counting from zero.
        batch: usize,
        /// What the underlying operation reported.
        message: String,
    },
}

impl core::fmt::Display for TrainError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NotCompiledIn { kind, feature } => write!(
                f,
                "this build has no {} backend; enable the `{feature}` feature",
                kind.name()
            ),
            Self::DeviceUnavailable { device } => write!(
                f,
                "{}:{} is not available on this machine",
                device.kind().name(),
                device.ordinal()
            ),
            Self::NoDeviceAvailable => {
                f.write_str("no backend is compiled into this build, so there is nowhere to run")
            }
            Self::CollectivesUnavailable { devices } => write!(
                f,
                "a {devices}-device run needs collectives, which are not implemented yet (DST-005)"
            ),
            Self::Step {
                epoch,
                batch,
                message,
            } => write!(f, "epoch {epoch}, batch {batch}: {message}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for TrainError {}

/// Attaches the position in the run to whatever went wrong.
///
/// A free function rather than a closure in the loop because it is used at four
/// different result types, and a closure would fix itself to the first.
fn at<T>(
    epoch: usize,
    batch: usize,
    result: incin_core::error::Result<T>,
) -> Result<T, TrainError> {
    result.map_err(|error| TrainError::Step {
        epoch,
        batch,
        message: error.to_string(),
    })
}

/// A device of `kind` at `ordinal`, where that names something.
///
/// `DeviceId`'s constructors are per family, and the preference walk holds a
/// `DeviceKind`. `None` for a family this build does not know how to address,
/// which is the honest answer for a `#[non_exhaustive]` enum.
fn device_at(kind: DeviceKind, ordinal: usize) -> Option<DeviceId> {
    match kind {
        DeviceKind::Cpu => (ordinal == 0).then(DeviceId::cpu),
        DeviceKind::Cuda => Some(DeviceId::cuda(ordinal)),
        // Metal resolves ordinal 0 only: the backend family has no
        // multi-ordinal device spelling yet, so higher ordinals are a
        // missing device, not a guess.
        DeviceKind::Metal => (ordinal == 0).then(|| DeviceId::metal(ordinal)),
        DeviceKind::Wgpu => Some(DeviceId::wgpu(ordinal)),
        _ => None,
    }
}

/// The Cargo feature that adds a backend family.
const fn feature_for(kind: DeviceKind) -> &'static str {
    match kind {
        DeviceKind::Cpu => "cpu",
        DeviceKind::Cuda => "cuda",
        DeviceKind::Metal => "metal",
        DeviceKind::Wgpu => "wgpu",
        // `DeviceKind` is `#[non_exhaustive]` outside `incin-core`. A family
        // added later has no feature name here, and naming the crate is more
        // useful than naming a guess.
        _ => "incin-backends",
    }
}

// ============================================================================
// The builder
// ============================================================================

/// Builds a [`Plan`], and with it a [`Trainer`].
///
/// Separate from `Trainer` because §2's example calls `.build()?`. The
/// validation happens once, before training, and its result is a report the
/// caller can read whether or not they go on to train.
#[derive(Debug, Clone)]
pub struct TrainerBuilder {
    preference: DevicePreference,
    epochs: usize,
    loss_scaling: LossScaling,
}

impl Default for TrainerBuilder {
    fn default() -> Self {
        Self {
            preference: DevicePreference::default(),
            epochs: 1,
            loss_scaling: LossScaling::None,
        }
    }
}

impl TrainerBuilder {
    /// Use exactly these devices, or fail.
    #[must_use]
    pub fn devices(mut self, devices: DeviceSet) -> Self {
        self.preference = DevicePreference::Exactly(devices);
        self
    }

    /// Resolve devices from a preference rather than naming them.
    #[must_use]
    pub fn device_preference(mut self, preference: DevicePreference) -> Self {
        self.preference = preference;
        self
    }

    /// How many passes over the data [`Trainer::fit`] should make. Defaults to
    /// one.
    #[must_use]
    pub fn epochs(mut self, epochs: usize) -> Self {
        self.epochs = epochs;
        self
    }

    /// Configures the loss scaling policy for mixed-precision training.
    #[must_use]
    pub fn loss_scaling(mut self, loss_scaling: LossScaling) -> Self {
        self.loss_scaling = loss_scaling;
        self
    }

    /// Validates the request against this machine.
    ///
    /// # Errors
    ///
    /// [`TrainError::NotCompiledIn`] or [`TrainError::DeviceUnavailable`] when
    /// the request cannot be satisfied, and [`TrainError::NoDeviceAvailable`]
    /// when a [`DevicePreference::Fastest`] resolution finds nothing. Never a
    /// substituted device.
    pub fn build(self) -> Result<Plan, TrainError> {
        self.build_on(&HostMachine)
    }

    /// [`build`](Self::build) against a given machine.
    ///
    /// The real entry point; `build` is this with [`HostMachine`]. Written that
    /// way round so the path a test exercises is the path a program runs.
    ///
    /// # Errors
    ///
    /// As [`build`](Self::build).
    pub fn build_on<M: Machine + ?Sized>(self, machine: &M) -> Result<Plan, TrainError> {
        let mut decisions = Vec::new();
        let devices = match &self.preference {
            DevicePreference::Exactly(requested) => {
                let kind = requested.primary().kind();
                if !machine.compiled_in(kind) {
                    return Err(TrainError::NotCompiledIn {
                        kind,
                        feature: feature_for(kind),
                    });
                }
                for &device in requested.devices() {
                    if !machine.has_device(device) {
                        return Err(TrainError::DeviceUnavailable { device });
                    }
                }
                decisions.push(Decision::new(
                    "devices-requested",
                    format!(
                        "{} {} device(s), named by the caller",
                        requested.len(),
                        kind.name()
                    ),
                ));
                requested.clone()
            }
            DevicePreference::Cpu => {
                let device = DeviceId::cpu();
                if !machine.compiled_in(DeviceKind::Cpu) {
                    return Err(TrainError::NotCompiledIn {
                        kind: DeviceKind::Cpu,
                        feature: "cpu",
                    });
                }
                if !machine.has_device(device) {
                    return Err(TrainError::DeviceUnavailable { device });
                }
                decisions.push(Decision::new("devices-requested", "the CPU, by request"));
                DeviceSet::cpu()
            }
            DevicePreference::Fastest => {
                let mut chosen = None;
                for &kind in FASTEST_ORDER {
                    if !machine.compiled_in(kind) {
                        decisions.push(Decision::new(
                            "family-not-compiled",
                            format!("{} skipped: not compiled into this build", kind.name()),
                        ));
                        continue;
                    }
                    let Some(device) = device_at(kind, 0) else {
                        continue;
                    };
                    if !machine.has_device(device) {
                        decisions.push(Decision::new(
                            "family-unavailable",
                            format!("{} skipped: compiled in but not present", kind.name()),
                        ));
                        continue;
                    }
                    chosen = Some(device);
                    break;
                }
                let device = chosen.ok_or(TrainError::NoDeviceAvailable)?;
                decisions.push(Decision::new(
                    "devices-resolved",
                    format!(
                        "{}:{} chosen as the fastest available family",
                        device.kind().name(),
                        device.ordinal()
                    ),
                ));
                // A one-device set from a resolved device cannot be empty, a
                // duplicate, or mixed.
                DeviceSet::new([device]).unwrap_or_else(|_| DeviceSet::cpu())
            }
            // `DevicePreference` is `#[non_exhaustive]` outside `incin-core`. A
            // variant added later has no resolution rule here, and refusing is
            // the only safe answer: the alternative is running somewhere the
            // caller did not ask for, which is the thing this module exists to
            // prevent.
            _ => return Err(TrainError::NoDeviceAvailable),
        };

        if devices.is_multi_device() {
            decisions.push(Decision::new(
                "collectives-required",
                format!(
                    "{} devices need collectives; DST-005 has not built them, so this plan \
                     describes a run it cannot execute",
                    devices.len()
                ),
            ));
        }
        decisions.push(Decision::new(
            "epochs",
            format!("{} pass(es) over the data", self.epochs),
        ));
        decisions.push(Decision::new(
            "loss-scaling",
            format!("{:?} policy configured", self.loss_scaling),
        ));

        Ok(Plan {
            devices,
            epochs: self.epochs,
            loss_scaling: self.loss_scaling,
            decisions,
        })
    }
}

// ============================================================================
// The trainer
// ============================================================================

/// What a completed [`Trainer::fit`] observed.
#[derive(Debug, Clone, PartialEq)]
pub struct FitOutcome {
    /// How many epochs ran.
    pub epochs: usize,
    /// How many batches were stepped in total, across all epochs.
    pub batches: usize,
    /// The loss of the last batch of the last epoch, if there was one.
    ///
    /// `None` for an empty dataset, which is not an error. It is a dataset
    /// with no batches, and reporting zero batches and no loss says so more
    /// honestly than a loss of `0.0`.
    pub final_loss: Option<f32>,
}

/// The automatic trainer from §2.
///
/// Owns the loop for forward, loss, backward, and step, plus the [`Plan`] that says
/// where it will run. The loss itself stays in caller code, passed to
/// [`fit`](Self::fit): what a model's loss is cannot be derived from the model,
/// and a trainer that guessed would be guessing at the one thing training is.
#[derive(Debug, Clone)]
pub struct Trainer {
    plan: Plan,
}

impl Trainer {
    /// Starts a builder.
    ///
    /// §2's example writes `Trainer::new(model, optimizer)`. The model and
    /// optimizer are not builder state here because nothing in the *plan*
    /// depends on them. Planning is about devices, and taking them early
    /// would mean the builder's type parameters propagated into every error
    /// this module returns.
    #[must_use]
    pub fn plan() -> TrainerBuilder {
        TrainerBuilder::default()
    }

    /// Wraps an already-built plan.
    #[must_use]
    pub fn new(plan: Plan) -> Self {
        Self { plan }
    }

    /// The plan this trainer was built with.
    #[must_use]
    pub fn report(&self) -> &Plan {
        &self.plan
    }

    /// Runs the training loop.
    ///
    /// `data` is re-iterated once per epoch, which is why it is `Clone`. A
    /// `&DataLoader` is the intended argument and is `Copy`.
    ///
    /// `loss` receives the model and one batch and returns the scalar to
    /// differentiate. It is a closure rather than a trait method because the
    /// loss is the one part of a training step that is genuinely the caller's.
    ///
    /// # Errors
    ///
    /// [`TrainError::CollectivesUnavailable`] if the plan names more than one
    /// device, and [`TrainError::Step`] carrying the epoch and batch if a
    /// forward pass, backward pass, or optimizer step fails.
    pub fn fit<B, M, O, D, Batch, F>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        data: D,
        mut loss: F,
    ) -> Result<FitOutcome, TrainError>
    where
        B: Backend + VariableBackend + AutogradBackend + HostInterop,
        O: Optimizer<B>,
        D: IntoIterator<Item = Batch> + Clone,
        F: FnMut(
            &mut M,
            Batch,
        ) -> incin_core::error::Result<
            Tensor<incin_core::shapes::Nil, B, f32, incin_core::tensor::grad::Grad>,
        >,
    {
        if self.plan.is_multi_device() {
            return Err(TrainError::CollectivesUnavailable {
                devices: self.plan.devices.len(),
            });
        }

        let mut batches = 0;
        let mut final_loss = None;
        for epoch in 0..self.plan.epochs {
            for (batch, item) in data.clone().into_iter().enumerate() {
                let value = at(epoch, batch, loss(model, item))?;
                let grads = at(epoch, batch, value.backward())?;
                at(epoch, batch, optimizer.step(&grads))?;
                final_loss = Some(at(epoch, batch, value.to_scalar::<f32>())?);
                batches += 1;
            }
        }

        Ok(FitOutcome {
            epochs: self.plan.epochs,
            batches,
            final_loss,
        })
    }

    /// Runs the training loop with mixed-precision loss scaling.
    ///
    /// Scales the computed loss before the backward pass, checks gradients for
    /// non-finite overflow (NaN/Inf), unscales gradients in-place, and steps
    /// the optimizer.
    ///
    /// # Errors
    ///
    /// As [`fit`](Self::fit).
    pub fn fit_scaled<B, M, O, D, Batch, F>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        scaler: &mut LossScaleState,
        data: D,
        mut loss: F,
    ) -> Result<FitOutcome, TrainError>
    where
        B: Backend
            + VariableBackend
            + AutogradBackend
            + HostInterop
            + incin_core::backend_authoring::Execute<incin_core::exec::catalog::op::MulScalar>
            + incin_core::optim::OptimizerBackend<f32>,
        <B as incin_core::backend_authoring::Execute<incin_core::exec::catalog::op::MulScalar>>::Output:
            Into<<B as incin_core::backend_authoring::StorageBackend>::Storage<f32>>,
        O: ScaledOptimizer<B>,
        D: IntoIterator<Item = Batch> + Clone,
        F: FnMut(
            &mut M,
            Batch,
        ) -> incin_core::error::Result<
            Tensor<incin_core::shapes::Nil, B, f32, incin_core::tensor::grad::Grad>,
        >,
    {
        if self.plan.is_multi_device() {
            return Err(TrainError::CollectivesUnavailable {
                devices: self.plan.devices.len(),
            });
        }

        let mut batches = 0;
        let mut final_loss = None;
        for epoch in 0..self.plan.epochs {
            for (batch, item) in data.clone().into_iter().enumerate() {
                let unscaled_loss_tensor = at(epoch, batch, loss(model, item))?;
                let current_scale = scaler.scale();
                let loss_for_backward = if (current_scale - 1.0).abs() > f32::EPSILON {
                    at(
                        epoch,
                        batch,
                        unscaled_loss_tensor.mul_scalar(current_scale as f64),
                    )?
                } else {
                    unscaled_loss_tensor.clone()
                };
                let mut grads = at(epoch, batch, loss_for_backward.backward())?;
                let _stepped = at(epoch, batch, optimizer.step_scaled(&mut grads, scaler))?;
                final_loss = Some(at(epoch, batch, unscaled_loss_tensor.to_scalar::<f32>())?);
                batches += 1;
            }
        }

        Ok(FitOutcome {
            epochs: self.plan.epochs,
            batches,
            final_loss,
        })
    }
}

#[cfg(test)]
mod device_order_tests {
    use super::*;

    /// The planner's family order must mirror detection's preference order.
    /// The two constants drifted once: detection ranked Metal ahead of WGPU
    /// while `Fastest` skipped Metal entirely, so a macOS machine with both
    /// features enabled planned onto WGPU. This test is the drift alarm.
    #[test]
    fn fastest_order_mirrors_detection_preference() {
        assert_eq!(FASTEST_ORDER, incin_backends::detect::PREFERENCE);
    }

    #[test]
    fn metal_resolves_only_ordinal_zero() {
        assert!(device_at(DeviceKind::Metal, 0).is_some());
        assert!(device_at(DeviceKind::Metal, 1).is_none());
    }

    #[test]
    fn metal_names_its_feature() {
        assert_eq!(feature_for(DeviceKind::Metal), "metal");
    }
}
