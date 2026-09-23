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
//! and `DST-011`. Multi-device *execution* needs per-batch gradient
//! synchronization: [`Trainer::fit`] refuses a multi-device plan with no
//! synchronizer attached ([`TrainError::CollectivesUnavailable`]) and one
//! whose attached synchronizer disagrees with the device count
//! ([`TrainError::SynchronizerMismatch`]); with a matching synchronizer it
//! runs, reducing gradients after every backward pass. No synchronizer
//! shipped here speaks to real ranks - the hardware-gated gap (NCCL adapter
//! wiring, device-resident buckets, per-rank data sharding, buffer
//! synchronization) is documented on [`GradientSynchronizer`] and
//! `incin_core::dist::sync`.
//!
//! Under the `distributed` feature a plan can also request
//! `ShardingSpec::Fsdp` (`#99`): `fit` then reduce-scatters (ZeRO-2) or
//! all-reduces-then-masks (ZeRO-1) gradients, steps, and all-gathers
//! parameters back into a full replica each step - through
//! `Trainer::with_fsdp_synchronizer`, refused before the first batch if
//! absent or mismatched. ZeRO-3 is refused at build
//! (`TrainError::UnsupportedShardingStage`): parameter-sharded execution
//! is not implemented, and a plan that describes a run this trainer cannot
//! execute is exactly what this module exists to prevent.

use incin_core::backend_authoring::Backend;
use incin_core::backend_authoring::{AutogradBackend, HostInterop, VariableBackend};
#[cfg(feature = "distributed")]
use incin_core::dist::fsdp::ZeROStage;
use incin_core::dist::sync::{
    FsdpSynchronizer, GradientSynchronizer, SyncError, all_reduce_model_gradients,
};
#[cfg(feature = "distributed")]
use incin_core::dist::sync::{
    all_gather_model_parameters, mask_gradients_to_owned_shard, reduce_scatter_model_gradients,
};
use incin_core::exec::{
    ExecutionPolicy, LossScaleState, LossScaling, PrecisionChoice, RuntimePrecisionPolicy,
};
use incin_core::nn::VisitParameters;
use incin_core::optim::{Optimizer, ScaledOptimizer};
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::{DeviceId, DeviceKind, DevicePreference, DeviceSet};
use incin_core::tensor::dtype::{ConstDType, DTypeDescriptor, f16};
use std::sync::Arc;

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

/// How parameters and gradients are sharded across the data-parallel axis (#99).
///
/// Set with [`TrainerBuilder::sharding`]; the plan records the choice as a
/// [`Decision`], and [`Trainer::fit`] executes it when an
/// [`FsdpSynchronizer`] is attached. Execution-only vocabulary: the
/// planning reports that describe sharded memory in detail
/// (`incin::experimental::distributed::FsdpPlan`) stay separate.
#[cfg(feature = "distributed")]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShardingSpec {
    /// Fully sharded data parallelism (ZeRO) over the data-parallel axis.
    ///
    /// Every step: gradients are reduced to this rank's owned slice
    /// (reduce-scatter for [`ZeROStage::ZeRO2`]; all-reduce then mask for
    /// [`ZeROStage::ZeRO1`]), the optimizer steps, and all parameters are
    /// all-gathered back into a full replica.
    ///
    /// [`ZeROStage::ZeRO3`] cannot execute here - parameters keep full
    /// storage on every rank - and is refused at build with
    /// [`TrainError::UnsupportedShardingStage`] rather than run as ZeRO-2.
    Fsdp {
        /// Which ZeRO partitioning stage to execute.
        stage: ZeROStage,
    },
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
    precision: RuntimePrecisionPolicy,
    decisions: Vec<Decision>,
    #[cfg(feature = "distributed")]
    sharding: Option<ShardingSpec>,
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

    /// Creates fresh loss scaling state from this plan's effective policy (`UX-001`).
    ///
    /// [`TrainerBuilder::precision`] supplies the default; a later
    /// [`TrainerBuilder::loss_scaling`] overrides it. Keep this state across
    /// [`Trainer::fit_scaled`] calls to preserve dynamic growth and backoff.
    /// This does not enable autocasting or change f32 parameter storage.
    #[must_use]
    pub fn loss_scale_state(&self) -> LossScaleState {
        LossScaleState::new(self.loss_scaling)
    }

    /// The runtime precision policy configured for this plan.
    ///
    /// `fit` and `fit_scaled` make it the ambient precision for the duration
    /// of the run; nothing else enforces it.
    #[must_use]
    pub fn precision(&self) -> RuntimePrecisionPolicy {
        self.precision
    }

    /// Every decision the planner made, in the order it made them.
    #[must_use]
    pub fn decisions(&self) -> &[Decision] {
        &self.decisions
    }

    /// The sharding strategy this plan executes, if any (#99).
    #[cfg(feature = "distributed")]
    #[must_use]
    pub fn sharding(&self) -> Option<ShardingSpec> {
        self.sharding
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
        #[cfg(feature = "distributed")]
        if let Some(ShardingSpec::Fsdp { stage }) = self.sharding {
            out.push_str(&format!("  • Sharding: FSDP / {stage:?}\n"));
        }
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
        // The `sharding` key exists only under `distributed`, so the
        // binding is mutated only there.
        #[cfg_attr(not(feature = "distributed"), allow(unused_mut))]
        let mut json = serde_json::json!({
            "devices": {
                "count": self.devices.len(),
                "primary": self.devices.primary().kind().name(),
                "is_multi_device": self.is_multi_device(),
            },
            "epochs": self.epochs,
            "decisions": decisions,
        });
        #[cfg(feature = "distributed")]
        if let Some(ShardingSpec::Fsdp { stage }) = self.sharding {
            json["sharding"] =
                serde_json::json!({ "strategy": "fsdp", "stage": format!("{stage:?}") });
        }
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
    /// The plan needs collectives and no gradient synchronizer is attached.
    ///
    /// [`Trainer::fit`] refuses a multi-device plan unless a synchronizer
    /// was attached with [`Trainer::with_synchronizer`]; this variant is
    /// that refusal, so a three-GPU request can never quietly train on one
    /// GPU. `DST-005` names the transports a real multi-rank synchronizer
    /// would be built on.
    CollectivesUnavailable {
        /// How many devices the plan named.
        devices: usize,
    },
    /// An attached [`GradientSynchronizer`] reports a world size that is
    /// not the plan's device count.
    ///
    /// A three-device plan backed by a two-rank synchronizer would attempt
    /// two ranks' worth of agreement for three shards of work, which is a
    /// hang or a wrong answer rather than an error, so the counts must
    /// match before the first batch.
    SynchronizerMismatch {
        /// How many devices the plan named.
        devices: usize,
        /// The world size the synchronizer reported.
        world_size: usize,
    },
    /// The plan asks for a ZeRO stage whose execution is not implemented (#99).
    ///
    /// [`ZeROStage::ZeRO3`] needs parameters sharded on every rank -
    /// persistent storage the trainer has no mechanism to shard or free -
    /// so [`TrainerBuilder::build`] refuses it at build time. Refusing is
    /// the contract: running ZeRO-3 as if it were ZeRO-2 would be a
    /// silent approximation of the one stage whose whole point is
    /// parameter memory.
    #[cfg(feature = "distributed")]
    UnsupportedShardingStage {
        /// The stage that was requested and cannot execute.
        stage: ZeROStage,
    },
    /// The plan is FSDP-sharded and no FSDP synchronizer is attached.
    ///
    /// A sharded plan needs the reduce-scatter/all-gather seam
    /// ([`Trainer::with_fsdp_synchronizer`]); without one there is no
    /// reduction to run, so [`Trainer::fit`] refuses before the first
    /// batch instead of stepping on rank-local gradients.
    #[cfg(feature = "distributed")]
    FsdpUnavailable {
        /// How many devices the plan named.
        devices: usize,
    },
    /// A mixed-f16 plan disabled loss scaling (`UX-001`, issue #2).
    ///
    /// The f16 active / exact-f32 accumulator contract requires scaling to
    /// protect small gradients from underflow, with non-finite detection and
    /// backoff handling overflow in dynamic mode. This is a planning safeguard,
    /// not a claim that the trainer performs f16 computation or autocasting.
    #[non_exhaustive]
    UnsupportedPrecision {
        /// The active dtype requested by the precision policy.
        active_dtype: DTypeDescriptor,
        /// The exact accumulator dtype requested by the precision policy.
        accumulator: DTypeDescriptor,
    },
    /// A forward pass, backward pass, gradient synchronization, or
    /// optimizer step failed.
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
                "a {devices}-device run needs gradient synchronization; attach a synchronizer \
                 with Trainer::with_synchronizer (DST-005 transports are not wired)"
            ),
            Self::SynchronizerMismatch {
                devices,
                world_size,
            } => write!(
                f,
                "gradient synchronizer world size {world_size} does not match the plan's \
                 {devices} device(s)"
            ),
            #[cfg(feature = "distributed")]
            Self::UnsupportedShardingStage { stage } => write!(
                f,
                "the plan asks for {stage:?}, whose sharded execution is not implemented; \
                 ZeRO-3 needs parameter sharding this trainer refuses to approximate"
            ),
            #[cfg(feature = "distributed")]
            Self::FsdpUnavailable { devices } => write!(
                f,
                "a {devices}-device FSDP-sharded run needs the FSDP synchronizer; attach one \
                 with Trainer::with_fsdp_synchronizer"
            ),
            Self::UnsupportedPrecision {
                active_dtype,
                accumulator,
            } => write!(
                f,
                "trainer plan requires loss scaling for active dtype {active_dtype:?} with exact accumulator {accumulator:?}"
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
    precision: RuntimePrecisionPolicy,
    #[cfg(feature = "distributed")]
    sharding: Option<ShardingSpec>,
}

impl Default for TrainerBuilder {
    fn default() -> Self {
        Self {
            preference: DevicePreference::default(),
            epochs: 1,
            loss_scaling: LossScaling::None,
            precision: RuntimePrecisionPolicy::default(),
            #[cfg(feature = "distributed")]
            sharding: None,
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

    /// Configures the effective loss scaling policy for mixed-precision training.
    ///
    /// Overrides the default from an earlier [`precision`](Self::precision) call.
    /// Disabling scaling for a mixed-f16 plan is rejected at build time.
    #[must_use]
    pub fn loss_scaling(mut self, loss_scaling: LossScaling) -> Self {
        self.loss_scaling = loss_scaling;
        self
    }

    /// Configures the runtime precision policy (e.g. AMP, mixed-bf16, fp32).
    ///
    /// Replaces any earlier loss scaling setting with this policy's default.
    /// The policy remains inspectable; this does not enable module autocasting.
    #[must_use]
    pub fn precision(mut self, precision: RuntimePrecisionPolicy) -> Self {
        self.precision = precision;
        self.loss_scaling = precision.loss_scaling();
        self
    }

    /// Configures the runtime precision policy.
    #[must_use]
    pub fn with_precision(self, precision: RuntimePrecisionPolicy) -> Self {
        self.precision(precision)
    }

    /// Shards parameters and gradients across the data-parallel axis (#99).
    ///
    /// [`ShardingSpec::Fsdp`] makes [`Trainer::fit`] execute
    /// reduce-scatter/mask plus parameter all-gather each step, which
    /// requires [`Trainer::with_fsdp_synchronizer`] to be attached before
    /// the first batch. [`ZeROStage::ZeRO3`] is refused by
    /// [`build`](Self::build) with
    /// [`TrainError::UnsupportedShardingStage`].
    ///
    /// Records a [`Decision`] so the plan says what will shard.
    #[cfg(feature = "distributed")]
    #[must_use]
    pub fn sharding(mut self, sharding: ShardingSpec) -> Self {
        self.sharding = Some(sharding);
        self
    }

    /// Validates the request against this machine.
    ///
    /// # Errors
    ///
    /// [`TrainError::NotCompiledIn`] or [`TrainError::DeviceUnavailable`] when
    /// the request cannot be satisfied, and [`TrainError::NoDeviceAvailable`]
    /// when a [`DevicePreference::Fastest`] resolution finds nothing. Never a
    /// substituted device. [`TrainError::UnsupportedPrecision`] if an f16-active
    /// policy with an exact-f32 accumulator has effective [`LossScaling::None`].
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
        if self.precision.active_dtype() == Some(<f16 as ConstDType>::DESCRIPTOR)
            && self.precision.accumulator()
                == PrecisionChoice::Exact(<f32 as ConstDType>::DESCRIPTOR)
            && self.loss_scaling == LossScaling::None
        {
            return Err(TrainError::UnsupportedPrecision {
                active_dtype: <f16 as ConstDType>::DESCRIPTOR,
                accumulator: <f32 as ConstDType>::DESCRIPTOR,
            });
        }

        #[cfg(feature = "distributed")]
        if let Some(ShardingSpec::Fsdp {
            stage: ZeROStage::ZeRO3,
        }) = self.sharding
        {
            // ZeRO-3's parameter-sharded execution does not exist; refusing
            // at build keeps a plan that cannot execute from ever
            // describing a run.
            return Err(TrainError::UnsupportedShardingStage {
                stage: ZeROStage::ZeRO3,
            });
        }

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
                    "{} devices need collectives; attach a gradient synchronizer with \
                     Trainer::with_synchronizer to execute this plan (DST-005 transports are \
                     not wired, so without one this plan describes a run it cannot execute)",
                    devices.len()
                ),
            ));
        }
        #[cfg(feature = "distributed")]
        if let Some(ShardingSpec::Fsdp { stage }) = self.sharding {
            decisions.push(Decision::new(
                "fsdp-sharding",
                match stage {
                    ZeROStage::ZeRO1 => format!(
                        "{stage:?} over the data-parallel axis: all-reduce then mask gradients \
                         to the owned slice, all-gather parameters after each step"
                    ),
                    // ZeRO-3 was refused above.
                    _ => format!(
                        "{stage:?} over the data-parallel axis: reduce-scatter gradients to \
                         the owned slice, all-gather parameters after each step"
                    ),
                },
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
        decisions.push(Decision::new(
            "precision",
            format!("{:?} runtime precision policy configured", self.precision),
        ));

        Ok(Plan {
            devices,
            epochs: self.epochs,
            loss_scaling: self.loss_scaling,
            precision: self.precision,
            decisions,
            #[cfg(feature = "distributed")]
            sharding: self.sharding,
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
///
/// An optional [`GradientSynchronizer`] attached with
/// [`with_synchronizer`](Self::with_synchronizer) runs after every backward
/// pass; without one, a multi-device plan is refused rather than faked.
/// A plan sharded with `ShardingSpec::Fsdp` takes the parallel route
/// through `with_fsdp_synchronizer` instead (#99).
#[derive(Debug, Clone)]
pub struct Trainer {
    plan: Plan,
    synchronizer: Option<Arc<dyn GradientSynchronizer>>,
    #[cfg(feature = "distributed")]
    fsdp: Option<Arc<dyn FsdpSynchronizer>>,
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
        Self {
            plan,
            synchronizer: None,
            #[cfg(feature = "distributed")]
            fsdp: None,
        }
    }

    /// The plan this trainer was built with.
    #[must_use]
    pub fn report(&self) -> &Plan {
        &self.plan
    }

    /// Attaches a gradient synchronizer used by [`fit`](Self::fit) and
    /// [`fit_scaled`](Self::fit_scaled).
    ///
    /// Every backward pass is followed by
    /// [`all_reduce_model_gradients`] against this synchronizer, before the
    /// optimizer step. The synchronizer's
    /// [`world size`](GradientSynchronizer::world_size) must equal the
    /// plan's device count or the run is refused with
    /// [`TrainError::SynchronizerMismatch`] - checked before the first
    /// batch, not discovered mid-collective.
    ///
    /// # Hardware-gated gap
    ///
    /// No transport-backed synchronizer ships with this crate: a real
    /// multi-rank run needs an implementation wired to a collective
    /// backend (`DST-005`'s transports), which is not runnable on this
    /// project's current hardware. [`SingleRankSynchronizer`] is the
    /// proven path, and the two-rank arithmetic is proven in the
    /// `dp2_network` tests against scripted peers.
    #[must_use]
    pub fn with_synchronizer(mut self, synchronizer: impl GradientSynchronizer + 'static) -> Self {
        self.synchronizer = Some(Arc::new(synchronizer));
        self
    }

    /// The attached gradient synchronizer, if any.
    #[must_use]
    pub fn synchronizer(&self) -> Option<&dyn GradientSynchronizer> {
        self.synchronizer.as_deref()
    }

    /// Attaches the FSDP synchronizer a sharded plan executes against (#99).
    ///
    /// Required by [`fit`](Self::fit) and [`fit_scaled`](Self::fit_scaled)
    /// when the plan carries [`ShardingSpec::Fsdp`]; a plan without one is
    /// refused with [`TrainError::FsdpUnavailable`] before the first
    /// batch. Its [`world size`](GradientSynchronizer::world_size) must
    /// equal the plan's device count or the run is refused with
    /// [`TrainError::SynchronizerMismatch`], exactly like
    /// [`with_synchronizer`](Self::with_synchronizer).
    ///
    /// Each step uses this synchronizer's reduce-scatter (ZeRO-2) or
    /// all-reduce-plus-mask (ZeRO-1) and parameter all-gather, in place of
    /// the plain all-reduce seam. [`SingleRankSynchronizer`] implements
    /// this trait as the identity, so a one-device sharded plan exercises
    /// the same walk a multi-rank run takes.
    ///
    /// # Hardware-gated gap
    ///
    /// As with [`with_synchronizer`](Self::with_synchronizer): no
    /// transport-backed implementation ships here. The protocol and the
    /// one-rank identity path are proven; multi-rank adapters wired to a
    /// collective backend are not runnable on this project's current
    /// hardware.
    #[cfg(feature = "distributed")]
    #[must_use]
    pub fn with_fsdp_synchronizer(mut self, synchronizer: impl FsdpSynchronizer + 'static) -> Self {
        self.fsdp = Some(Arc::new(synchronizer));
        self
    }

    /// The attached FSDP synchronizer, if any.
    #[cfg(feature = "distributed")]
    #[must_use]
    pub fn fsdp_synchronizer(&self) -> Option<&dyn FsdpSynchronizer> {
        self.fsdp.as_deref()
    }

    /// Checks the attached synchronizer against this plan before any
    /// batch runs.
    ///
    /// A multi-device plan with no synchronizer is
    /// [`CollectivesUnavailable`](TrainError::CollectivesUnavailable);
    /// an attached synchronizer whose world size is not the device count
    /// is [`SynchronizerMismatch`](TrainError::SynchronizerMismatch). An
    /// FSDP-sharded plan is validated against its FSDP synchronizer
    /// instead - absent is `TrainError::FsdpUnavailable` - and the plain
    /// seam, if also attached, still has to agree. The rank-inside-world
    /// check lives in [`all_reduce_model_gradients`] and the FSDP walks
    /// and surfaces as a [`TrainError::Step`] at the first batch.
    fn validate_synchronizer(&self) -> Result<(), TrainError> {
        let devices = self.plan.devices.len();
        #[cfg(feature = "distributed")]
        let fsdp_sharded = matches!(self.plan.sharding, Some(ShardingSpec::Fsdp { .. }));
        #[cfg(not(feature = "distributed"))]
        let fsdp_sharded = false;
        #[cfg(feature = "distributed")]
        if fsdp_sharded {
            match &self.fsdp {
                None => return Err(TrainError::FsdpUnavailable { devices }),
                Some(sync) if sync.world_size() != devices => {
                    return Err(TrainError::SynchronizerMismatch {
                        devices,
                        world_size: sync.world_size(),
                    });
                }
                Some(_) => {}
            }
        }
        match &self.synchronizer {
            // A sharded run takes its collective through the FSDP seam, so
            // the plain seam stays optional there even on many devices.
            None if self.plan.is_multi_device() && !fsdp_sharded => {
                Err(TrainError::CollectivesUnavailable { devices })
            }
            Some(sync) if sync.world_size() != devices => Err(TrainError::SynchronizerMismatch {
                devices,
                world_size: sync.world_size(),
            }),
            _ => Ok(()),
        }
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
    /// Each step runs inside an [`ExecutionPolicy`] scope holding this plan's
    /// [`RuntimePrecisionPolicy`], so every `ExecutionContext` the step builds
    /// carries the plan's precision and the caller's ambient policy is
    /// restored when this returns, errors included. The scope casts nothing:
    /// autocasting from an allowlist is not implemented.
    ///
    /// When a synchronizer is attached, gradients are reduced after the
    /// backward pass and before the optimizer step, via
    /// [`all_reduce_model_gradients`]. When the plan carries
    /// `ShardingSpec::Fsdp` (#99), the step instead reduce-scatters
    /// (ZeRO-2) or all-reduces-then-masks (ZeRO-1) gradients through the
    /// FSDP synchronizer, steps, and all-gathers parameters back into a
    /// full replica. This function does not shard `data` across ranks or
    /// migrate the model onto the plan's devices: those are the
    /// hardware-gated gaps documented on [`GradientSynchronizer`].
    ///
    /// # Errors
    ///
    /// [`TrainError::CollectivesUnavailable`] if the plan names more than
    /// one device and no synchronizer is attached,
    /// [`TrainError::SynchronizerMismatch`] if an attached synchronizer's
    /// world size is not the plan's device count, `FsdpUnavailable` if
    /// the plan is FSDP-sharded and no FSDP synchronizer is attached, and
    /// [`TrainError::Step`] carrying the epoch and batch if a forward
    /// pass, backward pass, gradient synchronization, parameter
    /// all-gather, or optimizer step fails.
    pub fn fit<B, M, O, D, Batch, F>(
        &self,
        model: &mut M,
        optimizer: &mut O,
        data: D,
        mut loss: F,
    ) -> Result<FitOutcome, TrainError>
    where
        B: Backend + VariableBackend + AutogradBackend + HostInterop,
        M: VisitParameters<B>,
        O: Optimizer<B>,
        D: IntoIterator<Item = Batch> + Clone,
        F: FnMut(
            &mut M,
            Batch,
        ) -> incin_core::error::Result<
            Tensor<incin_core::shapes::Nil, B, f32, incin_core::tensor::grad::Grad>,
        >,
    {
        self.validate_synchronizer()?;
        #[cfg(feature = "distributed")]
        let fsdp = match self.plan.sharding {
            Some(ShardingSpec::Fsdp { stage }) => Some((
                stage,
                self.fsdp.as_deref().ok_or(TrainError::FsdpUnavailable {
                    devices: self.plan.devices.len(),
                })?,
            )),
            None => None,
        };

        ExecutionPolicy::current()
            .with_precision(self.plan.precision)
            .scope(|| {
                let mut batches = 0;
                let mut final_loss = None;
                for epoch in 0..self.plan.epochs {
                    for (batch, item) in data.clone().into_iter().enumerate() {
                        let value = at(epoch, batch, loss(model, item))?;
                        let mut grads = at(epoch, batch, value.backward())?;
                        #[cfg(feature = "distributed")]
                        let mut sharded_step = false;
                        #[cfg(not(feature = "distributed"))]
                        let sharded_step = false;
                        #[cfg(feature = "distributed")]
                        if let Some((stage, fsdp)) = fsdp {
                            let reduced = match stage {
                                ZeROStage::ZeRO1 => {
                                    all_reduce_model_gradients(model, &mut grads, fsdp).and_then(
                                        |()| mask_gradients_to_owned_shard(model, &mut grads, fsdp),
                                    )
                                }
                                ZeROStage::ZeRO2 => {
                                    reduce_scatter_model_gradients(model, &mut grads, fsdp)
                                        .map(|_owned| ())
                                }
                                // `build` refuses this stage; the second
                                // check keeps a future construction path
                                // from running it as ZeRO-2.
                                ZeROStage::ZeRO3 => {
                                    return Err(TrainError::UnsupportedShardingStage { stage });
                                }
                            };
                            if let Err(error) = reduced {
                                return Err(TrainError::Step {
                                    epoch,
                                    batch,
                                    message: error.to_string(),
                                });
                            }
                            at(epoch, batch, optimizer.step(&grads))?;
                            if let Err(error) = all_gather_model_parameters(model, fsdp) {
                                return Err(TrainError::Step {
                                    epoch,
                                    batch,
                                    message: error.to_string(),
                                });
                            }
                            sharded_step = true;
                        }
                        if !sharded_step {
                            if let Some(sync) = self.synchronizer.as_deref()
                                && let Err(error) =
                                    all_reduce_model_gradients(model, &mut grads, sync)
                            {
                                return Err(TrainError::Step {
                                    epoch,
                                    batch,
                                    message: error.to_string(),
                                });
                            }
                            at(epoch, batch, optimizer.step(&grads))?;
                        }
                        final_loss = Some(at(epoch, batch, value.to_scalar::<f32>())?);
                        batches += 1;
                    }
                }

                Ok(FitOutcome {
                    epochs: self.plan.epochs,
                    batches,
                    final_loss,
                })
            })
    }

    /// Runs the training loop with mixed-precision loss scaling.
    ///
    /// Scales the computed loss before the backward pass, checks gradients for
    /// non-finite overflow (NaN/Inf), unscales gradients in-place, and steps
    /// the optimizer.
    ///
    /// The loop body runs under the same plan-precision [`ExecutionPolicy`]
    /// scope as [`fit`](Self::fit).
    ///
    /// When a synchronizer is attached, gradients are reduced after the
    /// scaled backward pass and before [`ScaledOptimizer::step_scaled`]
    /// unscales them. The loss scale is uniform, and a mean commutes with
    /// a uniform scale factor, so ranks only agree if they run identical
    /// [`LossScaleState`] policies - keep them in lockstep yourself.
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
        M: VisitParameters<B>,
        O: ScaledOptimizer<B>,
        D: IntoIterator<Item = Batch> + Clone,
        F: FnMut(
            &mut M,
            Batch,
        ) -> incin_core::error::Result<
            Tensor<incin_core::shapes::Nil, B, f32, incin_core::tensor::grad::Grad>,
        >,
    {
        self.validate_synchronizer()?;
        #[cfg(feature = "distributed")]
        let fsdp = match self.plan.sharding {
            Some(ShardingSpec::Fsdp { stage }) => Some((
                stage,
                self.fsdp.as_deref().ok_or(TrainError::FsdpUnavailable {
                    devices: self.plan.devices.len(),
                })?,
            )),
            None => None,
        };

        ExecutionPolicy::current()
            .with_precision(self.plan.precision)
            .scope(|| {
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
                                unscaled_loss_tensor
                                    .mul_scalar(current_scale as f64)
                                    .map(|scaled| scaled.forget_layout()),
                            )?
                        } else {
                            unscaled_loss_tensor.clone()
                        };
                        let mut grads = at(epoch, batch, loss_for_backward.backward())?;
                        #[cfg(feature = "distributed")]
                        let mut sharded_step = false;
                        #[cfg(not(feature = "distributed"))]
                        let sharded_step = false;
                        #[cfg(feature = "distributed")]
                        if let Some((stage, fsdp)) = fsdp {
                            let reduced = match stage {
                                ZeROStage::ZeRO1 => {
                                    all_reduce_model_gradients(model, &mut grads, fsdp).and_then(
                                        |()| mask_gradients_to_owned_shard(model, &mut grads, fsdp),
                                    )
                                }
                                ZeROStage::ZeRO2 => {
                                    reduce_scatter_model_gradients(model, &mut grads, fsdp)
                                        .map(|_owned| ())
                                }
                                // `build` refuses this stage; the second
                                // check keeps a future construction path
                                // from running it as ZeRO-2.
                                ZeROStage::ZeRO3 => {
                                    return Err(TrainError::UnsupportedShardingStage { stage });
                                }
                            };
                            if let Err(error) = reduced {
                                return Err(TrainError::Step {
                                    epoch,
                                    batch,
                                    message: error.to_string(),
                                });
                            }
                            // Unscales in place; zeros stay zero, so the
                            // owned slice survives the divide.
                            let _stepped =
                                at(epoch, batch, optimizer.step_scaled(&mut grads, scaler))?;
                            // Gather runs whether or not the step
                            // committed: ranks must issue the same
                            // collectives either way (lockstep scales are
                            // the caller's responsibility, as documented).
                            if let Err(error) = all_gather_model_parameters(model, fsdp) {
                                return Err(TrainError::Step {
                                    epoch,
                                    batch,
                                    message: error.to_string(),
                                });
                            }
                            sharded_step = true;
                        }
                        if !sharded_step {
                            if let Some(sync) = self.synchronizer.as_deref()
                                && let Err(error) =
                                    all_reduce_model_gradients(model, &mut grads, sync)
                            {
                                return Err(TrainError::Step {
                                    epoch,
                                    batch,
                                    message: error.to_string(),
                                });
                            }
                            let _stepped =
                                at(epoch, batch, optimizer.step_scaled(&mut grads, scaler))?;
                        }
                        final_loss =
                            Some(at(epoch, batch, unscaled_loss_tensor.to_scalar::<f32>())?);
                        batches += 1;
                    }
                }

                Ok(FitOutcome {
                    epochs: self.plan.epochs,
                    batches,
                    final_loss,
                })
            })
    }
}

/// A [`GradientSynchronizer`] for a world of one rank: the reduction is
/// the identity, because the mean of a single contribution is that
/// contribution.
///
/// It exists so a single-process run exercises the same
/// synchronize-after-backward path a multi-rank run takes - the round trip
/// through [`all_reduce_model_gradients`] touches every gradient - and so
/// tests can assert the path is value-preserving. Attaching it to a plan
/// with more (or fewer) than one device is refused with
/// [`TrainError::SynchronizerMismatch`], exactly like any other
/// synchronizer whose world size disagrees with the plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SingleRankSynchronizer;

impl GradientSynchronizer for SingleRankSynchronizer {
    fn world_size(&self) -> usize {
        1
    }

    fn rank(&self) -> usize {
        0
    }

    fn all_reduce_mean(&self, _values: &mut [f64]) -> Result<(), SyncError> {
        Ok(())
    }
}

impl FsdpSynchronizer for SingleRankSynchronizer {
    /// The whole world's shard is this rank's: identity.
    fn reduce_scatter_mean(&self, values: &[f64]) -> Result<Vec<f64>, SyncError> {
        Ok(values.to_vec())
    }

    /// One rank's gathered sequence is its shard, unchanged.
    fn all_gather(&self, shard: &[f64]) -> Result<Vec<f64>, SyncError> {
        Ok(shard.to_vec())
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
