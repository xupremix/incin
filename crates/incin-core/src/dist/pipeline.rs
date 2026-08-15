//! Typed two-stage pipeline plans and deterministic schedules.
//!
//! The first pipeline topology is exactly two ranks. A plan describes every
//! activation transfer from stage zero to stage one and every gradient
//! transfer in the reverse direction as one global `SendRecv` operation. Both
//! processes therefore preflight identical source/destination metadata instead
//! of independently guessing whether the next local call is a send or receive.
//!
//! Fully static callers select dtype, activation shape, microbatch count, and
//! schedule through types. [`Dyn`] callers use [`PipelinePlanBuilder::build_dyn`]
//! and traverse the matching runtime validation path.

use alloc::{vec, vec::Vec};

use half::{bf16, f16};
use typenum::{B1, IsLessOrEqual, NonZero, U1, U2, U4294967295, Unsigned};

use crate::dist::collective::{CollectiveError, CollectiveKind, StreamId};
use crate::dist::mesh::{Data, DeviceMesh, MeshSpec, Pipeline, TensorParallel};
use crate::dist::plan::{
    CollectivePlan, CollectivePlanBuilder, CollectiveTag, PlanError, SequenceToken,
};
use crate::shapes::{Dyn, Shape};
use crate::shapes::error::OperationKind;
use crate::tensor::dtype::{BuiltinDType, ConstDType, DTypeId};
use crate::shapes::error::ShapeError;

/// Exactly two pipeline stages and no data or tensor partitioning.
pub type TwoRankPipeline = MeshSpec<Data<U1>, TensorParallel<U1>, Pipeline<U2>>;

/// Floating activation/gradient dtypes supported by the first PP=2 path.
///
/// Static integer and block-quantized values have no implementation. [`Dyn`]
/// is admitted so [`validate_pipeline_dtype`] can apply the same rule at
/// runtime.
pub trait PipelineDType: crate::dist::CollectiveDType {}

impl PipelineDType for f32 {}
impl PipelineDType for f64 {}
impl PipelineDType for f16 {}
impl PipelineDType for bf16 {}
impl PipelineDType for Dyn {}

/// Runtime counterpart of [`PipelineDType`].
pub const fn validate_pipeline_dtype(dtype: DTypeId) -> Result<(), PipelineError> {
    match dtype {
        DTypeId::BF16 | DTypeId::F16 | DTypeId::F32 | DTypeId::F64 => Ok(()),
        DTypeId::U8 | DTypeId::U32 | DTypeId::I64 | DTypeId::Q8_0 | DTypeId::Bool => {
            Err(PipelineError::UnsupportedDType { dtype })
        }
    }
}

/// Pipeline execution policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineSchedule {
    /// Flush all forwards, then all backwards.
    GPipe,
    /// Alternate forward and backward work after the two-stage warmup.
    OneForwardOneBackward,
}

/// Type-level schedule choice for the static planning API.
pub trait StaticPipelineSchedule: 'static {
    /// Runtime projection stored in the immutable plan.
    const SCHEDULE: PipelineSchedule;
}

/// Static marker for [`PipelineSchedule::GPipe`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GPipe;

impl StaticPipelineSchedule for GPipe {
    const SCHEDULE: PipelineSchedule = PipelineSchedule::GPipe;
}

/// Static marker for [`PipelineSchedule::OneForwardOneBackward`].
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct OneForwardOneBackward;

impl StaticPipelineSchedule for OneForwardOneBackward {
    const SCHEDULE: PipelineSchedule = PipelineSchedule::OneForwardOneBackward;
}

/// Whether stage activations remain resident or are recomputed for backward.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActivationCheckpoint {
    /// Retain each activation until its backward use.
    Keep,
    /// Permit the executor to discard and recompute stage activations.
    Recompute,
}

/// Stable identity of the stage boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PipelineBoundaryId(u32);

impl PipelineBoundaryId {
    const MAX: u64 = (1_u64 << 31) - 1;

    /// Build a nonzero identity encodable with a microbatch and direction.
    pub const fn new(value: u64) -> Result<Self, PipelineError> {
        if value == 0 {
            Err(PipelineError::ReservedBoundaryId)
        } else if value > Self::MAX {
            Err(PipelineError::BoundaryIdTooLarge {
                maximum: Self::MAX,
                found: value,
            })
        } else {
            Ok(Self(value as u32))
        }
    }

    /// Numeric caller-defined identity.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0 as u64
    }
}

/// Direction and semantic payload of one stage-boundary transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineTransfer {
    /// Stage-zero activation consumed by stage one.
    ForwardActivation,
    /// Stage-one activation gradient consumed by stage zero.
    BackwardGradient,
}

impl PipelineTransfer {
    /// Source rank in the pure PP=2 mesh.
    #[must_use]
    pub const fn source_rank(self) -> usize {
        match self {
            Self::ForwardActivation => 0,
            Self::BackwardGradient => 1,
        }
    }

    /// Destination rank in the pure PP=2 mesh.
    #[must_use]
    pub const fn destination_rank(self) -> usize {
        match self {
            Self::ForwardActivation => 1,
            Self::BackwardGradient => 0,
        }
    }

    /// Stable semantic tag included in cross-rank preflight.
    #[must_use]
    pub const fn plan_tag(self, boundary: PipelineBoundaryId, microbatch: usize) -> CollectiveTag {
        let direction = match self {
            Self::ForwardActivation => 0,
            Self::BackwardGradient => 1,
        };
        CollectiveTag::new((boundary.get() << 33) | ((microbatch as u64) << 1) | direction)
    }
}

/// Compute work occupying one stage in one logical pipeline clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelineAction {
    /// Forward compute for one microbatch.
    Forward {
        /// Zero-based microbatch index.
        microbatch: usize,
    },
    /// Backward compute for one microbatch.
    Backward {
        /// Zero-based microbatch index.
        microbatch: usize,
    },
}

/// Warmup/steady/cooldown phase of a schedule clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PipelinePhase {
    /// Fill the two-stage pipeline.
    Warmup,
    /// Both stages can make useful progress.
    Steady,
    /// Drain outstanding backward work.
    Cooldown,
}

/// One logical clock with at most one compute action per stage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineClock {
    phase: PipelinePhase,
    stages: [Option<PipelineAction>; 2],
}

impl PipelineClock {
    /// Schedule phase.
    #[must_use]
    pub const fn phase(self) -> PipelinePhase {
        self.phase
    }

    /// Action assigned to `stage`, or `None` for a bubble.
    #[must_use]
    pub const fn stage(self, stage: usize) -> Option<PipelineAction> {
        if stage < 2 { self.stages[stage] } else { None }
    }
}

/// Schedule metadata and its explicit two-stage timeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineScheduleDescriptor {
    schedule: PipelineSchedule,
    microbatches: usize,
    checkpoint: ActivationCheckpoint,
    warmup_steps: usize,
    steady_steps: usize,
    cooldown_steps: usize,
    bubble_slots: usize,
    max_live_activations: [usize; 2],
    clocks: Vec<PipelineClock>,
}

impl PipelineScheduleDescriptor {
    /// Selected schedule.
    #[must_use]
    pub const fn schedule(&self) -> PipelineSchedule {
        self.schedule
    }

    /// Number of microbatches in the step.
    #[must_use]
    pub const fn microbatches(&self) -> usize {
        self.microbatches
    }

    /// Activation-checkpoint policy.
    #[must_use]
    pub const fn checkpoint(&self) -> ActivationCheckpoint {
        self.checkpoint
    }

    /// Logical clocks in the warmup.
    #[must_use]
    pub const fn warmup_steps(&self) -> usize {
        self.warmup_steps
    }

    /// Logical clocks in the steady phase.
    #[must_use]
    pub const fn steady_steps(&self) -> usize {
        self.steady_steps
    }

    /// Logical clocks in the cooldown.
    #[must_use]
    pub const fn cooldown_steps(&self) -> usize {
        self.cooldown_steps
    }

    /// Idle stage slots in the explicit timeline.
    #[must_use]
    pub const fn bubble_slots(&self) -> usize {
        self.bubble_slots
    }

    /// Peak retained forward activations per stage before checkpoint policy.
    #[must_use]
    pub const fn max_live_activations(&self) -> [usize; 2] {
        self.max_live_activations
    }

    /// Explicit two-stage timeline.
    #[must_use]
    pub fn clocks(&self) -> &[PipelineClock] {
        &self.clocks
    }
}

/// One activation or gradient transfer in total launch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineTransferDescriptor {
    boundary: PipelineBoundaryId,
    transfer: PipelineTransfer,
    microbatch: usize,
    elements: usize,
    dtype: DTypeId,
    sequence: SequenceToken,
}

impl PipelineTransferDescriptor {
    /// Stable stage-boundary identity.
    #[must_use]
    pub const fn boundary(self) -> PipelineBoundaryId {
        self.boundary
    }

    /// Forward activation or backward gradient.
    #[must_use]
    pub const fn transfer(self) -> PipelineTransfer {
        self.transfer
    }

    /// Zero-based microbatch index.
    #[must_use]
    pub const fn microbatch(self) -> usize {
        self.microbatch
    }

    /// Elements in the activation/gradient payload.
    #[must_use]
    pub const fn elements(self) -> usize {
        self.elements
    }

    /// Static or runtime-resolved dtype.
    #[must_use]
    pub const fn dtype(self) -> DTypeId {
        self.dtype
    }

    /// Position in the shared transport plan.
    #[must_use]
    pub const fn sequence(self) -> SequenceToken {
        self.sequence
    }
}

/// Immutable PP=2 transport plan plus its compute schedule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelinePlan {
    collective: CollectivePlan,
    boundary: PipelineBoundaryId,
    activation_shape: Vec<usize>,
    schedule: PipelineScheduleDescriptor,
    transfers: Vec<PipelineTransferDescriptor>,
}

impl PipelinePlan {
    /// Transport-neutral global communication plan.
    #[must_use]
    pub const fn collective_plan(&self) -> &CollectivePlan {
        &self.collective
    }

    /// Stable stage-boundary identity.
    #[must_use]
    pub const fn boundary(&self) -> PipelineBoundaryId {
        self.boundary
    }

    /// Per-microbatch activation shape.
    #[must_use]
    pub fn activation_shape(&self) -> &[usize] {
        &self.activation_shape
    }

    /// Schedule and bubble/activation evidence.
    #[must_use]
    pub const fn schedule(&self) -> &PipelineScheduleDescriptor {
        &self.schedule
    }

    /// Activation/gradient transfers in total launch order.
    #[must_use]
    pub fn transfers(&self) -> &[PipelineTransferDescriptor] {
        &self.transfers
    }

    /// Consume the wrapper for transport bootstrap.
    #[must_use]
    pub fn into_collective_plan(self) -> CollectivePlan {
        self.collective
    }
}

/// Builder for exactly two pipeline ranks.
pub struct PipelinePlanBuilder;

impl PipelinePlanBuilder {
    /// Build a plan whose dtype, activation shape, microbatch count, and
    /// schedule are all compile-time selected.
    pub fn build_static<K, S, Microbatches, Schedule>(
        mesh: &DeviceMesh<TwoRankPipeline>,
        rank: usize,
        boundary: PipelineBoundaryId,
        checkpoint: ActivationCheckpoint,
        stream: StreamId,
    ) -> Result<PipelinePlan, PipelineError>
    where
        K: ConstDType + BuiltinDType + PipelineDType,
        S: Shape,
        S::Arg: Default,
        Microbatches: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Schedule: StaticPipelineSchedule,
    {
        let shape_field = S::resolve(Default::default()).map_err(PipelineError::Shape)?;
        let dims = shape_field.clone();
        let elements = S::STATIC_NUMEL.ok_or_else(|| {
            PipelineError::Shape(ShapeError::TargetShapeRejected {
                operation: OperationKind::Storage,
                rank: S::RANK.unwrap_or(0),
            })
        })?;
        Self::build_checked(
            mesh,
            rank,
            boundary,
            dims.as_ref(),
            elements,
            K::DTYPE,
            Microbatches::USIZE,
            Schedule::SCHEDULE,
            checkpoint,
            stream,
        )
    }

    /// Runtime-selected counterpart of [`build_static`](Self::build_static).
    #[allow(clippy::too_many_arguments)]
    pub fn build_dyn(
        mesh: &DeviceMesh<TwoRankPipeline>,
        rank: usize,
        boundary: PipelineBoundaryId,
        activation_shape: &[usize],
        dtype: DTypeId,
        microbatches: usize,
        schedule: PipelineSchedule,
        checkpoint: ActivationCheckpoint,
        stream: StreamId,
    ) -> Result<PipelinePlan, PipelineError> {
        validate_pipeline_dtype(dtype)?;
        let elements = checked_elements(activation_shape)?;
        Self::build_checked(
            mesh,
            rank,
            boundary,
            activation_shape,
            elements,
            dtype,
            microbatches,
            schedule,
            checkpoint,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn build_checked(
        mesh: &DeviceMesh<TwoRankPipeline>,
        rank: usize,
        boundary: PipelineBoundaryId,
        activation_shape: &[usize],
        elements: usize,
        dtype: DTypeId,
        microbatches: usize,
        schedule: PipelineSchedule,
        checkpoint: ActivationCheckpoint,
        stream: StreamId,
    ) -> Result<PipelinePlan, PipelineError> {
        validate_pipeline_dtype(dtype)?;
        validate_microbatches(microbatches)?;
        if microbatches > u32::MAX as usize {
            return Err(PipelineError::TooManyMicrobatches {
                maximum: u32::MAX as usize,
                found: microbatches,
            });
        }
        validate_schedule_size(microbatches)?;

        let schedule_descriptor = build_schedule(schedule, microbatches, checkpoint)?;
        let transfer_order = transfer_order(schedule, microbatches);
        let mut inner = CollectivePlanBuilder::new(mesh);
        let mut previous = None;
        let mut transfers = Vec::with_capacity(transfer_order.len());
        for (transfer, microbatch) in transfer_order {
            let source = transfer.source_rank();
            let destination = transfer.destination_rank();
            let sequence = inner.push_send_recv_tagged(
                transfer.plan_tag(boundary, microbatch),
                rank,
                elements,
                dtype,
                source,
                destination,
                source,
                destination,
                stream,
                previous,
            )?;
            transfers.push(PipelineTransferDescriptor {
                boundary,
                transfer,
                microbatch,
                elements,
                dtype,
                sequence,
            });
            previous = Some(sequence);
        }

        let collective = inner.finish();
        validate_pipeline_descriptors(&collective, boundary, &transfers)?;
        Ok(PipelinePlan {
            collective,
            boundary,
            activation_shape: activation_shape.to_vec(),
            schedule: schedule_descriptor,
            transfers,
        })
    }
}

/// Runtime check matching `Microbatches: NonZero`.
pub const fn validate_microbatches(microbatches: usize) -> Result<(), PipelineError> {
    if microbatches == 0 {
        Err(PipelineError::ZeroMicrobatches)
    } else {
        Ok(())
    }
}

fn validate_schedule_size(microbatches: usize) -> Result<(), PipelineError> {
    microbatches
        .checked_mul(2)
        .and_then(|transfers| transfers.checked_add(2))
        .ok_or(PipelineError::ScheduleLengthOverflow { microbatches })
        .map(|_| ())
}

fn checked_elements(shape: &[usize]) -> Result<usize, PipelineError> {
    let mut elements = 1_usize;
    for &extent in shape {
        elements = elements
            .checked_mul(extent)
            .ok_or(ShapeError::ArithmeticOverflow {
                operation: OperationKind::Storage,
                expression: "pipeline activation dimensions",
            })?;
    }
    Ok(elements)
}

fn transfer_order(
    schedule: PipelineSchedule,
    microbatches: usize,
) -> Vec<(PipelineTransfer, usize)> {
    let mut order = Vec::with_capacity(microbatches * 2);
    match schedule {
        PipelineSchedule::GPipe => {
            for microbatch in 0..microbatches {
                order.push((PipelineTransfer::ForwardActivation, microbatch));
            }
            for microbatch in (0..microbatches).rev() {
                order.push((PipelineTransfer::BackwardGradient, microbatch));
            }
        }
        PipelineSchedule::OneForwardOneBackward => {
            order.push((PipelineTransfer::ForwardActivation, 0));
            for microbatch in 1..microbatches {
                order.push((PipelineTransfer::ForwardActivation, microbatch));
                order.push((PipelineTransfer::BackwardGradient, microbatch - 1));
            }
            order.push((PipelineTransfer::BackwardGradient, microbatches - 1));
        }
    }
    order
}

fn build_schedule(
    schedule: PipelineSchedule,
    microbatches: usize,
    checkpoint: ActivationCheckpoint,
) -> Result<PipelineScheduleDescriptor, PipelineError> {
    let clocks = match schedule {
        PipelineSchedule::GPipe => gpipe_clocks(microbatches),
        PipelineSchedule::OneForwardOneBackward => one_f_one_b_clocks(microbatches),
    };
    validate_clocks(&clocks, microbatches)?;
    let (warmup_steps, steady_steps, cooldown_steps) = match schedule {
        PipelineSchedule::GPipe => (microbatches + 1, 0, microbatches + 1),
        PipelineSchedule::OneForwardOneBackward => (2, microbatches * 2 - 2, 2),
    };
    let bubble_slots = clocks
        .iter()
        .flat_map(|clock| clock.stages)
        .filter(Option::is_none)
        .count();
    let max_live_activations = max_live_activations(&clocks, microbatches)?;
    Ok(PipelineScheduleDescriptor {
        schedule,
        microbatches,
        checkpoint,
        warmup_steps,
        steady_steps,
        cooldown_steps,
        bubble_slots,
        max_live_activations,
        clocks,
    })
}

fn gpipe_clocks(microbatches: usize) -> Vec<PipelineClock> {
    let mut clocks = Vec::with_capacity(microbatches * 2 + 2);
    for clock in 0..=microbatches {
        clocks.push(PipelineClock {
            phase: PipelinePhase::Warmup,
            stages: [
                (clock < microbatches).then_some(PipelineAction::Forward { microbatch: clock }),
                (clock > 0).then_some(PipelineAction::Forward {
                    microbatch: clock.saturating_sub(1),
                }),
            ],
        });
    }
    for clock in 0..=microbatches {
        clocks.push(PipelineClock {
            phase: PipelinePhase::Cooldown,
            stages: [
                (clock > 0).then_some(PipelineAction::Backward {
                    microbatch: microbatches.saturating_sub(clock),
                }),
                (clock < microbatches).then(|| PipelineAction::Backward {
                    microbatch: microbatches - 1 - clock,
                }),
            ],
        });
    }
    clocks
}

fn one_f_one_b_clocks(microbatches: usize) -> Vec<PipelineClock> {
    let mut clocks = Vec::with_capacity(microbatches * 2 + 2);
    clocks.push(PipelineClock {
        phase: PipelinePhase::Warmup,
        stages: [Some(PipelineAction::Forward { microbatch: 0 }), None],
    });
    clocks.push(PipelineClock {
        phase: PipelinePhase::Warmup,
        stages: [
            (microbatches > 1).then_some(PipelineAction::Forward { microbatch: 1 }),
            Some(PipelineAction::Forward { microbatch: 0 }),
        ],
    });
    clocks.push(PipelineClock {
        phase: if microbatches == 1 {
            PipelinePhase::Cooldown
        } else {
            PipelinePhase::Steady
        },
        stages: [None, Some(PipelineAction::Backward { microbatch: 0 })],
    });
    for microbatch in 1..microbatches {
        clocks.push(PipelineClock {
            phase: PipelinePhase::Steady,
            stages: [
                Some(PipelineAction::Backward {
                    microbatch: microbatch - 1,
                }),
                Some(PipelineAction::Forward { microbatch }),
            ],
        });
        clocks.push(PipelineClock {
            phase: if microbatch + 1 == microbatches {
                PipelinePhase::Cooldown
            } else {
                PipelinePhase::Steady
            },
            stages: [
                (microbatch + 1 < microbatches).then_some(PipelineAction::Forward {
                    microbatch: microbatch + 1,
                }),
                Some(PipelineAction::Backward { microbatch }),
            ],
        });
    }
    clocks.push(PipelineClock {
        phase: PipelinePhase::Cooldown,
        stages: [
            Some(PipelineAction::Backward {
                microbatch: microbatches - 1,
            }),
            None,
        ],
    });
    clocks
}

fn validate_clocks(clocks: &[PipelineClock], microbatches: usize) -> Result<(), PipelineError> {
    let mut counts = [[0_usize; 2]; 2];
    let mut forwarded = vec![[false; 2]; microbatches];
    let mut backwarded = vec![[false; 2]; microbatches];
    for clock in clocks {
        for (stage, action) in clock.stages.into_iter().enumerate() {
            let Some(action) = action else {
                continue;
            };
            let (direction, microbatch) = match action {
                PipelineAction::Forward { microbatch } => (0, microbatch),
                PipelineAction::Backward { microbatch } => (1, microbatch),
            };
            if microbatch >= microbatches {
                return Err(PipelineError::ScheduleMicrobatchOutOfRange {
                    stage,
                    microbatch,
                    microbatches,
                });
            }
            if direction == 0 {
                if forwarded[microbatch][stage] {
                    return Err(PipelineError::DuplicateScheduleAction {
                        stage,
                        microbatch,
                        direction: "forward",
                    });
                }
                forwarded[microbatch][stage] = true;
            } else {
                if !forwarded[microbatch][stage] {
                    return Err(PipelineError::BackwardBeforeForward { stage, microbatch });
                }
                if backwarded[microbatch][stage] {
                    return Err(PipelineError::DuplicateScheduleAction {
                        stage,
                        microbatch,
                        direction: "backward",
                    });
                }
                backwarded[microbatch][stage] = true;
            }
            counts[stage][direction] += 1;
        }
    }
    for (stage, stage_counts) in counts.iter().enumerate() {
        for (direction, &found) in stage_counts.iter().enumerate() {
            if found != microbatches {
                return Err(PipelineError::IncompleteSchedule {
                    stage,
                    direction,
                    expected: microbatches,
                    found,
                });
            }
        }
    }
    Ok(())
}

fn max_live_activations(
    clocks: &[PipelineClock],
    microbatches: usize,
) -> Result<[usize; 2], PipelineError> {
    let mut live = [0_usize; 2];
    let mut maximum = [0_usize; 2];
    let mut forwarded = vec![[false; 2]; microbatches];
    for clock in clocks {
        for (stage, action) in clock.stages.into_iter().enumerate() {
            match action {
                Some(PipelineAction::Forward { microbatch }) => {
                    if forwarded[microbatch][stage] {
                        return Err(PipelineError::DuplicateScheduleAction {
                            stage,
                            microbatch,
                            direction: "forward",
                        });
                    }
                    forwarded[microbatch][stage] = true;
                    live[stage] += 1;
                    maximum[stage] = maximum[stage].max(live[stage]);
                }
                Some(PipelineAction::Backward { microbatch }) => {
                    if !forwarded[microbatch][stage] || live[stage] == 0 {
                        return Err(PipelineError::BackwardBeforeForward { stage, microbatch });
                    }
                    live[stage] -= 1;
                }
                None => {}
            }
        }
    }
    if live != [0, 0] {
        return Err(PipelineError::LiveActivationsAfterSchedule { live });
    }
    Ok(maximum)
}

fn validate_pipeline_descriptors(
    collective: &CollectivePlan,
    boundary: PipelineBoundaryId,
    transfers: &[PipelineTransferDescriptor],
) -> Result<(), PipelineError> {
    if collective.descriptors().len() != transfers.len() {
        return Err(PipelineError::DescriptorCount {
            expected: transfers.len(),
            found: collective.descriptors().len(),
        });
    }
    for (index, (descriptor, transfer)) in
        collective.descriptors().iter().zip(transfers).enumerate()
    {
        let expected_kind = CollectiveKind::SendRecv {
            source: transfer.transfer.source_rank(),
            destination: transfer.transfer.destination_rank(),
        };
        if descriptor.kind() != expected_kind
            || descriptor.tag() != transfer.transfer.plan_tag(boundary, transfer.microbatch)
            || descriptor.sequence() != transfer.sequence
        {
            return Err(PipelineError::DescriptorMismatch { index });
        }
    }
    Ok(())
}

/// Failures while validating a PP=2 shape, schedule, or transfer plan.
#[non_exhaustive]
#[derive(thiserror::Error, Debug, Clone, PartialEq, Eq)]
pub enum PipelineError {
    /// Shared communication planning rejected a descriptor.
    #[error(transparent)]
    Plan(#[from] PlanError),
    /// Shared collective metadata rejected a descriptor.
    #[error(transparent)]
    Collective(#[from] CollectiveError),
    /// Dynamic activation cardinality overflowed.
    #[error(transparent)]
    Shape(#[from] ShapeError),
    /// Runtime `Dyn` dtype is not a supported activation/gradient type.
    #[error("dtype {dtype:?} is unsupported for pipeline activations and gradients")]
    UnsupportedDType {
        /// Rejected dtype.
        dtype: DTypeId,
    },
    /// Zero microbatches cannot make progress.
    #[error("a pipeline schedule requires at least one microbatch")]
    ZeroMicrobatches,
    /// Microbatch identity no longer fits the stable plan tag.
    #[error("pipeline supports at most {maximum} microbatches, found {found}")]
    TooManyMicrobatches {
        /// Largest encodable count.
        maximum: usize,
        /// Rejected count.
        found: usize,
    },
    /// The schedule/transfer count cannot be represented by this target.
    #[error("pipeline schedule length overflows for {microbatches} microbatches")]
    ScheduleLengthOverflow {
        /// Rejected count.
        microbatches: usize,
    },
    /// Boundary zero is reserved for unlabelled communication.
    #[error("pipeline boundary id zero is reserved")]
    ReservedBoundaryId,
    /// Boundary identity cannot fit beside microbatch and direction bits.
    #[error("pipeline boundary id {found} exceeds maximum {maximum}")]
    BoundaryIdTooLarge {
        /// Largest encodable identity.
        maximum: u64,
        /// Rejected identity.
        found: u64,
    },
    /// A generated clock references a nonexistent microbatch.
    #[error(
        "stage {stage} schedule references microbatch {microbatch}, but count is {microbatches}"
    )]
    ScheduleMicrobatchOutOfRange {
        /// Stage whose action is invalid.
        stage: usize,
        /// Rejected index.
        microbatch: usize,
        /// Configured count.
        microbatches: usize,
    },
    /// Backward work appeared without an earlier forward.
    #[error("stage {stage} schedules backward for microbatch {microbatch} before forward")]
    BackwardBeforeForward {
        /// Stage with invalid ordering.
        stage: usize,
        /// Microbatch with invalid ordering.
        microbatch: usize,
    },
    /// A stage/direction did not cover every microbatch.
    #[error("stage {stage} direction {direction} has {found} actions, expected {expected}")]
    IncompleteSchedule {
        /// Stage with incomplete work.
        stage: usize,
        /// Zero for forward, one for backward.
        direction: usize,
        /// Configured count.
        expected: usize,
        /// Generated count.
        found: usize,
    },
    /// One compute action was generated twice.
    #[error("stage {stage} schedules {direction} for microbatch {microbatch} more than once")]
    DuplicateScheduleAction {
        /// Stage with duplicate work.
        stage: usize,
        /// Duplicated microbatch.
        microbatch: usize,
        /// Forward or backward.
        direction: &'static str,
    },
    /// A supposedly complete schedule retains activations.
    #[error("pipeline schedule ends with live activations {live:?}")]
    LiveActivationsAfterSchedule {
        /// Remaining count per stage.
        live: [usize; 2],
    },
    /// Semantic and transport descriptor counts diverged.
    #[error("pipeline has {found} transport descriptors, expected {expected}")]
    DescriptorCount {
        /// Semantic transfer count.
        expected: usize,
        /// Transport descriptor count.
        found: usize,
    },
    /// A generated descriptor lost its semantic identity or direction.
    #[error("pipeline transport descriptor {index} disagrees with its semantic transfer")]
    DescriptorMismatch {
        /// Zero-based descriptor index.
        index: usize,
    },
}
