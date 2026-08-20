//! The stateless two-rank hybrid planner: entry points plus the analytical
//! feasibility and cost-estimation algorithm behind them.

use super::*;

/// Stateless two-rank hybrid planner.
#[derive(Debug, Default, Clone, Copy)]
pub struct HybridPlanner;

impl HybridPlanner {
    /// Plan from runtime-resolved (`Dyn`) workload and dtype values.
    pub fn plan_dyn(
        topology: &TwoRankPlanningTopology,
        dtype: DTypeId,
        workload: HybridWorkload,
        options: ParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError> {
        validate_hybrid_plan_dtype(dtype)?;
        validate_microbatches(workload.microbatches).map_err(|_| {
            HybridPlanError::ZeroWorkloadField {
                field: WorkloadField::Microbatches,
            }
        })?;
        if workload.microbatches > u32::MAX as usize {
            return Err(HybridPlanError::MicrobatchLimit {
                found: workload.microbatches,
                maximum: u32::MAX as usize,
            });
        }
        if options.remainder != ShardRemainderPolicy::Reject {
            return Err(HybridPlanError::UnsupportedRemainderPolicy {
                found: options.remainder,
            });
        }
        let limits = resolve_memory_limits(options.memory_limit, workload.device_capacity_bytes)?;
        let requested = requested_strategies(options.strategy)?;
        let mut feasible = Vec::new();
        let mut rejected = Vec::new();

        for strategy in [
            ParallelStrategyKind::Data,
            ParallelStrategyKind::Tensor,
            ParallelStrategyKind::Pipeline,
        ] {
            if !requested.contains(strategy) {
                rejected.push(RejectedStrategy {
                    strategy,
                    reason: match options.strategy {
                        ParallelStrategy::Auto { .. } => StrategyRejection::NotAllowed,
                        _ => StrategyRejection::NotSelected,
                    },
                });
                continue;
            }

            match build_candidate(
                strategy,
                topology,
                dtype,
                workload,
                options.schedule,
                limits,
            )? {
                Ok(candidate) => feasible.push(candidate),
                Err(reason) => rejected.push(RejectedStrategy { strategy, reason }),
            }
        }

        if feasible.is_empty() {
            return Err(HybridPlanError::NoFeasibleStrategy { rejected });
        }

        let pareto_frontier = pareto_frontier(&feasible);
        let chosen_index = choose_candidate(&feasible, options.objective);
        let chosen = feasible[chosen_index].clone();
        Ok(HybridPlanReport {
            objective: options.objective,
            chosen,
            feasible,
            pareto_frontier,
            rejected,
        })
    }

    /// Plan an automatic search whose logical values and dtype are all static.
    ///
    /// The bounds prove exact DP batch sharding, exact TP dimension and
    /// parameter sharding, nonzero bounded PP microbatches, and a floating
    /// dtype before runtime. Physical capacities and topology remain runtime
    /// observations and are still validated.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_auto_static<
        K,
        Batch,
        TensorExtent,
        Parameters,
        Activations,
        Microbatches,
        Schedule,
    >(
        topology: &TwoRankPlanningTopology,
        optimizer_state_copies: usize,
        device_capacity_bytes: [usize; 2],
        allowed: StrategySet,
        policy: StaticParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError>
    where
        K: ConstDType + BuiltinDType + HybridPlanDType,
        Batch: Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        TensorExtent:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Parameters:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Activations: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Microbatches: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Schedule: StaticPipelineSchedule,
    {
        let workload = HybridWorkload::new(
            Batch::USIZE,
            TensorExtent::USIZE,
            Parameters::USIZE,
            Activations::USIZE,
            Microbatches::USIZE,
            optimizer_state_copies,
            device_capacity_bytes,
        )?;
        Self::plan_dyn(
            topology,
            K::DTYPE,
            workload,
            ParallelOptions {
                strategy: ParallelStrategy::Auto { allowed },
                memory_limit: policy.memory_limit,
                remainder: policy.remainder,
                schedule: Schedule::SCHEDULE,
                objective: policy.objective,
            },
        )
    }

    /// Require a statically valid DP=2 plan.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_data_static<K, Batch, Parameters, Activations, Microbatches>(
        topology: &TwoRankPlanningTopology,
        tensor_shard_extent: usize,
        optimizer_state_copies: usize,
        device_capacity_bytes: [usize; 2],
        policy: StaticParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError>
    where
        K: ConstDType + BuiltinDType + HybridPlanDType,
        Batch: Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Parameters: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Activations: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Microbatches: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
    {
        let workload = HybridWorkload::new(
            Batch::USIZE,
            tensor_shard_extent,
            Parameters::USIZE,
            Activations::USIZE,
            Microbatches::USIZE,
            optimizer_state_copies,
            device_capacity_bytes,
        )?;
        plan_static_selected(
            topology,
            K::DTYPE,
            workload,
            ParallelStrategy::Data,
            PipelineSchedule::GPipe,
            policy,
        )
    }

    /// Require a statically valid TP=2 plan.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_tensor_static<K, TensorExtent, Parameters, Activations, Microbatches>(
        topology: &TwoRankPlanningTopology,
        batch_size: usize,
        optimizer_state_copies: usize,
        device_capacity_bytes: [usize; 2],
        policy: StaticParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError>
    where
        K: ConstDType + BuiltinDType + HybridPlanDType,
        TensorExtent:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Parameters:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Activations: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Microbatches: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
    {
        let workload = HybridWorkload::new(
            batch_size,
            TensorExtent::USIZE,
            Parameters::USIZE,
            Activations::USIZE,
            Microbatches::USIZE,
            optimizer_state_copies,
            device_capacity_bytes,
        )?;
        plan_static_selected(
            topology,
            K::DTYPE,
            workload,
            ParallelStrategy::Tensor,
            PipelineSchedule::GPipe,
            policy,
        )
    }

    /// Require a statically valid PP=2 plan and schedule.
    #[allow(clippy::too_many_arguments)]
    pub fn plan_pipeline_static<K, Parameters, Activations, Microbatches, Schedule>(
        topology: &TwoRankPlanningTopology,
        batch_size: usize,
        tensor_shard_extent: usize,
        optimizer_state_copies: usize,
        device_capacity_bytes: [usize; 2],
        policy: StaticParallelOptions,
    ) -> Result<HybridPlanReport, HybridPlanError>
    where
        K: ConstDType + BuiltinDType + HybridPlanDType,
        Parameters:
            Unsigned + NonZero + ShardDivisible<U2> + IsLessOrEqual<U4294967295, Output = B1>,
        Activations: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Microbatches: Unsigned + NonZero + IsLessOrEqual<U4294967295, Output = B1>,
        Schedule: StaticPipelineSchedule,
    {
        let workload = HybridWorkload::new(
            batch_size,
            tensor_shard_extent,
            Parameters::USIZE,
            Activations::USIZE,
            Microbatches::USIZE,
            optimizer_state_copies,
            device_capacity_bytes,
        )?;
        plan_static_selected(
            topology,
            K::DTYPE,
            workload,
            ParallelStrategy::Pipeline,
            Schedule::SCHEDULE,
            policy,
        )
    }
}

fn plan_static_selected(
    topology: &TwoRankPlanningTopology,
    dtype: DTypeId,
    workload: HybridWorkload,
    strategy: ParallelStrategy,
    schedule: PipelineSchedule,
    policy: StaticParallelOptions,
) -> Result<HybridPlanReport, HybridPlanError> {
    HybridPlanner::plan_dyn(
        topology,
        dtype,
        workload,
        ParallelOptions {
            strategy,
            memory_limit: policy.memory_limit,
            remainder: policy.remainder,
            schedule,
            objective: policy.objective,
        },
    )
}

fn requested_strategies(strategy: ParallelStrategy) -> Result<StrategySet, HybridPlanError> {
    let requested = match strategy {
        ParallelStrategy::Data => StrategySet::DATA,
        ParallelStrategy::Tensor => StrategySet::TENSOR,
        ParallelStrategy::Pipeline => StrategySet::PIPELINE,
        ParallelStrategy::Auto { allowed } => allowed,
    };
    if requested.is_empty() {
        Err(HybridPlanError::EmptyStrategySet)
    } else {
        Ok(requested)
    }
}

fn resolve_memory_limits(
    limit: MemoryLimit,
    capacities: [usize; 2],
) -> Result<[usize; 2], HybridPlanError> {
    match limit {
        MemoryLimit::PerRankBytes(bytes) => {
            if bytes == 0 {
                Err(HybridPlanError::ZeroMemoryLimit)
            } else {
                Ok([bytes; 2])
            }
        }
        MemoryLimit::PerDeviceFraction(fraction) => {
            if !fraction.is_finite() || fraction <= 0.0 || fraction > 1.0 {
                return Err(HybridPlanError::InvalidMemoryFraction);
            }
            let first = (capacities[0] as f64 * fraction).floor() as usize;
            let second = (capacities[1] as f64 * fraction).floor() as usize;
            if first == 0 || second == 0 {
                Err(HybridPlanError::ZeroMemoryLimit)
            } else {
                Ok([first, second])
            }
        }
    }
}

fn build_candidate(
    strategy: ParallelStrategyKind,
    topology: &TwoRankPlanningTopology,
    dtype: DTypeId,
    workload: HybridWorkload,
    schedule: PipelineSchedule,
    limits: [usize; 2],
) -> Result<Result<StrategyCandidate, StrategyRejection>, HybridPlanError> {
    let parameter_bytes = dtype
        .size_bytes(workload.parameter_elements, OperationKind::Storage)
        .map_err(HybridPlanError::Shape)?;
    let activation_bytes = dtype
        .size_bytes(
            workload.activation_elements_per_microbatch,
            OperationKind::Storage,
        )
        .map_err(HybridPlanError::Shape)?;
    let optimizer_bytes = checked_mul(
        parameter_bytes,
        workload.optimizer_state_copies,
        "parameter bytes * optimizer state copies",
    )?;
    let full_model_memory = checked_sum(
        &[parameter_bytes, parameter_bytes, optimizer_bytes],
        "parameters + gradients + optimizer states",
    )?;
    let compute_work = checked_sum(
        &[
            parameter_bytes,
            checked_mul(
                activation_bytes,
                workload.microbatches,
                "activation bytes * microbatches",
            )?,
        ],
        "parameters + step activations",
    )?;

    let (shards, collectives, memory, communication, pipeline_schedule, bubble_cost) =
        match strategy {
            ParallelStrategyKind::Data => {
                if !workload.batch_size.is_multiple_of(2) {
                    return Ok(Err(StrategyRejection::NonDivisible {
                        field: WorkloadField::BatchSize,
                        value: workload.batch_size,
                        degree: 2,
                    }));
                }
                let memory = checked_add_array(
                    [full_model_memory; 2],
                    [activation_bytes; 2],
                    "DP persistent + live activation",
                )?;
                let communication =
                    checked_mul(parameter_bytes, 2, "two-rank gradient all-reduce payload")?;
                (
                    vec![
                        ShardEvidence {
                            field: WorkloadField::BatchSize,
                            global: workload.batch_size,
                            per_rank: [workload.batch_size / 2; 2],
                        },
                        ShardEvidence {
                            field: WorkloadField::ParameterElements,
                            global: workload.parameter_elements,
                            per_rank: [workload.parameter_elements; 2],
                        },
                    ],
                    vec![CommunicationEvidence {
                        kind: PlanningCollectiveKind::AllReduce,
                        launches: 1,
                        bytes: communication,
                    }],
                    memory,
                    communication,
                    None,
                    0,
                )
            }
            ParallelStrategyKind::Tensor => {
                if !workload.tensor_shard_extent.is_multiple_of(2) {
                    return Ok(Err(StrategyRejection::NonDivisible {
                        field: WorkloadField::TensorShardExtent,
                        value: workload.tensor_shard_extent,
                        degree: 2,
                    }));
                }
                if !workload.parameter_elements.is_multiple_of(2) {
                    return Ok(Err(StrategyRejection::NonDivisible {
                        field: WorkloadField::ParameterElements,
                        value: workload.parameter_elements,
                        degree: 2,
                    }));
                }
                let sharded_model = full_model_memory / 2;
                let memory = checked_add_array(
                    [sharded_model; 2],
                    [activation_bytes; 2],
                    "TP sharded model + gathered activation",
                )?;
                let one_collective = activation_bytes;
                let communication =
                    checked_mul(one_collective, 2, "TP all-gather + reduce-scatter payload")?;
                (
                    vec![
                        ShardEvidence {
                            field: WorkloadField::TensorShardExtent,
                            global: workload.tensor_shard_extent,
                            per_rank: [workload.tensor_shard_extent / 2; 2],
                        },
                        ShardEvidence {
                            field: WorkloadField::ParameterElements,
                            global: workload.parameter_elements,
                            per_rank: [workload.parameter_elements / 2; 2],
                        },
                    ],
                    vec![
                        CommunicationEvidence {
                            kind: PlanningCollectiveKind::AllGather,
                            launches: 1,
                            bytes: one_collective,
                        },
                        CommunicationEvidence {
                            kind: PlanningCollectiveKind::ReduceScatter,
                            launches: 1,
                            bytes: one_collective,
                        },
                    ],
                    memory,
                    communication,
                    None,
                    0,
                )
            }
            ParallelStrategyKind::Pipeline => {
                if !workload.parameter_elements.is_multiple_of(2) {
                    return Ok(Err(StrategyRejection::NonDivisible {
                        field: WorkloadField::ParameterElements,
                        value: workload.parameter_elements,
                        degree: 2,
                    }));
                }
                let stage_model = full_model_memory / 2;
                let live = match schedule {
                    PipelineSchedule::GPipe => [workload.microbatches; 2],
                    PipelineSchedule::OneForwardOneBackward => {
                        [core::cmp::min(workload.microbatches, 2), 1]
                    }
                };
                let live_bytes = [
                    checked_mul(activation_bytes, live[0], "stage-zero activation residency")?,
                    checked_mul(activation_bytes, live[1], "stage-one activation residency")?,
                ];
                let memory = checked_add_array(
                    [stage_model; 2],
                    live_bytes,
                    "PP stage model + live activations",
                )?;
                let launches = checked_mul(
                    workload.microbatches,
                    2,
                    "pipeline forward + backward launches",
                )?;
                let communication = checked_mul(
                    activation_bytes,
                    launches,
                    "pipeline activation payload * launches",
                )?;
                let useful_slots = checked_mul(
                    workload.microbatches,
                    4,
                    "two-stage forward/backward useful slots",
                )?;
                let bubble_cost = (compute_work as u128).saturating_mul(4) / useful_slots as u128;
                (
                    vec![
                        ShardEvidence {
                            field: WorkloadField::ParameterElements,
                            global: workload.parameter_elements,
                            per_rank: [workload.parameter_elements / 2; 2],
                        },
                        ShardEvidence {
                            field: WorkloadField::Microbatches,
                            global: workload.microbatches,
                            per_rank: [workload.microbatches; 2],
                        },
                    ],
                    vec![CommunicationEvidence {
                        kind: PlanningCollectiveKind::SendRecv,
                        launches,
                        bytes: communication,
                    }],
                    memory,
                    communication,
                    Some(schedule),
                    bubble_cost,
                )
            }
        };

    for rank in 0..2 {
        if memory[rank] > limits[rank] {
            return Ok(Err(StrategyRejection::MemoryExceeded {
                rank,
                required: memory[rank],
                limit: limits[rank],
            }));
        }
    }

    let link_weight = match topology.link {
        LinkClass::SameDevice => 1_u128,
        LinkClass::HighBandwidth => 2,
        LinkClass::PeerCapable => 3,
        LinkClass::HostBounce => 6,
        LinkClass::Network => 8,
        LinkClass::Unreachable => {
            return Err(HybridPlanError::UnreachableLink {
                from_rank: 0,
                to_rank: 1,
            });
        }
    };
    let estimated_step_cost = (compute_work as u128 / 2)
        .saturating_add((communication as u128).saturating_mul(link_weight))
        .saturating_add(bubble_cost);

    Ok(Ok(StrategyCandidate {
        strategy,
        dtype,
        shards,
        collectives,
        per_rank_peak_memory: memory,
        memory_limits: limits,
        communication_bytes: communication,
        estimated_step_cost,
        topology_fingerprint: topology.fingerprint,
        link: topology.link,
        transport: topology.transport.clone(),
        schedule: pipeline_schedule,
    }))
}

fn choose_candidate(candidates: &[StrategyCandidate], objective: PlanObjective) -> usize {
    let mut chosen = 0;
    for index in 1..candidates.len() {
        let candidate = &candidates[index];
        let current = &candidates[chosen];
        let candidate_key = objective_key(candidate, objective);
        let current_key = objective_key(current, objective);
        if candidate_key < current_key
            || (candidate_key == current_key && candidate.strategy < current.strategy)
        {
            chosen = index;
        }
    }
    chosen
}

fn objective_key(candidate: &StrategyCandidate, objective: PlanObjective) -> u128 {
    match objective {
        PlanObjective::MinimizeStepTime => candidate.estimated_step_cost,
        PlanObjective::MinimizeMemory => core::cmp::max(
            candidate.per_rank_peak_memory[0],
            candidate.per_rank_peak_memory[1],
        ) as u128,
        PlanObjective::MinimizeCommunication => candidate.communication_bytes as u128,
    }
}

fn pareto_frontier(candidates: &[StrategyCandidate]) -> Vec<ParallelStrategyKind> {
    let mut frontier = Vec::new();
    for (index, candidate) in candidates.iter().enumerate() {
        let candidate_memory = core::cmp::max(
            candidate.per_rank_peak_memory[0],
            candidate.per_rank_peak_memory[1],
        );
        let dominated = candidates.iter().enumerate().any(|(other_index, other)| {
            if index == other_index {
                return false;
            }
            let other_memory =
                core::cmp::max(other.per_rank_peak_memory[0], other.per_rank_peak_memory[1]);
            let no_worse = other_memory <= candidate_memory
                && other.communication_bytes <= candidate.communication_bytes
                && other.estimated_step_cost <= candidate.estimated_step_cost;
            let strictly_better = other_memory < candidate_memory
                || other.communication_bytes < candidate.communication_bytes
                || other.estimated_step_cost < candidate.estimated_step_cost;
            no_worse && strictly_better
        });
        if !dominated {
            frontier.push(candidate.strategy);
        }
    }
    frontier
}

fn checked_mul(lhs: usize, rhs: usize, expression: &'static str) -> Result<usize, HybridPlanError> {
    lhs.checked_mul(rhs)
        .ok_or(HybridPlanError::ArithmeticOverflow { expression })
}

fn checked_sum(values: &[usize], expression: &'static str) -> Result<usize, HybridPlanError> {
    values.iter().try_fold(0_usize, |sum, &value| {
        sum.checked_add(value)
            .ok_or(HybridPlanError::ArithmeticOverflow { expression })
    })
}

fn checked_add_array(
    lhs: [usize; 2],
    rhs: [usize; 2],
    expression: &'static str,
) -> Result<[usize; 2], HybridPlanError> {
    Ok([
        lhs[0]
            .checked_add(rhs[0])
            .ok_or(HybridPlanError::ArithmeticOverflow { expression })?,
        lhs[1]
            .checked_add(rhs[1])
            .ok_or(HybridPlanError::ArithmeticOverflow { expression })?,
    ])
}
