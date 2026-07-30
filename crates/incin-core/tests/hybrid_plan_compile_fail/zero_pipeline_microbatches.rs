use incin_core::dist::{
    GPipe, HybridPlanner, MemoryLimit, PlanObjective, ShardRemainderPolicy,
    StaticParallelOptions, TwoRankPlanningTopology,
};
use incin_core::typenum::{U0, U4, U16};

fn zero_pipeline_microbatches(topology: &TwoRankPlanningTopology) {
    let _ = HybridPlanner::plan_pipeline_static::<f32, U16, U4, U0, GPipe>(
        topology,
        8,
        8,
        2,
        [10_000; 2],
        StaticParallelOptions {
            memory_limit: MemoryLimit::PerRankBytes(10_000),
            remainder: ShardRemainderPolicy::Reject,
            objective: PlanObjective::MinimizeMemory,
        },
    );
}

fn main() {}
