//! Integration coverage for `integer_dtype` on the documented public surface.
use incin_core::dist::{
    HybridPlanner, MemoryLimit, PlanObjective, ShardRemainderPolicy, StaticParallelOptions,
    TwoRankPlanningTopology,
};
use incin_core::typenum::{U4, U8, U16};

fn integer_dtype(topology: &TwoRankPlanningTopology) {
    let _ = HybridPlanner::plan_data_static::<u32, U8, U16, U4, U4>(
        topology,
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
