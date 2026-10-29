use core::marker::PhantomData;
use incin_core::dist::{
    CollectiveDType, HybridPlanDType, HybridPlanner, MemoryLimit, PipelineDType, PlanObjective,
    ShardRemainderPolicy, StaticParallelOptions, TwoRankPlanningTopology,
};
use incin_core::prelude::{ConstDType, DType, DTypeDescriptor, DTypeKey, DTypeKind};
use incin_core::tensor::dtype::StorageEncoding;
use incin_core::typenum::{U4, U8, U16};

#[derive(Clone, Debug, PartialEq)]
struct AcmePosit24;

impl DType for AcmePosit24 {
    type Arg = ();
    type Field = PhantomData<Self>;

    fn init(_: ()) -> Self::Field {
        PhantomData
    }

    fn descriptor(_: &Self::Field) -> DTypeDescriptor {
        Self::DESCRIPTOR
    }
}

impl ConstDType for AcmePosit24 {
    const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::new(
        DTypeKey::new("acme", "posit24", 1),
        DTypeKind::Opaque,
        StorageEncoding::scalar(3, 1),
    );
}

impl CollectiveDType for AcmePosit24 {}
impl PipelineDType for AcmePosit24 {}
impl HybridPlanDType for AcmePosit24 {}

fn custom_dtype_missing_builtin_id(topology: &TwoRankPlanningTopology) {
    let _ = HybridPlanner::plan_data_static::<AcmePosit24, U8, U16, U4, U4>(
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
