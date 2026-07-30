use incin_core::dist::{StreamId, TensorParallelId, TensorParallelPlanBuilder};
use incin_core::typenum::{U0, U3};

fn three_heads(builder: &mut TensorParallelPlanBuilder<'_>) {
    builder
        .push_attention_static::<f32, U0, U3>(
            TensorParallelId::new(1).unwrap(),
            4,
            StreamId::default(),
        )
        .unwrap();
}

fn main() {}
