//! PROPOSALS.md §3.8: "For three GPUs, valid examples are `DP=3`, `TP=3`, or
//! `PP=3`. A rectangular `2 × 2` mesh is not valid and must not be partially
//! populated implicitly."
//!
//! The `2 × 2` mesh below is a perfectly valid four-rank topology. What it is
//! not is a three-rank one, and a bound on `World` is what makes the difference
//! a compile error rather than three GPUs quietly running a plan written for
//! four.

use incin_core::dist::mesh::{Data, MeshSpec, Pipeline, TensorParallel, ValidMesh};
use incin_core::typenum::{U1, U2, U3};

fn on_three_ranks<M: ValidMesh<World = U3>>() -> usize {
    M::WORLD
}

fn main() {
    let _ = on_three_ranks::<MeshSpec<Data<U2>, TensorParallel<U2>, Pipeline<U1>>>();
}
