//! A mesh axis of degree zero describes no ranks, so `MeshSpec` has no
//! `ValidMesh` implementation and the world size cannot be named.

use incin_core::dist::mesh::{Data, MeshSpec, Pipeline, TensorParallel, ValidMesh};
use incin_core::typenum::{U0, U1, U3};

fn main() {
    let _ = <MeshSpec<Data<U0>, TensorParallel<U3>, Pipeline<U1>> as ValidMesh>::WORLD;
}
