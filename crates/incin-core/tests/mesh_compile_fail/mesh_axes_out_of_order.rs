//! The mesh axes are positional, and swapping two of them is the failure that
//! would otherwise be silent: `Data<U1> × Pipeline<U3> × TensorParallel<U1>`
//! has the same world size as the three-way tensor-parallel mesh it was meant
//! to be, and describes three sequential pipeline stages instead.

use incin_core::dist::mesh::{Data, MeshSpec, Pipeline, TensorParallel, ValidMesh};
use incin_core::typenum::{U1, U3};

fn main() {
    let _ = <MeshSpec<Data<U1>, Pipeline<U3>, TensorParallel<U1>> as ValidMesh>::WORLD;
}
