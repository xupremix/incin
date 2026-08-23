use incin::experimental::mesh;
use incin::prelude::*;

type MyMesh = mesh![dp = 2, tp = 4];

#[module]
pub struct ConflictModel<B: Backend> {
    #[parallel(mesh = MyMesh)]
    #[shard(mesh = MyMesh, axis = 0)]
    bad_layer: Linear<s![768, 256], B>,
}

fn main() {}
