use incin_core::dist::ShardDivisible;
use incin_core::typenum::{U3, U10};

fn requires_equal_shards<Extent: ShardDivisible<U3>>() {}

fn main() {
    // Ten elements cannot be partitioned into three integral local extents.
    requires_equal_shards::<U10>();
}
