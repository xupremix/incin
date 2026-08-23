//! Integration coverage for `literals_use_recursive_typenum_without_a_finite_alias_table` on the documented public surface.
#![recursion_limit = "512"]

use incin_core::prelude::{Shape, s};

type LiteralRange = s![0, 1, 64, 4096, 65_536, 1_000_000];
type RankTwoHundred = s![1; 200];

#[test]
fn literals_use_recursive_typenum_without_a_finite_alias_table() {
    assert_eq!(<LiteralRange as Shape>::STATIC_NUMEL, Some(0));
    let dims = <LiteralRange as Shape>::resolve(Default::default()).unwrap();
    assert_eq!(dims.as_ref(), &[0, 1, 64, 4096, 65_536, 1_000_000]);
}

#[test]
fn recursive_structural_shape_scales_to_rank_two_hundred() {
    assert_eq!(<RankTwoHundred as Shape>::RANK, Some(200));
    let dims = <RankTwoHundred as Shape>::resolve(Default::default()).unwrap();
    assert_eq!(dims.as_ref(), &[1; 200]);
}
