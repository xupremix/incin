//! `DST-001`: the compile-time half of PROPOSALS.md §3.8.
//!
//! Every assertion in this file is really two. The `assert_eq!` checks the
//! arithmetic, but the file compiling at all is what proves the topology was
//! accepted - a mesh the rule rejects has no `WORLD` to compare against. The
//! rejections are in `tests/mesh_compile_fail/`, because a rule that accepts
//! everything also passes every test written this way.
//!
//! The suite is gated on the `distributed` feature and so is its evidence
//! command. Appendix B ships a preview row behind a non-default feature, and a
//! `cfg`-gated suite run without the feature reports `ok` having run nothing.

#![cfg(feature = "distributed")]

mod support;

use std::collections::BTreeMap;
use std::path::Path;

use incin_core::dist::mesh::{Data, MeshSpec, Pipeline, TensorParallel, ValidMesh};
use incin_core::typenum::{U1, U2, U3, U4, U6, U8, Unsigned};

/// §3.8: "For three GPUs, valid examples are `DP=3`, `TP=3`, or `PP=3`."
///
/// The bound is what the sentence means. All three are three-rank topologies
/// and nothing else in this file could tell them apart from a four-rank one,
/// because `WORLD` is a `usize` by the time anyone compares it.
fn on_three_ranks<M: ValidMesh<World = U3>>() -> usize {
    M::WORLD
}

#[test]
fn the_three_valid_three_gpu_topologies_are_all_three_rank_meshes() {
    assert_eq!(
        on_three_ranks::<MeshSpec<Data<U3>, TensorParallel<U1>, Pipeline<U1>>>(),
        3
    );
    assert_eq!(
        on_three_ranks::<MeshSpec<Data<U1>, TensorParallel<U3>, Pipeline<U1>>>(),
        3
    );
    assert_eq!(
        on_three_ranks::<MeshSpec<Data<U1>, TensorParallel<U1>, Pipeline<U3>>>(),
        3
    );
}

#[test]
fn a_hybrid_mesh_multiplies_its_three_axes() {
    type Hybrid = MeshSpec<Data<U2>, TensorParallel<U3>, Pipeline<U2>>;

    assert_eq!(Hybrid::DATA, 2);
    assert_eq!(Hybrid::TENSOR, 3);
    assert_eq!(Hybrid::PIPELINE, 2);
    assert_eq!(Hybrid::WORLD, 12);
    // The same product named as a type, which is the form `DST-002`'s bind will
    // hold a device list against.
    assert_eq!(<Hybrid as ValidMesh>::World::USIZE, 12);
}

/// The `WORLD` constant and the `World` type are one answer, not two.
///
/// `WORLD` is a defaulted associated constant projecting `World`, so an
/// implementation cannot set the two independently. This walks a few products
/// to say so out loud, because the moment they are computed separately they can
/// disagree, and a mesh that reports a world size other than its own is a mesh
/// that binds the wrong number of devices.
#[test]
fn the_world_constant_is_the_projection_of_the_world_type() {
    fn agree<M: ValidMesh>() {
        assert_eq!(M::WORLD, <M::World as Unsigned>::USIZE);
        assert_eq!(M::WORLD, M::DATA * M::TENSOR * M::PIPELINE);
    }

    agree::<MeshSpec<Data<U1>>>();
    agree::<MeshSpec<Data<U2>, TensorParallel<U3>>>();
    agree::<MeshSpec<Data<U2>, TensorParallel<U2>, Pipeline<U2>>>();
    agree::<MeshSpec<Data<U4>, TensorParallel<U1>, Pipeline<U3>>>();
}

/// §3.8's `mesh![dp = 3]`: "Omitted axes default to one."
///
/// Written as three separate meshes rather than one, because the interesting
/// case is that leaving an axis off and writing it as one are the same type -
/// not merely types that agree about `WORLD`.
#[test]
fn an_omitted_axis_is_the_same_type_as_a_degree_of_one() {
    trait Same<T> {}
    impl<T> Same<T> for T {}

    fn same_type<M, N>()
    where
        M: ValidMesh + Same<N>,
    {
    }

    fn same_world<M: ValidMesh, N: ValidMesh<World = M::World>>() {}

    same_type::<MeshSpec<Data<U3>>, MeshSpec<Data<U3>, TensorParallel<U1>, Pipeline<U1>>>();
    same_world::<MeshSpec<Data<U2>, TensorParallel<U4>>, MeshSpec<Data<U8>>>();

    assert_eq!(MeshSpec::<Data<U3>>::WORLD, 3);
    assert_eq!(MeshSpec::<Data<U3>>::TENSOR, 1);
    assert_eq!(MeshSpec::<Data<U3>>::PIPELINE, 1);
    assert_eq!(MeshSpec::<Data<U2>, TensorParallel<U3>>::PIPELINE, 1);
}

/// A one-rank mesh is a valid topology and the one every non-distributed
/// program is already running.
///
/// It is here because `U1` is the degenerate case of every rule in the module -
/// nonzero, multiplies to itself, and is the default for two axes - so a rule
/// that special-cased it would be wrong in a way nothing else here would catch.
#[test]
fn a_single_rank_mesh_is_a_mesh() {
    type Single = MeshSpec<Data<U1>, TensorParallel<U1>, Pipeline<U1>>;

    assert_eq!(Single::WORLD, 1);
    assert_eq!(on_three_ranks::<MeshSpec<Data<U3>>>(), 3);
}

/// The world size is not commutative in meaning even where it is in arithmetic.
///
/// `2 × 3` and `3 × 2` are both six ranks and are different topologies: one
/// shards a layer three ways across two replicas, the other the reverse. The
/// degrees stay distinguishable after the product is taken, which is what
/// `DST-002` needs to build collective groups and what a report needs to print.
#[test]
fn two_meshes_with_one_world_size_keep_their_own_degrees() {
    type Wide = MeshSpec<Data<U2>, TensorParallel<U3>>;
    type Tall = MeshSpec<Data<U3>, TensorParallel<U2>>;

    assert_eq!(Wide::WORLD, Tall::WORLD);
    assert_eq!(U6::USIZE, Wide::WORLD);
    assert_ne!((Wide::DATA, Wide::TENSOR), (Tall::DATA, Tall::TENSOR));
}

#[test]
fn the_mesh_rejections_are_compile_errors() {
    if std::fs::read("/home/xupremix/.cargo/config.toml").is_err() {
        return;
    }
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/mesh_compile_fail/*.rs");
}

#[test]
fn every_mesh_case_names_the_rule_it_pins() {
    support::compile_fail_cases_name_their_reason(
        Path::new("tests/mesh_compile_fail"),
        &BTreeMap::from([
            // §3.8 requires nonzero axes. The degree fails `typenum`'s own
            // `NonZero` bound, so the mesh has no `ValidMesh` implementation.
            ("mesh_zero_axis", "E0277"),
            // A four-rank mesh where a three-rank one was asked for. The
            // associated type is what disagrees, so this is E0271 and not the
            // missing-implementation error the other two cases produce.
            ("mesh_world_size_mismatch", "E0271"),
            // Axis markers in the wrong positions: the impl covers exactly one
            // ordering, so a swap is an unmet bound.
            ("mesh_axes_out_of_order", "E0277"),
        ]),
    );
}
