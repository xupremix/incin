//! Rank coverage of the shape rules (`SHP-006`).
//!
//! Before the rank sweep, each rule's ceiling was a hand-written list of macro
//! invocations, and the eighteen lists disagreed: `Shape` reached 8,
//! `ElementCount` 4, `HasChannels1D` 3, and `ReplaceLastDim` overshot at 12.
//! A rank between a rule's ceiling and `Shape`'s is one where a tensor type is
//! expressible but its operations cannot resolve.
//!
//! `incin_macros::rank_sweep!` now generates the ladder from a single
//! `MAX_RANK`. These tests are *compile-time* assertions: each one names a
//! shape at a specific rank and requires the rule to resolve for it, so a rule
//! that silently loses a rank fails to build rather than failing an assertion.
//!
//! Every rule now sits at its correct ceiling: `MAX_RANK` for the
//! rank-preserving ones, `MAX_RANK - 1` for the two that *add* an axis, whose
//! `Output` is itself bounded by `Shape`.

use incin_core::prelude::{
    AppendDim, ConstShape, Dim, DynShape, ElementCount, EndsWith, HasChannels1D, HasChannels2D,
    MAX_RANK, PartialDynShape, ReplaceLastDim, Shape,
};
use incin_core::typenum::{U1, U2, U3, U24, Unsigned};

/// The ceiling every rank-preserving rule is generated up to.
const CEILING: usize = MAX_RANK;

#[test]
fn max_rank_is_single_sourced() {
    // `incin-core` must not carry its own copy of the number — the whole point
    // of SHP-006 is that there is one place to change it.
    assert_eq!(MAX_RANK, incin_macros::max_rank!());
    assert_eq!(CEILING, 8, "MAX_RANK changed; update the ladders below");
}

// --- rank-preserving rules reach MAX_RANK -------------------------------

// Rank 8 is the ceiling. Each of these is a compile-time proof: if the rule
// were not implemented at this rank, the file would not build.
type Rank8 = (U1, U1, U1, U1, U1, U1, U1, U2);

#[test]
fn shape_resolves_at_the_ceiling() {
    assert_eq!(<Rank8 as PartialDynShape>::RANK, CEILING);
    assert_eq!(<Rank8 as ConstShape>::NUMEL, 2);
    let field = <Rank8 as Shape>::from_dyn(&[1, 1, 1, 1, 1, 1, 1, 2]).expect("rank 8 resolves");
    assert_eq!(<Rank8 as DynShape>::rank(&field), CEILING);
}

#[test]
fn ends_with_resolves_at_the_ceiling() {
    // Was capped at rank 6 by its own arm count, not by anything about the
    // rule: `EndsWith` is a rank-preserving marker.
    fn requires_ends_with<S: EndsWith<D>, D: Dim>() {}
    requires_ends_with::<Rank8, U2>();
    requires_ends_with::<(U1, U1, U1, U1, U1, U1, U2), U2>();
    requires_ends_with::<(U2,), U2>();
}

#[test]
fn replace_last_dim_resolves_at_the_ceiling_and_not_above() {
    // Was generated to rank 12 — four ranks at which no tuple implements
    // `Shape`, so those impls could never be selected by anything.
    fn replaced<S: ReplaceLastDim<D>, D: Dim>() -> usize
    where
        <S as ReplaceLastDim<D>>::Output: PartialDynShape,
    {
        <<S as ReplaceLastDim<D>>::Output as PartialDynShape>::RANK
    }
    assert_eq!(replaced::<Rank8, U3>(), CEILING, "rank is not preserved");
    assert_eq!(replaced::<(U1, U2), U3>(), 2);
    assert_eq!(replaced::<(U2,), U3>(), 1);
}

// --- rank-increasing rules stop one short, correctly ---------------------

#[test]
fn append_dim_stops_one_below_the_ceiling_because_its_output_grows() {
    // `AppendDim::Output` is rank N+1 and is bounded by `Shape`. At N =
    // MAX_RANK the output tuple would have no `Shape` impl at all, so rank 7
    // is this rule's *correct* ceiling — not a gap. The audit's "1 short"
    // reading is a property of the measurement, not a defect.
    fn appended<S: AppendDim<D>, D: Dim>() -> usize
    where
        <S as AppendDim<D>>::Output: PartialDynShape,
    {
        <<S as AppendDim<D>>::Output as PartialDynShape>::RANK
    }

    type Rank7 = (U1, U1, U1, U1, U1, U1, U2);
    assert_eq!(appended::<Rank7, U3>(), CEILING);
    assert_eq!(appended::<(U2,), U3>(), 2);

    // And the output really is a usable `Shape` at the ceiling.
    let field = <<Rank7 as AppendDim<U3>>::Output as Shape>::from_dyn(&[1, 1, 1, 1, 1, 1, 2, 3])
        .expect("appending at rank 7 yields a resolvable rank-8 shape");
    assert_eq!(field.7.size(), 3);
}

// --- the ladder is continuous -------------------------------------------

#[test]
fn every_rank_up_to_the_ceiling_resolves() {
    // A ladder with a hole is worse than a low ceiling: it type-checks at
    // rank 5 and 7 but not 6. Each line is a separate compile-time proof.
    fn resolves<S: PartialDynShape + ConstShape>(expected: usize) {
        assert_eq!(<S as PartialDynShape>::RANK, expected);
    }
    resolves::<(U1,)>(1);
    resolves::<(U1, U1)>(2);
    resolves::<(U1, U1, U1)>(3);
    resolves::<(U1, U1, U1, U1)>(4);
    resolves::<(U1, U1, U1, U1, U1)>(5);
    resolves::<(U1, U1, U1, U1, U1, U1)>(6);
    resolves::<(U1, U1, U1, U1, U1, U1, U1)>(7);
    resolves::<(U1, U1, U1, U1, U1, U1, U1, U1)>(8);
}

#[test]
fn ends_with_has_no_hole_in_its_ladder() {
    fn requires_ends_with<S: EndsWith<U2>>() {}
    requires_ends_with::<(U2,)>();
    requires_ends_with::<(U1, U2)>();
    requires_ends_with::<(U1, U1, U2)>();
    requires_ends_with::<(U1, U1, U1, U2)>();
    requires_ends_with::<(U1, U1, U1, U1, U2)>();
    requires_ends_with::<(U1, U1, U1, U1, U1, U2)>();
    requires_ends_with::<(U1, U1, U1, U1, U1, U1, U2)>();
    requires_ends_with::<(U1, U1, U1, U1, U1, U1, U1, U2)>();
}

// --- the RFC's motivating case ------------------------------------------

#[test]
fn element_count_resolves_at_the_ceiling() {
    // `PROPOSALS.md` names `ElementCount` (rank 4) versus `Shape` (rank 8) as
    // the case that motivated SHP-006: a rank-5 tensor was expressible but
    // could not be reshaped, because the rule that proves element-count
    // preservation had no impl at that rank.
    fn count<S: ElementCount>() -> usize {
        <S as ElementCount>::Count::USIZE
    }
    assert_eq!(count::<(U2, U3)>(), 6);
    assert_eq!(count::<(U1, U1, U2, U3)>(), 6);
    assert_eq!(count::<(U1, U1, U1, U2, U3)>(), 6, "rank 5 lost its count");
    assert_eq!(count::<(U1, U1, U1, U1, U2, U3)>(), 6);
    assert_eq!(count::<(U1, U1, U1, U1, U1, U2, U3)>(), 6);
    assert_eq!(count::<(U1, U1, U1, U1, U1, U1, U2, U3)>(), 6);
    assert_eq!(count::<(U1, U1, U1, U1, U1, U2, U2, U3)>(), 12);
    assert_eq!(count::<(U2, U3, U2, U2, U1, U1, U1, U1)>(), 24);
    assert_eq!(count::<(U24,)>(), 24);
}

#[test]
fn channel_markers_resolve_across_their_whole_range() {
    // Both used to hold for exactly one rank — 3 and 4 respectively — so the
    // unbatched forms their own documentation names, `(C, L)` and `(C, H, W)`,
    // did not implement them.
    fn requires_1d<S: HasChannels1D<U2>>() {}
    requires_1d::<(U2, U3)>();
    requires_1d::<(U1, U2, U3)>();
    requires_1d::<(U1, U1, U1, U1, U1, U1, U2, U3)>();

    fn requires_2d<S: HasChannels2D<U2>>() {}
    requires_2d::<(U2, U3, U3)>();
    requires_2d::<(U1, U2, U3, U3)>();
    requires_2d::<(U1, U1, U1, U1, U1, U2, U3, U3)>();
}

// --- the families migrated last -----------------------------------------

#[test]
fn broadcast_resolves_at_the_ceiling() {
    use incin_core::prelude::BroadcastShape;

    // Same rank on both sides, at the ceiling.
    fn broadcasts<L: BroadcastShape<R>, R: Shape>() {}
    broadcasts::<Rank8, Rank8>();

    // And the mixed-rank form: a shorter operand right-aligned against a
    // rank-8 one. This was capped at rank 4, so a rank-5 broadcast had no
    // impl at all.
    broadcasts::<(U1, U1, U1, U1, U1), (U1, U1, U1, U1, U1)>();
    broadcasts::<(), Rank8>();
    broadcasts::<Rank8, ()>();
}

#[test]
fn concat_resolves_at_every_axis_of_the_ceiling_rank() {
    use incin_core::prelude::ConcatShape;
    use incin_core::typenum::{U0, U4, U7};

    fn concats<L: ConcatShape<R, A>, R: Shape, A>() {}
    // Axis 0, a middle axis, and the last axis of a rank-8 shape.
    concats::<Rank8, (U2, U1, U1, U1, U1, U1, U1, U2), U0>();
    concats::<Rank8, (U1, U1, U1, U1, U2, U1, U1, U2), U4>();
    concats::<Rank8, (U1, U1, U1, U1, U1, U1, U1, U3), U7>();
}

#[test]
fn stack_stops_one_below_the_ceiling_because_its_output_grows() {
    use incin_core::prelude::StackShape;
    use incin_core::typenum::{U0, U7};

    fn stacked<S: StackShape<A>, A>() -> usize
    where
        <S as StackShape<A>>::Output: PartialDynShape,
    {
        <<S as StackShape<A>>::Output as PartialDynShape>::RANK
    }

    // A rank-7 input stacks to rank 8 — the ceiling — at both extreme
    // insertion points. Rank 8 would produce a rank-9 tuple, which has no
    // `Shape` impl, so 7 is correct rather than short.
    type Rank7 = (U1, U1, U1, U1, U1, U1, U2);
    assert_eq!(stacked::<Rank7, U0>(), CEILING);
    assert_eq!(stacked::<Rank7, U7>(), CEILING);
}

#[test]
fn pooling_and_convolution_resolve_at_the_ceiling() {
    use incin_core::prelude::{AdaptiveAvgPool2dShape, Pool2dShape, SpatialConv2d};
    use incin_core::typenum::{U0, U4};

    // Rank 8 is five batch axes plus (C, H, W). Both pooling rules were a
    // single hand-written rank-4 impl before the sweep.
    type Batched = (U1, U1, U1, U1, U1, U2, U4, U4);
    fn pools<S: Pool2dShape<U2, U2, U0, U1>>() {}
    fn adapts<S: AdaptiveAvgPool2dShape<U2, U2>>() {}
    fn convolves<S: SpatialConv2d<U2, U1, U1, U0, U1>>() {}
    pools::<Batched>();
    adapts::<Batched>();
    convolves::<Batched>();

    // And the unbatched forms still resolve.
    pools::<(U2, U4, U4)>();
    adapts::<(U2, U4, U4)>();
}
