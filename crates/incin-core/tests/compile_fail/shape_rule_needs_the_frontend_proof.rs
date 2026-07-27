//! A lowering rule is reachable only for the operand shapes its frontend trait
//! relates. `BroadcastRule` implements `ShapeRule<(L, R)>` where
//! `L: BroadcastShape<R>` and nowhere else, so a pair the shape rule rejects
//! has no descriptor to lower to — the operation is refused before any
//! dimension is looked at, rather than at the point a kernel reads one.

use incin_core::exec::{BroadcastRule, ShapeRule};
use incin_core::typenum::{U3, U4, U5};

fn main() {
    // 4 and 5 are neither equal nor 1, so `(U3, U4)` does not broadcast to
    // `(U3, U5)` and `BroadcastRule` is not implemented for the pair.
    let _ = <BroadcastRule as ShapeRule<((U3, U4), (U3, U5))>>::lower(
        &((Default::default(), Default::default()), (Default::default(), Default::default())),
        (),
    );
}
