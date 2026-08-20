//! The type-level two-rank vocabulary.

use typenum::{U0, U1, Unsigned};

/// A type-level rank admitted by an exactly two-rank context.
///
/// This trait is sealed to [`U0`] and [`U1`]. A static `U2` rank therefore
/// fails to compile instead of reaching a runtime branch.
#[diagnostic::on_unimplemented(
    message = "`{Self}` is not a valid rank in a two-rank distributed context",
    label = "expected typenum::U0 or typenum::U1"
)]
pub trait StaticTwoRank: sealed::SealedRank + Unsigned + 'static {
    /// The rank encoded by this marker.
    const RANK: usize;
}

impl StaticTwoRank for U0 {
    const RANK: usize = 0;
}

impl StaticTwoRank for U1 {
    const RANK: usize = 1;
}

mod sealed {
    use typenum::{U0, U1};

    pub trait SealedRank {}
    impl SealedRank for U0 {}
    impl SealedRank for U1 {}
}
