//! A descriptor defined outside `incin-core` would carry no proof that
//! anything validated it, and no lowering rule could produce one. `Sealed` is
//! what makes that unrepresentable rather than merely discouraged.

use incin_core::exec::CanonicalOperation;

#[derive(Clone, Debug)]
struct RogueSpec;

impl CanonicalOperation for RogueSpec {}

fn main() {}
