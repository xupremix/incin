//! A descriptor defined outside `incin-core` would carry no proof that
//! anything validated it, and no lowering rule could produce one. `Sealed` is
//! what makes that unrepresentable rather than merely discouraged.

use incin_core::exec::{DescriptorSchemaVersion, OperationSpec};
use incin_core::prelude::{OperationKind, ShapeBuf};

#[derive(Clone, Debug)]
struct RogueSpec {
    output: ShapeBuf,
}

impl OperationSpec for RogueSpec {
    const KIND: OperationKind = OperationKind::Pointwise;
    const SCHEMA: DescriptorSchemaVersion = DescriptorSchemaVersion::CURRENT;

    fn output(&self) -> &ShapeBuf {
        &self.output
    }
}

fn main() {}
