//! Blocking the constructor is not enough on its own: a struct literal would
//! bypass it. Both halves of the seal are needed, so both are tested.

use incin_core::exec::{BroadcastSpec, ProofLevel, Validated};
use incin_core::prelude::ShapeBuf;

fn main() {
    let spec = BroadcastSpec::contiguous(
        &ShapeBuf::from_slice(&[2, 3]),
        &ShapeBuf::from_slice(&[2, 3]),
    )
    .unwrap();

    let _forged = Validated {
        descriptor: spec,
        proof: ProofLevel::Static,
    };
}
