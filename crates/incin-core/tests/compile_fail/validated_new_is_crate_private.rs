//! `Validated::new` is the whole seal. If an outside caller can reach it, any
//! descriptor can be stamped with any proof level, and a backend that trusts
//! the stamp is trusting nothing.

use incin_core::exec::{BroadcastSpec, ProofLevel, Validated};
use incin_core::prelude::ShapeBuf;

fn main() {
    let spec = BroadcastSpec::contiguous(
        &ShapeBuf::from_slice(&[2, 3]),
        &ShapeBuf::from_slice(&[2, 3]),
        None,
    )
    .unwrap();

    // A runtime-built descriptor claiming a compile-time proof: exactly the
    // forgery the crate-private constructor exists to prevent.
    let _forged = Validated::new(spec, ProofLevel::Static);
}
