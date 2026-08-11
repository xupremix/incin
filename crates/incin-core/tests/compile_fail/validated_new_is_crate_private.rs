//! `Validated::new` is the whole seal. If an outside caller can reach it, any
//! descriptor can be stamped with any proof level, and a backend that trusts
//! the stamp is trusting nothing.

use incin_core::backend_authoring::operations::NoAttributes;
use incin_core::exec::{Descriptor, LogicalTensorMeta, ProofLevel, Validated, op};
use incin_core::prelude::ShapeBuf;

fn main() {
    let shape = ShapeBuf::from_slice(&[2, 3]);
    let spec = Descriptor::<op::Add>::infer_runtime(
        NoAttributes,
        vec![
            LogicalTensorMeta { shape: Some(shape.clone()), dtype: None, device: None },
            LogicalTensorMeta { shape: Some(shape), dtype: None, device: None },
        ],
    ).unwrap().into_descriptor();

    // A runtime-built descriptor claiming a compile-time proof: exactly the
    // forgery the crate-private constructor exists to prevent.
    let _forged = Validated::new(spec, ProofLevel::Static);
}
