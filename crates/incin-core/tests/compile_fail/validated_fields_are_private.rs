//! Blocking the constructor is not enough on its own: a struct literal would
//! bypass it. Both halves of the seal are needed, so both are tested.

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

    let _forged = Validated {
        descriptor: spec,
        proof: ProofLevel::Static,
    };
}
