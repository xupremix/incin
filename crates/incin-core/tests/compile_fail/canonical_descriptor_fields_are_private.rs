use core::marker::PhantomData;
use incin_core::exec::catalog::{Descriptor, LogicalTensorMeta, NoAttributes, op};

fn main() {
    let _forged = Descriptor::<op::Add> {
        attributes: NoAttributes,
        outputs: vec![LogicalTensorMeta::unknown()],
        marker: PhantomData,
    };
}
