use core::marker::PhantomData;
use incin_core::prelude::{ConstDType, DType, DTypeDescriptor, DTypeKey, DTypeKind};
use incin_core::tensor::dtype::StorageEncoding;

#[derive(Clone, Debug, PartialEq)]
struct CustomLogical;

impl DType for CustomLogical {
    type Arg = ();
    type Field = PhantomData<Self>;

    fn init(_: ()) -> Self::Field {
        PhantomData
    }

    fn descriptor(_: &Self::Field) -> DTypeDescriptor {
        Self::DESCRIPTOR
    }
}

impl ConstDType for CustomLogical {
    const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::new(
        DTypeKey::new("consumer", "custom-logical", 1),
        DTypeKind::Opaque,
        StorageEncoding::scalar(2, 2),
    );
}

fn main() {
    assert_eq!(CustomLogical::DESCRIPTOR.builtin_id(), None);
}
