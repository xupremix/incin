use core::marker::PhantomData;
use incin_backends::dist::{CollectiveTuningProblem, TuneAllGather};
use incin_core::dist::mesh::TopologyFingerprint;
use incin_core::dist::{CollectiveDType, GroupId};
use incin_core::exec::Determinism;
use incin_core::prelude::{ConstDType, DType, DTypeDescriptor, DTypeKey, DTypeKind};
use incin_core::tensor::dtype::StorageEncoding;
use incin_core::typenum::U16;

#[derive(Clone, Debug, PartialEq)]
struct AcmePosit24;

impl DType for AcmePosit24 {
    type Arg = ();
    type Field = PhantomData<Self>;

    fn init(_: ()) -> Self::Field {
        PhantomData
    }

    fn descriptor(_: &Self::Field) -> DTypeDescriptor {
        Self::DESCRIPTOR
    }
}

impl ConstDType for AcmePosit24 {
    const DESCRIPTOR: DTypeDescriptor = DTypeDescriptor::new(
        DTypeKey::new("acme", "posit24", 1),
        DTypeKind::Opaque,
        StorageEncoding::scalar(3, 1),
    );
}

impl CollectiveDType for AcmePosit24 {}

fn custom_dtype_missing_builtin_id(topology: &TopologyFingerprint) {
    let _ = CollectiveTuningProblem::new_static::<AcmePosit24, U16, TuneAllGather>(
        GroupId::new(1, 2).unwrap(),
        topology,
        Determinism::Permitted,
        1024,
    );
}

fn main() {}
