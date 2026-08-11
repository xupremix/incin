use incin::backend_authoring::operations::{Descriptor, NoAttributes, OPERATION_CATALOG, op};
use incin::backend_authoring::{Execute, ExecutionDescriptor, Operation, StorageBackend, TensorBackend};

pub fn accepts_backend_contract<B, O>()
where
    B: TensorBackend<f32> + StorageBackend + Execute<O>,
    O: ExecutionDescriptor + Operation,
{
}

pub fn exact_descriptor_contract(_: Option<Descriptor<op::Add>>) -> (&'static str, NoAttributes) {
    (
        OPERATION_CATALOG
            .iter()
            .find(|row| row.name == "add")
            .expect("add is in the canonical catalog")
            .name,
        NoAttributes,
    )
}
