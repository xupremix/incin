use incin::backend_authoring::operations::{Descriptor, NoAttributes, OPERATION_CATALOG, op};
use incin::backend_authoring::{Backend, Execute, OperationSpec, StorageBackend};

pub fn accepts_backend_contract<B, O>()
where
    B: Backend + StorageBackend + Execute<O>,
    O: OperationSpec,
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
