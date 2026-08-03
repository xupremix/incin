use incin::backend_authoring::{Backend, Execute, OperationSpec, StorageBackend};

pub fn accepts_backend_contract<B, O>()
where
    B: Backend + StorageBackend + Execute<O>,
    O: OperationSpec,
{
}
