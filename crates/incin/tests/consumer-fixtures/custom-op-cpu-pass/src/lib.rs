use incin::backend_authoring::{
    Backend, Execute, ExecutionRequest, LogicalTensorMeta, Operation, OperationIdentity,
    OperationKey, StorageBackend,
};
use incin::prelude::{BackendError, Cpu, Device, Shape};
use incin_backends::cpu::{CpuBackendImpl, CpuStorage};

#[derive(Debug, Clone)]
pub struct CompanyOp;

impl Operation for CompanyOp {
    type Attributes = ();

    const KEY: OperationKey = OperationKey {
        namespace: std::borrow::Cow::Borrowed("company.example"),
        name: std::borrow::Cow::Borrowed("identity"),
        version: 1,
    };

    const IDENTITY: OperationIdentity = OperationIdentity::Custom(Self::KEY);

    fn infer_outputs(
        _: &(),
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, incin::backend_authoring::DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

impl Execute<CompanyOp> for CpuBackendImpl<Cpu> {
    type Output = CpuStorage;

    fn execute_shaped<S: Shape>(
        &self,
        request: ExecutionRequest<'_, CompanyOp, Self>,
    ) -> Result<Self::Output, BackendError> {
        request
            .inputs
            .first()
            .and_then(|input| input.downcast_ref::<CpuStorage>())
            .cloned()
            .ok_or_else(|| BackendError::unsupported(
                Self::BACKEND_NAME,
                incin::backend_authoring::UnsupportedReason::CustomOperation {
                    operation: CompanyOp::KEY,
                },
            ))
    }
}

fn accepts_external_executor<B>()
where
    B: Backend + Execute<CompanyOp>,
{
}

pub fn custom_operation_on_cpu_is_a_downstream_impl() {
    accepts_external_executor::<CpuBackendImpl<Cpu>>();
}
