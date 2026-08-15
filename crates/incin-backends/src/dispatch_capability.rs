use crate::dispatch::DispatchBackend;
use incin_core::backend_authoring::Backend;
use incin_core::backend_authoring::{AutogradBackend, HostInterop, VariableBackend};
use incin_core::error::Error;
use incin_core::exec::{Capabilities, PrecisionCapabilities, PrecisionRequest, ResolvedPrecision};
use incin_core::tensor::device::Device;

impl<D: Device> PrecisionCapabilities for DispatchBackend<D> {
    fn native_precision(
        &self,
        request: &PrecisionRequest,
    ) -> incin_core::error::Result<ResolvedPrecision> {
        #[cfg(feature = "cpu")]
        {
            crate::cpu::CpuBackendImpl::<incin_core::tensor::device::Cpu>::new()
                .native_precision(request)
        }
        #[cfg(not(feature = "cpu"))]
        {
            Err(Error::BackendUnavailable {
                backend: "DispatchBackend",
            })
        }
    }
}

impl<D: Device> Capabilities for DispatchBackend<D> {
    fn support(&self, query: &incin_core::exec::CapabilityQuery) -> incin_core::exec::SupportLevel {
        #[cfg(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "metal"))]
        {
            #[cfg(feature = "cpu")]
            {
                crate::cpu::CpuBackendImpl::<incin_core::tensor::device::Cpu>::default()
                    .support(query)
            }
            #[cfg(all(
                not(feature = "cpu"),
                any(feature = "wgpu", feature = "cuda", feature = "metal")
            ))]
            {
                let _ = query;
                incin_core::exec::SupportLevel::Native
            }
        }
        #[cfg(not(any(feature = "cpu", feature = "wgpu", feature = "cuda", feature = "metal")))]
        {
            let incin_core::exec::OperationIdentity::Builtin(operation) = &query.operation else {
                return incin_core::exec::SupportLevel::Unsupported(
                    incin_core::exec::UnsupportedReason::CustomOperation {
                        operation: match &query.operation {
                            incin_core::exec::OperationIdentity::Custom(operation) => {
                                operation.clone()
                            }
                            incin_core::exec::OperationIdentity::Builtin(_) => unreachable!(),
                        },
                    },
                );
            };
            incin_core::exec::SupportLevel::Unsupported(
                incin_core::exec::UnsupportedReason::Operation {
                    operation: *operation,
                },
            )
        }
    }
}
