use super::CpuBackendImpl;
use incin_core::backend_authoring::StorageBackend;
use incin_core::exec::{Capabilities, CapabilityQuery, SupportLevel};
use incin_core::tensor::device::{Cpu, Device};

impl<D: Device> Capabilities for CpuBackendImpl<D> {
    fn support(&self, query: &CapabilityQuery) -> SupportLevel {
        crate::capability::support(incin_core::tensor::device::DeviceKind::Cpu, query)
    }
}

pub(crate) const CPU_NAME: &str = <CpuBackendImpl<Cpu> as StorageBackend>::BACKEND_NAME;
