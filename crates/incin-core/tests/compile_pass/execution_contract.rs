extern crate incin_core as incin;

use incin_backends::cpu::CpuBackendImpl;
use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::exec::{
    Alignment, Descriptor, ExecutionContext, LogicalTensorMeta, TensorHandle, TensorMeta, op,
};
use incin_core::backend_authoring::operations::NoAttributes;
use incin_core::backend_authoring::{Execute, ExecutionRequest, StorageBackend, SupportsDType};
use incin_core::prelude::{BackendError, Cpu, DeviceId, DType, DTypeId, Local, Shape, ShapeBuf, Wgpu};

#[derive(Clone)]
struct Storage {
    metadata: TensorMeta,
}

struct Probe;

impl StorageBackend for Probe {
    const BACKEND_NAME: &'static str = "Probe";
    type Storage<K: DType> = Storage;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        &storage.metadata
    }
}

impl Execute<op::Add> for Probe {
    type Output = usize;

    fn execute_shaped<ShapeTy: Shape>(
        &self,
        request: ExecutionRequest<'_, op::Add, Self>,
    ) -> Result<Self::Output, BackendError> {
        assert_eq!(request.operation.descriptor().outputs().len(), 1);
        assert!(request.inputs[0].downcast_ref::<Storage>().is_some());
        Ok(request.inputs.len())
    }
}

fn supports<K: DType, B: SupportsDType<K>>() {}

fn main() {
    supports::<f64, CpuBackendImpl<Cpu>>();
    supports::<f32, WgpuBackendImpl<Wgpu>>();

    let shape = ShapeBuf::from_slice(&[2, 3]);
    let validated = Descriptor::<op::Add>::infer_runtime(
        NoAttributes,
        vec![
            LogicalTensorMeta { shape: Some(shape.clone()), dtype: None, device: None },
            LogicalTensorMeta { shape: Some(shape), dtype: None, device: None },
        ],
    ).unwrap();
    let storage = Storage {
        metadata: TensorMeta::contiguous(
            ShapeBuf::from_slice(&[2, 3]),
            DTypeId::F32.descriptor(),
            DeviceId::cpu(),
            Alignment::of::<f32>(),
            6,
        )
        .unwrap(),
    };
    let handles = [TensorHandle::from_storage::<Probe, f32, Local>(&storage)];
    let context = ExecutionContext::new(Probe);
    let request = ExecutionRequest {
        operation: &validated,
        inputs: &handles,
        context: &context,
    };
    assert_eq!(Execute::execute(context.backend(), request).unwrap(), 1);
}
