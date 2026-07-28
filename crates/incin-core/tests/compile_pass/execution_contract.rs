use incin_backends::cpu::CpuBackendImpl;
use incin_backends::wgpu::WgpuBackendImpl;
use incin_core::exec::{
    Alignment, BroadcastRule, BroadcastSpec, ExecutionContext, ShapeRule, TensorHandle, TensorMeta,
};
use incin_core::prelude::{
    BackendError, Cpu, DType, DeviceId, DTypeId, Execute, ExecutionRequest, Local, ShapeBuf,
    StorageBackend, SupportsDType, Wgpu,
};
use incin_core::typenum::{U2, U3};

struct Storage {
    metadata: TensorMeta,
}

struct Probe;

impl StorageBackend for Probe {
    type Storage<K: DType> = Storage;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        &storage.metadata
    }
}

impl Execute<BroadcastSpec> for Probe {
    type Output = usize;

    fn execute(
        &self,
        request: ExecutionRequest<'_, BroadcastSpec, Self>,
    ) -> Result<Self::Output, BackendError> {
        assert_eq!(request.operation.descriptor().output.dims(), &[2, 3]);
        assert!(request.inputs[0].downcast_ref::<Storage>().is_some());
        Ok(request.inputs.len())
    }
}

fn supports<K: DType, B: SupportsDType<K>>() {}

fn main() {
    supports::<f64, CpuBackendImpl<f32, Cpu>>();
    supports::<f32, WgpuBackendImpl<f32, Wgpu>>();

    let validated = <BroadcastRule as ShapeRule<((U2, U3), (U2, U3))>>::lower(
        &Default::default(),
        (),
    )
    .unwrap();
    let storage = Storage {
        metadata: TensorMeta::contiguous(
            ShapeBuf::from_slice(&[2, 3]),
            DTypeId::F32,
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
