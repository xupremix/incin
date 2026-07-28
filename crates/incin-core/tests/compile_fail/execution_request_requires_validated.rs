//! `ExecutionRequest` accepts the sealed `Validated<O>` proof, never a bare
//! descriptor that any caller can construct.

use incin_core::exec::{BroadcastSpec, ExecutionContext, TensorMeta};
use incin_core::prelude::{
    Cpu, DType, ExecutionRequest, ShapeBuf, StorageBackend, StrideBuf,
};

struct StorageOnly;

impl StorageBackend for StorageOnly {
    type Storage<K: DType> = TensorMeta;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage
    }
}

fn main() {
    let shape = ShapeBuf::from_slice(&[2, 3]);
    let strides = StrideBuf::contiguous_for(
        &shape,
        incin_core::prelude::OperationKind::Broadcast,
    )
    .unwrap();
    let descriptor = BroadcastSpec::new(&shape, &strides, &shape, &strides).unwrap();
    let context = ExecutionContext::new(StorageOnly);
    let inputs = [];
    let _: ExecutionRequest<'_, BroadcastSpec, StorageOnly> = ExecutionRequest {
        operation: &descriptor,
        inputs: &inputs,
        context: &context,
    };
}
