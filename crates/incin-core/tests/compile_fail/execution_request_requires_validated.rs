//! `ExecutionRequest` accepts the sealed `Validated<O>` proof, never a bare
//! descriptor that any caller can construct.

use incin_core::exec::{Descriptor, ExecutionContext, LogicalTensorMeta, TensorMeta, op};
use incin_core::backend_authoring::operations::NoAttributes;
use incin_core::backend_authoring::{ExecutionRequest, StorageBackend};
use incin_core::prelude::{Cpu, DType, ShapeBuf};

struct StorageOnly;

impl StorageBackend for StorageOnly {
    const BACKEND_NAME: &'static str = "StorageOnly";
    type Storage<K: DType> = TensorMeta;
    type Device = Cpu;

    fn metadata<K: DType>(storage: &Self::Storage<K>) -> &TensorMeta {
        storage
    }
}

fn main() {
    let shape = ShapeBuf::from_slice(&[2, 3]);
    let descriptor = Descriptor::<op::Add>::infer_runtime(
        NoAttributes,
        vec![
            LogicalTensorMeta { shape: Some(shape.clone()), dtype: None, device: None },
            LogicalTensorMeta { shape: Some(shape), dtype: None, device: None },
        ],
    ).unwrap().into_descriptor();
    let context = ExecutionContext::new(StorageOnly);
    let inputs = [];
    let _: ExecutionRequest<'_, Descriptor<op::Add>, StorageOnly> = ExecutionRequest {
        operation: &descriptor,
        inputs: &inputs,
        context: &context,
    };
}
