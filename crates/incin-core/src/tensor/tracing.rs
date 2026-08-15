use crate::err::{BackendError, Result};
use crate::exec::catalog::TraceDescriptor;
use crate::exec::spec::ExecutionDescriptor;
use crate::graph_recording::{Graph, TRACING_GRAPH, ValueId};
use crate::tensor::backend::Backend;
use crate::tensor::backend::SupportsDType;
use crate::tensor::backend::*;
use crate::tensor::device::DeviceId;
use crate::tensor::dtype::{DType, DTypeDescriptor};
// removed RefCell
// Private per B-3 (.agents/API_DESIGN.md "pub(crate) is default"): this used
// to be `pub`, letting any downstream crate `.lock()` the raw `Mutex<Graph>`
// and call arbitrary `Graph` methods directly, even though `Graph` itself is
// `pub(crate)`. The three functions below are the only operations downstream
// crates actually need (draining/snapshotting the graph, marking an input/
// output value) — everything else about `Graph`'s shape stays encapsulated.
/// Drain the process-wide tracing graph, returning everything recorded since
/// the last call (or since startup).
pub fn extract_graph() -> Graph {
    let mut b = TRACING_GRAPH.lock();
    core::mem::take(&mut *b)
}
/// Clone the process-wide tracing graph's current contents WITHOUT draining
/// it (unlike `extract_graph`) — used by telemetry to snapshot mid-flight.
/// Returns `None` if the graph is momentarily locked elsewhere.
pub fn tracing_graph_snapshot() -> Option<Graph> {
    TRACING_GRAPH.try_lock().map(|g| (*g).clone())
}

/// Mark `value_id` as a graph input (e.g. for ONNX export / visualization).
pub fn tracing_mark_input(value_id: ValueId) {
    TRACING_GRAPH.lock().mark_input(value_id);
}

/// Marks a traced tensor input while retaining its frontend shape proof.
pub fn tracing_mark_input_typed<S>(value_id: ValueId)
where
    S: crate::shapes::Shape + crate::exec::shape_projection::ShapeProjection,
{
    TRACING_GRAPH.lock().mark_input_with_shape::<S>(value_id);
}

/// Mark `value_id` as a graph output (e.g. for ONNX export / visualization).
pub fn tracing_mark_output(value_id: ValueId) {
    TRACING_GRAPH.lock().mark_output(value_id);
}

/// A transparent wrapper around any `Backend` `B` that additionally
/// records every operation as a node in the process-wide `TRACING_GRAPH`,
/// building up an exportable computation graph (used by ONNX export and
/// graph visualization) alongside the real computation `B` performs.
use crate::exec::capability::Capabilities;

#[derive(Clone)]
pub struct TracingBackend<B: Backend> {
    _marker: core::marker::PhantomData<B>,
}

impl<B: Backend> core::default::Default for TracingBackend<B> {
    fn default() -> Self {
        Self {
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B: Backend> Capabilities for TracingBackend<B> {
    fn support(&self, query: &crate::exec::CapabilityQuery) -> crate::exec::SupportLevel {
        B::default().support(query)
    }
}

#[derive(Clone)]
/// A `TracingBackend` storage handle: the real backend's storage plus the
/// `ValueId` identifying this tensor's node in the tracing graph.
pub struct TracingTensor<T> {
    /// The wrapped real backend storage.
    pub inner: T,
    /// This tensor's node id in the tracing graph.
    pub value_id: ValueId,
}

impl<T: crate::tensor::backend::ExecuteOutput> crate::tensor::backend::ExecuteOutput
    for TracingTensor<T>
{
}

#[derive(Clone)]
/// A `TracingBackend` variable handle: the real backend's `Var<K>` plus
/// the `ValueId` identifying this variable's node in the tracing graph.
pub struct TracingVar<V> {
    /// The wrapped real backend variable.
    pub inner: V,
    /// This variable's node id in the tracing graph.
    pub value_id: ValueId,
}

impl<V> From<TracingTensor<V>> for TracingVar<V> {
    fn from(storage: TracingTensor<V>) -> Self {
        Self {
            inner: storage.inner,
            value_id: storage.value_id,
        }
    }
}

impl<B: Backend> TracingBackend<B> {
    fn canonical_graph_attributes<O>(
        descriptor: &crate::exec::catalog::Descriptor<O>,
    ) -> Result<
        alloc::collections::BTreeMap<alloc::string::String, crate::graph_recording::AttributeValue>,
    >
    where
        O: crate::exec::catalog::Operation,
    {
        #[cfg(feature = "std")]
        {
            Ok(descriptor
                .trace_attributes()
                .map_err(|reason| BackendError::InvalidInput {
                    operation: crate::shapes::error::OperationKind::Storage,
                    reason,
                })?)
        }
        #[cfg(not(feature = "std"))]
        {
            let _ = descriptor;
            Ok(alloc::collections::BTreeMap::new())
        }
    }
}

impl<B: Backend> StorageBackend for TracingBackend<B> {
    const BACKEND_NAME: &'static str = "tracing";
    /// The wrapped backend's storage plus a tracing-graph node id.
    type Storage<K: super::dtype::DType> = TracingTensor<B::Storage<K>>;
    /// Delegates to `B`'s device type.
    type Device = B::Device;

    fn metadata<K: super::dtype::DType>(storage: &Self::Storage<K>) -> &crate::exec::TensorMeta {
        B::metadata(&storage.inner)
    }

    fn fresh_autograd_identity<K: super::dtype::DType>(
        storage: Self::Storage<K>,
    ) -> Self::Storage<K> {
        TracingTensor {
            inner: B::fresh_autograd_identity(storage.inner),
            value_id: storage.value_id,
        }
    }

    fn execution_storage<K: super::dtype::DType>(
        storage: &Self::Storage<K>,
    ) -> (&dyn core::any::Any, Option<usize>)
    where
        Self::Storage<K>: core::any::Any,
        B::Storage<K>: core::any::Any,
    {
        (&storage.inner, Some(storage.value_id))
    }
}

impl<B: Backend + crate::tensor::backend::Execute<O>, O: crate::exec::catalog::Operation>
    crate::tensor::backend::Execute<O> for TracingBackend<B>
{
    type Output = TracingTensor<B::Output>;

    fn execute(
        &self,
        request: crate::tensor::backend::ExecutionRequest<'_, O, Self>,
    ) -> core::result::Result<Self::Output, BackendError> {
        let inner_backend = B::default();
        let inner_context = crate::exec::ExecutionContext::from_scope(inner_backend.clone());
        let inner_inputs: alloc::vec::Vec<_> = request
            .inputs
            .iter()
            .map(crate::exec::request::TensorHandle::execution_view)
            .collect();
        let inner_res = inner_backend.execute(crate::tensor::backend::ExecutionRequest {
            operation: request.operation,
            inputs: &inner_inputs,
            context: &inner_context,
            payload: request.payload,
        })?;

        let output_id = {
            let mut g = TRACING_GRAPH.lock();
            let output_id = g.add_value(
                request
                    .operation
                    .descriptor()
                    .output_shape()
                    .cloned()
                    .or_else(|| {
                        request
                            .inputs
                            .first()
                            .map(|input| input.metadata().shape.clone())
                    })
                    .unwrap_or(crate::shapes::ShapeBuf::SCALAR)
                    .as_ref()
                    .to_vec(),
                request
                    .operation
                    .descriptor()
                    .trace_output_dtype(request.inputs)
                    .ok_or(BackendError::InvalidInput {
                        operation: crate::shapes::error::OperationKind::Storage,
                        reason: "tracing requires output dtype metadata",
                    })?,
                None,
            );
            let metadata = request.inputs.first().map(|input| input.metadata());
            if let Some(metadata) = metadata {
                let _ =
                    g.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
            }
            let inputs = request
                .inputs
                .iter()
                .filter_map(crate::exec::request::TensorHandle::tracing_value)
                .collect();
            let descriptor_payload = request
                .operation
                .descriptor()
                .trace_descriptor_payload()
                .map_err(|reason| BackendError::InvalidInput {
                    operation: crate::shapes::error::OperationKind::Storage,
                    reason,
                })?;
            g.add_node_with_descriptor_payload(
                request.operation.descriptor().trace_identity(),
                inputs,
                vec![output_id],
                Self::canonical_graph_attributes(request.operation.descriptor()).map_err(|_| {
                    BackendError::InvalidInput {
                        operation: crate::shapes::error::OperationKind::Storage,
                        reason: "tracing canonical attribute projection failed",
                    }
                })?,
                descriptor_payload,
            );
            output_id
        };

        Ok(TracingTensor {
            inner: inner_res,
            value_id: output_id,
        })
    }
}

impl<B: Backend> Backend for TracingBackend<B> {
    /// Delegates to `B`'s own inner backend (tracing is not itself a dispatch layer).
    type InnerBackend = B::InnerBackend;
}

impl<B: Backend + crate::tensor::backend::HostReadback> crate::tensor::backend::HostReadback
    for TracingBackend<B>
{
    fn float_to_vec1<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<f64>> {
        B::float_to_vec1(&t.inner)
    }

    fn int_to_vec1<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<i64>> {
        B::int_to_vec1(&t.inner)
    }
}

impl<B: Backend + crate::tensor::backend::HostInterop> crate::tensor::backend::HostInterop
    for TracingBackend<B>
{
    /// Delegates to `B::from_bytes`, additionally recording the result
    /// as a graph initializer (constant input) node.
    fn from_bytes<K: super::dtype::DType>(
        bytes: &[u8],
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as crate::tensor::backend::StorageBackend>::Storage<K>> {
        let inner = B::from_bytes(bytes, shape, dtype, device)?;
        let value_id = {
            let mut g = TRACING_GRAPH.lock();
            let id = g.add_value(shape.to_vec(), dtype, None);
            g.initializers.insert(id, bytes.to_vec());
            let metadata = B::metadata(&inner);
            let _ = g.set_value_placement(id, Some(metadata.device), Some(metadata.layout));
            id
        };
        Ok(TracingTensor { inner, value_id })
    }
    /// Delegates to `B::to_bytes`.
    fn to_bytes<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<u8>> {
        B::to_bytes(&t.inner)
    }
}

impl<B: Backend + crate::tensor::backend::AutogradBackend> crate::tensor::backend::AutogradBackend
    for TracingBackend<B>
{
    type Grads = B::Grads;

    fn backward<K: super::dtype::DType>(
        loss: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<Self::Grads> {
        B::backward(&loss.inner)
    }

    fn backward_with<K: super::dtype::DType>(
        loss: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
        seed: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<Self::Grads> {
        B::backward_with(&loss.inner, &seed.inner)
    }

    fn get_grad<K: super::dtype::DType>(
        _var: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
        _grads: &Self::Grads,
    ) -> Result<Option<<Self as crate::tensor::backend::StorageBackend>::Storage<K>>> {
        Ok(None)
    }
}

impl<B: VariableBackend> VariableBackend for TracingBackend<B> {
    type Var<K: DType> = TracingVar<<B as crate::tensor::backend::VariableBackend>::Var<K>>;

    fn var_as_tensor<K: DType>(var: &Self::Var<K>) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::var_as_tensor(&var.inner)?;
        Ok(TracingTensor {
            inner,
            value_id: var.value_id,
        })
    }

    fn var_from_tensor<K: DType>(
        tensor: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<Self::Var<K>> {
        let inner = B::var_from_tensor(&tensor.inner)?;
        Ok(TracingVar {
            inner,
            value_id: tensor.value_id,
        })
    }

    fn assign_var<K: DType>(
        var: &mut Self::Var<K>,
        tensor: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<()> {
        B::assign_var(&mut var.inner, &tensor.inner)
    }
}

impl<B: Backend + SupportsDType<K>, K: DType> SupportsDType<K> for TracingBackend<B> {
    fn resolve_dtype(field: &K::Field, device: &DeviceId) -> Result<DTypeDescriptor> {
        B::resolve_dtype(field, device)
    }
}

impl<B, NewD> StorageTransfer<NewD> for TracingBackend<B>
where
    B: Backend + StorageTransfer<NewD> + crate::tensor::backend::HostInterop,
    <B as StorageTransfer<NewD>>::Output: crate::tensor::backend::HostInterop,
    NewD: crate::tensor::device::Device,
{
    type Output = TracingBackend<<B as StorageTransfer<NewD>>::Output>;

    fn transfer_storage<K: super::dtype::DType>(
        storage: &Self::Storage<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as StorageBackend>::Storage<K>>
    where
        Self::Output: SupportsDType<K>,
    {
        let shape = B::shape(&storage.inner);
        let bytes = B::to_bytes(&storage.inner)?;
        let device_id = NewD::to_incin(device)?;
        let dtype_id = <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &device_id)?;
        let inner = <<B as StorageTransfer<NewD>>::Output as HostInterop>::from_bytes::<K>(
            &bytes, &shape, dtype_id, &device_id,
        )?;
        Ok(TracingTensor {
            inner,
            value_id: storage.value_id,
        })
    }
}

impl<B, NewD> TransferTo<NewD> for TracingBackend<B>
where
    B: Backend + TransferTo<NewD> + crate::tensor::backend::HostInterop,
    <B as StorageTransfer<NewD>>::Output: crate::tensor::backend::HostInterop,
    NewD: crate::tensor::device::Device,
    <B as StorageTransfer<NewD>>::Output: VariableBackend<Device = NewD>,
{
    fn transfer_var<K: DType>(
        variable: &Self::Var<K>,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as crate::tensor::backend::VariableBackend>::Var<K>>
    where
        Self::Output: SupportsDType<K>,
    {
        let source = B::var_as_tensor::<K>(&variable.inner)?;
        let shape = B::shape(&source);
        let bytes = B::to_bytes(&source)?;
        let device_id = NewD::to_incin(device)?;
        let dtype_id = <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &device_id)?;
        let storage = <<B as StorageTransfer<NewD>>::Output as HostInterop>::from_bytes::<K>(
            &bytes, &shape, dtype_id, &device_id,
        )?;
        let inner =
            <<B as StorageTransfer<NewD>>::Output as VariableBackend>::var_from_tensor(&storage)?;
        Ok(TracingVar {
            inner,
            value_id: variable.value_id,
        })
    }
}
