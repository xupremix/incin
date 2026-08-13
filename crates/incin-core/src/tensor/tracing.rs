use crate::exec::catalog::TraceDescriptor;
use crate::exec::spec::ExecutionDescriptor;
use crate::graph::{Graph, ValueId};
use crate::prelude::*;
use crate::tensor::backend::*;
// removed RefCell
use spin::{Lazy, Mutex};

// Private per B-3 (.agents/API_DESIGN.md "pub(crate) is default"): this used
// to be `pub`, letting any downstream crate `.lock()` the raw `Mutex<Graph>`
// and call arbitrary `Graph` methods directly, even though `Graph` itself is
// `pub(crate)`. The three functions below are the only operations downstream
// crates actually need (draining/snapshotting the graph, marking an input/
// output value) — everything else about `Graph`'s shape stays encapsulated.
pub(crate) static TRACING_GRAPH: Lazy<Mutex<Graph>> = Lazy::new(|| Mutex::new(Graph::new()));

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
pub fn tracing_mark_input_typed<S: crate::prelude::Shape>(value_id: ValueId) {
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
/// A `TracingBackend` variable handle: the real backend's `RawVar` plus
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
    fn traced_dtype<K: super::dtype::DType>(storage: &B::Storage<K>) -> DTypeDescriptor {
        B::storage_dtype(storage).unwrap_or_else(|| K::descriptor(&K::Field::default()))
    }

    fn record_value<K: super::dtype::DType>(storage: &B::Storage<K>) -> ValueId {
        let shape = B::shape(storage);
        let dtype = Self::traced_dtype(storage);
        let metadata = B::metadata(storage);
        let mut graph = TRACING_GRAPH.lock();
        let value_id = graph.add_value(shape.as_ref().to_vec(), dtype, None);
        let _ = graph.set_value_placement(value_id, Some(metadata.device), Some(metadata.layout));
        value_id
    }

    fn trace_canonical_unary<O, K: super::dtype::DType>(
        t: &TracingTensor<B::Storage<K>>,
        inner_res: &B::Storage<K>,
    ) -> Result<TracingTensor<B::Storage<K>>>
    where
        O: crate::exec::catalog::CanonicalOperation<
                Attributes = crate::exec::catalog::NoAttributes,
            >,
    {
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(
            crate::exec::catalog::NoAttributes,
            vec![crate::exec::catalog::LogicalTensorMeta {
                shape: Some(B::shape(&t.inner)),
                dtype: Some(Self::traced_dtype(&t.inner)),
                device: Some(B::metadata(&t.inner).device),
            }],
        )
        .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical unary descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let shape = B::shape(inner_res);
        let mut graph = TRACING_GRAPH.lock();
        let output_id =
            graph.add_value(shape.as_ref().to_vec(), Self::traced_dtype(inner_res), None);
        let metadata = B::metadata(inner_res);
        let _ = graph.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            vec![t.value_id],
            vec![output_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok(TracingTensor {
            inner: inner_res.clone(),
            value_id: output_id,
        })
    }

    fn trace_canonical_unary_with_attributes<O, K: super::dtype::DType>(
        t: &TracingTensor<B::Storage<K>>,
        inner_res: &B::Storage<K>,
        attributes: O::Attributes,
    ) -> Result<TracingTensor<B::Storage<K>>>
    where
        O: crate::exec::catalog::CanonicalOperation,
        O::Attributes: crate::exec::catalog::AttributeContract,
    {
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(
            attributes,
            vec![crate::exec::catalog::LogicalTensorMeta {
                shape: Some(B::shape(&t.inner)),
                dtype: Some(Self::traced_dtype(&t.inner)),
                device: Some(B::metadata(&t.inner).device),
            }],
        )
        .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical unary descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let shape = B::shape(inner_res);
        let mut graph = TRACING_GRAPH.lock();
        let output_id =
            graph.add_value(shape.as_ref().to_vec(), Self::traced_dtype(inner_res), None);
        let metadata = B::metadata(inner_res);
        let _ = graph.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            vec![t.value_id],
            vec![output_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok(TracingTensor {
            inner: inner_res.clone(),
            value_id: output_id,
        })
    }

    fn trace_canonical_shape_with_attributes<O, K: super::dtype::DType>(
        t: &TracingTensor<B::Storage<K>>,
        inner_res: &B::Storage<K>,
        attributes: O::Attributes,
    ) -> Result<TracingTensor<B::Storage<K>>>
    where
        O: crate::exec::catalog::CanonicalOperation,
        O::Attributes: crate::exec::catalog::AttributeContract,
    {
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(
            attributes,
            vec![crate::exec::catalog::LogicalTensorMeta {
                shape: Some(B::shape(&t.inner)),
                dtype: Some(Self::traced_dtype(&t.inner)),
                device: Some(B::metadata(&t.inner).device),
            }],
        )
        .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical shape descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let mut graph = TRACING_GRAPH.lock();
        let output_id = graph.add_value(
            B::shape(inner_res).as_ref().to_vec(),
            Self::traced_dtype(inner_res),
            None,
        );
        let metadata = B::metadata(inner_res);
        let _ = graph.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            vec![t.value_id],
            vec![output_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok(TracingTensor {
            inner: inner_res.clone(),
            value_id: output_id,
        })
    }

    fn trace_canonical_multi_shape<O, K: super::dtype::DType>(
        tensors: &[&TracingTensor<B::Storage<K>>],
        inner_res: &B::Storage<K>,
        attributes: O::Attributes,
    ) -> Result<TracingTensor<B::Storage<K>>>
    where
        O: crate::exec::catalog::CanonicalOperation,
        O::Attributes: crate::exec::catalog::AttributeContract,
    {
        let inputs = tensors
            .iter()
            .map(|tensor| crate::exec::catalog::LogicalTensorMeta {
                shape: Some(B::shape(&tensor.inner)),
                dtype: Some(Self::traced_dtype(&tensor.inner)),
                device: Some(B::metadata(&tensor.inner).device),
            })
            .collect();
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(attributes, inputs)
            .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical multi-input descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let mut graph = TRACING_GRAPH.lock();
        let output_id = graph.add_value(
            B::shape(inner_res).as_ref().to_vec(),
            Self::traced_dtype(inner_res),
            None,
        );
        let metadata = B::metadata(inner_res);
        let _ = graph.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            tensors.iter().map(|tensor| tensor.value_id).collect(),
            vec![output_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok(TracingTensor {
            inner: inner_res.clone(),
            value_id: output_id,
        })
    }

    fn trace_canonical_inputs_with_attributes<O, KOut: super::dtype::DType>(
        input_ids: alloc::vec::Vec<ValueId>,
        inputs: alloc::vec::Vec<crate::exec::catalog::LogicalTensorMeta>,
        inner_res: &B::Storage<KOut>,
        attributes: O::Attributes,
    ) -> Result<TracingTensor<B::Storage<KOut>>>
    where
        O: crate::exec::catalog::CanonicalOperation,
        O::Attributes: crate::exec::catalog::AttributeContract,
    {
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(attributes, inputs)
            .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical heterogeneous descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let mut graph = TRACING_GRAPH.lock();
        let output_id = graph.add_value(
            B::shape(inner_res).as_ref().to_vec(),
            Self::traced_dtype(inner_res),
            None,
        );
        let metadata = B::metadata(inner_res);
        let _ = graph.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            input_ids,
            vec![output_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok(TracingTensor {
            inner: inner_res.clone(),
            value_id: output_id,
        })
    }

    fn trace_canonical_binary<O, K: super::dtype::DType>(
        lhs: &TracingTensor<B::Storage<K>>,
        rhs: &TracingTensor<B::Storage<K>>,
        inner_res: &B::Storage<K>,
    ) -> Result<TracingTensor<B::Storage<K>>>
    where
        O: crate::exec::catalog::CanonicalOperation<
                Attributes = crate::exec::catalog::NoAttributes,
            >,
    {
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(
            crate::exec::catalog::NoAttributes,
            vec![
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&lhs.inner)),
                    dtype: Some(Self::traced_dtype(&lhs.inner)),
                    device: Some(B::metadata(&lhs.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&rhs.inner)),
                    dtype: Some(Self::traced_dtype(&rhs.inner)),
                    device: Some(B::metadata(&rhs.inner).device),
                },
            ],
        )
        .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical binary descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let mut graph = TRACING_GRAPH.lock();
        let output_id = graph.add_value(
            B::shape(inner_res).as_ref().to_vec(),
            Self::traced_dtype(inner_res),
            None,
        );
        let metadata = B::metadata(inner_res);
        let _ = graph.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            vec![lhs.value_id, rhs.value_id],
            vec![output_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok(TracingTensor {
            inner: inner_res.clone(),
            value_id: output_id,
        })
    }

    fn trace_canonical_binary_types<O, K1, K2, KOut>(
        lhs: &TracingTensor<B::Storage<K1>>,
        rhs: &TracingTensor<B::Storage<K2>>,
        inner_res: &B::Storage<KOut>,
    ) -> Result<TracingTensor<B::Storage<KOut>>>
    where
        O: crate::exec::catalog::CanonicalOperation<
                Attributes = crate::exec::catalog::NoAttributes,
            >,
        K1: super::dtype::DType,
        K2: super::dtype::DType,
        KOut: super::dtype::DType,
    {
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(
            crate::exec::catalog::NoAttributes,
            vec![
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&lhs.inner)),
                    dtype: Some(Self::traced_dtype(&lhs.inner)),
                    device: Some(B::metadata(&lhs.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&rhs.inner)),
                    dtype: Some(Self::traced_dtype(&rhs.inner)),
                    device: Some(B::metadata(&rhs.inner).device),
                },
            ],
        )
        .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical typed binary descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let mut graph = TRACING_GRAPH.lock();
        let output_id = graph.add_value(
            B::shape(inner_res).as_ref().to_vec(),
            Self::traced_dtype(inner_res),
            None,
        );
        let metadata = B::metadata(inner_res);
        let _ = graph.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            vec![lhs.value_id, rhs.value_id],
            vec![output_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok(TracingTensor {
            inner: inner_res.clone(),
            value_id: output_id,
        })
    }

    fn trace_canonical_binary_with_attributes<O, K: super::dtype::DType>(
        lhs: &TracingTensor<B::Storage<K>>,
        rhs: &TracingTensor<B::Storage<K>>,
        inner_res: &B::Storage<K>,
        attributes: O::Attributes,
    ) -> Result<TracingTensor<B::Storage<K>>>
    where
        O: crate::exec::catalog::CanonicalOperation,
        O::Attributes: crate::exec::catalog::AttributeContract,
    {
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(
            attributes,
            vec![
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&lhs.inner)),
                    dtype: Some(Self::traced_dtype(&lhs.inner)),
                    device: Some(B::metadata(&lhs.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&rhs.inner)),
                    dtype: Some(Self::traced_dtype(&rhs.inner)),
                    device: Some(B::metadata(&rhs.inner).device),
                },
            ],
        )
        .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical attributed binary descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let mut graph = TRACING_GRAPH.lock();
        let output_id = graph.add_value(
            B::shape(inner_res).as_ref().to_vec(),
            Self::traced_dtype(inner_res),
            None,
        );
        let metadata = B::metadata(inner_res);
        let _ = graph.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            vec![lhs.value_id, rhs.value_id],
            vec![output_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok(TracingTensor {
            inner: inner_res.clone(),
            value_id: output_id,
        })
    }

    fn trace_canonical_two_outputs<O, K, KOut>(
        t: &TracingTensor<B::Storage<K>>,
        first: &B::Storage<K>,
        second: &B::Storage<KOut>,
        attributes: O::Attributes,
    ) -> Result<(ValueId, ValueId)>
    where
        O: crate::exec::catalog::CanonicalOperation,
        O::Attributes: crate::exec::catalog::AttributeContract,
        K: super::dtype::DType,
        KOut: super::dtype::DType,
    {
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(
            attributes,
            vec![crate::exec::catalog::LogicalTensorMeta {
                shape: Some(B::shape(&t.inner)),
                dtype: Some(Self::traced_dtype(&t.inner)),
                device: Some(B::metadata(&t.inner).device),
            }],
        )
        .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical multi-output descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let mut graph = TRACING_GRAPH.lock();
        let first_id = graph.add_value(
            B::shape(first).as_ref().to_vec(),
            Self::traced_dtype(first),
            None,
        );
        let second_id = graph.add_value(
            B::shape(second).as_ref().to_vec(),
            Self::traced_dtype(second),
            None,
        );
        let metadata = B::metadata(first);
        let _ = graph.set_value_placement(first_id, Some(metadata.device), Some(metadata.layout));
        let metadata = B::metadata(second);
        let _ = graph.set_value_placement(second_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            vec![t.value_id],
            vec![first_id, second_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok((first_id, second_id))
    }

    fn trace_canonical_unary_types_with_attributes<O, KIn, KOut>(
        t: &TracingTensor<B::Storage<KIn>>,
        inner_res: &B::Storage<KOut>,
        attributes: O::Attributes,
    ) -> Result<TracingTensor<B::Storage<KOut>>>
    where
        O: crate::exec::catalog::CanonicalOperation,
        O::Attributes: crate::exec::catalog::AttributeContract,
        KIn: super::dtype::DType,
        KOut: super::dtype::DType,
    {
        let descriptor = crate::exec::catalog::Descriptor::<O>::infer_runtime(
            attributes,
            vec![crate::exec::catalog::LogicalTensorMeta {
                shape: Some(B::shape(&t.inner)),
                dtype: Some(Self::traced_dtype(&t.inner)),
                device: Some(B::metadata(&t.inner).device),
            }],
        )
        .map_err(|_| BackendError::InvalidInput {
            operation: crate::shapes::error::OperationKind::Storage,
            reason: "tracing canonical typed unary descriptor validation failed",
        })?;
        let payload = descriptor
            .descriptor()
            .trace_descriptor_payload()
            .map_err(|reason| BackendError::InvalidInput {
                operation: crate::shapes::error::OperationKind::Storage,
                reason,
            })?;
        let mut graph = TRACING_GRAPH.lock();
        let output_id = graph.add_value(
            B::shape(inner_res).as_ref().to_vec(),
            Self::traced_dtype(inner_res),
            None,
        );
        let metadata = B::metadata(inner_res);
        let _ = graph.set_value_placement(output_id, Some(metadata.device), Some(metadata.layout));
        graph.add_node_with_descriptor_payload(
            descriptor.descriptor().trace_identity(),
            vec![t.value_id],
            vec![output_id],
            alloc::collections::BTreeMap::new(),
            payload,
        );
        Ok(TracingTensor {
            inner: inner_res.clone(),
            value_id: output_id,
        })
    }
}

fn axis_attributes(
    axis: usize,
) -> alloc::collections::BTreeMap<alloc::string::String, crate::graph::AttributeValue> {
    let mut attributes = alloc::collections::BTreeMap::new();
    attributes.insert(
        alloc::string::String::from("axis"),
        crate::graph::AttributeValue::Int(axis as i64),
    );
    attributes
}

impl<B: Backend> crate::tensor::backend::StorageBackend for TracingBackend<B> {
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

    fn execute_shaped<S: crate::prelude::Shape>(
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
        let inner_res =
            inner_backend.execute_shaped::<S>(crate::tensor::backend::ExecutionRequest {
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
                alloc::collections::BTreeMap::new(),
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
    /// The wrapped backend's variable plus a tracing-graph node id.
    type RawVar = TracingVar<B::RawVar>;
    /// Delegates to `B`'s gradient collection type — tracing adds no gradient bookkeeping of its own.
    type Grads = B::Grads;
    /// Delegates to `B`'s own inner backend (tracing is not itself a dispatch layer).
    type InnerBackend = B::InnerBackend;

    /// Delegates to the wrapped backend's canonical `ShapeBuf`.
    fn shape<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> crate::shapes::ShapeBuf {
        B::shape(&t.inner)
    }

    fn storage_dtype<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Option<DTypeDescriptor> {
        B::storage_dtype(&t.inner)
    }

    fn storage_device<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Option<DeviceId> {
        B::storage_device(&t.inner)
    }

    fn format_tensor_display<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> alloc::string::String
    where
        Self: crate::tensor::backend::TensorOps<Self>,
    {
        use crate::tensor::backend::TensorOps;
        use crate::tensor::display::{Values, render};
        let shape = Self::shape(t);
        match Self::storage_dtype(t) {
            None => alloc::format!("<tensor: shape={shape:?}, dtype unknown to this backend>"),
            Some(dtype) if dtype.is_quantized() => alloc::format!(
                "<{} tensor: shape={shape:?}, not printable without dequantizing>",
                dtype.name()
            ),
            Some(dtype) if dtype.is_integer() => match Self::int_to_vec1(t) {
                Ok(values) => render(&shape, &Values::Int(values)),
                Err(err) => alloc::format!("<tensor: shape={shape:?}, values unavailable: {err}>"),
            },
            Some(_) => match Self::float_to_vec1(t) {
                Ok(values) => render(&shape, &Values::Float(values)),
                Err(err) => alloc::format!("<tensor: shape={shape:?}, values unavailable: {err}>"),
            },
        }
    }
    /// Renders a tensor's values and metadata for `Debug`.
    fn format_tensor_debug<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> alloc::string::String
    where
        Self: crate::tensor::backend::TensorOps<Self>,
    {
        Self::format_tensor_display::<K>(t)
    }

    /// Delegates to `B::var_as_tensor`, carrying the variable's tracing-graph node id over to the resulting storage.
    fn var_as_tensor<K: super::dtype::DType>(
        var: &<Self as Backend>::RawVar,
    ) -> Result<<Self as crate::tensor::backend::StorageBackend>::Storage<K>> {
        let inner = B::var_as_tensor(&var.inner)?;
        Ok(TracingTensor {
            inner,
            value_id: var.value_id,
        })
    }

    /// Delegates to `B::var_from_tensor`, carrying the storage's tracing-graph node id over to the resulting variable.
    fn var_from_tensor<K: super::dtype::DType>(
        t: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_from_tensor(&t.inner)?;
        Ok(TracingVar {
            inner,
            value_id: t.value_id,
        })
    }

    /// Delegates to `B::assign_var`.
    fn assign_var<K: super::dtype::DType>(
        var: &mut <Self as Backend>::RawVar,
        tensor: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<()> {
        B::assign_var(&mut var.inner, &tensor.inner)
    }

    /// Delegates to `B::backward` — tracing does not itself affect gradient computation.
    fn backward<K: super::dtype::DType>(
        loss: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<<Self as Backend>::Grads> {
        B::backward(&loss.inner)
    }

    fn backward_with<K: super::dtype::DType>(
        loss: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
        seed: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
    ) -> Result<<Self as Backend>::Grads> {
        B::backward_with(&loss.inner, &seed.inner)
    }

    /// Always `None` — tracing doesn't maintain its own gradient map;
    /// gradient lookup must go through the wrapped backend `B` directly.
    fn get_grad<K: super::dtype::DType>(
        _var: &<Self as crate::tensor::backend::StorageBackend>::Storage<K>,
        _grads: &<Self as Backend>::Grads,
    ) -> Result<Option<<Self as crate::tensor::backend::StorageBackend>::Storage<K>>> {
        Ok(None)
    }

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

impl<B: Backend + SupportsDType<K>, K: DType> SupportsDType<K> for TracingBackend<B> {
    fn resolve_dtype(field: &K::Field, device: &DeviceId) -> Result<DTypeDescriptor> {
        B::resolve_dtype(field, device)
    }
}

impl<B: Backend + CreationOps<B>> CreationOps<Self> for TracingBackend<B> {
    /// Delegates to `B::zeros`, additionally recording a new
    /// tracing-graph value node for the result.
    fn zeros<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::zeros(shape, dtype, device)?;
        let value_id = Self::record_value(&inner);
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::ones`, additionally recording a new
    /// tracing-graph value node for the result.
    fn ones<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::ones(shape, dtype, device)?;
        let value_id = Self::record_value(&inner);
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::full`, additionally recording a new
    /// tracing-graph value node for the result.
    fn full<K: super::dtype::DType>(
        val: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::full(val, shape, dtype, device)?;
        let value_id = Self::record_value(&inner);
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::arange`, additionally recording a new
    /// tracing-graph value node for the result.
    fn arange<K: super::dtype::DType>(
        start: f64,
        step: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::arange(start, step, shape, dtype, device)?;
        let value_id = Self::record_value(&inner);
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::linspace`, additionally recording a new
    /// tracing-graph value node for the result.
    fn linspace<K: super::dtype::DType>(
        start: f64,
        end: f64,
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::linspace(start, end, shape, dtype, device)?;
        let value_id = Self::record_value(&inner);
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::rand`, additionally recording a new
    /// tracing-graph value node for the result.
    fn rand<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::rand(shape, dtype, device)?;
        let value_id = Self::record_value(&inner);
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::randn`, additionally recording a new
    /// tracing-graph value node for the result.
    fn randn<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::randn(shape, dtype, device)?;
        let value_id = Self::record_value(&inner);
        Ok(TracingTensor { inner, value_id })
    }

    /// Delegates to `B::var_zeros`, additionally recording a new
    /// tracing-graph value node for the result.
    fn var_zeros<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_zeros::<K>(shape, dtype, device)?;
        let tensor = B::var_as_tensor::<K>(&inner)?;
        let value_id = Self::record_value(&tensor);
        Ok(TracingVar { inner, value_id })
    }

    /// Delegates to `B::var_ones`, additionally recording a new
    /// tracing-graph value node for the result.
    fn var_ones<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_ones::<K>(shape, dtype, device)?;
        let tensor = B::var_as_tensor::<K>(&inner)?;
        let value_id = Self::record_value(&tensor);
        Ok(TracingVar { inner, value_id })
    }

    /// Delegates to `B::var_rand`, additionally recording a new
    /// tracing-graph value node for the result.
    fn var_rand<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_rand::<K>(shape, dtype, device)?;
        let tensor = B::var_as_tensor::<K>(&inner)?;
        let value_id = Self::record_value(&tensor);
        Ok(TracingVar { inner, value_id })
    }

    /// Delegates to `B::var_randn`, additionally recording a new
    /// tracing-graph value node for the result.
    fn var_randn<K: super::dtype::DType>(
        shape: &[usize],
        dtype: DTypeDescriptor,
        device: &DeviceId,
    ) -> Result<<Self as Backend>::RawVar> {
        let inner = B::var_randn::<K>(shape, dtype, device)?;
        let tensor = B::var_as_tensor::<K>(&inner)?;
        let value_id = Self::record_value(&tensor);
        Ok(TracingVar { inner, value_id })
    }
}

impl<B, NewD> TransferTo<NewD> for TracingBackend<B>
where
    B: Backend + TransferTo<NewD>,
    NewD: crate::tensor::device::Device,
{
    type Output = TracingBackend<<B as TransferTo<NewD>>::Output>;

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
        let inner = <<B as TransferTo<NewD>>::Output as Backend>::from_bytes::<K>(
            &bytes, &shape, dtype_id, &device_id,
        )?;
        Ok(TracingTensor {
            inner,
            value_id: storage.value_id,
        })
    }

    fn transfer_var<K: DType>(
        variable: &Self::RawVar,
        dtype: &K::Field,
        device: &NewD::Field,
    ) -> Result<<Self::Output as Backend>::RawVar>
    where
        Self::Output: SupportsDType<K>,
    {
        let source = B::var_as_tensor::<K>(&variable.inner)?;
        let shape = B::shape(&source);
        let bytes = B::to_bytes(&source)?;
        let device_id = NewD::to_incin(device)?;
        let dtype_id = <Self::Output as SupportsDType<K>>::resolve_dtype(dtype, &device_id)?;
        let storage = <<B as TransferTo<NewD>>::Output as Backend>::from_bytes::<K>(
            &bytes, &shape, dtype_id, &device_id,
        )?;
        let inner = <<B as TransferTo<NewD>>::Output as Backend>::var_from_tensor(&storage)?;
        Ok(TracingVar {
            inner,
            value_id: variable.value_id,
        })
    }
}

impl<B: Backend + NumericOps<B>> NumericOps<Self> for TracingBackend<B> {
    /// Delegates to `B::add`, additionally recording an `OperationKind.Add` node.
    fn add<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::add(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary::<crate::exec::catalog::op::Add, K>(lhs, rhs, &inner)
    }

    /// Delegates to `B::sub`, additionally recording an `OperationKind.Sub` node.
    fn sub<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sub(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary::<crate::exec::catalog::op::Sub, K>(lhs, rhs, &inner)
    }

    /// Delegates to `B::mul`, additionally recording an `OperationKind.Mul` node.
    fn mul<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mul(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary::<crate::exec::catalog::op::Mul, K>(lhs, rhs, &inner)
    }

    /// Delegates to `B::div`, additionally recording an `OperationKind.Div` node.
    fn div<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::div(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary::<crate::exec::catalog::op::Div, K>(lhs, rhs, &inner)
    }
}

/// The error a tracing-only gap reports.
///
/// Identical to what `FloatOps`' removed default bodies produced, so this
/// changes where the refusal is written down, not what a caller observes.
fn untraceable<B: Backend>(op: &'static str) -> Error {
    Error::UnsupportedBackendOperation {
        op,
        backend: core::any::type_name::<TracingBackend<B>>(),
    }
}

/// Float operations the tracing graph has no node for.
///
/// [`OperationKind`] carries no variant for any of these, and ONNX export builds its
/// node list from that vocabulary, so recording them is not possible without
/// extending both. Delegating to `B` *without* recording would be worse than
/// refusing: the exported graph would silently omit the operation and stop
/// describing the model it came from.
///
/// Writing them out is the point. Until `EXE-009` these were inherited from
/// `FloatOps`' default bodies, which left an operation this backend cannot
/// trace indistinguishable, at this impl site, from one it can.
macro_rules! untraced_float_ops {
    (
        unary: $($unary:ident),* $(,)?;
        exponent: $($exponent:ident),* $(,)?;
        bounds: $($bounds:ident),* $(,)?;
        binary: $($binary:ident),* $(,)?;
    ) => {
        $(
            fn $unary<K: super::dtype::DType>(
                _t: &<Self as StorageBackend>::Storage<K>,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err(untraceable::<B>(stringify!($unary)))
            }
        )*
        $(
            fn $exponent<K: super::dtype::DType>(
                _t: &<Self as StorageBackend>::Storage<K>,
                _exponent: f64,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err(untraceable::<B>(stringify!($exponent)))
            }
        )*
        $(
            fn $bounds<K: super::dtype::DType>(
                _t: &<Self as StorageBackend>::Storage<K>,
                _min: f64,
                _max: f64,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err(untraceable::<B>(stringify!($bounds)))
            }
        )*
        $(
            fn $binary<K: super::dtype::DType>(
                _lhs: &<Self as StorageBackend>::Storage<K>,
                _rhs: &<Self as StorageBackend>::Storage<K>,
            ) -> Result<<Self as StorageBackend>::Storage<K>> {
                Err(untraceable::<B>(stringify!($binary)))
            }
        )*
    };
}

impl<B: Backend + FloatOps<B>> FloatOps<Self> for TracingBackend<B> {
    untraced_float_ops! {
        unary: sign, floor, ceil, round, log2, log10, sin, cos, tan, asin, acos,
               atan, sinh, cosh, asinh, acosh, atanh, erf, rsqrt, trunc, frac;
        exponent: powf;
        bounds: clamp;
        binary: atan2, fmod, remainder;
    }

    /// Delegates to `B::add_scalar_float`, additionally recording an
    /// `OperationKind::AddScalar` node.
    fn add_scalar_float<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::add_scalar_float(&t.inner, scalar)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::AddScalar, K>(
            t,
            &inner,
            crate::exec::catalog::ScalarAttributes { value: scalar },
        )
    }

    /// Delegates to `B::mul_scalar_float`, additionally recording an
    /// `OperationKind::MulScalar` node.
    fn mul_scalar_float<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        scalar: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mul_scalar_float(&t.inner, scalar)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::MulScalar, K>(
            t,
            &inner,
            crate::exec::catalog::ScalarAttributes { value: scalar },
        )
    }

    /// Delegates to `B::relu`, additionally recording an
    /// `OperationKind::Relu` node.
    fn relu<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::relu(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Relu, K>(t, &inner)
    }

    fn step<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::step(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Step, K>(t, &inner)
    }

    fn mish<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mish(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Mish, K>(t, &inner)
    }

    fn elu<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::elu(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Elu, K>(t, &inner)
    }

    /// Delegates to `B::gelu`, additionally recording an
    /// `OperationKind::Gelu` node.
    fn gelu<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::gelu(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Gelu, K>(t, &inner)
    }

    /// Delegates to `B::abs`, additionally recording an
    /// `OperationKind::Abs` node.
    fn abs<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::abs(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Abs, K>(t, &inner)
    }

    /// Delegates to `B::exp`, additionally recording an
    /// `OperationKind::Exp` node.
    fn exp<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::exp(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Exp, K>(t, &inner)
    }

    /// Delegates to `B::neg`, additionally recording an
    /// `OperationKind::Neg` node.
    fn neg<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::neg(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Neg, K>(t, &inner)
    }

    /// Delegates to `B::sqrt`, additionally recording an
    /// `OperationKind::Sqrt` node.
    fn sqrt<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sqrt(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Sqrt, K>(t, &inner)
    }

    /// Delegates to `B::log`, additionally recording an
    /// `OperationKind::Log` node.
    fn log<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::log(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Log, K>(t, &inner)
    }

    /// Delegates to `B::tanh`, additionally recording an
    /// `OperationKind::Tanh` node.
    fn tanh<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::tanh(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Tanh, K>(t, &inner)
    }

    /// Delegates to `B::sigmoid`, additionally recording an
    /// `OperationKind::Sigmoid` node.
    fn sigmoid<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sigmoid(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Sigmoid, K>(t, &inner)
    }

    /// Delegates to `B::swish`, additionally recording an
    /// `OperationKind::Swish` node.
    fn swish<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::swish(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::Swish, K>(t, &inner)
    }

    /// Delegates to `B::softmax`, additionally recording an
    /// `OperationKind::Softmax` node with the reduced `dim` as an attribute.
    fn softmax<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::softmax(&t.inner, dim)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::Softmax, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }
}

impl<B: Backend + ReductionOps<B>> ReductionOps<Self> for TracingBackend<B> {
    /// `OperationKind` has no product or cumulative-sum node, so these cannot be
    /// recorded. Refusing keeps an exported graph honest; delegating silently
    /// would drop the operation from the model the graph claims to describe.
    fn prod_all<K: super::dtype::DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(untraceable::<B>("prod_all"))
    }

    /// Untraceable for the same reason as [`prod_all`](Self::prod_all).
    fn prod_dim<K: super::dtype::DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(untraceable::<B>("prod_dim"))
    }

    /// Untraceable for the same reason as [`prod_all`](Self::prod_all).
    fn cumsum<K: super::dtype::DType>(
        _t: &<Self as StorageBackend>::Storage<K>,
        _dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        Err(untraceable::<B>("cumsum"))
    }

    /// Delegates to `B::sum_all`, additionally recording an `OperationKind::SumAll` node.
    fn sum_all<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sum_all(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::SumAll, K>(t, &inner)
    }

    /// Delegates to `B::mean_all`, additionally recording an `OperationKind::MeanAll` node.
    fn mean_all<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mean_all(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::MeanAll, K>(t, &inner)
    }

    /// Delegates to `B::max_all`, additionally recording an `OperationKind::MaxAll` node.
    fn max_all<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::max_all(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::MaxAll, K>(t, &inner)
    }

    /// Delegates to `B::min_all`, additionally recording an `OperationKind::MinAll` node.
    fn min_all<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::min_all(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::MinAll, K>(t, &inner)
    }

    /// Delegates to `B::sum_dim`, additionally recording an `OperationKind::SumDim` node.
    fn sum_dim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sum_dim(&t.inner, dim)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::SumDim, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::sum_keepdim`, additionally recording an `OperationKind::SumDim` node.
    fn sum_keepdim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sum_keepdim(&t.inner, dim)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::SumKeepDim, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::mean_dim`, additionally recording an `OperationKind::MeanDim` node.
    fn mean_dim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mean_dim(&t.inner, dim)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::MeanDim, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::mean_keepdim`, additionally recording an `OperationKind::MeanDim` node.
    fn mean_keepdim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mean_keepdim(&t.inner, dim)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::MeanKeepDim, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::max_dim`, additionally recording an `OperationKind::MaxDim` node.
    fn max_dim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::max_dim(&t.inner, dim)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::MaxDim, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::max_keepdim`, additionally recording an `OperationKind::MaxDim` node.
    fn max_keepdim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::max_keepdim(&t.inner, dim)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::MaxKeepDim, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::min_dim`, additionally recording an `OperationKind::MinDim` node.
    fn min_dim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::min_dim(&t.inner, dim)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::MinDim, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::min_keepdim`, additionally recording an `OperationKind::MinDim` node.
    fn min_keepdim<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::min_keepdim(&t.inner, dim)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::MinKeepDim, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::argmax`, additionally recording an `OperationKind::ArgMax` node.
    fn argmax<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        let inner = B::argmax(&t.inner, dim)?;
        Self::trace_canonical_unary_types_with_attributes::<crate::exec::catalog::op::ArgMax, K, KInt>(
            t,
            &inner,
            crate::exec::catalog::IndexReductionAttributes {
                axis: dim,
                dtype: Self::traced_dtype(&inner),
            },
        )
    }

    /// Delegates to `B::argmin`, additionally recording an `OperationKind::ArgMin` node.
    fn argmin<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: Option<usize>,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        let inner = B::argmin(&t.inner, dim)?;
        Self::trace_canonical_unary_types_with_attributes::<crate::exec::catalog::op::ArgMin, K, KInt>(
            t,
            &inner,
            crate::exec::catalog::IndexReductionAttributes {
                axis: dim,
                dtype: Self::traced_dtype(&inner),
            },
        )
    }

    /// Delegates to `B::topk`, recording both outputs (values and indices)
    /// in the tracing graph under `OperationKind::TopK` with `k`, `dim`, and
    /// `largest` stored as node attributes.
    fn topk<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: usize,
        dim: usize,
        largest: bool,
    ) -> Result<(
        <Self as StorageBackend>::Storage<K>,
        <Self as StorageBackend>::Storage<KInt>,
    )> {
        let (v_inner, i_inner) = B::topk(&t.inner, k, dim, largest)?;

        let (v_id, i_id) =
            Self::trace_canonical_two_outputs::<crate::exec::catalog::op::TopK, K, KInt>(
                t,
                &v_inner,
                &i_inner,
                crate::exec::catalog::TopKAttributes {
                    k,
                    axis: dim,
                    largest,
                    index_dtype: Self::traced_dtype(&i_inner),
                },
            )?;

        Ok((
            TracingTensor {
                inner: v_inner,
                value_id: v_id,
            },
            TracingTensor {
                inner: i_inner,
                value_id: i_id,
            },
        ))
    }

    /// Delegates to `B::argsort`, recording an `OperationKind::Argsort` node
    /// with `dim` and `descending` as node attributes.
    fn argsort<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        descending: bool,
    ) -> Result<<Self as StorageBackend>::Storage<KInt>> {
        let inner = B::argsort(&t.inner, dim, descending)?;
        Self::trace_canonical_unary_types_with_attributes::<
            crate::exec::catalog::op::Argsort,
            K,
            KInt,
        >(
            t,
            &inner,
            crate::exec::catalog::ArgsortAttributes {
                axis: dim,
                descending,
                index_dtype: Self::traced_dtype(&inner),
            },
        )
    }
}

impl<B: Backend + TensorOps<B>> TensorOps<Self> for TracingBackend<B> {
    /// Delegates to `B::tensor_to_dtype`, additionally recording an `OperationKind::ToDtype` node.
    fn tensor_to_dtype<K1: super::dtype::DType, K2: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K1>,
        dtype: DTypeDescriptor,
    ) -> Result<<Self as StorageBackend>::Storage<K2>> {
        let inner = B::tensor_to_dtype::<K1, K2>(&t.inner, dtype)?;
        Self::trace_canonical_unary_types_with_attributes::<crate::exec::catalog::op::ToDType, K1, K2>(
            t,
            &inner,
            crate::exec::catalog::DTypeAttributes { dtype },
        )
    }

    /// Delegates to `B::broadcast_as`, additionally recording an `OperationKind::Broadcast` node.
    fn broadcast_as<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::broadcast_as(&t.inner, shape)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::BroadcastAs, K>(
            t,
            &inner,
            crate::exec::catalog::ShapeAttributes {
                shape: shape.to_vec(),
            },
        )
    }

    /// Delegates to `B::reshape`, additionally recording an
    /// `OperationKind::Reshape` node with the target shape stored as a constant
    /// initializer input (matching ONNX's `Reshape` operator signature).
    fn reshape<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::reshape(&t.inner, shape)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::ReshapeExact, K>(
            t,
            &inner,
            crate::exec::catalog::ShapeAttributes {
                shape: shape.to_vec(),
            },
        )
    }

    /// Delegates to `B::transpose`, additionally recording an
    /// `OperationKind::Transpose` node with the resulting permutation as a `perm` attribute.
    fn transpose<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim1: usize,
        dim2: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::transpose(&t.inner, dim1, dim2)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::TransposeExact, K>(
            t,
            &inner,
            crate::exec::catalog::TransposeAttributes {
                first: dim1,
                second: dim2,
            },
        )
    }

    /// Delegates to `B::narrow`, additionally recording an `OperationKind::Narrow` node.
    fn narrow<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        start: usize,
        len: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::narrow(&t.inner, dim, start, len)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::Narrow, K>(
            t,
            &inner,
            crate::exec::catalog::NarrowAttributes {
                axis: dim,
                start,
                length: len,
            },
        )
    }

    /// Delegates to `B::concat`, additionally recording an
    /// `OperationKind::Concat` node with `dim` as an `axis` attribute.
    fn concat<K: super::dtype::DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inners: Vec<&B::Storage<K>> = tensors.iter().map(|t| &t.inner).collect();
        let inner = B::concat(&inners, dim)?;
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::ConcatExact, K>(
            tensors,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::stack`, additionally recording an
    /// `OperationKind::Stack` node with `dim` as an `axis` attribute.
    fn stack<K: super::dtype::DType>(
        tensors: &[&<Self as StorageBackend>::Storage<K>],
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inners: Vec<&B::Storage<K>> = tensors.iter().map(|t| &t.inner).collect();
        let inner = B::stack(&inners, dim)?;
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::StackExact, K>(
            tensors,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::slice`, additionally recording an `OperationKind::Slice` node.
    fn slice<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        ranges: &[(usize, usize)],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::slice(&t.inner, ranges)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::SliceExact, K>(
            t,
            &inner,
            crate::exec::catalog::SliceAttributes {
                ranges: ranges.to_vec(),
            },
        )
    }

    /// Delegates to `B::flatten`, additionally recording a
    /// (shape-only) `OperationKind::Reshape` node.
    fn flatten<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        start_dim: usize,
        end_dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::flatten(&t.inner, start_dim, end_dim)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::FlattenExact, K>(
            t,
            &inner,
            crate::exec::catalog::FlattenAttributes {
                start_axis: start_dim,
                end_axis: end_dim,
            },
        )
    }

    /// Delegates to `B::squeeze`, additionally recording a
    /// (shape-only) `OperationKind::Reshape` node.
    fn squeeze<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::squeeze(&t.inner, dim)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::ReshapeExact, K>(
            t,
            &inner,
            crate::exec::catalog::ShapeAttributes {
                shape: B::shape(&inner).as_ref().to_vec(),
            },
        )
    }

    /// Delegates to `B::broadcast_left`, additionally recording an `OperationKind::Broadcast` node.
    fn broadcast_left<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        shape: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::broadcast_left(&t.inner, shape)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::BroadcastLeft, K>(
            t,
            &inner,
            crate::exec::catalog::ShapeAttributes {
                shape: shape.to_vec(),
            },
        )
    }

    /// Delegates to `B::matmul`, additionally recording an `OperationKind::MatMul` node.
    fn matmul<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::matmul(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary::<crate::exec::catalog::op::MatMulExact, K>(lhs, rhs, &inner)
    }
    /// Delegates to `B::float_to_scalar`.
    fn float_to_scalar<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<f64> {
        B::float_to_scalar(&t.inner)
    }
    /// Delegates to `B::float_to_vec1`.
    fn float_to_vec1<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<f64>> {
        B::float_to_vec1(&t.inner)
    }
    /// Delegates to `B::int_to_scalar`.
    fn int_to_scalar<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<i64> {
        B::int_to_scalar(&t.inner)
    }
    /// Delegates to `B::int_to_vec1`.
    fn int_to_vec1<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<alloc::vec::Vec<i64>> {
        B::int_to_vec1(&t.inner)
    }

    /// Delegates to `B::where_cond`, recording an `OperationKind::WhereCond` node
    /// whose inputs are the mask and both branches.
    fn where_cond<K: super::dtype::DType>(
        mask: &<Self as StorageBackend>::Storage<bool>,
        on_true: &<Self as StorageBackend>::Storage<K>,
        on_false: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::where_cond::<K>(&mask.inner, &on_true.inner, &on_false.inner)?;
        Self::trace_canonical_inputs_with_attributes::<crate::exec::catalog::op::WhereCond, K>(
            alloc::vec![mask.value_id, on_true.value_id, on_false.value_id],
            alloc::vec![
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&mask.inner)),
                    dtype: Some(Self::traced_dtype(&mask.inner)),
                    device: Some(B::metadata(&mask.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&on_true.inner)),
                    dtype: Some(Self::traced_dtype(&on_true.inner)),
                    device: Some(B::metadata(&on_true.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&on_false.inner)),
                    dtype: Some(Self::traced_dtype(&on_false.inner)),
                    device: Some(B::metadata(&on_false.inner).device),
                },
            ],
            &inner,
            crate::exec::catalog::NoAttributes,
        )
    }

    /// Delegates to `B::gather`, recording an `OperationKind::Gather` node.
    fn gather<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::gather::<K, KInt>(&t.inner, dim, &index.inner)?;
        Self::trace_canonical_inputs_with_attributes::<crate::exec::catalog::op::Gather, K>(
            alloc::vec![t.value_id, index.value_id],
            alloc::vec![
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&t.inner)),
                    dtype: Some(Self::traced_dtype(&t.inner)),
                    device: Some(B::metadata(&t.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&index.inner)),
                    dtype: Some(Self::traced_dtype(&index.inner)),
                    device: Some(B::metadata(&index.inner).device),
                },
            ],
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::scatter`, recording an `OperationKind::Scatter` node whose
    /// inputs are the target, the index, and the source.
    fn scatter<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
        src: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::scatter::<K, KInt>(&t.inner, dim, &index.inner, &src.inner)?;
        Self::trace_canonical_inputs_with_attributes::<crate::exec::catalog::op::Scatter, K>(
            alloc::vec![t.value_id, index.value_id, src.value_id],
            alloc::vec![
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&t.inner)),
                    dtype: Some(Self::traced_dtype(&t.inner)),
                    device: Some(B::metadata(&t.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&index.inner)),
                    dtype: Some(Self::traced_dtype(&index.inner)),
                    device: Some(B::metadata(&index.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&src.inner)),
                    dtype: Some(Self::traced_dtype(&src.inner)),
                    device: Some(B::metadata(&src.inner).device),
                },
            ],
            &inner,
            crate::exec::catalog::ScatterAttributes {
                axis: dim,
                duplicate_indices: crate::exec::catalog::DuplicateIndexRule::LastWriteWins,
            },
        )
    }

    /// Delegates to `B::index_select`, recording an `OperationKind::IndexSelect` node.
    fn index_select<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        index: &<Self as StorageBackend>::Storage<KInt>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::index_select::<K, KInt>(&t.inner, dim, &index.inner)?;
        Self::trace_canonical_inputs_with_attributes::<crate::exec::catalog::op::IndexSelect, K>(
            alloc::vec![t.value_id, index.value_id],
            alloc::vec![
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&t.inner)),
                    dtype: Some(Self::traced_dtype(&t.inner)),
                    device: Some(B::metadata(&t.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&index.inner)),
                    dtype: Some(Self::traced_dtype(&index.inner)),
                    device: Some(B::metadata(&index.inner).device),
                },
            ],
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::masked_fill`, recording an `OperationKind::MaskedFill` node.
    fn masked_fill<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        mask: &<Self as StorageBackend>::Storage<bool>,
        value: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::masked_fill::<K>(&t.inner, &mask.inner, value)?;
        Self::trace_canonical_inputs_with_attributes::<crate::exec::catalog::op::MaskedFill, K>(
            alloc::vec![t.value_id, mask.value_id],
            alloc::vec![
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&t.inner)),
                    dtype: Some(Self::traced_dtype(&t.inner)),
                    device: Some(B::metadata(&t.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&mask.inner)),
                    dtype: Some(Self::traced_dtype(&mask.inner)),
                    device: Some(B::metadata(&mask.inner).device),
                },
            ],
            &inner,
            crate::exec::catalog::ScalarAttributes { value },
        )
    }

    /// Delegates to `B::unsqueeze`, recording an `OperationKind::Unsqueeze` node.
    fn unsqueeze<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::unsqueeze(&t.inner, dim)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::UnsqueezeExact, K>(
            t,
            &inner,
            crate::exec::catalog::AxisAttributes { axis: dim },
        )
    }

    /// Delegates to `B::repeat`, recording an `OperationKind::Repeat` node.
    fn repeat<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        repeats: &[usize],
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::repeat(&t.inner, repeats)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::Repeat, K>(
            t,
            &inner,
            crate::exec::catalog::RepeatAttributes {
                repeats: repeats.to_vec(),
            },
        )
    }

    /// Delegates to `B::pad`, recording an `OperationKind::Pad` node.
    fn pad<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        padding: &[(usize, usize)],
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::pad(&t.inner, padding, val)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::Pad, K>(
            t,
            &inner,
            crate::exec::catalog::PadAttributes {
                padding: padding.to_vec(),
                value: val,
            },
        )
    }

    /// Delegates to `B::triu`, recording an `OperationKind::Triu` node.
    fn triu<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::triu(&t.inner, k)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::Triu, K>(
            t,
            &inner,
            crate::exec::catalog::DiagonalAttributes { offset: k },
        )
    }

    /// Delegates to `B::tril`, recording an `OperationKind::Tril` node.
    fn tril<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::tril(&t.inner, k)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::Tril, K>(
            t,
            &inner,
            crate::exec::catalog::DiagonalAttributes { offset: k },
        )
    }

    /// Delegates to `B::diag`, recording an `OperationKind::Diag` node.
    fn diag<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        k: i64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::diag(&t.inner, k)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::Diag, K>(
            t,
            &inner,
            crate::exec::catalog::DiagonalAttributes { offset: k },
        )
    }

    /// Delegates to `B::cmp_eq`, recording an `OperationKind::CmpEq` node.
    fn cmp_eq<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_eq(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary_types::<crate::exec::catalog::op::CmpEq, K, K, bool>(
            lhs, rhs, &inner,
        )
    }

    /// Delegates to `B::cmp_ne`, recording an `OperationKind::CmpNe` node.
    fn cmp_ne<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_ne(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary_types::<crate::exec::catalog::op::CmpNe, K, K, bool>(
            lhs, rhs, &inner,
        )
    }

    /// Delegates to `B::cmp_lt`, recording an `OperationKind::CmpLt` node.
    fn cmp_lt<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_lt(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary_types::<crate::exec::catalog::op::CmpLt, K, K, bool>(
            lhs, rhs, &inner,
        )
    }

    /// Delegates to `B::cmp_le`, recording an `OperationKind::CmpLe` node.
    fn cmp_le<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_le(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary_types::<crate::exec::catalog::op::CmpLe, K, K, bool>(
            lhs, rhs, &inner,
        )
    }

    /// Delegates to `B::cmp_gt`, recording an `OperationKind::CmpGt` node.
    fn cmp_gt<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_gt(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary_types::<crate::exec::catalog::op::CmpGt, K, K, bool>(
            lhs, rhs, &inner,
        )
    }

    /// Delegates to `B::cmp_ge`, recording an `OperationKind::CmpGe` node.
    fn cmp_ge<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::cmp_ge(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary_types::<crate::exec::catalog::op::CmpGe, K, K, bool>(
            lhs, rhs, &inner,
        )
    }

    /// Delegates to `B::logical_and`, recording an `OperationKind::LogicalAnd` node.
    fn logical_and(
        lhs: &<Self as StorageBackend>::Storage<bool>,
        rhs: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::logical_and(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary_types::<crate::exec::catalog::op::LogicalAnd, bool, bool, bool>(
            lhs, rhs, &inner,
        )
    }

    /// Delegates to `B::logical_or`, recording an `OperationKind::LogicalOr` node.
    fn logical_or(
        lhs: &<Self as StorageBackend>::Storage<bool>,
        rhs: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::logical_or(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary_types::<crate::exec::catalog::op::LogicalOr, bool, bool, bool>(
            lhs, rhs, &inner,
        )
    }

    /// Delegates to `B::logical_not`, recording an `OperationKind::LogicalNot` node.
    fn logical_not(
        t: &<Self as StorageBackend>::Storage<bool>,
    ) -> Result<<Self as StorageBackend>::Storage<bool>> {
        let inner = B::logical_not(&t.inner)?;
        Self::trace_canonical_unary::<crate::exec::catalog::op::LogicalNot, bool>(t, &inner)
    }

    /// Delegates to `B::sub_scalar`, recording an `OperationKind::SubScalar` node.
    fn sub_scalar<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::sub_scalar(&t.inner, val)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::SubScalar, K>(
            t,
            &inner,
            crate::exec::catalog::ScalarAttributes { value: val },
        )
    }

    /// Delegates to `B::div_scalar`, recording an `OperationKind::DivScalar` node.
    fn div_scalar<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        val: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::div_scalar(&t.inner, val)?;
        Self::trace_canonical_unary_with_attributes::<crate::exec::catalog::op::DivScalar, K>(
            t,
            &inner,
            crate::exec::catalog::ScalarAttributes { value: val },
        )
    }

    /// Delegates to `B::maximum`, recording an `OperationKind::Maximum` node.
    fn maximum<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::maximum(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary::<crate::exec::catalog::op::Maximum, K>(lhs, rhs, &inner)
    }

    /// Delegates to `B::minimum`, recording an `OperationKind::Minimum` node.
    fn minimum<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::minimum(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary::<crate::exec::catalog::op::Minimum, K>(lhs, rhs, &inner)
    }

    /// Delegates to `B::abs_diff`, recording an `OperationKind::AbsDiff` node.
    fn abs_diff<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::abs_diff(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary::<crate::exec::catalog::op::AbsDiff, K>(lhs, rhs, &inner)
    }

    /// Delegates to `B::lerp`, recording an `OperationKind::Lerp` node.
    fn lerp<K: super::dtype::DType>(
        start: &<Self as StorageBackend>::Storage<K>,
        end: &<Self as StorageBackend>::Storage<K>,
        weight: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::lerp(&start.inner, &end.inner, weight)?;
        Self::trace_canonical_binary_with_attributes::<crate::exec::catalog::op::Lerp, K>(
            start,
            end,
            &inner,
            crate::exec::catalog::LerpAttributes { weight },
        )
    }

    /// Delegates to `B::addmm`, recording an `OperationKind::Addmm` node whose inputs
    /// are the added matrix and both multiplicands.
    fn addmm<K: super::dtype::DType>(
        mat: &<Self as StorageBackend>::Storage<K>,
        mat1: &<Self as StorageBackend>::Storage<K>,
        mat2: &<Self as StorageBackend>::Storage<K>,
        beta: f64,
        alpha: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::addmm(&mat.inner, &mat1.inner, &mat2.inner, beta, alpha)?;
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::Addmm, K>(
            &[mat, mat1, mat2],
            &inner,
            crate::exec::catalog::AddmmAttributes { alpha, beta },
        )
    }

    /// Delegates to `B::bmm`, recording an `OperationKind::Bmm` node.
    fn bmm<K: super::dtype::DType>(
        lhs: &<Self as StorageBackend>::Storage<K>,
        rhs: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::bmm(&lhs.inner, &rhs.inner)?;
        Self::trace_canonical_binary::<crate::exec::catalog::op::BatchedMatMul, K>(lhs, rhs, &inner)
    }

    /// Delegates to `B::scaled_dot_product_attention`, recording an
    /// `OperationKind::ScaledDotProductAttention` node. The optional mask becomes a
    /// fourth input only when one was supplied, so the recorded arity matches
    /// the call.
    fn scaled_dot_product_attention<K: super::dtype::DType>(
        q: &<Self as StorageBackend>::Storage<K>,
        k: &<Self as StorageBackend>::Storage<K>,
        v: &<Self as StorageBackend>::Storage<K>,
        mask: Option<&<Self as StorageBackend>::Storage<K>>,
        scale: Option<f64>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::scaled_dot_product_attention(
            &q.inner,
            &k.inner,
            &v.inner,
            mask.map(|m| &m.inner),
            scale,
        )?;
        let tensors = if let Some(m) = mask {
            alloc::vec![q, k, v, m]
        } else {
            alloc::vec![q, k, v]
        };
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::ScaledDotProductAttention, K>(
            &tensors,
            &inner,
            crate::exec::catalog::AttentionAttributes {
                scale,
                has_mask: mask.is_some(),
            },
        )
    }

    /// Delegates to `B::unfold`, recording an `OperationKind::Unfold` node.
    fn unfold<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        dim: usize,
        size: usize,
        step: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::unfold(&t.inner, dim, size, step)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::Unfold, K>(
            t,
            &inner,
            crate::exec::catalog::UnfoldAttributes {
                axis: dim,
                size,
                step,
            },
        )
    }

    /// Delegates to `B::pixel_shuffle`, recording an `OperationKind::PixelShuffle` node.
    fn pixel_shuffle<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        upscale_factor: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::pixel_shuffle(&t.inner, upscale_factor)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::PixelShuffle, K>(
            t,
            &inner,
            crate::exec::catalog::PixelShuffleAttributes { upscale_factor },
        )
    }

    /// Delegates to `B::group_norm`, recording an `OperationKind::GroupNorm` node.
    fn group_norm<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        groups: usize,
        eps: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::group_norm(&t.inner, groups, eps)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::GroupNorm, K>(
            t,
            &inner,
            crate::exec::catalog::GroupNormAttributes {
                groups,
                epsilon: eps,
            },
        )
    }

    /// Delegates to `B::instance_norm`, recording an `OperationKind::InstanceNorm` node.
    fn instance_norm<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        eps: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::instance_norm(&t.inner, eps)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::InstanceNorm, K>(
            t,
            &inner,
            crate::exec::catalog::EpsilonAttributes { epsilon: eps },
        )
    }
}

impl<B: Backend + crate::tensor::backend::ModuleOps<B>> ModuleOps<Self> for TracingBackend<B> {
    /// Delegates to `B::conv1d`, additionally recording an
    /// `OperationKind::Conv1d` node with `strides`/`pads`/`dilations` attributes.
    fn conv1d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner_bias = bias.map(|b| &b.inner);
        let inner = B::conv1d(
            &x.inner,
            &weight.inner,
            inner_bias,
            stride,
            padding,
            dilation,
            groups,
        )?;
        let mut tensors = alloc::vec![x, weight];
        if let Some(b) = bias {
            tensors.push(b);
        }
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::Conv1dExact, K>(
            &tensors,
            &inner,
            crate::exec::catalog::Conv1dAttributes {
                stride,
                padding,
                dilation,
                groups,
                has_bias: bias.is_some(),
            },
        )
    }

    /// Delegates to `B::conv2d`, additionally recording an
    /// `OperationKind::Conv2d` node with `strides`/`pads`/`dilations` attributes.
    fn conv2d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner_bias = bias.map(|b| &b.inner);
        let inner = B::conv2d(
            &x.inner,
            &weight.inner,
            inner_bias,
            stride,
            padding,
            dilation,
            groups,
        )?;
        let mut tensors = alloc::vec![x, weight];
        if let Some(b) = bias {
            tensors.push(b);
        }
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::Conv2dExact, K>(
            &tensors,
            &inner,
            crate::exec::catalog::Conv2dAttributes {
                stride: [stride, stride],
                padding: [padding, padding],
                dilation: [dilation, dilation],
                groups,
                has_bias: bias.is_some(),
            },
        )
    }

    /// Delegates to `B::max_pool2d`, additionally recording an `OperationKind::MaxPool2d` node.
    fn max_pool2d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::max_pool2d(&x.inner, kernel_size, stride, padding, dilation)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::MaxPool2d, K>(
            x,
            &inner,
            crate::exec::catalog::Pool2dAttributes {
                kernel: [kernel_size.0, kernel_size.1],
                stride: [stride.0, stride.1],
                padding: [padding.0, padding.1],
                dilation: [dilation.0, dilation.1],
            },
        )
    }

    /// Delegates to `B::avg_pool2d`, additionally recording an `OperationKind::AvgPool2d` node.
    fn avg_pool2d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::avg_pool2d(&x.inner, kernel_size, stride, padding)?;
        Self::trace_canonical_shape_with_attributes::<crate::exec::catalog::op::AvgPool2d, K>(
            x,
            &inner,
            crate::exec::catalog::AvgPool2dAttributes {
                kernel: [kernel_size.0, kernel_size.1],
                stride: [stride.0, stride.1],
                padding: [padding.0, padding.1],
            },
        )
    }

    /// Delegates to `B::adaptive_avg_pool2d`, additionally recording an `OperationKind::AdaptiveAvgPool2d` node.
    fn adaptive_avg_pool2d<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        output_size: (usize, usize),
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::adaptive_avg_pool2d(&x.inner, output_size)?;
        Self::trace_canonical_shape_with_attributes::<
            crate::exec::catalog::op::AdaptiveAvgPool2dExact,
            K,
        >(
            x,
            &inner,
            crate::exec::catalog::AdaptivePool2dAttributes {
                output: [output_size.0, output_size.1],
            },
        )
    }

    /// Delegates to `B::embedding`, additionally recording an `OperationKind::Embedding` node.
    fn embedding<K: super::dtype::DType, KInt: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<KInt>,
        w: &<Self as StorageBackend>::Storage<K>,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::embedding(&t.inner, &w.inner)?;
        Self::trace_canonical_binary_types::<crate::exec::catalog::op::EmbeddingExact, KInt, K, K>(
            t, w, &inner,
        )
    }

    /// Delegates to `B::layer_norm`, additionally recording an `OperationKind::LayerNorm` node.
    fn layer_norm<K: super::dtype::DType>(
        x: &<Self as StorageBackend>::Storage<K>,
        weight: &<Self as StorageBackend>::Storage<K>,
        bias: Option<&<Self as StorageBackend>::Storage<K>>,
        eps: f32,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::layer_norm(&x.inner, &weight.inner, bias.map(|b| &b.inner), eps)?;
        let mut inputs = alloc::vec![x];
        inputs.push(weight);
        if let Some(bias) = bias {
            inputs.push(bias);
        }
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::LayerNorm, K>(
            &inputs,
            &inner,
            crate::exec::catalog::LayerNormAttributes {
                normalized_shape: B::shape(&weight.inner).as_ref().to_vec(),
                epsilon: f64::from(eps),
                has_bias: bias.is_some(),
            },
        )
    }

    /// Delegates to `B::batch_norm`, additionally recording an `OperationKind::BatchNorm` node.
    fn batch_norm<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: Option<&<Self as StorageBackend>::Storage<K>>,
        b: Option<&<Self as StorageBackend>::Storage<K>>,
        rm: Option<&<Self as StorageBackend>::Storage<K>>,
        rv: Option<&<Self as StorageBackend>::Storage<K>>,
        e: f32,
        momentum: f64,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::batch_norm(
            &t.inner,
            w.map(|x| &x.inner),
            b.map(|x| &x.inner),
            rm.map(|x| &x.inner),
            rv.map(|x| &x.inner),
            e,
            momentum,
        )?;
        let mut inputs = alloc::vec![t];
        if let Some(w) = w {
            inputs.push(w);
        }
        if let Some(b) = b {
            inputs.push(b);
        }
        if let Some(rm) = rm {
            inputs.push(rm);
        }
        if let Some(rv) = rv {
            inputs.push(rv);
        }
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::BatchNorm, K>(
            &inputs,
            &inner,
            crate::exec::catalog::BatchNormAttributes {
                epsilon: f64::from(e),
                momentum,
                training: rm.is_none(),
                has_weight: w.is_some(),
                has_bias: b.is_some(),
                has_running_mean: rm.is_some(),
                has_running_variance: rv.is_some(),
            },
        )
    }

    /// Delegates to `B::conv_transpose2d`, additionally recording an `OperationKind::ConvTranspose2d` node.
    fn conv_transpose2d<K: super::dtype::DType>(
        t: &<Self as StorageBackend>::Storage<K>,
        w: &<Self as StorageBackend>::Storage<K>,
        b: Option<&<Self as StorageBackend>::Storage<K>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
        dilation: usize,
        groups: usize,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner_b = b.map(|b| &b.inner);
        let inner = B::conv_transpose2d(
            &t.inner,
            &w.inner,
            inner_b,
            stride,
            padding,
            output_padding,
            dilation,
            groups,
        )?;
        let mut inputs = alloc::vec![t, w];
        if let Some(b) = b {
            inputs.push(b);
        }
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::ConvTranspose2d, K>(
            &inputs,
            &inner,
            crate::exec::catalog::ConvTranspose2dAttributes {
                stride: [stride, stride],
                padding: [padding, padding],
                output_padding: [output_padding, output_padding],
                dilation: [dilation, dilation],
                groups,
                has_bias: b.is_some(),
            },
        )
    }
}

impl<B: Backend + crate::tensor::backend::LossOps<B>> LossOps<Self> for TracingBackend<B> {
    /// Delegates to `B::cross_entropy_loss`, additionally recording an `OperationKind::CrossEntropyLoss` node.
    fn cross_entropy_loss<K: super::dtype::DType, KInt: super::dtype::DType>(
        logits: &<Self as StorageBackend>::Storage<K>,
        targets: &<Self as StorageBackend>::Storage<KInt>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::cross_entropy_loss(&logits.inner, &targets.inner, reduction)?;
        Self::trace_canonical_inputs_with_attributes::<crate::exec::catalog::op::CrossEntropyLoss, K>(
            alloc::vec![logits.value_id, targets.value_id],
            alloc::vec![
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&logits.inner)),
                    dtype: Some(Self::traced_dtype(&logits.inner)),
                    device: Some(B::metadata(&logits.inner).device),
                },
                crate::exec::catalog::LogicalTensorMeta {
                    shape: Some(B::shape(&targets.inner)),
                    dtype: Some(Self::traced_dtype(&targets.inner)),
                    device: Some(B::metadata(&targets.inner).device),
                },
            ],
            &inner,
            crate::exec::catalog::LossAttributes {
                reduction: match reduction {
                    crate::nn::loss::Reduction::None => crate::exec::catalog::LossReduction::None,
                    crate::nn::loss::Reduction::Mean => crate::exec::catalog::LossReduction::Mean,
                    crate::nn::loss::Reduction::Sum => crate::exec::catalog::LossReduction::Sum,
                },
            },
        )
    }

    /// Delegates to `B::mse_loss`, additionally recording an `OperationKind::MseLoss` node.
    fn mse_loss<K: super::dtype::DType>(
        predictions: &<Self as StorageBackend>::Storage<K>,
        targets: &<Self as StorageBackend>::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::mse_loss(&predictions.inner, &targets.inner, reduction)?;
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::MseLoss, K>(
            &[predictions, targets],
            &inner,
            crate::exec::catalog::LossAttributes {
                reduction: match reduction {
                    crate::nn::loss::Reduction::None => crate::exec::catalog::LossReduction::None,
                    crate::nn::loss::Reduction::Mean => crate::exec::catalog::LossReduction::Mean,
                    crate::nn::loss::Reduction::Sum => crate::exec::catalog::LossReduction::Sum,
                },
            },
        )
    }

    /// Delegates to `B::l1_loss`, additionally recording an `OperationKind::L1Loss` node.
    fn l1_loss<K: super::dtype::DType>(
        predictions: &<Self as StorageBackend>::Storage<K>,
        targets: &<Self as StorageBackend>::Storage<K>,
        reduction: crate::nn::loss::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::l1_loss(&predictions.inner, &targets.inner, reduction)?;
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::L1Loss, K>(
            &[predictions, targets],
            &inner,
            crate::exec::catalog::LossAttributes {
                reduction: match reduction {
                    crate::nn::loss::Reduction::None => crate::exec::catalog::LossReduction::None,
                    crate::nn::loss::Reduction::Mean => crate::exec::catalog::LossReduction::Mean,
                    crate::nn::loss::Reduction::Sum => crate::exec::catalog::LossReduction::Sum,
                },
            },
        )
    }

    /// Delegates to `B::bce_with_logits_loss`, additionally recording an `OperationKind::BceWithLogitsLoss` node.
    fn bce_with_logits_loss<K: super::dtype::DType>(
        logits: &<Self as StorageBackend>::Storage<K>,
        targets: &<Self as StorageBackend>::Storage<K>,
        _r: crate::nn::loss::Reduction,
    ) -> Result<<Self as StorageBackend>::Storage<K>> {
        let inner = B::bce_with_logits_loss(&logits.inner, &targets.inner, _r)?;
        Self::trace_canonical_multi_shape::<crate::exec::catalog::op::BceWithLogitsLoss, K>(
            &[logits, targets],
            &inner,
            crate::exec::catalog::LossAttributes {
                reduction: match _r {
                    crate::nn::loss::Reduction::None => crate::exec::catalog::LossReduction::None,
                    crate::nn::loss::Reduction::Mean => crate::exec::catalog::LossReduction::Mean,
                    crate::nn::loss::Reduction::Sum => crate::exec::catalog::LossReduction::Sum,
                },
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::backend::dummy::DummyBackend;
    use crate::tensor::device::Cpu;

    type B = TracingBackend<DummyBackend<Cpu>>;

    /// Looks up the `OperationKind` of the node that produced `value_id`, robust
    /// to other tests concurrently adding unrelated nodes to the shared
    /// process-wide `TRACING_GRAPH`.
    fn op_for_output(value_id: ValueId) -> OperationKind {
        let g = TRACING_GRAPH.lock();
        let operation = g
            .nodes
            .iter()
            .find(|n| n.outputs.contains(&value_id))
            .unwrap_or_else(|| panic!("no node found producing value {value_id}"))
            .operation
            .clone();
        match operation {
            crate::exec::OperationIdentity::Builtin(operation) => operation,
            crate::exec::OperationIdentity::Custom(_) => {
                panic!("tracing test expects a built-in operation")
            }
        }
    }

    #[test]
    /// Each unary activation must record its own `OperationKind`, not a
    /// different op's — regression test for a bug where abs/exp/neg/
    /// sqrt/log/tanh/sigmoid/swish all recorded `OperationKind::Relu` (copy-paste
    /// from `relu`'s implementation), silently corrupting ONNX export /
    /// graph visualization for any model using those activations.
    fn unary_activations_record_their_own_op_type() {
        let t: <B as StorageBackend>::Storage<f32> = TracingTensor {
            inner: alloc::vec![2, 3],
            value_id: 0,
        };

        let cases: [(
            fn(&<B as StorageBackend>::Storage<f32>) -> Result<<B as StorageBackend>::Storage<f32>>,
            OperationKind,
        ); 9] = [
            (<B as FloatOps<B>>::relu::<f32>, OperationKind::Relu),
            (<B as FloatOps<B>>::abs::<f32>, OperationKind::Abs),
            (<B as FloatOps<B>>::exp::<f32>, OperationKind::Exp),
            (<B as FloatOps<B>>::neg::<f32>, OperationKind::Neg),
            (<B as FloatOps<B>>::sqrt::<f32>, OperationKind::Sqrt),
            (<B as FloatOps<B>>::log::<f32>, OperationKind::Log),
            (<B as FloatOps<B>>::tanh::<f32>, OperationKind::Tanh),
            (<B as FloatOps<B>>::sigmoid::<f32>, OperationKind::Sigmoid),
            (<B as FloatOps<B>>::swish::<f32>, OperationKind::Swish),
        ];

        for (f, expected_op) in cases {
            let out = f(&t).unwrap();
            assert_eq!(
                op_for_output(out.value_id),
                expected_op,
                "wrong OperationKind recorded for this activation"
            );
        }
    }
}
