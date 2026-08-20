//! Core communicator management and collective execution for NCCL transport.

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream, DeviceRepr};
use cudarc::nccl::{Comm, Id, NcclType};
use incin_core::dist::placement::Placement;
use incin_core::dist::{
    AgreedPlan, CollectiveDType, CollectiveDescriptor, CollectiveError, CollectiveKind,
    CollectivePlan, DataParallelDType, DistributedContext, DistributedContextHandle, GradientId,
    PipelineBoundaryId, PipelineDType, PipelineTransfer, TensorParallelCollective,
    TensorParallelDType, TensorParallelError, TensorParallelId, validate_collective_dtype,
    validate_data_parallel_dtype, validate_pipeline_dtype, validate_tensor_parallel_dtype,
};
use incin_core::exec::ReduceOp;
use incin_core::shapes::Shape;
use incin_core::tensor::base::Tensor;
use incin_core::tensor::device::Device;
use incin_core::tensor::dtype::{DType, DTypeId};
use incin_core::tensor::grad::RequiresGrad;

use crate::cuda::backend::CudaBackendImpl;
use crate::cuda::storage::{CudaBuffer, CudaStorage};
use crate::cuda::tape::CudaGrads;
use crate::dist::nccl::buffer::{NcclBuffer, NcclEvent};
use crate::dist::nccl::config::{
    BootstrapRole, TwoRankBootstrapConfig, WORLD, bootstrap_from_context,
};
use crate::dist::nccl::error::{NcclTransportError, catch_nccl_panic, nccl_error};
use crate::dist::nccl::wire::{exchange_bootstrap, id_from_bytes, id_to_bytes};

/// One NCCL communicator bound to one agreed plan and one of two ranks.
#[derive(Debug)]
pub struct NcclTransport {
    comm: Comm,
    context: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    plan: CollectivePlan,
    agreed: AgreedPlan,
    cursor: usize,
    rank: usize,
    device_ordinal: usize,
    distributed_context: Option<DistributedContextHandle>,
}

impl NcclTransport {
    /// Bootstrap over TCP and initialize a two-rank NCCL communicator.
    ///
    /// Rank zero must use [`TwoRankBootstrapConfig::root`] and rank one must
    /// use [`TwoRankBootstrapConfig::peer`] with the root's reachable address.
    /// Plan agreement happens before `ncclCommInitRank`.
    pub fn connect(
        config: TwoRankBootstrapConfig,
        plan: CollectivePlan,
        device_ordinal: usize,
    ) -> Result<Self, NcclTransportError> {
        Self::connect_inner(config, plan, device_ordinal, None)
    }

    /// Initialize NCCL from an already-agreed distributed context.
    ///
    /// Rank, reachable root address, deadline, and process-local CUDA ordinal
    /// come from rendezvous rather than being repeated by the caller. Any
    /// communicator initialization or later launch failure invalidates the
    /// context's shared fail-stop handle.
    pub fn connect_context<M, R>(
        context: &DistributedContext<M, R>,
        plan: CollectivePlan,
    ) -> Result<Self, NcclTransportError> {
        context.ensure_active()?;
        let config = bootstrap_from_context(context);
        let handle = context.handle();
        match Self::connect_inner(
            config,
            plan,
            context.local_cuda_device(),
            Some(handle.clone()),
        ) {
            Ok(transport) => Ok(transport),
            Err(error) => {
                handle.invalidate();
                Err(error)
            }
        }
    }

    fn connect_inner(
        config: TwoRankBootstrapConfig,
        plan: CollectivePlan,
        device_ordinal: usize,
        distributed_context: Option<DistributedContextHandle>,
    ) -> Result<Self, NcclTransportError> {
        if config.timeout.is_zero() {
            return Err(NcclTransportError::InvalidTimeout);
        }
        let root_id = match config.role {
            BootstrapRole::Root { .. } => {
                Some(catch_nccl_panic("create unique id", Id::new)?.map_err(nccl_error)?)
            }
            BootstrapRole::Peer { .. } => None,
        };
        let root_bytes = root_id.as_ref().map(id_to_bytes);
        let bootstrap = exchange_bootstrap(config, plan.summary(), root_bytes)?;
        let id = root_id.unwrap_or_else(|| id_from_bytes(bootstrap.unique_id));
        let context = crate::cuda::gpu::cuda_cache::try_get_cuda_device(device_ordinal)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let stream = context.default_stream();
        let rank = config.rank();
        let comm = catch_nccl_panic("initialize communicator", || {
            Comm::from_rank(stream.clone(), rank, WORLD, id)
        })?
        .map_err(nccl_error)?;
        Ok(Self {
            comm,
            context,
            stream,
            plan,
            agreed: bootstrap.agreed,
            cursor: 0,
            rank,
            device_ordinal,
            distributed_context,
        })
    }

    /// This process's communicator rank.
    #[must_use]
    pub const fn rank(&self) -> usize {
        self.rank
    }

    /// Fixed communicator cardinality.
    #[must_use]
    pub const fn world_size(&self) -> usize {
        WORLD
    }

    /// Cross-rank plan agreement established by bootstrap.
    #[must_use]
    pub const fn agreed_plan(&self) -> AgreedPlan {
        self.agreed
    }

    /// Number of descriptors already submitted.
    #[must_use]
    pub const fn cursor(&self) -> usize {
        self.cursor
    }

    /// Launch the next descriptor for a statically checked dtype or for `Dyn`.
    ///
    /// Static unsupported dtypes cannot satisfy [`CollectiveDType`]. `Dyn`
    /// reaches the same method and is checked against the descriptor at runtime.
    pub fn execute<K: CollectiveDType>(
        &mut self,
        input: &NcclBuffer<K>,
    ) -> Result<(NcclBuffer<K>, NcclEvent), NcclTransportError> {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        validate_launch(
            descriptor,
            self.cursor,
            input.dtype(),
            input.elements(),
            input.bytes(),
        )?;

        let mut output = self
            .stream
            .alloc_zeros::<u8>(descriptor.output_bytes())
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        launch_by_dtype(
            &self.comm,
            &self.stream,
            self.rank,
            descriptor.kind(),
            descriptor.dtype(),
            input.device_bytes(),
            &mut output,
            descriptor.input_elements(),
            descriptor.output_elements(),
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let result = NcclBuffer::try_from_device_bytes(
            output,
            descriptor.output_elements(),
            input.dtype.clone(),
        )?;
        let completion = NcclEvent {
            event,
            group: descriptor.group(),
            sequence: descriptor.sequence(),
            stream: descriptor.stream(),
            kind: descriptor.kind(),
            distributed_context: self.distributed_context.clone(),
        };
        self.cursor += 1;
        context_guard.commit();
        Ok((result, completion))
    }

    /// Mean-reduce one parameter gradient from a typed CUDA tensor.
    ///
    /// `K` is inferred from the tensor. Static integer and quantized tensors
    /// cannot satisfy [`DataParallelDType`]; a `Tensor<..., Dyn, ...>` reaches
    /// this method and its runtime dtype is checked before allocation or
    /// launch. The plan descriptor must carry the same stable [`GradientId`]
    /// and DP=2 `Partial<Mean> -> Replicated` semantics.
    pub fn synchronize_gradient<S, D, K, G, P>(
        &mut self,
        id: GradientId,
        parameter: &Tensor<S, CudaBackendImpl<D>, K, G, P>,
        gradients: &mut CudaGrads,
    ) -> Result<NcclEvent, NcclTransportError>
    where
        S: Shape,
        D: Device,
        K: DataParallelDType,
        G: RequiresGrad,
        P: Placement,
    {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let dtype = parameter
            .builtin_dtype_id()
            .ok_or(NcclTransportError::Collective(
                CollectiveError::UnsupportedDType {
                    dtype: parameter.dtype().builtin_id().unwrap_or(DTypeId::F32),
                },
            ))?;
        validate_data_parallel_dtype(dtype)?;
        let parameter_storage = parameter.inner();
        let input = gradients
            .get(parameter_storage.id)
            .cloned()
            .ok_or(NcclTransportError::MissingGradient { id })?;
        let (output, event) = self.execute_gradient_storage(id, dtype, &input)?;
        gradients.grads.insert(parameter_storage.id, output);
        context_guard.commit();
        Ok(event)
    }

    /// Execute one TP=2 linear/attention collective directly from CUDA tensor
    /// storage and return its flat rank-ordered result.
    ///
    /// Static integer and quantized tensors cannot satisfy
    /// [`TensorParallelDType`]. `Dyn` reaches the same method and is checked
    /// before allocation or launch. For all-gather, the flat result is ordered
    /// by rank shard; a higher layer must reassemble non-leading tensor axes
    /// before presenting the result as an ordinary row-major tensor.
    pub fn execute_tensor_parallel_flat<S, D, K, G, P>(
        &mut self,
        id: TensorParallelId,
        collective: TensorParallelCollective,
        input: &Tensor<S, CudaBackendImpl<D>, K, G, P>,
    ) -> Result<(CudaStorage, NcclEvent), NcclTransportError>
    where
        S: Shape,
        D: Device,
        K: TensorParallelDType,
        G: RequiresGrad,
        P: Placement,
    {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let dtype = input
            .builtin_dtype_id()
            .ok_or(NcclTransportError::Collective(
                CollectiveError::UnsupportedDType {
                    dtype: input.dtype().builtin_id().unwrap_or(DTypeId::F32),
                },
            ))?;
        validate_tensor_parallel_dtype(dtype)?;
        let input = input.inner();
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        validate_tensor_parallel_launch(
            descriptor,
            self.cursor,
            id,
            collective,
            dtype,
            input.buffer.len,
            input.buffer.data.len(),
        )?;
        if input.buffer.device_id != self.device_ordinal {
            return Err(NcclTransportError::TensorParallelDevice {
                expected: self.device_ordinal,
                found: input.buffer.device_id,
            });
        }
        if input.meta.offset_elements() != 0
            || input.meta.layout() != incin_core::exec::LayoutClass::Contiguous
        {
            return Err(NcclTransportError::TensorParallelLayout {
                offset: input.meta.offset_elements(),
                layout: input.meta.layout(),
            });
        }

        let mut output = self
            .stream
            .alloc_zeros::<u8>(descriptor.output_bytes())
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        launch_by_dtype(
            &self.comm,
            &self.stream,
            self.rank,
            descriptor.kind(),
            descriptor.dtype(),
            &input.buffer.data,
            &mut output,
            descriptor.input_elements(),
            descriptor.output_elements(),
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let buffer = CudaBuffer {
            len: descriptor.output_elements(),
            dtype: dtype.descriptor(),
            data: Arc::new(output),
            device: self.context.clone(),
            device_id: self.device_ordinal,
        };
        let storage = CudaStorage::try_new(Arc::new(buffer), vec![descriptor.output_elements()])
            .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
        let completion = NcclEvent {
            event,
            group: descriptor.group(),
            sequence: descriptor.sequence(),
            stream: descriptor.stream(),
            kind: descriptor.kind(),
            distributed_context: self.distributed_context.clone(),
        };
        self.cursor += 1;
        context_guard.commit();
        Ok((storage, completion))
    }

    /// Execute one TP=2 collective and reassemble the replicated logical shape
    /// on CUDA.
    ///
    /// NCCL all-gather returns rank-major shards. When the sharded tensor axis
    /// is not leading, this method materializes the required axis movement on
    /// the same CUDA stream before recording the returned completion event.
    pub fn execute_tensor_parallel<S, D, K, G, P>(
        &mut self,
        id: TensorParallelId,
        collective: TensorParallelCollective,
        input: &Tensor<S, CudaBackendImpl<D>, K, G, P>,
        global_shape: &[usize],
    ) -> Result<(CudaStorage, NcclEvent), NcclTransportError>
    where
        S: Shape,
        D: Device,
        K: TensorParallelDType,
        G: RequiresGrad,
        P: Placement,
    {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        let input_shape = input.inner().meta.shape().dims();
        let local_shape =
            validate_tensor_parallel_shapes(descriptor, collective, input_shape, global_shape)?;

        let (flat, first_event) = self.execute_tensor_parallel_flat(id, collective, input)?;
        let storage = reassemble_tensor_parallel_storage::<D, K>(
            &flat,
            collective,
            &local_shape,
            global_shape,
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        context_guard.commit();
        Ok((
            storage,
            NcclEvent {
                event,
                group: first_event.group,
                sequence: first_event.sequence,
                stream: first_event.stream,
                kind: first_event.kind,
                distributed_context: first_event.distributed_context,
            },
        ))
    }

    /// Execute one PP=2 activation or gradient transfer from a typed CUDA
    /// tensor.
    ///
    /// Both ranks submit a same-shaped tensor. The source tensor is sent and
    /// copied to the returned source storage; the destination's input is a
    /// placeholder overwritten by `recv`. This symmetric call shape lets `K`
    /// be inferred on both processes while the immutable descriptor carries
    /// one global source/destination contract.
    pub fn execute_pipeline<S, D, K, G, P>(
        &mut self,
        boundary: PipelineBoundaryId,
        transfer: PipelineTransfer,
        microbatch: usize,
        input: &Tensor<S, CudaBackendImpl<D>, K, G, P>,
    ) -> Result<(CudaStorage, NcclEvent), NcclTransportError>
    where
        S: Shape,
        D: Device,
        K: PipelineDType,
        G: RequiresGrad,
        P: Placement,
    {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let dtype = input
            .builtin_dtype_id()
            .ok_or(NcclTransportError::Collective(
                CollectiveError::UnsupportedDType {
                    dtype: input.dtype().builtin_id().unwrap_or(DTypeId::F32),
                },
            ))?;
        validate_pipeline_dtype(dtype)?;
        let input = input.inner();
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        validate_pipeline_launch(
            descriptor,
            self.cursor,
            boundary,
            transfer,
            microbatch,
            dtype,
            input.buffer.len,
            input.buffer.data.len(),
        )?;
        if input.buffer.device_id != self.device_ordinal {
            return Err(NcclTransportError::PipelineDevice {
                expected: self.device_ordinal,
                found: input.buffer.device_id,
            });
        }
        if input.meta.offset_elements() != 0
            || input.meta.layout() != incin_core::exec::LayoutClass::Contiguous
        {
            return Err(NcclTransportError::PipelineLayout {
                offset: input.meta.offset_elements(),
                layout: input.meta.layout(),
            });
        }

        let mut output = self
            .stream
            .alloc_zeros::<u8>(descriptor.output_bytes())
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        launch_by_dtype(
            &self.comm,
            &self.stream,
            self.rank,
            descriptor.kind(),
            descriptor.dtype(),
            &input.buffer.data,
            &mut output,
            descriptor.input_elements(),
            descriptor.output_elements(),
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let buffer = CudaBuffer {
            len: descriptor.output_elements(),
            dtype: dtype.descriptor(),
            data: Arc::new(output),
            device: self.context.clone(),
            device_id: self.device_ordinal,
        };
        let storage = CudaStorage::try_new(Arc::new(buffer), input.meta.shape().dims().to_vec())
            .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
        let completion = NcclEvent {
            event,
            group: descriptor.group(),
            sequence: descriptor.sequence(),
            stream: descriptor.stream(),
            kind: descriptor.kind(),
            distributed_context: self.distributed_context.clone(),
        };
        self.cursor += 1;
        context_guard.commit();
        Ok((storage, completion))
    }

    fn execute_gradient_storage(
        &mut self,
        id: GradientId,
        dtype: DTypeId,
        input: &CudaStorage,
    ) -> Result<(CudaStorage, NcclEvent), NcclTransportError> {
        let mut context_guard = ContextOperationGuard::new(&self.distributed_context)?;
        let descriptor =
            self.plan
                .descriptors()
                .get(self.cursor)
                .ok_or(NcclTransportError::PlanExhausted {
                    collectives: self.plan.descriptors().len(),
                })?;
        validate_gradient_launch(
            descriptor,
            self.cursor,
            id,
            dtype,
            input.buffer.len,
            input.buffer.data.len(),
        )?;
        if input.buffer.device_id != self.device_ordinal {
            return Err(NcclTransportError::GradientDevice {
                expected: self.device_ordinal,
                found: input.buffer.device_id,
            });
        }
        if input.meta.offset_elements() != 0
            || input.meta.layout() != incin_core::exec::LayoutClass::Contiguous
        {
            return Err(NcclTransportError::GradientLayout {
                offset: input.meta.offset_elements(),
                layout: input.meta.layout(),
            });
        }

        let mut output = self
            .stream
            .alloc_zeros::<u8>(descriptor.output_bytes())
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        launch_by_dtype(
            &self.comm,
            &self.stream,
            self.rank,
            descriptor.kind(),
            descriptor.dtype(),
            &input.buffer.data,
            &mut output,
            descriptor.input_elements(),
            descriptor.output_elements(),
        )?;
        let event = self
            .stream
            .record_event(None)
            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
        let buffer = CudaBuffer {
            len: descriptor.output_elements(),
            dtype: dtype.descriptor(),
            data: Arc::new(output),
            device: self.context.clone(),
            device_id: self.device_ordinal,
        };
        let storage = CudaStorage::try_new(Arc::new(buffer), input.meta.shape().dims().to_vec())
            .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
        let completion = NcclEvent {
            event,
            group: descriptor.group(),
            sequence: descriptor.sequence(),
            stream: descriptor.stream(),
            kind: descriptor.kind(),
            distributed_context: self.distributed_context.clone(),
        };
        self.cursor += 1;
        context_guard.commit();
        Ok((storage, completion))
    }
}

pub(crate) struct ContextOperationGuard {
    handle: Option<DistributedContextHandle>,
    committed: bool,
}

impl ContextOperationGuard {
    pub(crate) fn new(
        handle: &Option<DistributedContextHandle>,
    ) -> Result<Self, NcclTransportError> {
        if let Some(handle) = handle {
            handle.ensure_active()?;
        }
        Ok(Self {
            handle: handle.clone(),
            committed: false,
        })
    }

    pub(crate) fn commit(&mut self) {
        self.committed = true;
    }
}

impl Drop for ContextOperationGuard {
    fn drop(&mut self) {
        if !self.committed
            && let Some(handle) = &self.handle
        {
            handle.invalidate();
        }
    }
}

pub(crate) fn validate_gradient_launch(
    descriptor: &CollectiveDescriptor,
    cursor: usize,
    id: GradientId,
    dtype: DTypeId,
    elements: usize,
    bytes: usize,
) -> Result<(), NcclTransportError> {
    validate_data_parallel_dtype(dtype)?;
    if descriptor.tag().get() != id.get() {
        return Err(NcclTransportError::GradientIdentity {
            expected: descriptor.tag().get(),
            found: id.get(),
        });
    }
    let expected_source = incin_core::dist::PlacementKind::Partial {
        reduction: ReduceOp::Mean,
    };
    if descriptor.kind() != CollectiveKind::AllReduce(ReduceOp::Mean)
        || descriptor.source() != expected_source
        || descriptor.destination() != incin_core::dist::PlacementKind::Replicated
    {
        return Err(NcclTransportError::NotDataParallelGradient {
            kind: descriptor.kind(),
            from_placement: descriptor.source(),
            to_placement: descriptor.destination(),
        });
    }
    validate_launch(descriptor, cursor, dtype, elements, bytes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_tensor_parallel_launch(
    descriptor: &CollectiveDescriptor,
    cursor: usize,
    id: TensorParallelId,
    collective: TensorParallelCollective,
    dtype: DTypeId,
    elements: usize,
    bytes: usize,
) -> Result<(), NcclTransportError> {
    validate_tensor_parallel_dtype(dtype)?;
    let expected_tag = collective.plan_tag(id).get();
    if descriptor.tag().get() != expected_tag {
        return Err(NcclTransportError::TensorParallelIdentity {
            expected: descriptor.tag().get(),
            found: expected_tag,
        });
    }
    let valid = match collective {
        TensorParallelCollective::ColumnOutputGather { tensor_axis }
        | TensorParallelCollective::AttentionHeadGather { tensor_axis } => {
            descriptor.kind() == CollectiveKind::AllGather
                && descriptor.source()
                    == incin_core::dist::PlacementKind::Sharded { axis: tensor_axis }
                && descriptor.destination() == incin_core::dist::PlacementKind::Replicated
        }
        TensorParallelCollective::RowOutputSum => {
            descriptor.kind() == CollectiveKind::AllReduce(ReduceOp::Sum)
                && descriptor.source()
                    == incin_core::dist::PlacementKind::Partial {
                        reduction: ReduceOp::Sum,
                    }
                && descriptor.destination() == incin_core::dist::PlacementKind::Replicated
        }
    };
    if !valid {
        return Err(NcclTransportError::NotTensorParallelOperation {
            expected: collective,
            kind: descriptor.kind(),
            from_placement: descriptor.source(),
            to_placement: descriptor.destination(),
        });
    }
    validate_launch(descriptor, cursor, dtype, elements, bytes)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn validate_pipeline_launch(
    descriptor: &CollectiveDescriptor,
    cursor: usize,
    boundary: PipelineBoundaryId,
    transfer: PipelineTransfer,
    microbatch: usize,
    dtype: DTypeId,
    elements: usize,
    bytes: usize,
) -> Result<(), NcclTransportError> {
    validate_pipeline_dtype(dtype)?;
    let expected = transfer.plan_tag(boundary, microbatch).get();
    if descriptor.tag().get() != expected {
        return Err(NcclTransportError::PipelineIdentity {
            expected,
            found: descriptor.tag().get(),
        });
    }
    let expected_kind = CollectiveKind::SendRecv {
        source: transfer.source_rank(),
        destination: transfer.destination_rank(),
    };
    let expected_source = incin_core::dist::PlacementKind::PipelineStage {
        index: transfer.source_rank(),
    };
    let expected_destination = incin_core::dist::PlacementKind::PipelineStage {
        index: transfer.destination_rank(),
    };
    if descriptor.kind() != expected_kind
        || descriptor.source() != expected_source
        || descriptor.destination() != expected_destination
    {
        return Err(NcclTransportError::NotPipelineTransfer {
            expected: transfer,
            kind: descriptor.kind(),
            from_placement: descriptor.source(),
            to_placement: descriptor.destination(),
        });
    }
    validate_launch(descriptor, cursor, dtype, elements, bytes)
}

pub(crate) fn validate_tensor_parallel_shapes(
    descriptor: &CollectiveDescriptor,
    collective: TensorParallelCollective,
    input_shape: &[usize],
    global_shape: &[usize],
) -> Result<Vec<usize>, NcclTransportError> {
    let global_elements = checked_shape_elements(global_shape)?;
    if global_elements != descriptor.output_elements() {
        return Err(NcclTransportError::TensorParallelOutputElements {
            expected: descriptor.output_elements(),
            found: global_elements,
        });
    }

    let expected_local = match collective {
        TensorParallelCollective::ColumnOutputGather { tensor_axis }
        | TensorParallelCollective::AttentionHeadGather { tensor_axis } => {
            let Some(&global_extent) = global_shape.get(tensor_axis) else {
                return Err(TensorParallelError::AxisOutOfBounds {
                    axis: tensor_axis,
                    rank: global_shape.len(),
                }
                .into());
            };
            let local_extent = incin_core::dist::validate_two_way_extent(
                match collective {
                    TensorParallelCollective::ColumnOutputGather { .. } => {
                        incin_core::dist::TensorParallelDimension::OutputFeatures
                    }
                    TensorParallelCollective::AttentionHeadGather { .. } => {
                        incin_core::dist::TensorParallelDimension::AttentionHeads
                    }
                    TensorParallelCollective::RowOutputSum => unreachable!(),
                },
                global_extent,
            )?;
            let mut expected = global_shape.to_vec();
            expected[tensor_axis] = local_extent;
            expected
        }
        TensorParallelCollective::RowOutputSum => global_shape.to_vec(),
    };
    if input_shape != expected_local {
        return Err(NcclTransportError::TensorParallelShape {
            expected: expected_local,
            found: input_shape.to_vec(),
        });
    }
    Ok(expected_local)
}

pub(crate) fn reassemble_tensor_parallel_storage<D: Device, K: DType>(
    flat: &CudaStorage,
    collective: TensorParallelCollective,
    local_shape: &[usize],
    global_shape: &[usize],
) -> Result<CudaStorage, NcclTransportError>
where
    K: TensorParallelDType,
{
    let mut storage = match collective {
        TensorParallelCollective::ColumnOutputGather { tensor_axis }
        | TensorParallelCollective::AttentionHeadGather { tensor_axis } => {
            let mut rank_major = Vec::with_capacity(local_shape.len() + 1);
            rank_major.push(WORLD);
            rank_major.extend_from_slice(local_shape);
            let mut storage = CudaBackendImpl::<D>::reshape::<K>(flat, &rank_major)
                .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
            for position in 0..tensor_axis {
                storage = CudaBackendImpl::<D>::transpose::<K>(&storage, position, position + 1)
                    .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
            }
            storage
        }
        TensorParallelCollective::RowOutputSum => flat.clone(),
    };
    storage = CudaBackendImpl::<D>::reshape::<K>(&storage, global_shape)
        .map_err(|error| NcclTransportError::InvalidBuffer(error.to_string()))?;
    Ok(storage)
}

pub(crate) fn checked_shape_elements(shape: &[usize]) -> Result<usize, NcclTransportError> {
    shape.iter().copied().try_fold(1usize, |elements, extent| {
        elements
            .checked_mul(extent)
            .ok_or(NcclTransportError::TensorParallel(
                TensorParallelError::ElementCountOverflow,
            ))
    })
}

pub(crate) fn validate_launch(
    descriptor: &CollectiveDescriptor,
    cursor: usize,
    dtype: DTypeId,
    elements: usize,
    bytes: usize,
) -> Result<(), NcclTransportError> {
    if descriptor.sequence().get() != cursor as u64 {
        return Err(NcclTransportError::Sequence {
            expected: cursor as u64,
            found: descriptor.sequence().get(),
        });
    }
    if descriptor.group().ranks() != WORLD {
        return Err(NcclTransportError::GroupCardinality {
            expected: WORLD,
            found: descriptor.group().ranks(),
        });
    }
    validate_collective_dtype(dtype)?;
    if dtype != descriptor.dtype() {
        return Err(NcclTransportError::DType {
            expected: descriptor.dtype(),
            found: dtype,
        });
    }
    if elements != descriptor.input_elements() {
        return Err(NcclTransportError::Elements {
            expected: descriptor.input_elements(),
            found: elements,
        });
    }
    if bytes != descriptor.input_bytes() {
        return Err(NcclTransportError::BufferBytes {
            expected: descriptor.input_bytes(),
            found: bytes,
        });
    }
    validate_reduction(descriptor.kind(), descriptor.dtype())
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_by_dtype(
    comm: &Comm,
    stream: &Arc<CudaStream>,
    rank: usize,
    kind: CollectiveKind,
    dtype: DTypeId,
    input: &CudaSlice<u8>,
    output: &mut CudaSlice<u8>,
    input_elements: usize,
    output_elements: usize,
) -> Result<(), NcclTransportError> {
    macro_rules! launch {
        ($element:ty) => {
            launch_typed::<$element>(
                comm,
                stream,
                rank,
                kind,
                input,
                output,
                input_elements,
                output_elements,
            )
        };
    }
    match dtype {
        DTypeId::U8 => launch!(u8),
        DTypeId::U32 => launch!(u32),
        DTypeId::I64 => launch!(i64),
        DTypeId::BF16 => launch!(half::bf16),
        DTypeId::F16 => launch!(half::f16),
        DTypeId::F32 => launch!(f32),
        DTypeId::F64 => launch!(f64),
        DTypeId::Q8_0 => Err(CollectiveError::UnsupportedDType { dtype }.into()),
        _ => Err(CollectiveError::UnsupportedDType { dtype }.into()),
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn launch_typed<T: DeviceRepr + NcclType>(
    comm: &Comm,
    stream: &Arc<CudaStream>,
    rank: usize,
    kind: CollectiveKind,
    input: &CudaSlice<u8>,
    output: &mut CudaSlice<u8>,
    input_elements: usize,
    output_elements: usize,
) -> Result<(), NcclTransportError> {
    let input = unsafe { input.transmute::<T>(input_elements) }.ok_or(
        NcclTransportError::Protocol("input reinterpretation failed"),
    )?;
    let mut output = unsafe { output.transmute_mut::<T>(output_elements) }.ok_or(
        NcclTransportError::Protocol("output reinterpretation failed"),
    )?;
    match kind {
        CollectiveKind::AllReduce(op) => {
            comm.all_reduce(&input, &mut output, &nccl_reduce(op))
                .map_err(nccl_error)?;
        }
        CollectiveKind::AllGather => {
            comm.all_gather(&input, &mut output).map_err(nccl_error)?;
        }
        CollectiveKind::ReduceScatter(op) => {
            comm.reduce_scatter(&input, &mut output, &nccl_reduce(op))
                .map_err(nccl_error)?;
        }
        CollectiveKind::AllToAll => {
            if input_elements != output_elements || !input_elements.is_multiple_of(WORLD) {
                return Err(CollectiveError::NonDivisible {
                    elements: input_elements,
                    ranks: WORLD,
                }
                .into());
            }
            let chunk = input_elements / WORLD;
            let (input0, input1) = input.split_at(chunk);
            let (mut output0, mut output1) = output.split_at_mut(chunk);
            catch_nccl_panic("launch all-to-all group", || {
                match rank {
                    0 => {
                        stream
                            .memcpy_dtod(&input0, &mut output0)
                            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
                        let mut group = comm.group();
                        group.send(input1, 1).map_err(nccl_error)?;
                        group.recv(output1, 1).map_err(nccl_error)?;
                    }
                    1 => {
                        stream
                            .memcpy_dtod(&input1, &mut output1)
                            .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
                        let mut group = comm.group();
                        group.send(input0, 0).map_err(nccl_error)?;
                        group.recv(output0, 0).map_err(nccl_error)?;
                    }
                    found => {
                        return Err(NcclTransportError::RemoteRank { expected: 1, found });
                    }
                }
                Ok(())
            })??;
        }
        CollectiveKind::SendRecv {
            source,
            destination,
        } => {
            if source >= WORLD {
                return Err(CollectiveError::PeerOutOfRange {
                    endpoint: "source",
                    rank: source,
                    ranks: WORLD,
                }
                .into());
            }
            if destination >= WORLD {
                return Err(CollectiveError::PeerOutOfRange {
                    endpoint: "destination",
                    rank: destination,
                    ranks: WORLD,
                }
                .into());
            }
            if source == destination {
                return Err(CollectiveError::SamePeer { rank: source }.into());
            }
            if input_elements != output_elements {
                return Err(NcclTransportError::Elements {
                    expected: input_elements,
                    found: output_elements,
                });
            }
            match rank {
                rank if rank == source => {
                    stream
                        .memcpy_dtod(&input, &mut output)
                        .map_err(|error| NcclTransportError::Cuda(error.to_string()))?;
                    comm.send(&input, destination as i32).map_err(nccl_error)?;
                }
                rank if rank == destination => {
                    comm.recv(&mut output, source as i32).map_err(nccl_error)?;
                }
                found => {
                    return Err(NcclTransportError::LocalRank {
                        rank: found,
                        world: WORLD,
                    });
                }
            }
        }
        _ => return Err(NcclTransportError::UnsupportedCollective),
    }
    Ok(())
}

pub(crate) fn validate_reduction(
    kind: CollectiveKind,
    dtype: DTypeId,
) -> Result<(), NcclTransportError> {
    let op = match kind {
        CollectiveKind::AllReduce(op) | CollectiveKind::ReduceScatter(op) => op,
        CollectiveKind::AllGather | CollectiveKind::AllToAll | CollectiveKind::SendRecv { .. } => {
            return Ok(());
        }
        _ => return Err(NcclTransportError::UnsupportedCollective),
    };
    incin_core::dist::validate_collective_reduction(dtype, op).map_err(Into::into)
}

pub(crate) const fn nccl_reduce(op: ReduceOp) -> cudarc::nccl::ReduceOp {
    match op {
        ReduceOp::Sum => cudarc::nccl::ReduceOp::Sum,
        ReduceOp::Mean => cudarc::nccl::ReduceOp::Avg,
        ReduceOp::Max => cudarc::nccl::ReduceOp::Max,
        ReduceOp::Min => cudarc::nccl::ReduceOp::Min,
        ReduceOp::Prod => cudarc::nccl::ReduceOp::Prod,
    }
}
