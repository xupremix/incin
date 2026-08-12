//! Executable CPU lowering for the compiled graph plan.

use alloc::format;
use alloc::vec::Vec;

use incin_core::backend_authoring::{Execute, ExecutionRequest};
use incin_core::compiled::CompiledPlan;
use incin_core::exec::catalog::{CanonicalOperation, CapturedDescriptor, Descriptor, op};
use incin_core::exec::{ExecutionContext, OperationIdentity, TensorHandle};
use incin_core::prelude::{Cpu, DTypeDescriptor, Dyn, Error, Result};

use super::{CpuBackendImpl, CpuStorage};

/// Inputs and outputs for one executable CPU compiled invocation.
#[derive(Debug, Clone)]
pub struct CpuCompiledInvocation {
    pub inputs: Vec<CpuStorage>,
}

/// CPU compiled plan after backend lowering admission.
#[derive(Debug, Clone)]
pub struct CpuCompiledPlan {
    plan: CompiledPlan,
}

impl CpuCompiledPlan {
    /// Admits only nodes with a captured descriptor and an implemented CPU
    /// canonical lowering.
    pub fn try_new(plan: &CompiledPlan) -> Result<Self> {
        for node in &plan.graph.nodes {
            let supported = matches!(
                node.operation,
                OperationIdentity::Builtin(
                    incin_core::prelude::OperationKind::Add
                        | incin_core::prelude::OperationKind::Relu
                        | incin_core::prelude::OperationKind::MatMulExact
                        | incin_core::prelude::OperationKind::Linear
                        | incin_core::prelude::OperationKind::Addmm
                )
            );
            if !supported {
                return Err(Error::Msg(format!(
                    "compiled CPU lowering does not support {:?}",
                    node.operation
                )));
            }
            if node.descriptor_payload.is_none() {
                return Err(Error::Msg(format!(
                    "compiled CPU node {} has no captured descriptor",
                    node.id
                )));
            }
        }
        Ok(Self { plan: plan.clone() })
    }

    pub fn run(&self, invocation: CpuCompiledInvocation) -> Result<Vec<CpuStorage>> {
        invocation.run_admitted(&self.plan)
    }
}

impl CpuCompiledInvocation {
    /// Constructs an invocation from owned CPU input storages.
    #[must_use]
    pub fn new(inputs: Vec<CpuStorage>) -> Self {
        Self { inputs }
    }
}

impl CpuCompiledInvocation {
    /// Executes the plan using the planner's buffer slots.
    pub fn run(self, plan: &CompiledPlan) -> Result<Vec<CpuStorage>> {
        let admitted = CpuCompiledPlan::try_new(plan)?;
        self.run_admitted(&admitted.plan)
    }

    fn run_admitted(self, plan: &CompiledPlan) -> Result<Vec<CpuStorage>> {
        let input_meta = self
            .inputs
            .iter()
            .map(|storage| (storage.shape.to_vec(), storage.dtype))
            .collect::<Vec<(Vec<usize>, DTypeDescriptor)>>();
        plan.verify_inputs(&input_meta)?;
        let mut slots = (0..plan.memory_plan.peak_live_slots())
            .map(|_| None::<CpuStorage>)
            .collect::<Vec<_>>();
        for (value_id, storage) in plan.graph.inputs.iter().copied().zip(self.inputs) {
            let slot = plan
                .memory_plan
                .assignments()
                .get(&value_id)
                .ok_or_else(|| {
                    Error::Msg(format!("missing allocation for input value {value_id}"))
                })?;
            slots[slot.index()] = Some(storage);
        }

        let backend = CpuBackendImpl::<Cpu>::default();
        let context = ExecutionContext::new(backend);
        for node in &plan.graph.nodes {
            let OperationIdentity::Builtin(operation) = node.operation else {
                return Err(Error::Msg(
                    "compiled CPU execution rejects custom operations".into(),
                ));
            };
            let payload = node.descriptor_payload.as_ref().ok_or_else(|| {
                Error::Msg(format!("node {operation} has no captured descriptor"))
            })?;
            let inputs = node
                .inputs
                .iter()
                .map(|value_id| {
                    let slot = plan
                        .memory_plan
                        .assignments()
                        .get(value_id)
                        .ok_or_else(|| {
                            Error::Msg(format!("missing allocation for input value {value_id}"))
                        })?;
                    slots[slot.index()].as_ref().ok_or_else(|| {
                        Error::Msg(format!(
                            "input value {value_id} is not available at node {}",
                            node.id
                        ))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let output = match operation {
                incin_core::prelude::OperationKind::Add => {
                    execute::<op::Add>(&context, payload, &inputs)?
                }
                incin_core::prelude::OperationKind::Relu => {
                    execute::<op::Relu>(&context, payload, &inputs)?
                }
                incin_core::prelude::OperationKind::MatMulExact => {
                    execute::<op::MatMulExact>(&context, payload, &inputs)?
                }
                incin_core::prelude::OperationKind::Linear => {
                    execute::<op::Linear>(&context, payload, &inputs)?
                }
                incin_core::prelude::OperationKind::Addmm => {
                    execute::<op::Addmm>(&context, payload, &inputs)?
                }
                _ => {
                    return Err(Error::Msg(format!(
                        "compiled CPU lowering does not support {operation}"
                    )));
                }
            };
            let output_id = *node
                .outputs
                .first()
                .ok_or_else(|| Error::Msg(format!("node {operation} has no output")))?;
            let slot = plan
                .memory_plan
                .assignments()
                .get(&output_id)
                .ok_or_else(|| {
                    Error::Msg(format!("missing allocation for output value {output_id}"))
                })?;
            slots[slot.index()] = Some(output);
        }

        plan.graph
            .outputs
            .iter()
            .map(|value_id| {
                let slot = plan
                    .memory_plan
                    .assignments()
                    .get(value_id)
                    .ok_or_else(|| {
                        Error::Msg(format!("missing allocation for output value {value_id}"))
                    })?;
                slots[slot.index()]
                    .clone()
                    .ok_or_else(|| Error::Msg(format!("output value {value_id} was not produced")))
            })
            .collect()
    }
}

fn execute<O>(
    context: &ExecutionContext<CpuBackendImpl<Cpu>>,
    payload: &incin_core::graph::DescriptorPayload,
    inputs: &[&CpuStorage],
) -> Result<CpuStorage>
where
    O: CanonicalOperation,
    O::Attributes: incin_core::exec::catalog::AttributeContract,
    CpuBackendImpl<Cpu>: incin_core::backend_authoring::Execute<O, Output = CpuStorage>,
{
    let captured = CapturedDescriptor::from_payload(O::ID, payload.schema, payload.payload.clone());
    let descriptor: Descriptor<O> = captured
        .decode()
        .map_err(|error| Error::Msg(format!("invalid captured descriptor: {error}")))?;
    let logical_inputs = inputs
        .iter()
        .map(|storage| incin_core::exec::catalog::LogicalTensorMeta {
            shape: Some(storage.shape.clone().into()),
            dtype: Some(storage.dtype),
            device: Some(storage.device),
        })
        .collect();
    let validated = Descriptor::<O>::infer_runtime(descriptor.attributes().clone(), logical_inputs)
        .map_err(|error| Error::Msg(format!("runtime descriptor validation failed: {error}")))?;
    let handles = inputs
        .iter()
        .map(|storage| TensorHandle::from_storage::<CpuBackendImpl<Cpu>, Dyn, _>(*storage))
        .collect::<Vec<_>>();
    context
        .backend()
        .execute_shaped::<Dyn>(ExecutionRequest {
            operation: &validated,
            inputs: &handles,
            context,
            payload: None,
        })
        .map_err(|error| Error::Msg(format!("compiled CPU operation failed: {error}")))
}
