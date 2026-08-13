//! Executable CPU lowering for the compiled graph plan.

use alloc::format;
use alloc::vec::Vec;

use incin_core::backend_authoring::{Execute, ExecutionRequest};
use incin_core::compiled::CompiledPlan;
use incin_core::exec::catalog::{CanonicalOperation, CapturedDescriptor, Descriptor, op};
use incin_core::exec::{ExecutionContext, OperationIdentity, TensorHandle};
use incin_core::prelude::{Cpu, DTypeDescriptor, Dyn, Error, Result};

use super::{CpuBackendImpl, CpuStorage};

macro_rules! cpu_compiled_operations {
    ($callback:ident) => {
        $callback!(
            Add => op::Add,
            Sub => op::Sub,
            Mul => op::Mul,
            Div => op::Div,
            Relu => op::Relu,
            ReshapeExact => op::ReshapeExact,
            BroadcastAs => op::BroadcastAs,
            MatMulExact => op::MatMulExact,
            TransposeExact => op::TransposeExact,
            Narrow => op::Narrow,
            FlattenExact => op::FlattenExact,
            SliceExact => op::SliceExact,
            ConcatExact => op::ConcatExact,
            StackExact => op::StackExact,
            SumDim => op::SumDim,
            SumKeepDim => op::SumKeepDim,
            Linear => op::Linear,
            Addmm => op::Addmm,
        );
    };
}

macro_rules! supports_cpu_operation {
    ($($kind:ident => $descriptor:ty,)*) => {
        fn supports_cpu_operation(operation: incin_core::prelude::OperationKind) -> bool {
            matches!(operation, $(incin_core::prelude::OperationKind::$kind)|*)
        }
    };
}

cpu_compiled_operations!(supports_cpu_operation);

/// Canonical metadata for one operation admitted by the compiled CPU path.
///
/// The operation identity and semantic fields come from the core catalog.
/// This type only reports the backend lowering subset and does not define a
/// second operation vocabulary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuCompiledSupport {
    pub operation: incin_core::prelude::OperationKind,
    pub name: &'static str,
    pub descriptor: &'static str,
    pub execution_site: incin_core::exec::ExecutionSite,
    pub capture_eligible: bool,
}

macro_rules! compiled_support_report {
    ($($kind:ident => $descriptor:ty,)*) => {
        fn compiled_support_report_impl() -> core::result::Result<Vec<CpuCompiledSupport>, &'static str> {
            let mut report = Vec::new();
            $(push_compiled_support(
                &mut report,
                incin_core::prelude::OperationKind::$kind,
            )?;)*
            Ok(report)
        }
    };
}

fn push_compiled_support(
    report: &mut Vec<CpuCompiledSupport>,
    operation: incin_core::prelude::OperationKind,
) -> core::result::Result<(), &'static str> {
    let entry = incin_core::exec::catalog::catalog_entry(operation)
        .ok_or("compiled CPU operation is missing from the canonical catalog")?;
    if !entry.site.is_backend_executable() {
        return Err("compiled CPU operation is not backend executable");
    }
    report.push(CpuCompiledSupport {
        operation,
        name: entry.name,
        descriptor: entry.descriptor,
        execution_site: entry.site,
        capture_eligible: entry.capture_eligible,
    });
    Ok(())
}

cpu_compiled_operations!(compiled_support_report);

/// Returns the canonical metadata for operations admitted by compiled CPU.
#[must_use]
pub fn compiled_support() -> core::result::Result<Vec<CpuCompiledSupport>, &'static str> {
    compiled_support_report_impl()
}

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

/// An admitted executable CPU function backed by one immutable compiled plan.
///
/// Admission happens once at construction. Each invocation only validates its
/// runtime inputs against the captured guards and executes the admitted nodes.
#[derive(Debug, Clone)]
pub struct CpuCompiledFunction {
    plan: CpuCompiledPlan,
}

impl CpuCompiledPlan {
    /// Lowers a generic compiled plan into an executable CPU plan.
    ///
    /// This is the executable compilation boundary. It rejects operations
    /// without a canonical CPU lowering and descriptors that cannot be
    /// decoded and revalidated before any invocation is attempted.
    pub fn compile(plan: &CompiledPlan) -> Result<Self> {
        for node in &plan.graph.nodes {
            if !matches!(node.operation, OperationIdentity::Builtin(operation) if supports_cpu_operation(operation))
            {
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
            let payload = node.descriptor_payload.as_ref().ok_or_else(|| {
                Error::Msg(format!(
                    "compiled CPU node {} has no captured descriptor",
                    node.id
                ))
            })?;
            validate_cpu_descriptor_dispatch(
                match &node.operation {
                    OperationIdentity::Builtin(operation) => *operation,
                    OperationIdentity::Custom(key) => {
                        return Err(Error::Msg(format!(
                            "compiled CPU lowering does not support custom operation {key:?}"
                        )));
                    }
                },
                payload,
            )?;
        }
        Ok(Self { plan: plan.clone() })
    }

    /// Creates a reusable executable function from a generic compiled plan.
    pub fn function(plan: &CompiledPlan) -> Result<CpuCompiledFunction> {
        Ok(CpuCompiledFunction {
            plan: Self::compile(plan)?,
        })
    }

    pub fn run(&self, invocation: CpuCompiledInvocation) -> Result<Vec<CpuStorage>> {
        invocation.run_admitted(&self.plan)
    }
}

impl CpuCompiledFunction {
    /// Compiles and admits a generic plan once for repeated invocation.
    pub fn compile(plan: &CompiledPlan) -> Result<Self> {
        CpuCompiledPlan::function(plan)
    }

    /// Invokes the admitted function with ordered graph inputs.
    pub fn run(&self, invocation: CpuCompiledInvocation) -> Result<Vec<CpuStorage>> {
        self.plan.run(invocation)
    }

    /// Returns the number of caller-provided graph inputs.
    #[must_use]
    pub fn input_count(&self) -> usize {
        self.plan.plan.graph.inputs.len()
    }

    /// Returns the number of graph outputs produced by the function.
    #[must_use]
    pub fn output_count(&self) -> usize {
        self.plan.plan.graph.outputs.len()
    }
}

macro_rules! validate_cpu_descriptor_dispatch {
    ($($kind:ident => $descriptor:ty,)*) => {
        fn validate_cpu_descriptor_dispatch(
            operation: incin_core::prelude::OperationKind,
            payload: &incin_core::graph::DescriptorPayload,
        ) -> Result<()> {
            match operation {
                $(incin_core::prelude::OperationKind::$kind => {
                    validate_descriptor::<$descriptor>(payload)
                })*
                _ => Err(Error::Msg(format!(
                    "compiled CPU lowering does not support {operation}"
                ))),
            }
        }
    };
}

cpu_compiled_operations!(validate_cpu_descriptor_dispatch);

fn validate_descriptor<O>(payload: &incin_core::graph::DescriptorPayload) -> Result<()>
where
    O: CanonicalOperation,
    O::Attributes: incin_core::exec::catalog::AttributeContract,
{
    let captured = CapturedDescriptor::from_payload(O::ID, payload.schema, payload.payload.clone());
    let descriptor: Descriptor<O> = captured
        .decode()
        .map_err(|error| Error::Msg(format!("invalid captured descriptor: {error}")))?;
    descriptor
        .revalidate()
        .map_err(|error| Error::Msg(format!("captured descriptor validation failed: {error}")))?;
    Ok(())
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
        let admitted = CpuCompiledPlan::compile(plan)?;
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

        let backend = CpuBackendImpl::<Cpu>::default();
        for (value_id, bytes) in &plan.graph.initializers {
            let value = plan.graph.value_metadata.get(value_id).ok_or_else(|| {
                Error::Msg(format!(
                    "initializer value {value_id} has no captured metadata"
                ))
            })?;
            let storage = <CpuBackendImpl<Cpu> as incin_core::prelude::Backend>::from_bytes::<Dyn>(
                bytes,
                &value.shape,
                value.dtype,
                &incin_core::prelude::DeviceId::cpu(),
            )?;
            let slot = plan
                .memory_plan
                .assignments()
                .get(value_id)
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "missing allocation for initializer value {value_id}"
                    ))
                })?;
            slots[slot.index()] = Some(storage);
        }

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
            let output = dispatch_cpu_operation(operation, &context, payload, &inputs)?;
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

macro_rules! dispatch_cpu_operation {
    ($($kind:ident => $descriptor:ty,)*) => {
        fn dispatch_cpu_operation(
            operation: incin_core::prelude::OperationKind,
            context: &ExecutionContext<CpuBackendImpl<Cpu>>,
            payload: &incin_core::graph::DescriptorPayload,
            inputs: &[&CpuStorage],
        ) -> Result<CpuStorage> {
            match operation {
                $(incin_core::prelude::OperationKind::$kind => {
                    execute::<$descriptor>(context, payload, inputs)
                })*
                _ => Err(Error::Msg(format!(
                    "compiled CPU lowering does not support {operation}"
                ))),
            }
        }
    };
}

cpu_compiled_operations!(dispatch_cpu_operation);

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
