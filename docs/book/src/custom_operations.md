# Custom and fused operations

There are three different forms of fusion:

1. A new semantic operation, such as `BiasGelu`, is an `Operation` with
   attributes and output inference, plus a backend `Execute<YourOperation>`.
2. A faster implementation of an existing operation stays behind that
   operation's existing `Execute` implementation. It does not add a second
   public operation hierarchy.
3. Combining several graph nodes belongs to compiler lowering and is outside
   this API task.

The executable authoring contract is exercised in
`crates/incin-core/tests/custom_operation.rs` and the downstream backend
fixture under `crates/incin/tests/consumer-fixtures`. Those fixtures cover
operation identity, serializable attributes, output inference, validation,
capability admission, and execution payloads.

The smallest useful implementation has four pieces:

1. Define an `Operation` type and its serializable attributes.
2. Add output inference and validation to the canonical operation catalog.
3. Implement `Execute<YourOperation>` for each backend that supports it.
4. Register capability admission and test the operation through the public
   consumer fixture.

The compact `CompanyIdentity` operation in
`crates/incin-core/tests/custom_operation.rs` is the reference implementation:
it defines serializable attributes, implements the canonical operation
contract, validates the input metadata, and executes through the backend
dispatch path. The downstream fixture invokes that operation through the
public authoring API, which is the important compatibility check for external
backend crates.

Here is the compact shape of the author-facing operation declaration. The
backend implementation uses the same `Execute<Identity>` request path as the
built-in catalog and returns its backend storage type.

```rust,ignore
use incin_core::backend_authoring::{
    Descriptor, DescriptorError, Execute, ExecutionRequest, LogicalTensorMeta, Operation,
    OperationKey,
};
use incin_core::err::BackendError;
use incin_core::exec::catalog::NoAttributes;
use std::borrow::Cow;

#[derive(Clone, Debug)]
struct Identity;

impl Operation for Identity {
    type Attributes = NoAttributes;

    const KEY: OperationKey = OperationKey {
        namespace: Cow::Borrowed("example.org"),
        name: Cow::Borrowed("identity"),
        version: 1,
    };

    fn infer_outputs(
        _: &Self::Attributes,
        inputs: &[LogicalTensorMeta],
    ) -> Result<Vec<LogicalTensorMeta>, DescriptorError> {
        Ok(inputs.first().cloned().into_iter().collect())
    }
}

// The backend supplies the storage type and executes the validated request.
impl Execute<Identity> for MyBackend {
    type Output = MyStorage;

    fn execute(
        &self,
        request: ExecutionRequest<'_, Identity, Self>,
    ) -> Result<Self::Output, BackendError> {
        let _input = request.inputs.first().ok_or(BackendError::InvalidInput {
            operation: incin_core::shapes::error::OperationKind::Storage,
            reason: "identity needs one input",
        })?;
        // Decode the checked handle and launch the backend copy kernel here.
        todo!("backend-specific storage copy")
    }
}

// `infer_invocation` creates a validated `Descriptor<Identity>` before the
// caller invokes the same canonical dispatch path as built-in operations.
let invocation = Identity::infer_invocation(NoAttributes, logical_inputs)?;
let descriptor: &Descriptor<Identity> = invocation.descriptor();
let output = incin_core::exec::dispatch::execute_shaped::<
    Identity,
    MyBackend,
    incin_core::shapes::Dyn,
>(
    &context,
    NoAttributes,
    &[input_handle],
    &output_shape,
)?;
```

The real fixture fills in metadata validation, capability admission, and
backend execution. Keep the operation key stable once published, serialize
all attributes, and route execution through the validated descriptor request.

For a real fused operation such as `BiasGelu`, the same pattern applies. The
operation accepts activation and bias handles, validates their broadcast
relationship, infers the output metadata, and dispatches one backend kernel.
It should not introduce a parallel executor or construct a `TensorMeta` from
unchecked fields. If the operation is built from existing tensor methods
instead, document it as a composition rather than as a new fused catalog
entry.

Custom autodiff registration is not part of the current extension contract.
Unless a custom operation is composed from existing differentiable tensor
operations, document it as forward-only.
