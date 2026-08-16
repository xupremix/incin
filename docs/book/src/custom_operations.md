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

For example, a fused `BiasGelu` operation would accept the activation and bias
handles, validate their broadcast relationship, infer the output metadata, and
dispatch one backend kernel. It should not introduce a parallel executor or
construct a `TensorMeta` from unchecked fields. If the operation is built from
existing tensor methods instead, document it as a composition rather than as a
new fused catalog entry.

Custom autodiff registration is not part of the current extension contract.
Unless a custom operation is composed from existing differentiable tensor
operations, document it as forward-only.
