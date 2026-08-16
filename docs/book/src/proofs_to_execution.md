# From proofs to execution

The frontend follows one route:

```text
Tensor<S, B, K, G>
  -> operation and descriptor
  -> validation and capability admission
  -> Execute<O> with an ExecutionRequest
  -> backend storage
```

The concrete Rust `Shape` type is erased before the executor runs. The
executor receives checked `TensorHandle` values whose `TensorMeta` contains
runtime shape, strides, offset, dtype, device, layout, and alignment. It also
receives a validated descriptor with operation attributes and inferred output
metadata.

Typed entry points preserve useful proof provenance in `ShapeEvidence`:
proof level, static rank, and static element count. Dynamic entry points carry
dynamic evidence. This gives a backend enough information for safe kernel
selection without making `Execute<O>` generic over the caller's shape type.

The backend capability query additionally carries operation, dtype, layout,
rank, training mode, and math mode. Shape evidence is not a substitute for
runtime metadata: kernels must continue to use the checked handle and
validated descriptor for actual execution.
