# Production panic/unwrap/expect classification

The raw workspace inventory is archived in
`test-results/panic-unwrap-expect-inventory.txt`; the final inventory is
archived separately after implementation. Occurrences were reviewed by
boundary, not mechanically rewritten.

## Converted recoverable paths

- Tensor binary and scalar operator overloads now return the same typed
  `Result` as their named methods.
- CUDA and WGPU unary backward recipes propagate kernel, broadcast, device,
  upload, download, and buffer-map errors through the tape.
- CUDA context creation and WGPU initial adapter/device creation fail through
  `BackendError::Execution`.
- WGPU empty scalar and extremum readback no longer fabricates zero/index 0.
- CPU, WGPU, and Candle integer readback rejects lossy float conversion.
- CPU integer fill/ranges, embedding indices, and cross-entropy targets reject
  non-finite, fractional, negative, or out-of-range values as applicable.
- Adam/AdamW state load and every optimizer step prepare before commit and
  preserve state on failure.
- Safetensors path metadata produces macro diagnostics for empty/invalid Rust
  identifiers instead of panicking.
- DataLoader rejects zero batch size during construction and mutex poison no
  longer aborts a worker. MNIST path derivation no longer unwraps filesystem
  components.
- Shape-index inference no longer unwraps an optional dynamic dimension.

## Statically proven internal transitions

This class includes backend-created contiguous metadata rebuilt from the exact
allocation just checked, unique ownership immediately after a fresh device
allocation, fixed-width chunks after an exact remainder check, and shape-only
reshapes whose element-count equality was already proven. Their messages name
the proof. They are internal invariant assertions, not untrusted-input error
handling.

WGPU's internal `get_device_state` assertion is reachable only after a
fallible initial buffer constructor has installed the state. Dispatch parameter
buffers and readback necessarily have an existing storage buffer and therefore
cross that proof first.

## Test/debug assertions

Unit/integration tests, compile fixtures, examples demonstrating deliberate
panic capture, and Graphify/test visualization panels use `unwrap`, `expect`,
or deliberate `panic!` as assertions. These are not product recovery paths.

## Process boundaries

CLI/LSP executable plumbing contains assertions for child pipes configured by
the same process and test harness frames. A failure terminates that command;
it does not unwind through the tensor library API.

## Compatibility exceptions requiring later descriptor migration

- Ordinary CPU arithmetic and storage materialization reject unsupported Q8_0
  output construction through `Error::UnsupportedDType`; direct Q8_0 reads
  dequantize from their block scale without forging a float-storage identity.
  FND-004 still owns the exact quantized operation inventory and semantics.
- The legacy `StateDict::state_dict` trait has an infallible export signature.
  Optimizer state created by validated candidate commits is representable, but
  changing the workspace-wide serialization trait is deferred to later
  module conformance work.
- Panics deliberately exposed by visualization test panels remain explicit
  test/debug behavior.

None of these exceptions is treated as a successful unsupported operation.
