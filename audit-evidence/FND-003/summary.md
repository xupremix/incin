# FND-003 — DONE

The FND-003 acceptance gate passes at the recorded pre-commit checkout
`418497534cb7cde3d7542e11f09237a32bb6c83a` with the changes in this task's
diff.

## Frozen contract

- Recoverable public failures use a coherent typed vocabulary for shape,
  dtype/conversion, placement, unsupported capability, overflow, backend,
  autograd, module/state, malformed artifacts, I/O/resources, and internal
  invariants. External diagnostic text is bounded to 512 UTF-8 bytes.
- Tensor binary and scalar operators return `Result` instead of panicking.
- Float-to-integer conversion is exact by default. NaN, infinity, fractional
  values, and out-of-range values are rejected unless an explicit truncation
  or saturation policy defines the behavior.
- SGD, Adam, and AdamW validate and prepare a complete update set before
  mutation. A failed commit restores every parameter byte-for-byte; Adam and
  AdamW publish state tensors and advance their counters only after the full
  commit succeeds.
- Backend variable assignment is failure-atomic for one variable. The
  feature-gated CPU failure injector proves multi-parameter rollback.
- Recoverable CUDA/WGPU/Candle/Metal initialization, execution, readback,
  autograd, macro metadata, loader construction, and model/data paths
  propagate typed errors rather than aborting or fabricating values.

## Acceptance evidence

The resolved default/no-default/std/CPU and independent CUDA/WGPU/Metal/Candle
feature checks pass. Focused failure tests, isolated facade contracts, exact
workspace Clippy, the full workspace suite, explicit doctests, rustdoc warnings,
task-local formatting, diff hygiene, and `cargo public-api` pass. Initial
failing runs are retained and followed by passing reruns.

The repository-wide formatter baseline remains non-clean outside this task's
Rust diff. `cargo semver-checks` is blocked because its forced all-feature
scratch build resolves a newer Candle dtype enum and cannot build rustdoc; the
exact failure is archived and is not presented as a semver result.

No CUDA or Metal hardware-execution claim is made. FND-003 does not freeze
operation semantics or replace the legacy operation-family architecture; those
remain FND-004 and FND-005 respectively.
