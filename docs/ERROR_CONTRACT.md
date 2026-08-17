# Error and Failure Contract

This document freezes the FND-003 public failure vocabulary. Public and
backend operations return `incin::Result<T>` for recoverable failures. Error
values carry an operation identity and bounded metadata; tensor contents are
never embedded in diagnostics.

## Public categories

| Failure category | Typed representation |
|---|---|
| Shape, rank, axis, broadcast | `Error::Shape(ShapeError)` |
| Dtype and conversion | `Error::DTypeMismatch`, `Error::InvalidConversion` |
| Device and placement | `Error::PlacementMismatch`, `Error::DeviceMismatch`, storage/device variants |
| Unsupported operation/capability | `BackendError::Unsupported`, `Error::UnsupportedBackendOperation`, `Error::UnsupportedDType` |
| Support policy refusal | `CanonicalError::Policy(PolicyViolation)`, preserved as `Error::Policy` |
| Arithmetic/allocation overflow | `Error::ArithmeticOverflow`, `Error::AllocationOverflow`, `ShapeError::ArithmeticOverflow` |
| Backend execution | `Error::Backend(BackendError::Execution)` |
| Autograd/non-finite policy | `Error::Backward(BackwardError)` |
| Module/state dictionary | `Error::InvalidModuleState` |
| Malformed model/data/artifact | `Error::MalformedArtifact` |
| I/O and resource bounds | `Error::Io`, `Error::ResourceLimit` |
| Internal invariant | `Error::InternalInvariant` |

`ErrorMessage` truncates external diagnostic text at a UTF-8 boundary to
`MAX_ERROR_MESSAGE_BYTES`. New error paths use it for parser, driver, I/O, and
backend messages. Legacy free-form variants remain temporarily for source
compatibility, but are not the contract for new foundation code.

## Numeric conversion

Implicit float-to-integer conversion is not permitted at public scalar and
index boundaries. `FloatToIntPolicy::Exact` rejects NaN, infinity, fractional
values, and values outside the destination range. `Truncate` and `Saturate`
must be named explicitly. CPU and WGPU/Candle integer readback, CPU integer
fill/range creation, embedding indices, and cross-entropy targets use exact
conversion.

## Mutation and rollback

`Backend::assign_var` is failure-atomic for one variable: `Err` means its prior
bytes remain intact. SGD, Adam, and AdamW validate and prepare every candidate
before mutation. Commit failure restores all parameter snapshots. Adam and
AdamW publish moment maps and advance the step counter only after every
parameter commit succeeds. State-dictionary loads validate complete paired
moments in temporary maps before replacing live state.

## Panics and process boundaries

Recoverable tensor operators return `Result`, including Rust `+`, `-`, `*`,
and `/` overloads. Backend launch, device initialization, buffer readback,
autograd recipe, macro metadata, data-loader construction, and model/data I/O
fail through typed results. Remaining `panic!`, `unwrap`, and `expect` sites
must be one of:

- a test/debug assertion;
- a process boundary where termination is the declared behavior;
- a statically proven internal transition immediately following validation.

The reviewed workspace classification and remaining compatibility exceptions
are archived in `audit-evidence/FND-003/panic-classification.md`.
