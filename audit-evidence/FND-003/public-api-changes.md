# FND-003 public API changes

## Added at `incin::*` and `incin::prelude::*`

- `BackendError`, `BackwardError`, `NonFiniteSite`
- `ConversionFailure`, `FloatToIntPolicy`
- `ErrorMessage`
- `convert_f64_to_i64`

## Changed

- Tensor `Add`, `Sub`, `Mul`, and `Div` operator outputs now return
  `incin::Result<Tensor<...>>` instead of panicking on backend/runtime failure.
- Tensor scalar `Add` and `Mul` operator outputs now return `Result`.
- `DataLoader::new` now returns `incin_core::Result<DataLoader<...>>` and
  rejects a zero batch size.
- `ScalarValue::to_i64` now requires a `FloatToIntPolicy` and returns `Result`.
- `Backend::assign_var` now documents failure-atomic behavior as an authoring
  requirement.

## Feature-gated test surface

- `incin::test_utils::fail_assign_on` and `AssignFailureGuard` are available
  only under `test-utils` for deterministic rollback tests.

No compatibility shim silently restores the former panicking or truncating
behavior.
