# FND-001 public facade migration

Reviewed against `cargo public-api -p incin` output archived before and after the change. Paths not listed as moved are either retained explicitly or removed because they were implementation details accidentally reachable through a wildcard.

| Previous path | Current path | Disposition |
|---|---|---|
| `incin::model!`, `incin::import_model!` | `incin::experimental::model!`, `incin::experimental::import_model!` | Moved; fail-closed ONNX expansion remains unstable. |
| `incin::mesh!`, `incin::prelude::mesh!`, `incin::macros::mesh!` | `incin::experimental::mesh!` | Moved; distributed declarations are experimental. |
| `incin::prelude::{parallel!, placement!}` | `incin::experimental::{parallel!, placement!}` | Moved. |
| `incin::compile::*` | `incin::experimental::compiled::*` with `compiled` | Moved and feature-gated. |
| `incin::tuning::*` | `incin::experimental::tuning::*` with `autotune` | Moved and feature-gated. |
| `incin::dist::*` | `incin::experimental::distributed::*` with `distributed` | Moved and explicitly allow-listed. |
| `incin::dist::mesh::*` | `incin::experimental::distributed::mesh::*` | Moved; exported distributed macros now expand through this path. |
| `incin::train::*` | `incin::experimental::training::*` with `train` | Moved. |
| `incin::plan_report::*` | `incin::experimental::training::plan_report::*` with `train` | Moved. |
| `incin::tune_report::*` | `incin::experimental::tuning_report::*` | Moved. |
| backend operation and execution contracts reachable through broad imports | `incin::backend_authoring::{...}` with `backend-authoring` | Moved to an explicit authoring allow-list. |
| `incin::test_utils::*` wildcard | `incin::test_utils::DummyBackend` with `test-utils` | Narrowed to the intended test backend. |
| `incin::prelude::{Graph, OpType}` and graph/compiler/tracing/raw-storage internals | none | Removed from the end-user facade. Import from owning internal crates only when developing Incin itself. |
| `incin::prelude::{SupportsDType, TransferTo}` | `incin::backend_authoring::{SupportsDType, TransferTo}` | Moved; normal tensor users do not name these backend extension traits. |
| `incin::prelude` autoref fallback traits | none | Removed from ordinary imports. A doc-hidden macro ABI supports `#[module]` without exposing them in the prelude. |
| `incin::prelude::{BTreeMap, String, Vec, format!}` | standard/alloc library paths | Removed; not Incin contracts. |
| wildcard contents of `incin::{nn,metrics,data,transforms,hub}` | same module paths for reviewed names | Retained through explicit allow-lists; unreviewed owning-crate internals are no longer forwarded. |

Stable additions at the root and prelude are the deliberate dtype/device/gradient marker contracts: `BoolDType`, `FloatDType`, `IntDType`, `PlainDType`, `QuantDType`, `Q8_0`, `TensorElement`, `Device`, `DeviceKind`, `DevicePreference`, `DeviceSet`, `DeviceSetError`, `RequiresGrad`, `bf16`, and `f16` as applicable.

`incin::__macro_support` and `incin_core::__macro_support` are `#[doc(hidden)]` implementation namespaces required by procedural macro hygiene. They are not ordinary-import tiers or stable user contracts.
