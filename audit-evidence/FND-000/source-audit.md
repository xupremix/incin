# FND-000 Source Audit

Snapshot inspected: `fa8d2030141b04bc7c0dfccb382bfa60647223cf` on
branch `develop`.

The following findings were reproduced from the current checkout before
production changes:

| Finding | Reproduction | Result |
|---|---|---|
| Public wildcard re-exports at the facade/core boundary | `rg -n 'pub use .*::\*' crates/incin/src/lib.rs crates/incin-core/src/lib.rs crates/incin-core/src/tensor/mod.rs crates/incin-backends/src/lib.rs` | 13 matching declarations. |
| Required methods in the legacy operation-family trait block | `sed -n '235,870p' crates/incin-core/src/tensor/backend.rs \| rg '^    fn ' \| wc -l` | 142 methods. |
| Concrete typed operation descriptors | Source inspection of `crates/incin-core/src/exec/spec.rs` and descriptor modules | Six: `BroadcastSpec`, `MatMulSpec`, `ReductionSpec`, `Conv2dSpec`, `Pool2dSpec`, and `ReshapeSpec`. |
| Descriptor-to-legacy CPU adapters | Inspection of `crates/incin-backends/src/cpu/executor.rs` | Descriptor executors call legacy tensor/module/reduction methods. |
| Capability declarations | Inspection of `crates/incin-backends/src/capability.rs` | Static backend capability tables are maintained separately from `Execute<O>` implementations. |
| Forgeable invariant tuple fields | Workspace search for public tuple structs | Public fields include checked sizes, buffer slots, IDs, `Dyn`, and runtime device selectors. |
| Compiled false success | Inspection of `crates/incin-core/src/compiled/fold.rs` | Folding and prepacking returned cloned graphs as successful results. |
| ONNX false success/fabrication | Inspection of `crates/incin-macros/src/onnx.rs` | Unknown ranks defaulted to rank four, control flow could generate panics, initializers became zero parameters, and a no-op loader returned success. |
| Workspace validation claim | Archived `cargo test --workspace` baseline | Exit 101 in two current-toolchain `trybuild` snapshot mismatches; the historical aggregate completion claim was not reproduced. |

## Graph snapshot

Graphify 0.9.30 was available, but `graphify-out/GRAPH_REPORT.md` identifies
commit `11518760` while this audit began at `fa8d2030`. The graph was therefore
treated as stale and was not used to claim current source relationships. The
exact verification output is archived in
`test-results/graphify-verification.txt`.

## Missing supporting inputs

`docs/FOUNDATION_REVIEW_AND_EXECUTION_PLAN.md`, `docs/PROJECT_STATUS.md`, and
`audit-evidence/FND-000/source-audit.md` were absent both from the worktree and
the inspected Git history. The latter two have been created from current source
and command evidence. No substitute review-plan content was invented.
