# Public API tiers for 0.1.0

Every module in a shipped crate is assigned one of four tiers. The tier states
what a 0.1.0 consumer may rely on and what the project may still change without
a major version bump. Module counts are declared items in the reviewed
baselines under this directory, so the table is checkable against
`tools/check-public-api-baseline.py` rather than being a prose claim.

## Tier definitions

| Tier | Meaning | Compatibility intent |
| --- | --- | --- |
| **S — stable user API** | What ordinary users write against. | Breaking changes need a major bump. |
| **X — expert / backend authoring** | Needed to implement a backend, a codegen target, or a visualization plugin. Documented, but narrower and more likely to move. | Breaking changes are announced in the changelog; a minor bump may carry them before 1.0. |
| **M — intentional internal / macro ABI** | Reachable only because macro expansion or a trait implementation requires it. Marked `#[doc(hidden)]`. | No compatibility promise. See [hidden-items.md](hidden-items.md). |
| **P — preview** | Feature-gated, excluded from the baselines by feature selection. | No compatibility promise. See [../PROJECT_STATUS.md](../PROJECT_STATUS.md). |

Tier **I — implementation detail** is the absence of a row: it is not `pub`, so
it does not appear in a baseline. The governing rule is
[API_DESIGN.md](../API_DESIGN.md): `pub(crate)` is the default, and a module
earns `pub` only by being named in the table below.

## `incin` — the facade

| Module | Items | Tier | Notes |
| --- | ---: | :---: | --- |
| `prelude` | 154 | S | The ordinary user tier. Guarded by `tools/check-public-api.sh`, which rejects wildcard exports and a denylist of implementation names. |
| `nn` | 80 | S | Layers and module traits. |
| `doctor` | 41 | S | User-facing diagnostics. |
| `optim` | 15 | S | Optimizers. |
| `state` | 12 | S | Checkpoint and state-dict entry points. |
| `data` | 12 | S | Dataset and loader surface. |
| `metrics` | 8 | S | |
| `types` | 8 | S | |
| `transforms` | 7 | S | |
| `advanced` | 26 | X | Escape hatch for users who need below-prelude control. |
| `macros` | 7 | M | Macro expansion support. |
| `experimental` | 7 | P | Preview namespaces. |

## `incin-core`

| Module | Items | Tier | Notes |
| --- | ---: | :---: | --- |
| `exec` | 2530 | X | Descriptors, capabilities, `Execute<O>`. The canonical execution contract, and the single largest surface in the workspace. Expert tier because implementing a backend requires it. |
| `nn` | 1162 | S | |
| `shapes` | 1014 | S | Compile-time shape system. Large because shape proofs are type-level; most entries are generated trait impls. |
| `prelude` | 710 | S | |
| `backend_authoring` | 538 | X | Named for its tier. The gate asserts `backend_authoring::legacy` never reappears. |
| `tensor` | 429 | S | `tensor::backend` is `pub(crate)`, enforced by the gate. |
| `optim` | 77 | S | |
| `metrics`, `io`, `distributions`, `loss`, `error`, `graph`, `types`, `serialization` | 6–34 each | S | |
| `advanced` | 45 | X | |
| `dist` | 22 | X | Distributed placement. |
| `resource` | 9 | X | |
| `onnx` | 7 | S | |
| `autograd` | 7 | X | Tape contract used by backend implementations. |
| `experimental` | 1 | P | |
| `paranoid_audit` | 1 | M | |

### Reading `err` versus `error` in the baselines

`incin_core::err` is `pub(crate)`; the supported path is
`incin_core::error`, a re-export facade over it. `cargo-public-api` renders a
type by its defining module, so roughly 300 signatures in
`incin-backends-cpu.txt` print `incin_core::err::Result` even though no public
module of that name exists. `err` is therefore tier I, not a second public
error path, and nothing in a baseline that spells `incin_core::err::` is a
supported import.

## `incin-backends`

| Module | Items | Tier | Notes |
| --- | ---: | :---: | --- |
| `cpu` | 284 | X | Reference backend. Its `execute` methods take `ExecutionRequest<'_, Op, Self>`, so the descriptor path is the only entry. |
| `target` | 72 | X | Target-first allocation and construction. |
| `dispatch` | 72 | X | Runtime backend selection for `Dyn`. |
| `external` | 67 | X | Conformance harness for third-party backends. |
| `prelude` | 50 | X | Documented in `lib.rs` as the single import surface for backend authors. |
| `codegen` | 25 | X | `render_wgsl` / `render_msl` / `render_cuda` on `PointwiseOpSpec`. Deliberately frozen: this is what a new accelerator backend needs. |
| `nn_target` | 6 | X | |
| `detect` | 6 | X | Runtime device detection. |
| `simd` | 5 | X | Compile-time lane-width resolution (`simd_lanes` is a `const fn`) plus AVX2 detection. Kept public deliberately: it has a module doc comment and a curated crate-root re-export, and it resolves const-generic parameters rather than executing operations. |
| `capability`, `capability_docs`, `backend_kind` | 1–5 each | X | Capability registry and its generated documentation. |
| `iteration` | 0 | I | **Changed for 0.1.0.** Was `pub`; now `pub(crate)`. See below. |
| `tuning` | 0 | P | Feature-gated behind `autotune`; absent from the baseline feature set. |

## `incin-viz`

| Module | Items | Tier | Notes |
| --- | ---: | :---: | --- |
| `panels` | 57 | X | Plugin-facing panel surface. |
| `app` | 14 | X | |
| `transport_reader` | 6 | X | |
| `dispatch` | 6 | X | |
| `err` | 4 | X | |

`incin-viz-plugin-api` is expert tier in its entirety; it exists to be
implemented against.

## Changes made for the 0.1.0 freeze

### `incin_backends::iteration` demoted to implementation detail

`iteration` exposed exactly one item, `tile_2d`, a 2D loop-tiling helper that
takes runtime `rows`/`cols` and sits below the descriptor contract. It matched
API_DESIGN.md's "Internal Modules" rule directly, it had no module
documentation and no crate-root re-export, and its only consumer outside the
crate was an integration test whose own header described it as "the documented
public surface", a justification that existed only because the test was out
of crate. Internal callers already used the `crate::iteration::` path, so no
call site changed.

The three test functions were moved verbatim into `iteration.rs` under
`#[cfg(test)] mod tests` rather than dropped, and still run
(`iteration::tests::test_tile_2d_*`). `tools/check-public-api.sh` now asserts
the module stays `pub(crate)`, so the removal cannot silently regress.

### Considered and deliberately kept

`simd` and `codegen` were both reviewed as privatization candidates and kept.
Neither executes operations or bypasses a capability query; both are documented
surfaces that a backend author needs. Freezing them is a decision recorded here,
not an accident of visibility, which was the failure mode the review was
looking for.
