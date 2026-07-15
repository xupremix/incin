# API Design & Anti-Breaking-Changes Guidelines

**IMPORTANT INSTRUCTIONS FOR ALL AI AGENTS WORKING ON THIS REPOSITORY**

This codebase follows an extremely strict policy regarding the public API surface. Every `pub` item you declare represents a commitment that we cannot break without bumping the major version (or the minor version pre-1.0).

To ensure we can evolve the implementation freely (e.g., rewriting CUDA kernels, changing WGSL pipeline caching, or refactoring the CPU runtime), **we must not expose implementation details**.

## The Core Rule: `pub(crate)` is Default

By default, any new module, struct, enum, trait, or function **MUST** use `pub(crate)` rather than `pub`, unless you have explicit confirmation that it needs to be accessible by downstream consumers.

### Examples of what MUST NOT be `pub`:
- **Dispatch functions:** `pub fn dispatch_matmul(...)` — NO. Use `pub(crate)`. Downstream users do not launch shaders manually.
- **Internal State:** `pub struct WgpuDeviceState` — NO. Use `pub(crate)`.
- **Raw Buffers:** `pub struct WgpuBuffer` — NO. Use `pub(crate)`.
- **Internal Modules:** `pub mod ops; pub mod tape;` — NO. Use `pub(crate) mod ops; pub(crate) mod tape;`.

### Examples of what MAY be `pub`:
- The concrete backend implementation structs: `pub struct WgpuBackend;`
- The types required to satisfy the `Backend` trait associated types: `pub struct WgpuVar; pub struct WgpuGrads;`
- Re-exports of core traits like `Backend`, `ModuleOps`, `OptimizerOps`.

## Exposing Trait Implementations Safely
When a type is `pub` (e.g., `WgpuStorage` because it's the `Backend::Storage` associated type), **its internal fields should remain private**.
- Correct:
  ```rust
  pub struct WgpuStorage {
      pub(crate) buffer: Arc<WgpuBuffer>,
      pub(crate) shape: Vec<usize>,
  }
  ```
- Incorrect:
  ```rust
  pub struct WgpuStorage {
      pub buffer: Arc<WgpuBuffer>, // LEAKED: users can read/write raw memory
      pub shape: Vec<usize>,       // LEAKED: users can mutate shape without strides
  }
  ```

## Working in this Codebase
When implementing a new feature or operation:
1. Always start with `pub(crate)`.
2. Do not add `pub` functions to internal modules just to make them accessible from another internal module — `pub(crate)` handles that perfectly.
3. If an existing `pub` item clearly looks like an internal implementation detail, you should flag it or change it to `pub(crate)`.
4. Ensure all unit tests (which live in the same crate) can still access `pub(crate)` items just fine. There is no need for `pub` in order to test things internally.
