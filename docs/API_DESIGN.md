# Public API Design & Encapsulation Guidelines

**INSTRUCTIONS FOR ALL DEVELOPERS AND AI ASSISTANTS WORKING ON THIS REPOSITORY**

This codebase follows a strict policy regarding the public API surface. Every `pub` item declared represents a long-term contract that cannot be broken without a major version bump.

To ensure we can evolve the implementation freely (e.g., rewriting CUDA kernels, changing WGSL pipeline caching, or refactoring the CPU runtime), **we must not expose internal implementation details**.

---

## 1. The Core Rule: `pub(crate)` is Default

By default, any module, struct, enum, trait, or function **MUST** use `pub(crate)` rather than `pub`, unless it explicitly needs to be accessible by downstream library consumers.

### Examples of what MUST NOT be `pub`:
- **Dispatch functions:** `pub fn dispatch_matmul(...)` $\rightarrow$ **Use `pub(crate)`**. Downstream users do not launch shaders/kernels manually.
- **Internal State:** `pub struct WgpuDeviceState` $\rightarrow$ **Use `pub(crate)`**.
- **Raw Memory Buffers:** `pub struct WgpuBuffer` $\rightarrow$ **Use `pub(crate)`**.
- **Internal Modules:** `pub mod ops; pub mod tape;` $\rightarrow$ **Use `pub(crate) mod ops; pub(crate) mod tape;`**.

### Examples of what MAY be `pub`:
- Concrete backend implementations: `pub type IncinBackend<D = Cpu> = NativeBackend<D>;`
- Associated types satisfying the `Backend` trait: `pub struct CpuVar; pub struct CpuGrads;`
- Re-exports of core traits (`Backend`); operation execution uses the
  descriptor `Execute<O>` contract and does not expose historical operation
  families.

---

## 2. Exposing Trait Implementations Safely
When a type is `pub` (e.g., `CpuStorage` because it satisfies `Backend::Storage`), **its internal fields must remain private**:

```rust
// Correct:
pub struct CpuStorage {
    pub(crate) buffer: Arc<CpuBuffer>,
    pub(crate) shape: Vec<usize>,
}

// Incorrect (LEAKED internal state):
pub struct CpuStorage {
    pub buffer: Arc<CpuBuffer>, // LEAKED: users can mutate raw memory
    pub shape: Vec<usize>,       // LEAKED: users can mutate shape without updating strides
}
```

---

## 3. Best Practices for Features & Operations
1. Always start with `pub(crate)`.
2. Do not add `pub` functions to internal modules just to make them accessible from another internal module. Use `pub(crate)` for cross-module internal access.
3. If an existing `pub` item clearly looks like an internal implementation detail, flag it or change it to `pub(crate)`. The policy applies to code already in the tree, not only to new code.
4. Ensure unit tests (which live in the same crate) use `pub(crate)` items without exposing them to public API surface.

---

This file is the single copy of this policy. Agent-facing directories may link
to it; they must not paraphrase it. See `PROPOSALS.md` §2.10 and decision
`D-011` in Appendix C.
