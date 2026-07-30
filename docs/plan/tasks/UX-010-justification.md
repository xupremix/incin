# UX-010 Justification — Typed `einsum!` Macro

**Date:** 2026-07-30  
**Tier:** exploratory  
**Deps:** EXE-003, DST-003  

## Justification

1. **Unification of contraction operations:** `einsum!` subsumes `matmul`, `batch_matmul`, `dot`, `outer`, `trace`, and arbitrary index contractions under a single, readable subscript notation (e.g., `einsum!("bik,bkj->bij"; a, b)`). This eliminates the need to choose between six different APIs for the same conceptual operation.

2. **Ergonomic ML workflow:** Transformer attention, low-rank adapters, and einops-style rearrangements all benefit from a single `einsum!` call. `batch_matmul` in the prelude is a workaround for what `einsum!` provides naturally.

3. **Compile-time dimension validation:** The subscript parser extracts repeated indices and validates that:
   - each repeated index appears in exactly two operands (or once for a trace),
   - the output subscript's indices are a subset of the input indices,
   - error messages are emitted at macro expansion time.

4. **Zero runtime overhead:** All subscript parsing and index validation happens at proc-macro expand time; the generated code is a plain function call.

## Decision

Implement `einsum!` as a declarative proc-macro in `crates/incin-macros/src/einsum.rs` that:
- Parses a subscript string literal and operand expressions,
- Validates subscript structure at compile time,
- Emits a call to a runtime `einsum_impl` function accepting the subscript and operand slices,
- Adds parity tests against `matmul` results.
