# UX-011 Usability Evaluation - `parallel!` Block Macro

**Date:** 2026-07-30  
**Tier:** exploratory  
**Deps:** DST-004  

## Evaluation & Usability Evidence

1. **Context & Problem:** While `#[parallel]` and `#[shard]` attributes (`UX-004`) annotate struct field placements on `#[module]` types, inline computation blocks inside functions or training loops need a clean way to introduce scoped mesh execution contexts without defining a separate struct module.

2. **Syntax Design:**
   ```rust
   // Block macro form:
   let result = parallel!(mesh => {
       // Code executed within the scoped mesh topology context
       x + y
   });
   ```

3. **Usability Conclusion:** Implementing `parallel!` as a expression-block macro provides clear scoping for distributed operations and complements the declarative `#[parallel]` attribute macro.

4. **Implementation Plan:**
  - Add `parallel_block.rs` proc macro parsing `parallel!(mesh_expr => block_expr)` and `parallel!(block_expr)`.
  - Re-export `parallel` macro in `incin-macros` and `incin` prelude.
  - Add integration test `crates/incin-macros/tests/parallel_block.rs`.
