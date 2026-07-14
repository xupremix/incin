# No_Std Migration Status

This document tracks the ongoing migration to make the `kindle-core`, `kindle-native`, and `kindle-backends` crates `no_std` compatible.

## What is Done
1. **High-Performance SIMD Matmul**: We implemented high-performance SIMD matmul algorithms (`avx2`, `fma`, dot products for `Q8_0` and nested block iterations for `f32`) and added strictly scoped `unsafe` compliance for Rust 2024.
2. **Global Replacements**: Replaced all `std::` components across `kindle-core`, `kindle-native`, and `kindle-backends` with `core::`, `alloc::`, and `hashbrown::` equivalents where applicable.
3. **Cargo.toml Update**: Gated non-compatible `std` dependencies (like `bincode` and `safetensors`) behind `std` optional features in our `Cargo.toml` files, ensuring `no_std` isn't constrained by them.
4. **ThreadLocal Replacement**: Substituted `thread_local!` instances with `spin::Mutex` inside `tape.rs` and `tracing.rs`.

## What is Left to do
There are a few syntax errors and type-inference issues in `tracing.rs` and `serialize.rs` resulting from the automated search/replace phase. Specifically, when you return, we need to:

- **Fix `Graph::new()`**: Complete the rewrite of `Graph::new()` with `spin::Lazy` since `Mutex::new` requires `const fn`.
- **Fix Syntax in `tracing.rs`**: Patch a few missing parenthesis brackets `})` from the `TRACING_GRAPH.with` rewrites inside `tracing.rs`.
- **ONNX Exporter Gate**: Feature-gate the ONNX exporter (`std::fs` usages) in `kindle-core`.
- **Fix Type-Inference**: Resolve a minor `k.clone()` type-inference hiccup inside `serialize.rs`.
- **Validation**: Validate the final build by running:
  ```bash
  cargo check --no-default-features -p kindle-core -p kindle-native -p kindle-backends
  ```

*You can provide this file to the AI assistant in a new conversation to seamlessly pick up where we left off.*
