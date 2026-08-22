# The Incin Wiki

Welcome to the **Incin Knowledge Base & Architecture Wiki**.

Incin is a compile-time shape-checked, capability-driven deep learning framework built in Rust. This wiki provides technical documentation, architecture deep dives, backend implementation specifications, and contributor guides.

---

## 🧭 Topic Navigation

### 1. Architecture & Type System
* **[Architecture Deep Dive](Architecture-Deep-Dive.md)**: Type-level shape representations, tensor metadata, runtime vs static shapes, and tape-based reverse-mode autograd.
* **[Shape & Typenum Internals](Shape-and-Typenum-Internals.md)**: How `typenum` binary trees (`UInt<...>`, `B0`, `B1`) encode arbitrary tensor dimensions and matrix contraction rules at compile time.
* **[Compiled Graph & Kernel Fusion](Compiled-Graph-and-Fusion.md)**: Graph tracing, constant folding, horizontal & vertical kernel fusion, and offline plan tuning.

### 2. Backend & Hardware Development
* **[Backend Authoring Guide](Backend-Authoring-Guide.md)**: Step-by-step guide to writing a new execution backend (`StorageBackend`, `Execute<Op>`, memory transfers, stream synchronization).
* **[Memory Model & Zero-Copy Views](Architecture-Deep-Dive.md#memory-model-and-strides)**: Contiguous buffers, non-contiguous strided slicing (`narrow`), transposition, and broadcasting.

### 3. Developer & AI Tooling
* **[Agent Skills & IDE Setup](Agent-Skills-and-IDE-Setup.md)**: Using Incin AI Agent Skills in Antigravity, Cursor, and Claude, plus configuring `incin-lsp` for VS Code and Neovim.
* **[CLI & Diagnostic Interceptor](Troubleshooting-and-FAQ.md#using-cargo-incin)**: Using `cargo incin check`, `cargo incin doctor`, and `cargo incin inspect`.

### 4. Governance, Soundness & Contributing
* **[Contributing & Soundness Gates](Contributing-and-Soundness-Gates.md)**: Toolchains, running Miri, AddressSanitizer, ThreadSanitizer, code conventions, and pull request checklist.
* **[Troubleshooting & FAQ](Troubleshooting-and-FAQ.md)**: Common compilation errors, typenum translation tips, out-of-memory avoidance, and device configuration.

---

## 🔗 Quick Links
* **Repository**: [github.com/xupremix/incin](https://github.com/xupremix/incin)
* **The Incin Book**: [xupremix.github.io/incin](https://xupremix.github.io/incin)
* **API Documentation**: [docs.rs/incin](https://docs.rs/incin)
