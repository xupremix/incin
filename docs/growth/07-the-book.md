# 07 - The Incin Book ("The Incinnomicon")

> **Historical plan.** The mdBook described here now exists under
> `docs/book/`. For current chapter status and user guidance, use
> `docs/book/src/SUMMARY.md`, `docs/README.md`, and the repository README.
> The task list below records the original adoption plan; it is not a claim
> that the Book is absent.

> **Depends on:** the features it documents (`01`–`06`) - write each chapter as
> its feature lands, never speculatively ahead. **Effort:** ongoing. **Priority:**
> the long-tail adoption engine - a great book is why people *stay*.

## Goal

A rustnomicon-quality book that teaches a newcomer **every technique and idiom of
Incin**, from "install Rust" to "export a quantized model to Ollama" - with a
special, honest focus on the thing that trips people up (the type-level shape
system) and the thing that delights them (compile-time safety + the TUI). The
tone: like *The Rustonomicon* and *The Little Book of Rust Macros* - precise,
example-first, unafraid of the hard parts, but always motivating *why* before
*how*.

## Tooling & location

- **mdBook.** Location: `docs/book/` (committed; distinct from `docs/growth/`).
- `docs/book/book.toml`, `docs/book/src/SUMMARY.md` (the chapter map),
  `docs/book/src/**/*.md`.
- **Executable snippets remain the goal, not a current blanket claim.** Wire
  important book snippets to doctests or executable fixtures, preferably by
  pulling code via `{{#include ../../.. /crates/incin/examples/…}}` so examples
  cannot rot. `mdbook build docs/book` checks the Markdown book; the repository
  documents which Rust snippets are actually compiled.
- Add a CI job: `mdbook build docs/book` + `mdbook test docs/book` (the latter
  against a crate that re-exports the prelude). Document the exact commands in
  `docs/book/README.md`.
- Deploy to GitHub Pages (or docs.rs sidecar) on release.

## Installing / reading the book (historical rollout plan)

This section is written in advance so whoever scaffolds `07.1` implements the
access path the rest of this plan already assumes, rather than inventing one
ad hoc. The rollout notes below predate the current `docs/book/` source tree.
They are kept as historical planning context rather than current availability
claims.

Once `07.1` lands:
- **Read it online:** a GitHub Pages URL, published by CI on merge to the
  default branch (root README should gain a "Book" badge/link at that point - 
  do not add one before the Pages job actually exists and has published once).
- **Read it locally, from source:**
  ```bash
  cargo install mdbook mdbook-linkcheck
  mdbook serve docs/book --open
  ```
  `serve` live-rebuilds on save, which matters here specifically because
  every chapter pulls real code via `{{#include}}` - editing the source
  example it points at should reflow the book immediately.
- **Build it once (e.g. for CI or a static deploy) instead of serving:**
  `mdbook build docs/book` (output in `docs/book/book/`, gitignored).

## Structure (`SUMMARY.md` outline)

Each chapter lists **learning objectives** and the **repo artifact** it draws
from, so a writer (or agent) knows exactly what to demonstrate and where the
ground-truth code is.

### Part I - Getting Started (make them succeed in 10 minutes)
1. **Why Incin?** - the shapes-are-types pitch, honest positioning (safety +
   deployment + observability, *not* research flexibility). Objective: reader
   can articulate the one differentiator. Artifact: `README.md`.
2. **Install & `cargo incin new`** - Rust toolchain, `protoc`, first project in
   two commands. Objective: a running MNIST loop with the live TUI. Artifact:
   doc `05` scaffolder, `examples/mnist_training.rs`.
3. **Your first tensor** - `Tensor::<s![2,3]>::zeros`, `Dyn`, backends. Objective:
   create/inspect tensors on CPU. Artifact: `examples/tensors`, README quickstart.

### Part II - The Type-Level Shape System (the hard, differentiating core)
4. **`s!` and the shape type** - what `s![2,3]` expands to, `DynShape` and
   `PartialDynShape`. Objective: read a `Tensor<…>` type and know
   its shape. Artifact: `shapes/shape.rs`, `incin-macros` `s`.
5. **Reading typenum (and why you rarely have to)** - `UInt`/`UTerm` demystified,
   and the tooling (`cargo incin`, the IDE extension) that means you read
   decimals, not this. Objective: never be scared by a `UInt<…>` again.
   Artifact: `incin-diagnostics`, doc `01`/`02`.
6. **Shape-changing ops** - reshape, transpose, broadcast, concat, stack, matmul
  - each with a *deliberately broken* example and the (readable) compile error
   it produces. Objective: predict which ops compile. Artifact: `shapes/*`, the
   `compile_fail/*` snapshots.
7. **Named dimensions** - `symbolic_dim!`, `Tensor<[Batch, Feature]>`, the
   transpose-safety demo. Objective: use names to make axis bugs impossible.
   Artifact: doc `03`, `examples/named_tensors`.
8. **`idx!` slicing** - Python-style indexing at the type level. Objective:
   slice/narrow with confidence. Artifact: `shapes/idx.rs`, `examples/idx_demo`.

### Part III - Building & Training Models
9. **`#[module]` and `Parameters`/`StateDict`** - defining models, how the macro
   derives traversal. Objective: build a multi-layer model. Artifact:
   `incin-macros/src/module.rs`, `nn/module.rs`.
10. **`Sequential`, `seq!`, `seq_type!`** - composing layers, PyTorch-flat state
    dict keys. Artifact: `nn/`, README.
11. **Autograd, `Grad`/`NoGrad`, and no `zero_grad()`** - gradients as a *type*;
    `.detach()`; why forgetting `zero_grad` is impossible here. Objective:
    understand the compile-time no_grad advantage. Artifact: `tensor/grad.rs`,
    `cpu/tape.rs`.
12. **Optimizers & checkpointing** - `SGD`/`Adam`/`AdamW`, `state_dict` resume.
    Artifact: `optim/`, `tests/optim_tests.rs`.
13. **Losses & metrics.** Artifact: `nn/loss`, `metrics.rs`.
14. **Data loading** - `Dataset`, `DataLoader` as an `Iterator`, parallel
    batching. Artifact: `incin-data`, `examples/dataloader`.
15. **A full training run + the live TUI** - end-to-end MNIST/CNN, watched in the
    terminal, reading the anomaly panel. Artifact: doc `05`, `incin-viz`.

### Part IV - Backends & Performance
16. **Backends & `TransferTo`** - write once, run on CPU/CUDA/WGPU; move a model
    between them. Artifact: `incin-backends`, `examples/backends`.
17. **Mixed precision & dtypes** - the dtype policy, low-precision storage with
    f32 compute. Artifact: `dtype_policy.rs`, `PROPOSALS.md` §3.
18. **CUDA internals (advanced)** - NVRTC codegen, autotuning. Objective: know
    *why* it is fast. Artifact: `cuda/{kernel,tuning}.rs`,
    `PROPOSALS.md` §§3.6–3.7.
19. **Compile-time model stats** - `PARAMS`/`FLOPS` consts, budgets. Artifact:
    doc `04`.

### Part V - Interop & Deployment
20. **ONNX import (`import_model!`)** - compile-time ONNX → typed struct.
    Artifact: `incin-macros` `import_model`, `tests/onnx_import.rs`.
21. **SafeTensors & `from_pretrained`** - load HF weights. Artifact: `nn/save`,
    `incin::hub`.
22. **GGUF & MLX export** - quantize, export, run in Ollama; inspect files.
    Artifact: doc `06`, `io/`.
23. **Single-binary & browser deployment.** Artifact: doc `06`.

### Part VI - Extending Incin (the "nomicon" deep end)
24. **Writing a custom layer** (implementing `Module`/`Parameters`/`StateDict`
    by hand).
25. **Custom `symbolic_dim!` patterns & shape traits** - for library authors.
26. **Writing a `incin-viz` panel plugin.** Artifact: `incin-viz-plugin-api`,
    `incin-viz/examples/custom_panel.rs`.
27. **The macro internals** - how `s!`/`idx!`/`#[module]` generate their types
    (for contributors). Artifact: `incin-macros`.

### Appendices
- **A. PyTorch → Incin Rosetta** - a big two-column table (`torch.nn.Linear` →
  `Linear<s![…]>`, `optim.SGD(model.parameters())` → `SGD::new(model.
  parameters(), …)`, `with torch.no_grad()` → `.detach()`/`NoGrad`, `x.shape` →
  the inlay hint, etc.). This is the single most-linked page for switchers - 
  write it first and keep it exhaustive.
- **B. Error message decoder** - common compile errors and what they mean.
- **C. Glossary** - typenum, `Dim`, `Shape`, backend, tape, dtype policy.

## Task list
1. **07.1** - scaffold mdBook (`book.toml`, `SUMMARY.md` with the outline above),
   CI build+test job, Pages deploy. Ship with Part I only.
2. **07.2** - write **Appendix A (Rosetta)** next - highest leverage for
   switchers, and it does not depend on unshipped features.
3. **07.3** - write Part II (shape system) - the hard part people most need help
   with; pair with doc `01`/`02` landing.
4. **07.4+** - one chapter per feature as it lands; never document unshipped
   APIs. Update `SUMMARY.md` status as chapters complete.

## Verification
- `mdbook build docs/book` (no broken links: add `mdbook-linkcheck`).
- `mdbook test docs/book` - **every** code block compiles against the current
  crates. A CI failure here blocks merge.
- Each chapter's "Artifact" line must point at code that still exists (grep-check
  in CI, or review).

## Risks / DO-NOT
- **DO-NOT** write a chapter for a feature that has not shipped - the book must
  never describe an API the code does not have. This is the book analogue of the
  repo's "trust the code" rule.
- **DO-NOT** paste code snippets that are not compiled by CI. Use includes from
  real examples wherever possible.
- **DO-NOT** bury the differentiator - Part I chapter 1 and Appendix A must both
  land in the first release; they are what convert a browser into a trier.
