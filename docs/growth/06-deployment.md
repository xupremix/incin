# 06 — Deployment (single binary · GGUF loop · browser/WASM)

> **Status:** PARTIAL — GGUF Q8_0 export + inspector already landed (see
> `crates/kindle-core/src/io/`). **Effort:** Low (packaging/benchmark) →
> Medium-High (WASM). **Priority:** the credibility close — "safe *and* it ships
> anywhere."

## Goal

Three deployment stories PyTorch handles badly:
1. **Single static binary, zero Python** — `cargo build --release` → one file,
   no Docker, no CUDA-runtime install for CPU inference.
2. **Train in Rust → export GGUF → run in Ollama/llama.cpp** — full loop, no
   Python anywhere.
3. **Same model in the browser** via WGPU/WASM — PyTorch's browser story is
   famously weak.

## Grounding

- GGUF export with **real** Q8_0 block quantization + F32 passthrough, and a
  full GGUF metadata/tensor-table inspector, already exist:
  `crates/kindle-core/src/io/{gguf.rs,inspect.rs,mlx.rs,mod.rs}`,
  `cargo kindle inspect`, and `crates/kindle-core/tests/export_test.rs`.
- Native CPU backend is dependency-light; WGPU backend works
  (`crates/kindle/examples/backends` runs ops on `WgpuB`).
- WASM blocker is documented in `IDEAS.md` "Open questions": `getrandom 0.2`
  refuses to build for bare `wasm32-unknown-unknown` without an entropy opt-in;
  this blocks `CreationOps::rand`/`randn` and therefore any WASM build.

## Workstream A — single-binary story (Effort: Low)

### Task 06.A1 — a `deploy/` example: train → save → load → infer in one binary
A minimal example crate that trains a tiny model, saves safetensors, and has an
`--infer` mode that loads and runs it — all CPU, no network. Prove the release
binary is self-contained: `ldd` shows no exotic deps; record its size.

### Task 06.A2 — the benchmark/comparison artifact
A reproducible `bench/` comparing: Kindle release-binary size + cold-start vs. a
minimal PyTorch inference Docker image size + cold-start, for the same model.
This is a *marketing artifact* as much as a benchmark — put the numbers in the
README and the book's deployment chapter. Be honest and reproducible (script it).

## Workstream B — GGUF loop completion (Effort: Medium)

### Task 06.B1 — extend quantization beyond Q8_0
`gguf.rs` currently implements F32 + Q8_0 and **explicitly rejects** the other
`QuantScheme`s (by design, after the recent fix). Implement the next-most-wanted
scheme, `Q4_K_M` (or `Q4_0` first as a stepping stone), end-to-end: the backend
`QuantizedOps::quantize` for the new `QuantDType` (mirror
`crates/kindle-backends/src/cpu/ops/quant.rs`'s `BlockQ8_0` path), the
`to_bytes` block serialization (`crates/kindle-backends/src/cpu/mod.rs` Q8_0
arm), and the exporter's per-tensor eligibility rule (`gguf.rs` `save`). Add a
byte-layout regression test in `export_test.rs` like the Q8_0 one.
**Verify against a real consumer:** load the exported file in `llama.cpp`/Ollama
and confirm it runs — GGUF is only "done" when a real runtime accepts it, not
when our inspector likes it.

### Task 06.B2 — round-trip parity test
Quantize → dequantize → compare against a reference; assert error under a
tolerance (the Q8_0 fidelity test at `cpu/ops/quant.rs::tests` is the model).

## Workstream C — browser/WASM (Effort: Medium-High)

### Task 06.C1 — resolve the getrandom/WASM decision (needs a product call)
`IDEAS.md` lists three options. Recommended default: **`getrandom` `js` feature
as a `wasm32`-only target dependency** for a browser demo (commits to a JS host,
which a browser demo already assumes), while leaving an issue open for the
`custom` hook for non-JS WASM hosts. **Flag this to the user before committing**
— it is a dependency-surface decision, not purely mechanical.

### Task 06.C2 — a browser inference demo
Compile a WGPU-backed inference path to `wasm32`, load a small model, run it in a
`<canvas>`/page. The `TransferTo` design means the model type is the same as
native; only the backend type param changes. Ship it as a static page (no
server). This is the highest-effort, highest-shareability item — scope it as its
own milestone after A and B land.

## Verification
- A/B: `cargo build --release -p <deploy-example>`; run the benchmark script;
  `cargo test -p kindle-core --test export_test`.
- B: **plus** a manual "does llama.cpp/Ollama load it" check — record the exact
  command and output in the PR.
- C: `cargo build --target wasm32-unknown-unknown …` (document the exact feature
  flags); manual browser smoke test.

## Risks / DO-NOT
- **DO-NOT** mark a new GGUF quant scheme "done" until a real external runtime
  loads it. Our inspector accepting it is necessary, not sufficient. (This is
  the deployment analogue of the "compiles clean ≠ verified" rule.)
- **DO-NOT** silently add the `getrandom` `js` feature workspace-wide — scope it
  to the `wasm32` target so native builds are unaffected, and get sign-off
  (06.C1).
- **DO-NOT** regress the native binary's dependency-lightness by pulling browser
  deps into core crates; keep WASM glue in a separate example/crate.

## Demo scripts
- *"My PyTorch inference image is 2.1 GB. My Kindle model is one 5 MB binary —
  same model, same output."*
- *"Trained in Rust. Exported to GGUF. Running in Ollama. Zero Python touched
  this model, ever."*
- *"I trained this in Rust and it's running in your browser tab. No server."*
