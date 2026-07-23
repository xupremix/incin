# 05 — Observability + `cargo kindle new` / `watch` (btop for training)

> **Depends on:** nothing hard — the TUI, telemetry, and plugin API already
> exist. **Effort:** Medium (wiring + scaffolding). **Priority:** high viral
> ceiling — terminal dashboards are catnip for dev creators.

## Goal

1. **30-second onboarding:** `cargo kindle new mnist` scaffolds a complete,
   compiling training project with a working loop and telemetry pre-wired.
2. **Live in-terminal dashboard:** `cargo kindle watch` (or running the training
   binary + the viz TUI) shows loss curves, gradient/weight norms, GPU/CPU
   memory, and the model graph — **no TensorBoard, no browser, no Python.**
3. **One-line instrumentation:** a training loop opts in with minimal code (a
   `#[kindle::observe]`-style helper or a couple of `reporter.scalar(...)`
   calls), not a manual telemetry-plumbing chore.

## Grounding (this is ~80% built already)

- `kindle-telemetry` (`crates/kindle-telemetry/src/`): `events.rs` defines
  `Event` with `Scalar`, `GradientNorm`, `WeightNorm`, `Memory`, `Epoch`,
  `Hyperparam`, `GraphSnapshot`; `emitter.rs`, `reporter.rs`, `run_dir.rs`, and
  `transport/{file,socket}.rs` handle emission and out-of-process transport.
- `kindle-viz` (`crates/kindle-viz/src/`): a ratatui-style TUI with
  `panels/{loss,scalar,norms,system,graph}.rs`, `app.rs`, `dispatch.rs`,
  `transport_reader.rs`, and a `main.rs` binary. It tails a telemetry transport
  and renders panels.
- `kindle-viz-plugin-api` (`crates/kindle-viz-plugin-api/src/`): a
  plugin/panel/keymap API — the TUI is **already extensible**, an
  under-marketed strength.
- `cargo-kindle` (`crates/kindle/src/bin/cargo-kindle.rs`) is the CLI to extend
  with `new` and `watch` subcommands (it already dispatches subcommands and has
  `inspect`/`translate`).

## Task list

### Task 05.1 — the `#[kindle::observe]` / reporter ergonomic
Confirm the current minimal code to emit a scalar (read `reporter.rs`'s public
API). If emitting loss each step already requires only
`reporter.scalar("loss", step, value)`, document that as the idiom and skip the
macro. If it requires manual transport/emitter setup, add a one-call constructor
(`Reporter::to_run_dir("runs/mnist")`) so a user needs exactly:
```rust
let mut rep = Reporter::to_run_dir("runs/mnist")?;
// in the loop:
rep.scalar("loss", step, loss_value);
rep.gradient_norm(step, &grads);   // if a helper doesn't exist, add one
```
Only build a proc-macro `#[observe]` if the plain API is genuinely clunky —
prefer the smaller surface.

### Task 05.2 — `cargo kindle watch`
Add a `watch` subcommand to `cargo-kindle.rs` that launches the `kindle-viz` TUI
pointed at the default run dir (or `--run-dir`). Effectively a thin spawn/handoff
to the existing viz binary, so `watch` is discoverable from the one CLI users
already know. Support `--socket` vs `--file` transport selection matching
`kindle-telemetry`'s transports.

### Task 05.3 — `cargo kindle new <template>`
Add a `new` subcommand that writes a ready-to-run project from an embedded
template. Templates (start with `mnist`, then `cnn`, `mlp`):
- `Cargo.toml` depending on `kindle` (+ `kindle-data`) with the right features;
- `src/main.rs`: a full training loop (model via `#[module]`, `DataLoader`,
  `AdamW`, loss), telemetry pre-wired to `runs/`, and a comment pointing at
  `cargo kindle watch`;
- a `README.md` with the two-command quickstart.
Model the training loop on the existing
`crates/kindle/examples/mnist_training.rs` and `native_training_demo/` so it is
guaranteed idiomatic and compiling. Embed templates with `include_str!` so the
CLI stays a single binary.

### Task 05.4 — a `panic`/anomaly panel
There is already a `panels/panic_test.rs`. Wire a real **anomaly panel**: when a
`GradientNorm`/`Scalar` event is NaN/Inf, surface it prominently ("⚠ NaN in
`fc2` at step 1043"). Kindle already has `backward_with_nan_check`
(`cpu/tape.rs`) — connect that provenance to a telemetry event so the panel can
name the offending layer. This is a strong "it debugs itself" beat.

### Task 05.5 — the graph panel from real static shapes
`panels/graph.rs` + the `GraphSnapshot` event already exist. Ensure the emitted
graph carries the **static** shapes (Kindle knows them exactly, unlike a runtime
tracer) so the TUI graph shows real dims on every edge. This is a differentiator
over PyTorch's runtime-traced graphs.

## Verification
- Standard loop, plus build the templates: `cargo kindle new mnist /tmp/x &&
  (cd /tmp/x && cargo build)` must succeed and the loop must run a few steps.
- `cargo test -p kindle-viz` (panel tests exist: `tests/panels.rs`).
- Manual: run a scaffolded project, `cargo kindle watch`, confirm loss curve and
  norms animate.

## Risks / DO-NOT
- **DO-NOT** fork or reinvent the TUI — extend `kindle-viz` and its plugin API.
- **DO-NOT** make `watch` require a GPU; CPU runs must render the dashboard.
- **DO-NOT** hardcode template contents that will bit-rot — generate the
  training loop from the *same* patterns as the maintained examples, and add a
  CI check that `cargo kindle new mnist` output compiles (Task 05.3 acceptance).
- **DO-NOT** block the training loop on the viz process; telemetry is
  out-of-process by design (`transport/socket.rs`) — keep it non-blocking.

## Demo script
Two commands: `cargo kindle new mnist && cargo kindle watch`. Loss curve draws
live in the terminal, grad-norm bars pulse, a NaN lights up the anomaly panel in
red and names the layer. Caption: *"btop, but it's training a neural net — no
browser, no Python, one binary."*
