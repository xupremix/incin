# Command-line tooling: `cargo incin`

`cargo incin` is two tools in one binary: a Cargo interceptor that
translates typenum shape errors into readable hints as you compile, and a
model tool for inspecting checkpoints, probing devices, and watching
training telemetry. Everything below is printed by the tool itself — run
`cargo incin --help` when in doubt.

## The interceptor: readable shape errors

`check`, `build`, `test`, `run` (plus `bench`, `doc`, `fix`, `clippy`)
forward to Cargo with real-time typenum error translation. A wall of
`typenum` trait bounds becomes a hint naming the mismatched shapes; anything
else delegates straight to `cargo <subcommand>` untouched, so `fmt`, `tree`,
plugins, and future built-ins keep working with no update to this tool.
`--raw` disables translation and shows the raw compiler diagnostics.

```sh
cargo incin check
cargo incin test -- --nocapture
cargo incin translate "S![2, 3] vs S![3, 2]"
```

`translate` renders the same hints for pasted text, which is how you decode
an error from CI logs without recompiling.

## `doctor`: what this machine can run

```sh
cargo incin doctor
cargo incin doctor --json
```

`doctor` reports devices, enabled features, caches, and capability probes —
the same registrations that generate `docs/capabilities.md`, queried live.
When a chapter says an operation "would run if the device were present,"
`doctor` tells you whether it is. `--check-updates` asks crates.io for a
newer `incin`.

## `inspect`, `plan`, `tune`: models and execution

- `cargo incin inspect <model>` prints metadata for `.safetensors`,
  `.gguf`, or `.onnx` files without loading weights into a model.
- `cargo incin plan [--json] [--devices DEV] [--epochs N]` generates the
  execution-plan report: which operations lower where, before you train.
- `cargo incin tune [--json] [--clear] [--offline]` inspects and manages
  the autotune cache of benchmarked CUDA launch configurations.

## `new`, `skills`, `watch`: projects and telemetry

- `cargo incin new mnist [path]` scaffolds a ready-to-run training
  project from a template.
- `cargo incin skills list|install` manages the Incin agent skills for AI
  assistants, so a coding agent working in your tree gets the same
  contracts this book describes.
- `cargo incin watch [--run-id ID | --run-dir PATH]` launches the live
  telemetry TUI over a training run's directory. The run side is the
  `incin-telemetry` emitter; the watch side reads its files, no socket,
  no daemon.

When a compile error names shapes you cannot parse, reach for this
chapter's first section before the [shape-error
how-to](./howto_debug_shapes_errors.md) — translation first, forensics
second.
