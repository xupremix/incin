# Experimental and specialized surfaces

Everything here is feature-gated and carries no compatibility guarantee. The
point of this chapter is to say plainly what each one *is* today, because
several are design surfaces rather than working features and the difference
matters when you're deciding what to build on.

## Quantization

The quantized backend-authoring contract has three operations: `quantize`
(float → compressed blocks), `dequantize` (blocks → float, lossy), and
`quantized_matmul` (two quantized operands → `f32`, without fully
dequantizing first). The only quantized representation any backend implements
is `Q8_0`.

There is **no `Tensor::quantize` method** - quantization is reachable only
through the backend trait, not the stable tensor surface. The CPU backend
implements all three. They are registered `training = false` because their
kernels push no tape entry: advertising them for training would promise a
gradient that never arrives.

## Distributed

Feature `distributed`. This is a **planning and typing surface, not an
execution path.**

What exists and works: typed device meshes (`mesh!`), compile-time and runtime
tensor placements (`placement!`, `Sharded`/`Replicated`/`Partial`), FSDP and
ZeRO stage descriptors, data-parallel/tensor-parallel/pipeline plan builders,
collective descriptors, and a substantial body of validation that rejects
inconsistent plans.

What does not exist: the execution. `Trainer::fit` refuses a multi-device plan
with `TrainError::CollectivesUnavailable` rather than silently running on one
device - which is the right failure, but it is a failure. Transports are
separate opt-ins (`distributed-reference` for an in-process deterministic
transport, `distributed-nccl` for two-host CUDA), and there is no end-to-end
distributed training path.

Treat the whole subsystem as research-grade: excellent for exploring what a
typed distributed plan should look like, not something to train a model with.

## Autotune

Feature `autotune` (implies `cuda`). Tuning configuration and inspection types
for CUDA launch parameters - `AutotunePolicy`, `KernelSignature`,
`PersistentTuningCache`, `TuningSelection` and friends, exposed under
`incin::experimental::tuning`. Cache records are validated on deserialization
(see [Invariants](./invariants.md)) rather than trusted as bytes.

## Compiled execution

Feature `compiled`. This is a preview-only CPU reference evaluator under
`incin::experimental::compiled`, not a stable compiler interface. The generic
plan builds symbolic guards and liveness information.
`CpuCompiledPlan::compile` performs CPU admission, and
`CpuCompiledInvocation` runs the admitted descriptor-backed subset. Unsupported
operations and malformed descriptors fail during admission.

`CompiledArtifact` serializes a preview plan snapshot for inspection and local
testing. It is not a deployment format or a portable ABI. A loader checks the
artifact format and the caller-supplied compatibility major/minor values (patch
values may differ); it does not verify the running framework version.

## Telemetry

Feature `telemetry`. Backend telemetry hooks plus an emitter/reporter pair in
the `incin-telemetry` crate. With this on, `cargo incin doctor` additionally
reports the telemetry run directory.

## Visualization

The `incin-viz` crate - a TUI for inspecting graphs, with a plugin API in
`incin-viz-plugin-api`. The `tui_graph_demo` example in the repository is the
working entry point.

## ONNX

Two macros, `model!` and `import_model!`, expand a `.onnx` graph into typed
Rust at compile time. Support is **partial and fail-closed**: initializers,
unknown rank, control flow, custom domains, and unsupported node types are all
macro-expansion errors rather than silently-wrong generated code.

There is also an ONNX *exporter* (`incin_core::onnx_exporter`). Both directions
should be understood as tooling for a known, simple graph shape rather than
general ONNX interoperability. If you have an arbitrary model from the wild,
expect it to be rejected.

## The preview `Trainer`

Feature `train`. `incin::experimental::training::Trainer` has a real `fit`:
forward, backward, optimizer step, per epoch and batch, with a closure
supplying the loss (because the loss is the one part of a training step that
is genuinely the caller's). It works for single-device training.

It is in `experimental` because the interface may change without a migration
path, and because the multi-device half of its plan surface does not execute.
Writing the loop yourself - as [Training](./training.md) shows - is neither
harder nor less supported.
