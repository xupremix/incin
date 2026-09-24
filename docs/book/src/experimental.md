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

The stable tensor facade sits on top of these ops rather than beside them:
`Tensor::quantize(axis)` and `Tensor::dequantize::<Kout>()` (#93) dispatch
the same catalog descriptors, so validation, capability admission and the
gradient node stay the backend's — see [Quantization](./quantization.md).
CPU and CUDA advertise all three operations; WGPU and Metal advertise
none. The CPU implementations record an identity straight-through tape
entry under gradient recording (#93), `quantized_matmul` records none, and
CUDA's quantize kernels record no tape entry yet. The `training` flag on
those capability rows gates `ExecutionPolicy`'s training flag rather than
gradient recording: CPU's `quantize`/`dequantize` rows declare
`training = true` (a training-policy call is admitted and records the STE
entry), while `quantized_matmul` is `training = false` on every backend
and all three rows are `false` on CUDA, whose kernels record no tape entry
yet.

## Distributed

Feature `distributed`. Most of this surface is still **planning and typing**,
but one lowering executes.

What exists and works as typed planning: device meshes (`mesh!`), compile-time
and runtime tensor placements (`placement!`, `Sharded`/`Replicated`/`Partial`),
FSDP and ZeRO stage descriptors, data-parallel/tensor-parallel/pipeline plan
builders, collective descriptors, and a substantial body of validation that
rejects inconsistent plans.

What executes: FSDP/ZeRO data parallel (#99) through the preview trainer.
ZeRO-1 (all-reduce, then mask gradients to the rank's owned slice) and ZeRO-2
(reduce-scatter) plus a parameter all-gather after every optimizer step run
via `Trainer::with_fsdp_synchronizer`. That lowering is a host-side `f64`
protocol proven on CPU against scripted peers and a two-rank trajectory equal
to the single-device full-batch reference. See
[Distributed planning](./distributed.md) for the exact boundary.

What does not exist: multi-host transport in a default build.
`Trainer::fit` refuses a multi-device plan without the matching synchronizer
(`TrainError::CollectivesUnavailable` / `TrainError::FsdpUnavailable`) rather
than silently running on one device. Transports are separate opt-ins
(`distributed-reference` for an in-process deterministic transport,
`distributed-nccl` for two-host CUDA), ZeRO-3 is refused at plan build, and
real multi-rank NCCL execution stays gated on the unset
`HARDWARE_CUDA_RUNNER` (#82).

Treat the non-FSDP plan surface as research-grade — excellent for exploring
what a typed distributed plan should look like — and the FSDP path as a
CPU-proven preview, not a substitute for a distributed runtime.

## Autotune

Feature `autotune` (implies `cuda`). Tuning configuration and inspection types
for CUDA launch parameters: `AutotunePolicy`, `KernelSignature`,
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

The `incin-viz` crate is a TUI for inspecting graphs, with a plugin API in
`incin-viz-plugin-api`. The `tui_graph_demo` example in the repository is the
working entry point.

## ONNX

Two macros, `model!` and `import_model!`, expand a `.onnx` graph into typed
Rust at compile time.

Dense f32 initializers are embedded into typed `Param` module state with
exact IEEE-754 bit fidelity. Support remains **fail-closed**: sparse
initializers, unknown rank, non-f32 initializers, control flow, custom domains,
and unsupported node types produce macro-expansion compile diagnostics rather
than fabricating unverified values.

There is also an ONNX *exporter* (`incin_core::onnx_exporter`). It writes an
intermediate `value_info` entry for every node output not already named by the
graph's inputs, initializers, or outputs, so `OnnxImporter` — which performs
no shape inference of its own and refuses any output it cannot look up —
reconstructs a multi-node chain from the file instead of guessing. Structural
round-trip properties (export then import preserves inputs, outputs,
initializers, node wiring, attributes, and every value's shape and dtype) and
fail-closed fuzz sweeps over malformed bytes live in
`crates/incin-core/tests/onnx_roundtrip.rs` and
`crates/incin-core/tests/onnx_fuzz.rs`. Both directions should be understood
as tooling for supported graph topologies rather than arbitrary model
interoperability: the macros fail closed at compile time, and the
exporter/importer pair returns errors rather than writing or accepting
unverified values.

## The preview `Trainer`

Feature `train`. `incin::experimental::training::Trainer` has a real `fit`:
forward, backward, optimizer step, per epoch and batch, with a closure
supplying the loss (because the loss is the one part of a training step that
is genuinely the caller's). It works for single-device training.

It is in `experimental` because the interface may change without a migration
path, and because the multi-device half of its plan surface does not execute.
Writing the loop yourself (as [Training](./training.md) shows) is neither
harder nor less supported.

Precision policy is retained and inspectable. `Plan::loss_scale_state()` derives
fresh scaler state from the plan's effective loss scaling setting: `precision()`
supplies the default, and a later `loss_scaling()` overrides it. Pass that state
to `fit_scaled` and retain it across calls for dynamic growth and backoff. Plan
construction rejects an f16-active policy with an exact-f32 accumulator and
scaling disabled via `TrainError::UnsupportedPrecision`; bf16 does not require
scaling. Scaling protects small f16 gradients from underflow, while dynamic
backoff handles non-finite gradients by skipping the optimizer update.

Both `fit` and `fit_scaled` install the plan's **dispatch-time autocast** for
the duration of the run (`incin_core::exec::autocast`, issue #2). Under a
mixed policy, allowlisted operands are cast at admission when the backend's
capability rows admit the active dtype (for example `matmul` and
`broadcast_add` under `mixed_bf16`); master weights stay f32 because
parameter creation is outside the allowlist. This is not module-boundary
autocasting and not a separate master-weight copy — it is an allowlist, not
an ambient cast of every float op. The precision fixtures pin both sides:
`mixed_bf16_autocasts_allowlisted_dispatch_operands` / the f16 twin assert the
cast, while `mixed_bf16_retains_f32_master_weights` asserts the weights do
not move. Mixed-precision memory and throughput gains still depend on which
backend rows admit the low-precision dtype.
