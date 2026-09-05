# Summary

[Introduction](./introduction.md)

# Getting started

- [Installation](./installation.md)
- [Editor integrations](./editor_integrations.md)
- [Quickstart](./quickstart.md)

# Core concepts

- [Tensors](./tensors.md)
- [Shapes: static, dynamic, and mixed](./shapes.md)
- [Advanced shapes](./advanced_shapes.md)
- [Layout: proving where elements live](./layout.md)
- [Autograd](./autograd.md)
- [Errors](./errors.md)

# Building models

- [Layers and `#[module]`](./building_models.md)
- [Sequential models](./sequential.md)
- [A small Transformer-style block](./transformer.md)

# Training

- [Losses, optimizers, and schedulers](./training.md)
- [Data loading](./data_loading.md)
- [Quantization](./quantization.md)
- [Distributed planning](./distributed.md)
- [Metrics](./metrics.md)
- [Saving and loading](./saving_loading.md)

# Backends

- [CPU, and what actually runs on GPU today](./backends.md)
- [The target API and canonical dispatch](./target_api.md)
- [Backend authoring](./backend_authoring.md)
- [Backend conformance](./backend_conformance.md)
- [From proofs to execution](./proofs_to_execution.md)
- [Custom and fused operations](./custom_operations.md)

# Reference

- [The macro reference](./macros.md)
- [Every feature flag](./feature_flags.md)
- [Invariants and proof types](./invariants.md)
- [Experimental surfaces](./experimental.md)
- [Coming from PyTorch](./pytorch_cheatsheet.md)
- [0.1.0 release notes](./release_notes.md)
- [What's not finished yet](./whats_not_finished.md)

# Deep dive

The ideas underneath everything above: how an operation travels from a typed
method call to a running kernel, which guarantees hold at which stage, and
how to extend the system with your own backends, devices, and dtypes.
Concept-first by design; the hands-on contracts live in the backend
authoring and custom operations chapters above.

- [The layered architecture](./deep_architecture.md)
- [Type semantics](./deep_type_semantics.md)
- [Lowering: from descriptor to kernel](./deep_lowering.md)
- [Proofs: how claims are checked](./deep_proofs.md)
- [What the macros guarantee](./deep_macros_internals.md)
- [Deep autograd: tapes, recipes, and custom training ops](./deep_autograd.md)
