# Introduction

Incin is a Rust deep learning framework built around one idea: a tensor's shape
can live in the type system, and the compiler will check it. Write
`Tensor<s![768, 256], Backend>` and a `matmul` against something the wrong
width does not compile — you don't wait for a runtime crash three epochs into
a training run. Dynamic shapes are just as first-class (`Tensor<Dyn, Backend>`)
for the parts of a model that genuinely are dynamic (batch size, sequence
length), and the two compose in the same program.

This book is a hands-on tour. Every code block in it is checked against the
actual crate — most were lifted directly from a working end-to-end smoke test
covering tensors, autograd, every common layer type, a full training loop,
metrics, and checkpoint save/load, all run on CPU before being written down
here. Where something is not finished, the book says so plainly rather than
describing the design intent as if it were shipped — see
[What's not finished yet](./whats_not_finished.md) for the current honest
state, especially around GPU backends.

## How this book is organized

- **Getting started** gets a tensor on screen in five lines.
- **Core concepts** covers the type-level shape system, autograd, and the
  error contract — the things that shape every other chapter.
- **Building models** and **Training** are the day-to-day chapters: layers,
  losses, optimizers, data loading, checkpoints.
- **Backends** is the honest one: what runs where, today, measured rather
  than assumed.
- **Reference** has a PyTorch-to-Incin cheatsheet and the running list of
  known gaps.

## Where else to look

`docs/GUIDE.md` in the repository is the architectural companion to this
book — the type-level shape system's internals, the canonical execution
path, backend authoring, and the idioms the codebase itself follows. This
book is task-oriented ("how do I train a model"); `GUIDE.md` is
concept-oriented ("how does the shape-proof system work"). The generated
`docs/OPERATION_SEMANTICS.md` and `docs/capabilities.md` are the exhaustive,
always-current per-operation reference this book does not try to duplicate.
