# Introduction

Incin is a Rust deep learning framework built around one idea: a tensor's shape
can live in the type system, and the compiler will check it. Write
`Tensor<s![768, 256], Backend>` and a `matmul` against something the wrong
width does not compile; you don't wait for a runtime crash three epochs into
a training run. Dynamic shapes are just as first-class (`Tensor<Dyn, Backend>`)
for the parts of a model that genuinely are dynamic (batch size, sequence
length), and the two compose in the same program.

This book is a hands-on tour. Rust snippets are checked where they are wired
into doctests or executable fixtures; prose snippets are not promised to be
automatically executed. The examples below are drawn from the current crate
and its CPU-focused smoke tests. Where something is not finished, the book
says so plainly rather than describing design intent as if it were shipped. See
[What's not finished yet](./whats_not_finished.md) for the current honest
state, especially around GPU backends.

## How this book is organized

- **Getting started** gets a tensor on screen in five lines.
- **Core concepts** covers the type-level shape system, autograd, and the
  error contract: the things that shape every other chapter.
- **Building models** and **Training** are the day-to-day chapters: layers,
  losses, optimizers, data loading, checkpoints.
- **Backends** is the honest one: what runs where, today, measured rather
  than assumed.
- **Reference** has a PyTorch-to-Incin cheatsheet and the running list of
  known gaps.
- **Deep dive** is for when you want to know why, not just how: five
  chapters on the execution model (the layer stack, type semantics,
  lowering and dispatch, the proof stages) and on extending Incin with
  your own backends, devices, and dtypes.

## Where else to look

- The source code, issue tracker, and releases are hosted on [GitHub](https://github.com/xupremix/incin).
- The [Deep dive](./deep_architecture.md) part of this book explains the
  execution system itself: how a call becomes a kernel, where each guarantee
  comes from, and where you can hook in. This book is task-oriented ("how do
  I train a model"); those chapters are concept-oriented ("why does a shape
  mismatch fail at compile time"). The generated
  `docs/OPERATION_SEMANTICS.md` and `docs/capabilities.md` are the exhaustive,
  always-current per-operation reference this book does not try to duplicate.
