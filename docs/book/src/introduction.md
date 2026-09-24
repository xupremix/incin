# Introduction

Incin is a Rust deep learning framework built around one idea: the two
commonest training failures — a wrong shape and an ignored error — should
be caught by the compiler and the type system, not three epochs into a run.
A tensor's shape lives in the type (`Tensor<s![768, 256], Backend>`), so a
`matmul` against the wrong width does not compile; everything that can
still fail at runtime returns a typed `Result` that names the failure and
fails closed — a backend that cannot do what you asked refuses it with a
typed reason rather than doing something else quietly. Dynamic shapes
(`Tensor<Dyn, Backend>`) are first-class too, for the parts of a model that
genuinely are dynamic (batch size, sequence length), and the two compose in
the same program.

## Who it's for

- Rust programmers who want to train or run models without leaving the
  type system — shapes checked at compile time, errors checked at runtime.
- Anyone who has been bitten by a silent broadcast, a shape mismatch
  discovered mid-run, or an error nobody checked.
- People coming from PyTorch: [Coming from PyTorch](./pytorch_cheatsheet.md)
  maps the vocabulary across.

## A taste of Incin

```rust
use incin::prelude::*;

let x = Cpu.randn(shape![2, 8])?;
let layer = Linear::<s![8, 4]>::build(())?;
let h = ReLU.forward(layer.forward(x)?)?;
println!("{:?}", h.dims()); // [2, 4]
# Ok::<(), incin::Error>(())
```

Five lines: a tensor whose shape is a compile-time proof, a static `8 -> 4`
layer that accepts only width-8 input, a forward pass, and a shape print.
Feed the layer a wrong width and it does not compile; ask a backend for an
operation it lacks and you get a typed refusal, never a wrong answer.

## What actually runs today

CPU is the complete backend, verified across the whole operation catalog.
The CUDA, WGPU, and Metal backends are previews: narrower operation
coverage, partial hardware evidence, and typed refusals where a kernel is
missing. [Backends](./backends.md) is the measured table — no aspiration
in it — and [What's not finished yet](./whats_not_finished.md) tracks the
gaps the book knows about. This book says what is unfinished rather than
describing design intent as if it shipped.

## Where to go next

**Start with the [Quickstart](./quickstart.md)**: from `cargo add incin` to
a trained model and a saved checkpoint in about thirty minutes, every
section a complete program you run yourself. Then the rest of the book:

- **Getting started** — [Installation](./installation.md), editor setups.
- **Core concepts** — the type-level shape system, autograd, and the error
  contract: the pieces that shape every other chapter.
- **Building models** and **Training** — layers, losses, optimizers, data
  loading, checkpoints.
- **Backends** — the honest chapter: what runs where, measured rather than
  assumed.
- **Reference** — the PyTorch cheatsheet, every feature flag, and the
  running list of known gaps.
- **Deep dive** — for when you want to know why, not just how: how a call
  becomes a kernel, where each guarantee comes from, and how to extend
  Incin with your own backends, devices, and dtypes.

Rust snippets are checked where they are wired into the book's doctest
suite or an executable fixture; snippets marked otherwise, and prose-only
examples, are not promised to be executed. Where else to look: the source
code, issue tracker, and releases are on
[GitHub](https://github.com/xupremix/incin), and the generated
`docs/OPERATION_SEMANTICS.md` and `docs/capabilities.md` are the exhaustive,
always-current per-operation reference this book does not try to duplicate.
