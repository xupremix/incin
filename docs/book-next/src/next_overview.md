# In development: start here

This book is the companion to [the stable book](../#/introduction). The
stable book documents what Incin can do today; this one documents what is
being built next, while it is being built: the preview backends closing
their operation gaps, the experimental execution surfaces growing into
stable ones, and the implementation stories behind each step.

## How to read these chapters

Every chapter here starts where a stable chapter stops. "WGPU: the
operation-gap campaign" assumes you have read
[CPU, and what actually runs on GPU today](../#/backends) and picks up at
the preview rows. "Distributed execution" assumes
[Distributed planning](../#/distributed) and covers what happens when plans
meet real ranks. Each chapter names the tracking issues, shows the actual
implementation shape with an example you can run or compile, and ends with
an honest boundary: what is proven, what is merely compiled, and what still
needs hardware, design work, or both.

Nothing here is a promise. A chapter describes the tree as it is, with the
rough edges visible. When an in-development surface stabilizes, its chapter
here shrinks to a pointer and the stable book gains the real documentation.
That migration — not a big-bang rewrite — is how this book stays honest.

## The map

| Chapter | Theme | Status | Tracking |
|---|---|---|---|
| [WGPU gaps](./next_wgpu_gaps.md) | indexing, masks, cross-entropy, attention on the software adapter | landing in batches, runtime-verified | #91 |
| [CUDA](./next_cuda.md) | fused GEMM epilogues, SDPA dispatch, dtype-parametric kernels | implemented, hardware-proven outside CI | #85, #90 |
| [Metal](./next_metal.md) | pointwise coverage, norms, SDPA; spatial still empty | host-verified, device runs on Apple Silicon | #92 |
| [Quantized training](./next_quantized_training.md) | Q8_0 straight-through on CPU/CUDA; fp8 and block formats next | STE done, formats in flight | #93, #94, #95 |
| [Distributed execution](./next_distributed_execution.md) | tensor-parallel matmul, FSDP/ZeRO lowering, NCCL transport | CPU-proven lowering; transport hardware-gated | #97, #99 |
| [Compiled graphs](./next_compiled_fusion.md) | legality-checked pointwise fusion, CPU reference evaluator | CPU half landed; device dispatch next | #112 |

Two more frontiers — [mixed-precision
autocast](../#/training) and streaming checkpoints — already live in the
working tree and are covered from the user's side in the stable book; their
remaining work is hardening and release hardening, tracked under #2 and
#13.

## Versions and truth

The two books share one site and one header switcher. The switcher keeps
you on the same chapter when both books carry it, and drops you at the
other book's introduction otherwise. The generated references are shared:
capability truth lives in `docs/capabilities.md` no matter which book you
are reading, and the API reference beside the stable book covers the same
tree.

If a Next chapter ever disagrees with the source it cites, the source is
right and the chapter is stale — file it as such. File paths are given as
`crate/path:line` so you can check every claim in under a minute.

A note on links for contributors: inside this book, chapters link to each
other as `./other_chapter.md` and to stable chapters as `../#/<slug>`. The
stable book never links here; it documents stable reality, and the switcher
is the only crossing.
