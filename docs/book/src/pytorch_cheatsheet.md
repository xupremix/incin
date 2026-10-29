# Coming from PyTorch

This page is the cross-reference gateway, not a tutorial: every row maps one
PyTorch spelling to its Incin counterpart and points at the chapter that
explains it. Every Incin spelling below was checked against the source before
it was written; where a call has no example anywhere in the workspace, the row
says so instead of inventing one.

Two habits carry over directly: `?` is on almost everything (see
[Errors](./errors.md)), and the training loop has the same shape as PyTorch's.

Three do not:

- **Shapes, dtypes, and gradient tracking are type parameters.** `s![768, 256]`,
  `f64`, `Grad`/`NoGrad` are part of the type, so a class of mistakes you find
  at runtime in PyTorch is found at compile time here.
- **Failures are values.** Ordinary bad input returns a typed `Error`; it does
  not raise an exception, and operator syntax is the one deliberate panic
  boundary.
- **`backward()` hands you a value.** There is no `.grad` attribute to
  accumulate into and therefore no `zero_grad()`.

## Concept mapping

### Tensors, shapes, and dtypes

| PyTorch | Incin | Note |
|---|---|---|
| `torch.tensor([1, 2, 3])` | `tensor![1, 2, 3]?` | Integer literals default to `i64`, the same default `torch.tensor` has. `target.tensor([1_i64, 2, 3])?` is the target-first form. |
| `torch.zeros(2, 3)` | `Cpu.zeros(shape![2, 3])?` | Concrete code targets first; generic code uses `Tensor::<s![2, 3], B>::zeros(())?`. |
| `torch.randn(2, 3)` | `Cpu.randn(shape![2, 3])?` | Also `rand`, `ones`, `full`, `arange`, `linspace`, `from_slice`. See [Tensors](./tensors.md). |
| `torch.zeros(2, 3, dtype=torch.float64)` | `Cpu.dtype::<f64>()?.zeros(shape![2, 3])?` | Rebinding the target, or the `K` type parameter: `Tensor::<s![2, 3], B, f64>::zeros(())?`. |
| `torch.zeros(2, 3, device=dev, dtype=dt)` | `target.zeros([2, 3])?` with `target: Target<Native, Dyn>` | Runtime-chosen device and dtype live on the `Target`, not on each call. See [The target API](./target_api.md). |
| `x.shape` | `x.dims()` | A `ShapeBuf`; `x.dims().as_ref()` is `&[usize]`. |
| `x.numel()` | `x.numel()` | |
| `x.dtype` | `x.dtype()`, or the `K` type parameter | Runtime: a `DTypeDescriptor`. Compile time: `K`. |
| `x.to(torch.float64)` | `x.to_dtype::<f64>()?` | Allocating in a dtype and *executing* in one are separate questions; see the capability refusal in [Tensors](./tensors.md). |
| `x.reshape(...)` / `x.view(...)` | `x.reshape(shape![3, 4])?` | `reshape` builds a new geometry; `to_shape` re-asserts a shape *type* over the same dims and fails if they disagree. |
| `torch.cat([a, b], dim=0)` | `a.concat(&b, axis!(0))?` | |
| `torch.stack([a, b], dim=0)` | `a.stack(&b, axis!(0))?` | |
| `x[0]` / `x[:, 1:3]` | `x.get(i![0, ..])?` / `x.get(i![.., 1..3])?` | Runtime indexing yields a `Dyn` result; the shape you proved is not carried forward. |

### Gradients

| PyTorch | Incin | Note |
|---|---|---|
| `x.requires_grad_()` | `x.require_grad()` | Consumes `x` and returns a `Grad` tensor. The change is in the type, not a runtime flag. |
| `x.detach()` | `x.detach()` | Returns a `NoGrad` tensor with the same values. |
| `with torch.no_grad():` | `incin_core::exec::GradMode::Disabled.scope(\|\| { ... })` | Thread-local, and it can only *tighten* recording. There is deliberately no facade alias for it. |
| `loss.backward()` | `let grads = loss.backward()?;` | Only on `Tensor<..., Grad, ...>` with a scalar (rank-0) shape; returns `Gradients<B>`. |
| `param.grad` | `grads.require(&param)?` | Explicit lookup by tensor. `grads.get(&param)?` is the optional form; neither is an attribute on the parameter. |
| `optimizer.zero_grad()` | — | Not needed. Each `backward()` produces a fresh `Gradients` value; nothing accumulates on the parameter. |
| `torch.autograd.backward(loss, grad_outputs=seed)` | `loss.backward_with(&seed)?` | Signature only — no call site in the workspace; the seed must match the loss's own shape. |
| `retain_graph=True` / `create_graph=True` | — | Not present. A second `backward()` from the same loss returns `BackwardError::GraphConsumed`, and there are no second-order gradients. |

### Layers and modules

| PyTorch | Incin | Note |
|---|---|---|
| `nn.Linear(768, 256)` | `Linear::<s![768, 256], B>::build(())?` | In/out features are the shape type, not constructor arguments. The layer only accepts `[.., 768]` input and only produces `[.., 256]` output. |
| `nn.Sequential(a, b, c)` | `seq!(a, b, c)` with `type T = SeqTy!(A, B, C)` | Two layers can be `Sequential<A, B>` directly; `seq!`/`SeqTy!` right-nest anything longer. See [Sequential](./sequential.md). |
| `F.relu(x)` / `nn.ReLU()` | `x.relu()?` / `ReLU.forward(x)?` | Both exist. The method is the short form; the unit-layer form is what a `seq!` list holds. |
| `F.gelu(x)` / `F.sigmoid(x)` | `x.gelu()?` / `x.sigmoid()?` | Same pattern for `tanh`, `swish`, `mish`, `elu`, `relu`. A `#[module]`-free call still returns `Result`. |
| `nn.Dropout(0.1)` | `Dropout::new(0.1)` | Mode-sensitive: it needs `model.train()` / `model.eval()` to change behavior. |
| `nn.Flatten()` | `Flatten::new(axis!(1), axis!(-1))` | For a statically known range, `x.flatten(axis!(1), axis!(-1))?` keeps the exact output shape. |
| `nn.Embedding(16, 4)` | `Embedding::<s![16, 4], B>::build(())?` | Shape is `(Vocab, EmbedDim)`. The index tensor's element type matches the layer's `K`, so indices are written as integer-valued floats. |
| `nn.LayerNorm(8)` | `LayerNorm::<s![8], B>::build(1e-5_f32)?` | The epsilon is the `build` argument the static shape did not already fix. |
| `nn.BatchNorm2d(4)` | `BatchNorm2d::<s![4], B>::build((1e-5_f32, 0.1_f32))?` | `(eps, momentum)`. |
| `nn.Conv2d(1, 4, 3)` | `Conv2d::<s![4, 1, 3, 1, 0, 1], B>::build(())?` | Six-wide shape: `(Out, In, Kernel, Stride, Padding, Dilation)`, not positional constructor args. |
| `class MyNet(nn.Module)` | `#[module] struct MyNet { ... }` with `fn forward(&self, x) -> Result<...>` | `#[module]` derives parameter, state, mode, and device visitors by walking fields. See [Layers and `#[module]`](./building_models.md). |
| `model.parameters()` | `ParameterGroup::<B, f32>::from_module(&model)?` | Or let `from_module` on the optimizer collect them for you. |
| `y = self.W(x)` reused across Q/K/V | `self.q.forward(x.clone())?` | `Module::forward` takes its input by value, so every extra use of one tensor needs `.clone()` — a reference-count bump, not a data copy. Losses are the exception: they take `&pred, &target`. |

### The training loop

| PyTorch | Incin | Note |
|---|---|---|
| `optim.SGD(model.parameters(), lr=0.1)` | `SGD::<B>::from_module(&model, 0.1)?` | `from_group(ParameterGroup::from_module(&model)?, lr)` is the explicit-group form. |
| `optim.AdamW(model.parameters(), lr=1e-3)` | `AdamW::<B>::from_module(&model, 1e-3)?` | Parameters are collected through the module visitor, keyed by name. |
| `optimizer.step()` | `optim.step(&grads)?` | Takes the `Gradients` explicitly instead of reading accumulated `.grad` fields. A step that reaches *no* parameter is an error, not a silent no-op. |
| `optimizer.zero_grad()` | — | See the gradients table above. |
| `scheduler.step()` | `sched.step(); optim.lr = sched.get_lr();` | The scheduler does not own the optimizer; `lr` is a public field you copy into. |
| `nn.MSELoss()(pred, target)` | `MSELoss::new().forward(&pred, &target)?` | Same default reduction (`Mean`). `MSELoss::<Sum>::with_reduction()` picks another; `Mean`, `Sum`, `NoneReduction` live in `incin::nn`. |
| `F.cross_entropy(logits, labels)` | `logits.cross_entropy_loss(&labels)?` | Labels are `i64` class indices, as in PyTorch. `CrossEntropyLoss::new().forward(&logits, &labels)?` is the module form. |
| `torch.nn.utils.clip_grad_norm_(params, 1.0)` | `clip_grad_norm(&group, &mut grads, 1.0)?` | Returns the norm *before* rescaling — the number worth logging. Clip before `step`. |
| `torch.save(model.state_dict(), "m.pt")` | `model.save(Format::Safetensors, Path::new("m.safetensors"))?` | safetensors, not pickle. See [Saving and loading](./saving_loading.md). |
| `model.load_state_dict(torch.load("m.pt"))` | `model.load(Format::Safetensors, Path::new("m.safetensors"))?` | Transactional: every path is checked before a write commits, so a partial load cannot happen. |
| `DataLoader(dataset, batch_size=4)` | `DataLoader::builder(dataset).batch_size(4).build()?` | Builder setters are infallible; validation happens in `build`. Scalar samples become `Vec<T>`, tuples collate field-wise. |

### Modes, devices, and inference

| PyTorch | Incin | Note |
|---|---|---|
| `model.train()` / `model.eval()` | `model.train()` / `model.eval()` / `model.set_training(bool)` | The `TrainMode` trait, implemented recursively by `#[module]`. `Dropout` responds; `Linear` and friends are zero-cost no-ops. |
| `with torch.no_grad(): ...` | `incin_core::exec::GradMode::Disabled.scope(\|\| ...)` | A **different axis** from `train()`/`eval()`: mode changes what layers do, `GradMode` changes what the tape records. You often want both. |
| `torch.device("cuda" if ... else "cpu")` | `let target: Target<Native, Dyn> = Target::new((), detect_device().unwrap_or_else(DeviceId::cpu), ());` | `Target`/`Native` come from `incin_backends::target`, `detect_device` from `incin_backends::detect`. It probes the machine and returns a `DeviceId`, which the target then carries. See [Backends](./backends.md). |
| `x.to(device)` | `x.to_device(&device_arg)?` | Device is part of the type on the receiving end: the result backend is `<B as StorageTransfer<D2>::Output>` and the result is `NoGrad`. |
| `model.to(device)` | `model.to_device(&device_arg)?` | The module-level `ToDevice` trait (in the prelude) hands back a relocated model rather than mutating one. |
| `pin_memory()` / non-blocking copies | — | Not present. There is no host-pinned staging path; you read values back with `to_vec1`/`to_scalar`, which allocates on the host. |

`incin_core::exec::ExecutionPolicy::with_training(true)` is **not** a
PyTorch-shaped API and is not the counterpart of `model.train()`. It sets the
execution policy's *training* capability flag, which is what a capability row
declaring `training = false` refuses (`"training is unsupported for quantize"`,
[#93](./quantization.md)) — a claim about which kernels were measured, not a
model mode.

## API recipes

Real signatures. Where a spelling has no call site in the workspace, the row
says "signature only" rather than showing you a snippet that has never run.

| PyTorch | Incin | Note |
|---|---|---|
| `a @ b` / `torch.matmul(a, b)` | `a.matmul(&b)?` | Broadcasts batch axes; the contracting pair must agree, statically when the shapes are static. |
| `a.t()` / `a.transpose(0, 1)` | `a.transpose(axis!(0), axis!(1))?` | Static, named, and signed selectors all share one axis argument. |
| `x.view(3, 4)` / `x.view(-1, 4)` | `x.reshape(shape![3, 4])?` | A literal extent stays static and provable. A runtime extent is a `usize` *value* in the macro (`shape![n, 4]`, `n: usize`) — the word `usize` is not one. For `-1`, the spelled form is `x.reshape_infer(InferShape::<Dyn>::new(vec![None, Some(4)]))?`. |
| `x.flatten(1, -1)` | `x.flatten(axis!(1), axis!(-1))?` | |
| `torch.cat([a, b], dim=0)` | `a.concat(&b, axis!(0))?` | Two operands per call; chain for more. |
| `torch.stack([a, b], dim=0)` | `a.stack(&b, axis!(0))?` | Inserts a new axis of extent 2. |
| `torch.chunk(x, 2, dim=0)` / `torch.split(x, 2, 0)` | `x.chunk(2, axis!(0))?` / `x.split(2, axis!(0))?` | Count first, then axis. |
| `x[0]` / `x[:, 1:3]` / `x[..., -1]` | `x.get(i![0, ..])?` / `x.get(i![.., 1..3])?` / `x.get(i![.., .., -1])?` | `i!` supports negative indices and signed ranges; the result is `Dyn`. |
| `x.narrow(1, 2, 3)` | `x.try_narrow(1isize, 2, 3)?` | Takes `self` by value, so the original is moved. |
| `x.squeeze(1)` / `x.unsqueeze(0)` | `x.try_squeeze(1isize)?` / `x.unsqueeze(axis!(0))?` | `try_squeeze` consumes; `unsqueeze` borrows. |
| `torch.index_select(x, 0, idx)` | `x.index_select(axis!(0), &idx)?` | `idx` is an integer tensor. |
| `torch.gather(x, 1, idx)` | `x.gather(axis!(1), &idx)?` | Accepts the `u32` indices that `topk` and `argsort` return. |
| `torch.topk(x, 2, dim=1)` | `x.topk(2, axis!(1), true)?` | `(values, indices)`, both `NoGrad`. |
| `x.argmax(dim=1)` | `x.argmax(axis!(1))?` | Returns `u32` and **drops** the axis; there is no `keepdim` variant of `argmax`. |
| `a + b` (broadcast) | `a + b`, `a.try_add(&b)?`, or `a.broadcast_add(&b)?` | Three spellings, three failure modes: operator panics on a dynamic/backend error, `try_add` returns it, `broadcast_add` refuses at compile time when the shapes cannot broadcast. |
| `x.expand(...)` / `x.repeat(...)` | `x.broadcast_to::<Dyn>(vec![4, 3])?` / `x.repeat(&[2])?` | `expand` is an alias of `broadcast_to`; the target must be a broadcast extension of the source. The static-target form takes the shape type's `Arg`, which is a nested unit tuple — `broadcast_to::<s![4, 3]>(((), ((), ())))` — so the `Dyn` spelling is the readable one. `repeat` tiles an axis; `repeat_interleave(2, 0)` puts copies of each element next to it. |
| `x.to(torch.float64)` | `x.to_dtype::<f64>()?` | |
| `x.sum(dim=1)` / `x.sum(dim=1, keepdim=True)` | `x.sum(axis!(1))?` / `x.sum_keepdim(axis!(1))?` | Same split for `mean` → `mean` / `mean_keepdim`. |
| `x.sum()` / `x.mean()` | `x.clone().sum_all()?` / `x.clone().mean_all()?` | These consume the tensor — clone first if you still need it. |
| `F.softmax(x, dim=-1)` / `F.log_softmax(x, dim=-1)` | `x.softmax(axis!(-1))?` / `x.log_softmax(axis!(-1))?` | |
| `F.cross_entropy(logits, y)` | `logits.cross_entropy_loss(&y)?` | Reduction is a type parameter: `cross_entropy_loss_with::<NoneReduction, _, _, _, _>(&y)?`. |
| `F.mse_loss(p, t)` | `p.mse_loss(&t)?` | `mse_loss_with::<Sum, _, _, _>` for another reduction. |
| `F.pad(x, (1, 1, 1, 1))` | `x.pad(&[(1, 1), (1, 1)], 0.0)?` | One `(before, after)` pair **per axis, in axis order** — not PyTorch's last-axis-first pad tuple. |
| `F.embedding(idx, weight)` | `emb.forward(idx)?` where `emb = Embedding::<s![V, D], B>::build(())?` | `idx` is written as integer-valued floats (the layer's own `K`), not an integer dtype. |
| `torch.where(mask, a, b)` | `mask.where_cond(&a, &b)?` | The receiver is the **condition**. The mask may broadcast into the data; one that would enlarge it does not compile. |
| `x.masked_fill(mask, 0.0)` | `x.masked_fill(&mask, 0.0)?` | Same directional rule: output keeps the input's shape type. |
| `torch.clamp(x, -1.0, 1.0)` | `x.clamp(-1.0, 1.0)?` | |
| `F.one_hot(idx, num_classes)` | `idx.one_hot::<N>()?` | The result is a **`bool`** tensor, not `int64` as PyTorch's is. |
| `torch.sort(x, dim=0)` / `torch.argsort(x, dim=0)` | `x.sort(0, false)?` / `x.argsort(0, false)?` | These two take a plain `usize` axis, not `axis!(...)` and not a negative one — unlike every other row here. Stable in both directions. |
| `a == b` | `a.eq(&b)?` | Returns a `bool` tensor. |

The rows above, spelled out:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 3], B>::ones(())?;
let b = Tensor::<s![3, 2], B>::ones(())?;

// matmul, transpose, reshape
let product = a.matmul(&b)?;                       // a @ b
let transposed = a.transpose(axis!(0), axis!(1))?; // a.t()
let same = a.clone().reshape(shape![3, 2])?;       // a.view(3, 2)
let flat = a.reshape_infer(InferShape::<Dyn>::new(vec![None, Some(3)]))?; // a.view(-1, 3)

// reductions and softmax
let by_row = a.sum_keepdim(axis!(1))?;             // a.sum(1, keepdim=True)
let scored = product.softmax(axis!(-1))?;          // F.softmax(product, dim=-1)
let guess = a.argmax(axis!(1))?;                   // a.argmax(dim=1) -> u32

// stacking and concatenation
let joined = a.concat(&a, axis!(0))?.concat(&a, axis!(0))?; // torch.cat([a, a, a], 0)
let stacked = a.stack(&a, axis!(0))?;              // torch.stack([a, a], 0)

// slicing and selection
let first = a.get(i![0, ..])?;                     // a[0]
let window = a.get(i![.., 1..3])?;                 // a[:, 1:3]
let cut = a.clone().try_narrow(0isize, 1, 1)?;     // a.narrow(0, 1, 1)
let top = a.topk(1, axis!(1), true)?;              // a.topk(1, dim=1)
let one_hot = guess.one_hot::<4>()?;               // F.one_hot(guess, 4) -> bool

// dtype and broadcast
let doubles = a.to_dtype::<f64>()?;                // a.to(torch.float64)
let wider = a.try_add(&Tensor::<s![3], B>::ones(())?)?; // a + b, recoverable
let expanded = Tensor::<s![3], B>::ones(())?.broadcast_to::<Dyn>(vec![4, 3])?; // x.expand(...)
let expanded_static = Tensor::<s![3], B>::ones(())?.broadcast_to::<s![4, 3]>(((), ((), ())))?;
let clamped = wider.clamp(-1.0, 1.0)?;             // torch.clamp(wider, -1., 1.)
# Ok::<(), incin::Error>(())
```

A full training loop is on [the training chapter](./training.md); the loop
shape is identical to PyTorch's, with `loss.backward()?` producing the value
that `optim.step(&grads)?` consumes.

## Where Incin deliberately differs

### Compile-time shape proofs, not runtime shape checks

A shape type is a proof. `Tensor<s![2, 3], B>` and `Tensor<s![3, 2], B>` are
different types, so the pair that would raise a `RuntimeError` in PyTorch does
not compile here:

```rust,compile_fail
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 3], B>::ones(())?;
let b = Tensor::<s![3, 2], B>::ones(())?;
let c = &a + &b; // does not compile: [2, 3] and [3, 2] do not broadcast together
# Ok::<(), incin::Error>(())
```

The same is true of a layer fed the wrong feature count — `Linear::<s![3, 4],
B>` given a `s![2, 5]` input is refused at `forward` with `its last dimension
must be ...` (the fixture is
`crates/incin-core/tests/compile_fail/forward_linear_static_mismatch.rs`):

```rust,compile_fail
use incin::prelude::*;
type B = DefaultBackend;

let layer = Linear::<s![3, 4], B>::build(())?;
let input = Tensor::<s![2, 5], B>::zeros(())?; // 5 != 3
let _ = layer.forward(input);
# Ok::<(), incin::Error>(())
```

[Shapes](./shapes.md) has the whole system; [Advanced shapes](./advanced_shapes.md)
has broadcasting, `reshape` vs `to_shape`, and slicing.

### `Dyn` when you cannot prove it — and only then

`s![2, 3]` is a type, `Dyn` is "nothing established yet", and a mixed shape
like `s![usize, 128]` is the middle ground where only some axes are runtime
values. `Dyn` is not a worse `Tensor`; it is an honest one. Two conversions
connect them: `into_dyn()` always succeeds, `to_shape::<s![2, 3]>()?` checks
at runtime and returns a typed error when the dims disagree. Code that is
generic over a batch size uses `dim!(Batch)` rather than lying with a
constant.

### Typed errors instead of exceptions

Every fallible operation returns `Result<T, Error>` — a dimension mismatch, an
out-of-range index, and an unsupported backend operation are three distinct
variants rather than three strings. Backend refusals are typed too
(`UnsupportedReason::DType`, `::Layout`, `::Rank`), which is what makes
backend gaps queryable instead of only readable. See [Errors](./errors.md).

One deliberate exception: operator syntax (`+`, `-`, `*`, `/`) is a
convenience boundary and panics with the fixed message
``incin tensor operator `+` failed`` — no error contents, no tensor contents.
Reach for `try_add` / `try_sub` / `try_mul` / `try_div` whenever the failure
must stay recoverable, and for `add_exact` and friends when the operands are
required to match exactly.

### Quantization is a block dtype with an STE gradient (#93)

PyTorch has no per-tensor `quantize` in autograd; its QAT path is
`FakeQuantize`. Incin exposes the block conversion as tensor operations —
`quantize(axis)` and `dequantize::<Kout>()` — and they record a tape entry
whose backward passes the cotangent through unchanged: the straight-through
estimator, documented as an approximation everywhere it appears (#93,
Decision 2). `dequantize(quantize(x))` therefore has exactly the identity
backward.

```rust,no_run
use incin::prelude::*;

// 64 elements on the only axis: two whole Q8_0 blocks.
let x = Cpu.zeros(shape![64])?;
let q = x.quantize(-1)?;          // same shape, dtype Q8_0
let back = q.dequantize::<f32>()?; // lossy: the f16 block scale rounds
# Ok::<(), incin::Error>(())
```

The rest of the contract is [the quantization chapter](./quantization.md):
`Q8_0` is the only quantized representation in 0.2.0, admission is per
operation (#93, Decision 8) rather than a blanket wall, and block divisibility
is proved twice — a `const { assert! }` at monomorphization for static shapes,
a typed error for `Dyn`.

### No in-place mutation, no view aliasing

There is no `x.add_(y)`, no `relu_()`, and no write-through view. In-place
mutation and aliasing are listed under "What stays future" in
[Deep autograd](./deep_autograd.md): ownership, views, allocation identity,
and autograd versioning need their own design first. Operations return new
tensors, and `.clone()` — the call you make when `forward` takes its input by
value — is a reference-count bump on shared storage, not a copy of the data.

## Gotchas

| You see | Why | Fix |
|---|---|---|
| ``error[E0277]: the trait bound `Q8_0: FloatCapable` is not satisfied`` | Per-operation dtype admission (#93, Decision 3): 25 unary methods are bounded on `FloatCapable`, and `Q8_0` deliberately does not implement it — a block has no per-element `f32` to read. | `dequantize::<f32>()?` first, or don't call that method on quantized storage. The same shape of error, `Q8_0: QuantCapable`, refuses `dequantize` on a float. |
| `error[E0080]: evaluation panicked: the last axis of a Q8_0 block must be a multiple of 32` | Static block-divisibility proof, checked at monomorphization (#93, Decision 4). | Make the last axis a whole multiple of 32 — reshape or pad. The error points at the `quantize` call site, not at a kernel. |
| `quantize: axis 1 has extent 48, which is not a multiple of the Q8_0 block size 32; ...` | The same rule, when the extent is only known at run time (`Dyn`), so it cannot be a compile error. | Same fix; the message names the axis, the actual extent, and the required multiple. |
| `backend 'Cpu' refused the request: dtype i64 is unsupported for add` | The capability tables, the error text, and the kernel are generated from one source, so an operation with no kernel for that dtype refuses instead of promoting silently. | Cast with `to_dtype::<f32>()?`, or ask the registry first — `incin_core::exec::CapabilityQuery`, which is what `cargo incin doctor` does. |
| `error[E0277]: ... its last dimension must be ...` on `layer.forward(x)` | The layer's weight shape is part of its type: `Linear<s![768, 256], B>` accepts only `[.., 768]`. | Reshape the input, or change the layer's shape parameter so the two agree in one place. |
| `error[E0308]: mismatched types` on `t1.add_exact(&t2)` | `*_exact` operations require `ShapeEq`; there is no implicit rebroadcast. | Use `try_add`/`+` to broadcast, or make the shapes equal. |
| ``panic: incin tensor operator `+` failed`` | Operator syntax is the documented panic boundary: a fixed message with no error details. | `a.try_add(&b)?` (recoverable) or `a.broadcast_add(&b)?` (compile-time). |
| `backward pass found no recorded operations: the tape is empty or was already drained` | The tape is drained by the walk that consumes it; there is no `retain_graph`. This is also what a `backward()` on a *different thread* from the forward pass produces, because the tape is thread-local. | One forward per backward, on one thread. Run two optimizers on one loss by doing two forward passes. |
| `optimizer.step()` refuses a `Gradients` that matches no parameter | Nearly always a `Gradients` from a different model, or from a `backward` that ran elsewhere. | Compute `grads` from this model's forward pass. Note also that a committed step reassigns parameter storage, so old `Gradients` cannot be reused for a second step. |
| ``unsupported safetensors dtype q8_0`` when saving | The safetensors writer refuses block dtypes (it always has); postcard round-trips them. | `Format::Postcard` for a module whose parameters are `Q8_0`. |
| `training is unsupported for quantized_matmul` (or for `quantize` on CUDA) | The capability row's `training = false` gates `ExecutionPolicy`'s training flag: `quantized_matmul` is `false` on every backend and all three quantized rows are `false` on CUDA, while CPU's `quantize`/`dequantize` rows are `true` and admit the flag. | Drop the flag — or run the boundary on CPU. It says nothing about differentiability — the STE tape node is gated by gradient recording instead (#93). |
| `Device mismatch: left .., right ..` | Two backends are two different *types*, so most of the time this arrives as `error[E0308]` naming both backend types rather than as a runtime message; a runtime mismatch inside one backend renders as `Device mismatch: left …, right …`. | Move one operand with `.to_device(..)`, or fix the construction so both operands have the same backend type. |
| `error[E0277]: ... ChannelsLast ... Contiguous` on `reshape_view` | A view over strided storage would alias, and aliasing is not implemented. | Use `reshape`, which materializes. |
| Output of `x.get(i![...])` is `Tensor<Dyn, ...>` and now nothing type-checks | Runtime indexing cannot prove its result's extents. | `to_shape::<s![..]>()?` to re-assert a static shape (checked at run time), or keep the dynamic shape and let downstream code use `usize` axes. |

## What 0.2.0 does not have

Stated plainly so nothing above has to hedge. Verified against
[What's not finished yet](./whats_not_finished.md), with the quantization
scope in [Quantization](./quantization.md):

- **No `torch.compile`-style graph compiler.** "Compiled execution" in Incin
  is `incin::experimental::compiled`, a CPU reference evaluator with no stable
  facade contract, no optimized backend, no deployment target, and no portable
  artifact ABI. The `OperationSpec` IR exists; there is no graph fusion pass
  over it.
- **No TorchScript or FX export path**, and no Python runtime at all. The
  import/export surfaces that do exist are ONNX (`model!` / `import_model!`,
  feature-gated, fail-closed, feed-forward subset only — see
  [Experimental](./experimental.md)) and GGUF (export only).
- **No flash/online-softmax attention.** Evaluation materializes scores; see
  [What's not finished yet](./whats_not_finished.md) for exactly which paths
  are fused and which are composed.
- **No `retain_graph`, `create_graph`, second-order gradients, activation
  checkpointing, or in-place mutation.**
- **GPU training is partial evidence, not a hardware proof.** CPU is the only
  backend verified across the complete catalog; CUDA value tests are
  `#[ignore]`d without `HARDWARE_CUDA_RUNNER`.
- **Distributed**: one executed tier (FSDP/ZeRO-1, ZeRO-2 on CPU against
  scripted peers); ZeRO-3, tensor parallel, and pipeline parallel are planning
  surfaces.
- **Quantization** (0.2.0 scope, #93): no GPTQ/AWQ or any post-training
  search, no clip-based STE, no quantized parameter initialization, no
  Q4_0/NVFP4/MXFP4 tensor dtypes, no `quantize`/`dequantize` on WGPU or Metal.

If a claim in this page ever disagrees with `docs/capabilities.md`,
`docs/OPERATION_SEMANTICS.md`, or the compile-fail fixtures in
`crates/incin-core/tests/compile_fail/`, the generated document and the
fixtures are right and this page is stale — file it.

## The trade

The biggest structural difference is not any one API. Shape and
dtype mismatches, tensors that shouldn't require a gradient, and layers fed
the wrong feature count are, as much as possible, compile errors here rather
than runtime exceptions. Code that "just runs" in PyTorch because Python
doesn't check any of that ahead of time often needs its shapes made
explicit, by writing `s![768, 256]` rather than trusting two `768`s a hundred
lines apart to agree, to compile in Incin at all. That's the trade this
library is built around, not a friction to work around.
