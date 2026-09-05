# Autograd

Gradient tracking is a type parameter, `G`, defaulting to `NoGrad`. A
`Tensor<S, B, K, Grad>` records a tape entry for the operations that produce
it; a `Tensor<S, B, K, NoGrad>` never does; the distinction is visible in
the type, not just at runtime.

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 2], B>::ones(())?.require_grad();
let b = Tensor::<s![2, 2], B>::full(3.0, ())?;

let c = &a * &b;
let loss = c.sum_all()?;

let grads = loss.backward()?;
let grad_a = grads.require(&a)?;
# Ok::<(), incin::Error>(())
```

`backward()` walks the tape from `loss` and returns a backend-typed
`Gradients` handle. Use `grads.get(&tensor)` for an optional gradient or
`grads.require(&tensor)` when the gradient is required. Both return an
ordinary detached tensor, so backend storage details stay out of model code.

## Turning tracking off

> Gradient recording scopes are explicit policy scopes. The facade does not
> expose a convenience alias for them.

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 2], B>::ones(())?.require_grad();
let b = Tensor::<s![2, 2], B>::ones(())?;

// Nothing inside this closure records a tape entry, regardless of what
// operations run or what G their operands carry.
let c = incin_core::exec::GradMode::Disabled.scope(|| &a * &b);
# Ok::<(), incin::Error>(())
```

`GradMode::Disabled.scope` is a scoped, thread-local override. It can only *tighten*
recording, never loosen it. An operation on an already-`NoGrad` tensor reads
no thread-local at all; the common `Grad`-tensor path is the one that
consults the scope.

For a single tensor rather than a whole closure, build it as `NoGrad`
directly instead:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let target = Tensor::<s![3, 2], B, f32, NoGrad>::zeros(())?;
# Ok::<(), incin::Error>(())
```

This is the usual shape for a loss function's `target` argument; see
[Training](./training.md): the label data isn't something you differentiate
with respect to, so it's typed `NoGrad` rather than merely happening to have
no gradient at runtime.

## `detach` and `require_grad`

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let a = Tensor::<s![2, 2], B>::ones(())?.require_grad();
let stopped: Tensor<s![2, 2], B, f32, NoGrad> = a.detach();
let resumed: Tensor<s![2, 2], B, f32, Grad> = stopped.require_grad();
# Ok::<(), incin::Error>(())
```

`detach` cuts a tensor off from the graph it was part of without changing its
values; `require_grad` does the reverse. Both change `G` in the type, so a
detached tensor accidentally used where a `Grad` one is expected is a compile
error, not a silently-missing gradient discovered during a debugging
session.

## One backward per graph

The tape is drained into the walk that consumes it: a second `backward()`
from the same loss finds an empty graph and returns only the seed. There is
no `retain_graph` or `create_graph`, and no second-order gradients — running
`backward()` twice, or differentiating through a gradient (gradient
penalties, meta-learning, Hessian-vector products), is not expressible yet.

In practice this means one forward per backward. Two optimizers (or a GAN's
discriminator and generator steps) sharing one loss must each run their own
forward pass first; reusing a spent `Gradients` matches nothing, and a step
that matches nothing is refused rather than silently committing. To train
through your own operations under these rules, see [Custom and fused
operations](./custom_operations.md).
