# Autograd

Gradient tracking is a type parameter, `G`, defaulting to `NoGrad`. A
`Tensor<S, B, K, Grad>` records a tape entry for the operations that produce
it; a `Tensor<S, B, K, NoGrad>` never does  -  the distinction is visible in
the type, not just at runtime.

```rust,no_run
use incin::prelude::*;
use incin::backend_authoring::AutogradBackend;
type B = DefaultBackend;

let a = Tensor::<s![2, 2], B>::ones(())?.require_grad();
let b = Tensor::<s![2, 2], B>::full(3.0, ())?;

let c = &a * &b;
let loss = c.sum_all()?;

let grads = loss.backward()?;
let grad_a = B::get_grad::<f32>(a.inner(), grads.as_backend())?
    .expect("a participated in the computation");
# Ok::<(), incin::Error>(())
```

`backward()` walks the tape from `loss` and returns `Gradients`;
`Backend::get_grad` reads one tensor's gradient back out of it.

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

This is the usual shape for a loss function's `target` argument  -  see
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
