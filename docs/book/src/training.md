# Losses, optimizers, and schedulers

## Losses

Every loss module's `forward` takes references, and the target argument is
`NoGrad` — the label data isn't something you differentiate with respect to,
and that's stated in the type rather than left as a convention to remember:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let pred = Tensor::<s![3, 2], B>::ones(())?;
let target = Tensor::<s![3, 2], B, f32, NoGrad>::zeros(())?;

let loss = MSELoss::<Mean>::new().forward(&pred, &target)?;
let loss2 = L1Loss::<Mean>::new().forward(&pred, &target)?;
# Ok::<(), incin::Error>(())
```

`Mean` is one of three reduction modes (`Mean`, `Sum`, `NoneReduction`) that
parameterize a loss's own type — `MSELoss<Sum>` sums instead of averaging,
and `MSELoss<NoneReduction>` returns the per-element loss unreduced, each a
different type rather than a runtime flag.

## Optimizers

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let model = Linear::<s![4, 2], B>::build(())?;

// `from_module` collects every trainable Param the model owns (recursively,
// through #[module] and Sequential), keyed by name.
let mut optim = AdamW::<B>::from_module(&model, 1e-2)?;

let x = Tensor::<s![3, 4], B>::ones(())?;
let target = Tensor::<s![3, 2], B, f32, NoGrad>::zeros(())?;

let pred = model.forward(x.require_grad())?;
let loss = MSELoss::<Mean>::new().forward(&pred, &target)?;
let grads = loss.backward()?;
optim.step(&grads)?;
# Ok::<(), incin::Error>(())
```

`SGD` and `AdamW` use `from_module(&model, lr)` for the normal module path, or
`from_group(ParameterGroup::from_module(&model)?, lr)` when a caller needs to
assemble a homogeneous group explicitly. Both use `step(&grads)`. The
learning rate is a public field (`optim.lr`), not hidden behind a setter.

## Learning rate schedulers

A scheduler doesn't own the optimizer; it tracks its own state and you copy
its current value across after each step:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let model = Linear::<s![4, 2], B>::build(())?;
let mut optim = AdamW::<B>::from_module(&model, 1e-2)?;
let mut sched = CosineAnnealingLR::new(1e-2, 1e-4, 100); // start, min, total steps

let x = Tensor::<s![3, 4], B>::ones(())?;
let target = Tensor::<s![3, 2], B, f32, NoGrad>::zeros(())?;

for _step in 0..3 {
    let pred = model.forward(x.clone().require_grad())?;
    let loss = MSELoss::<Mean>::new().forward(&pred, &target)?;
    let grads = loss.backward()?;
    optim.step(&grads)?;

    sched.step();
    optim.lr = sched.get_lr();
}
# Ok::<(), incin::Error>(())
```

`ConstantLR`, `LinearLR`, `CosineAnnealingLR`, and `StepLR` all implement the
same `LRScheduler` trait (`get_lr() -> f64`, `step(&mut self)`), so swapping
one for another is a one-line change.

## The whole loop

Putting it together — model, loss, optimizer, scheduler, several steps:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

fn train() -> Result<()> {
    let model = Linear::<s![4, 2], B>::build(())?;
    let mut optim = AdamW::<B>::from_module(&model, 1e-2)?;
    let mut sched = CosineAnnealingLR::new(1e-2, 1e-4, 10);

    let x = Tensor::<s![8, 4], B>::rand(())?;
    let target = Tensor::<s![8, 2], B, f32, NoGrad>::zeros(())?;

    for epoch in 0..10 {
        let pred = model.forward(x.clone().require_grad())?;
        let loss = MSELoss::<Mean>::new().forward(&pred, &target)?;
        let grads = loss.backward()?;
        optim.step(&grads)?;
        sched.step();
        optim.lr = sched.get_lr();

        let value = loss.to_scalar::<f32>()?;
        println!("epoch {epoch}: loss = {value}");
    }
    Ok(())
}
# fn main() -> Result<()> { train() }
```

For a real dataset instead of a fixed tensor, see [Data
loading](./data_loading.md) — the loop shape is identical, just with a
`DataLoader` iteration in place of the fixed `x`.
