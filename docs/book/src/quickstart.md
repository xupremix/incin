# Quickstart

A tensor, some arithmetic, and a gradient — the shortest useful program:

```rust,no_run
use incin::prelude::*;

fn main() -> Result<()> {
    let a = Tensor::<s![2, 2], DefaultBackend>::ones(())?;
    let b = Tensor::<s![2, 2], DefaultBackend>::full(3.0, ())?;

    let c = a.mul(&b)?;
    let loss = c.sum_all()?;

    let grads = loss.backward()?;
    let grad_a = DefaultBackend::get_grad::<f32>(a.inner(), grads.as_backend())?
        .expect("a participated in the computation, so it has a gradient");

    println!("d(loss)/d(a) = {:?}", grad_a);
    Ok(())
}
```

`s![2, 2]` is a **type**, not a value — `a`'s shape is checked at compile
time. `a.mul(&b)` requires `b`'s shape type to match exactly (`ShapeEq`); a
`[2, 3]` operand there is a compile error, not a panic three lines later.

## A tiny model

```rust,no_run
use incin::prelude::*;

type Backend = DefaultBackend;

fn main() -> Result<()> {
    let layer = Linear::<s![8, 4], Backend>::build(())?;
    let x = Tensor::<s![2, 8], Backend>::ones(())?;

    let h = layer.forward(x)?;
    let h = ReLU.forward(h)?;

    assert_eq!(h.dims().as_ref(), &[2, 4]);
    Ok(())
}
```

`Linear<s![8, 4], Backend>` is a layer with a statically known `8 -> 4`
weight shape. Feed it anything but a `[.., 8]` input and, again, it's a
compile error.

## One training step

```rust,no_run
use incin::prelude::*;

type Backend = DefaultBackend;

fn main() -> Result<()> {
    let model = Linear::<s![4, 2], Backend>::build(())?;
    let x = Tensor::<s![3, 4], Backend>::ones(())?;
    let target = Tensor::<s![3, 2], Backend, f32, NoGrad>::zeros(())?;

    let mut optim = Adam::<Backend>::new(model.parameters(), 1e-2);

    let pred = model.forward(x)?;
    let loss = MSELoss::<Mean>::new().forward(&pred, &target)?;
    let grads = loss.backward()?;
    optim.step(&grads)?;

    Ok(())
}
```

That's the whole shape of a training step: forward, a loss module, `backward`,
`optimizer.step`. The rest of this book fills in the pieces — more layer
types, real datasets, schedulers, metrics, and checkpointing.
