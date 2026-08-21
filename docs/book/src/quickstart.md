# Quickstart

A tensor, some arithmetic, and a gradient - the shortest useful program:

```rust,no_run
use incin::prelude::*;

fn main() -> Result<()> {
    let a = Cpu.ones(shape![2, 2])?.require_grad();
    let b = Cpu.full(shape![2, 2], 3.0)?;

    let c = &a * &b;
    let loss = c.sum_all()?;

    let grads = loss.backward()?;
    let grad_a = grads
        .require(&a)?
        .to_vec1::<f32>()?;

    println!("d(loss)/d(a) = {:?}", grad_a);
    Ok(())
}
```

`shape![2, 2]` produces a statically-checked type-level shape proof. `a`'s shape is checked at compile
time. Checked arithmetic uses the same broadcasting rules as ordinary
operators, while `add_exact`, `mul_exact`, and their siblings request strict
equal-shape behavior explicitly.

## A tiny model

```rust,no_run
use incin::prelude::*;

type Backend = DefaultBackend;

fn main() -> Result<()> {
    let layer = Linear::<s![8, 4], Backend>::build(())?;
    let x = Cpu.randn(shape![2, 8])?;

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
    let x = Cpu.randn(shape![3, 4])?;
    let target = Cpu.zeros(shape![3, 2])?;

    let mut optim = AdamW::<Backend>::from_module(&model, 1e-2)?;

    let pred = model.forward(x.require_grad())?;
    let loss = MSELoss::<Mean>::new().forward(&pred, &target)?;
    let grads = loss.backward()?;
    optim.step(&grads)?;

    Ok(())
}
```

That's the whole shape of a training step: forward, a loss module, `backward`,
`optimizer.step`. The rest of this book fills in the pieces - more layer
types, real datasets, schedulers, metrics, and checkpointing.
