# Quickstart

Thirty minutes from `cargo add incin` to a trained model and a checkpoint on
disk, hands on the whole way. Every section ends with one complete program:
paste it into `src/main.rs` and `cargo run` it. If the dependency is not in
your project yet, do [Installation](./installation.md) first — the default
CPU build is all this page needs.

```bash
cargo add incin
```

Everything below runs on **CPU**, the one backend Incin verifies across the
complete operation catalog. The accelerator backends are previews with
narrower coverage: [Backends](./backends.md) has the measured table, and
[What's not finished yet](./whats_not_finished.md) the honest gaps.

## 1. First tensor

A tensor's shape lives in its type: `shape![2, 3]` is a compile-time proof
of a 2 x 3 tensor, so `x`'s shape is settled before the program runs, and a
later mix-up fails where you wrote it rather than three epochs into a
training run. Checked arithmetic (`try_add`, `try_mul`, and friends) uses
the same broadcasting rules as the operators, while `add_exact`,
`mul_exact`, and their siblings request strictly equal shapes.

`Cpu` is the target-first way to allocate: name the device, get the tensor.
The program below builds a 2 x 3 tensor, broadcasts a length-3 bias across
both rows, and doubles the result.

### Run it

```rust
use incin::prelude::*;

fn main() -> Result<()> {
    let x = Cpu.tensor([[1.0_f32, 2.0, 3.0], [4.0, 5.0, 6.0]])?;
    let bias = Cpu.full(shape![3], 0.5)?;
    let y = &x + &bias; // [2, 3] + [3]: broadcast, checked
    let doubled = &y * 2.0;

    println!("y:\n{y}");
    println!("doubled:\n{doubled}");
    Ok(())
}
```

## 2. Autograd: one backward pass

`.require_grad()` marks a tensor for tracking; every operation on it records
a tape entry. `backward()` walks the tape from the loss and returns the
`Gradients` handle, `.require(&a)` fetches one tensor's gradient out of it,
and `.to_vec1::<f32>()` reads it back to the host.

The math is visible in the result: `loss = sum(a * b)`, so
`d(loss)/d(a)` is `b` everywhere — the printed gradient is `3.0` in every
slot, one per element of `a`.

Backward passes need gradient-tracking inputs, which is what
`.require_grad()` on `a` is for; tensors default to not tracking
(`NoGrad`), so tape entries are recorded only where you asked for them.
The [Autograd](./autograd.md) chapter covers scopes, what `backward`
consumes, and how gradients are stored.

### Run it

```rust
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

## 3. A tiny model

`Linear<s![8, 4], Backend>` is a layer whose weight shape is a static
`8 -> 4` fact: it accepts anything shaped `[.., 8]`, produces `[.., 4]`,
and every other width is a compile error, checked where you wrote it.

### Run it — a forward pass

```rust
use incin::prelude::*;

type Backend = DefaultBackend;

fn main() -> Result<()> {
    let layer = Linear::<s![8, 4], Backend>::build(())?;
    let x = Cpu.randn(shape![2, 8])?;

    let h = layer.forward(x)?;
    let h = ReLU.forward(h)?;

    assert_eq!(h.dims().as_ref(), &[2, 4]);
    println!("h: {:?}", h.dims());
    Ok(())
}
```

### A training loop in three lines

An optimizer built with `from_module` collects every trainable parameter the
model owns, keyed by name — one line, once, before the loop. After that a
step is exactly three lines, and that shape never changes from this model
to MNIST to a transformer: compute the loss, run `backward()`, hand the
gradients to `optim.step`. Collect fresh gradients for every step (one
`Gradients` value drives one step), which the loop below does by
construction because `backward()` runs again on each iteration. Watch the
loss fall from step to step. That is the whole shape of training — forward,
loss module, `backward`, `optimizer.step`; real runs differ in the data
feeding `x` and in the bookkeeping around the loop (schedulers, clipping,
metrics), not in these four lines.

### Run it — five steps

```rust
use incin::prelude::*;

type Backend = DefaultBackend;

fn main() -> Result<()> {
    let model = Linear::<s![4, 2], Backend>::build(())?;
    let mut optim = AdamW::<Backend>::from_module(&model, 1e-2)?;
    let x = Cpu.randn(shape![3, 4])?;
    let target = Cpu.zeros(shape![3, 2])?;

    for step in 0..5 {
        let pred = model.forward(x.clone().require_grad())?;

        // The whole training step, three lines:
        let loss = MSELoss::new().forward(&pred, &target)?;
        let grads = loss.backward()?;
        optim.step(&grads)?;

        println!("step {step}: loss = {:.4}", loss.to_scalar::<f32>()?);
    }
    Ok(())
}
```

## 4. Save it, and what's next

Checkpointing is one trait — `ModelExt`, in the prelude — over `save` and
`load` with an explicit `Format`. Loading is transactional: every path in
the file is validated against the module before anything is written, so a
snapshot missing a parameter is refused rather than half-applied. The
program below saves a trained layer, reloads it into a freshly built one,
and checks both produce identical output.

- [Training](./training.md) — losses, optimizers, schedulers, gradient
  clipping: the bookkeeping around the four lines above.
- [Building models](./building_models.md) — the layer catalog and
  `#[module]` for composing real networks.
- [Saving and loading](./saving_loading.md) — formats, sharded
  checkpoints, ONNX/GGUF export.
- [Shapes](./shapes.md) — static, dynamic, and mixed shapes in depth.
- [Data loading](./data_loading.md) feeds real batches where this page
  used fixed tensors; [Errors](./errors.md) is the contract behind every
  `?`.

### Run it

```rust
use incin::prelude::*;

type Backend = DefaultBackend;

fn main() -> Result<()> {
    let model = Linear::<s![4, 2], Backend>::build(())?;
    let path = std::env::temp_dir()
        .join(format!("incin_quickstart_{}.safetensors", std::process::id()));
    model.save(Format::Safetensors, &path)?;

    let mut reloaded = Linear::<s![4, 2], Backend>::build(())?;
    reloaded.load(Format::Safetensors, &path)?;

    let x = Tensor::<s![1, 4], Backend>::ones(())?;
    let before = model.forward(x.clone())?;
    let after = reloaded.forward(x)?;
    assert_eq!(before.to_vec1::<f32>()?, after.to_vec1::<f32>()?);

    println!("reloaded model matches the saved one");
    let _ = std::fs::remove_file(&path);
    Ok(())
}
```
