# Run a training loop

A training loop in Incin is four moves: record a forward pass, walk it
backwards, hand the resulting gradients to an optimizer, and let the optimizer
commit the update. This chapter is the doing half of [Losses, optimizers, and
schedulers](./training.md) — complete programs plus the exact refusal you get
when one of the four moves is wrong.

Everything below compiles. Failure snippets are written so you can see the
message Incin prints before you hit it.

## 1. Record, backprop, step

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

fn train_step(model: &Linear<s![4, 2], B>, optim: &mut AdamW<B>) -> Result<f32> {
    let x = Tensor::<s![8, 4], B>::rand(())?;
    let target = Tensor::<s![8, 2], B, f32, NoGrad>::zeros(())?;

    let pred = model.forward(x.require_grad())?;
    let loss = MSELoss::new().forward(&pred, &target)?;
    let grads = loss.backward()?;
    optim.step(&grads)?;
    loss.to_scalar::<f32>()
}
# fn main() -> Result<()> {
#     let model = Linear::<s![4, 2], B>::build(())?;
#     let mut optim = AdamW::<B>::from_module(&model, 1e-2)?;
#     train_step(&model, &mut optim)?;
#     Ok(())
# }
```

Three things worth noticing:

- `x.require_grad()` attaches gradient markers to the *input*. Model
  parameters already carry them, so the forward pass records whatever it needs
  without extra ceremony.
- `loss.backward()` returns a fresh `Gradients<B>`. There is no module-level
  gradient buffer to clear — see recipe 2.
- `optim.step(&grads)` takes the gradients by reference and commits. A
  scheduler is the usual fourth line; see [Schedulers](./training.md).

## 2. There is no `zero_grad`, and a spent `Gradients` is refused

Because gradients are produced per backward pass instead of accumulating into
the model, the loop needs no clearing step. The flip side: the value you
handed to `step` is tied to the parameter storage that existed *before* the
update, so reusing it after a commit finds nothing to update.

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
let model = Linear::<s![4, 2], B>::build(())?;
let mut optim = AdamW::<B>::from_module(&model, 1e-2)?;

let x = Tensor::<s![8, 4], B>::rand(())?.require_grad();
let target = Tensor::<s![8, 2], B, f32, NoGrad>::zeros(())?;
let pred = model.forward(x.clone())?;
let loss = MSELoss::new().forward(&pred, &target)?;
let grads = loss.backward()?;

optim.step(&grads)?; // commits: parameters move to new storage
if let Err(err) = optim.step(&grads) {
    // "adamw_step: invalid module or state dictionary: no parameter in this
    //  group received a gradient: the backward pass did not reach it. A tape
    //  is thread-local, so a backward call on a thread other than the one
    //  that recorded the forward pass drains an empty graph and produces
    //  exactly this state."
    println!("{err}");
}
# Ok(())
# }
```

The same refusal fires when `backward()` never reached the group at all —
no optimizer call, a detached forward, or a backward on a thread other than
the one that recorded the forward pass (the tape is thread-local).

If you need a gradient for a specific parameter, ask for it rather than
guessing:

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
let a = Cpu.randn(shape![4, 1])?.require_grad();
let c = Cpu.randn(shape![1, 4])?.require_grad();
let grads = a.matmul(&c)?.sum_all()?.backward()?;

let bystander = Cpu.randn(shape![4])?.require_grad();
if let Err(err) = grads.require(&bystander) {
    // "backward recipe for storage could not produce a gradient: the backward
    //  pass produced no gradient for this tensor"
    println!("{err}");
}
# Ok(())
# }
```

## 3. Catch a branch your forward pass skipped

`step` skips parameters that received no gradient — an unused parameter
genuinely has nothing to apply — and commits the rest. `step_strict` refuses
that partial coverage instead, so a silently detached branch cannot hide
inside an otherwise successful run.

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

#[module]
pub struct SkippedHead {
    used: Linear<s![4, 2], B>,
    unused: Linear<s![4, 2], B>,
}

impl SkippedHead {
    fn new() -> Result<Self> {
        Ok(Self {
            used: Linear::build(())?,
            unused: Linear::build(())?,
        })
    }

    fn forward(&self, x: Tensor<s![3, 4], B>) -> Result<Tensor<s![3, 2], B, f32, Grad>> {
        self.used.forward(x)
    }
}

# fn main() -> Result<()> {
let model = SkippedHead::new()?;
let x = Tensor::<s![3, 4], B>::ones(())?;
let grads = model.forward(x)?.sum_all()?.backward()?;

let mut strict = AdamW::<B>::from_module(&model, 1e-2)?;
if let Err(err) = strict.step_strict(&grads) {
    // "adamw_step_strict: invalid module or state dictionary: strict step
    //  requires every parameter in this group to have received a gradient,
    //  but only some did: a parameter the forward pass did not use, or a
    //  detached branch, was silently skipped."
    println!("{err}");
}

let mut lenient = AdamW::<B>::from_module(&model, 1e-2)?;
lenient.step(&grads)?; // PyTorch-compatible: commits the covered half
# Ok(())
# }
```

Both spellings refuse a step that reached *zero* parameters (recipe 2). The
difference is only what happens at `0 < covered < total`.

## 4. Put the model in the right mode

`train()` / `eval()` (or `set_training(bool)`) walk the whole module tree.
Leaf layers such as `Linear` are a no-op; layers with behaviour that differs
between the two modes — `Dropout` is the usual one — respond to it.

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
let mut drop = Dropout::new(0.5);
let x = Tensor::<s![1, 8], B>::ones(())?;

drop.eval();
assert_eq!(drop.forward(x.clone())?.to_vec1::<f32>()?, vec![1.0; 8]);

drop.train();
let dropped = drop.forward(x)?.to_vec1::<f32>()?;
assert!(dropped.iter().any(|&v| v == 0.0) || dropped.iter().all(|&v| v == 2.0));
# Ok(())
# }
```

Switch back before the next batch — a model left in `eval()` after a
validation pass trains with dropout disabled and nothing tells you.

## 5. Run inference without recording

Layer mode and tape mode are two different switches. `TrainMode` decides what
layers do; `GradMode` decides whether the tape records anything.

```rust,no_run
use incin::prelude::*;
use incin_core::exec::GradMode;

type B = DefaultBackend;

# fn main() -> Result<()> {
let mut model = Linear::<s![4, 2], B>::build(())?;
let x = Tensor::<s![3, 4], B>::ones(())?;

model.eval();
let logits = GradMode::Disabled.scope(|| model.forward(x.clone()))?;
let best = logits.to_vec1::<f32>()?;
println!("{best:?}");

// Backpropagating through a forward that recorded nothing is refused:
let logits = GradMode::Disabled.scope(|| model.forward(x))?;
if let Err(err) = logits.sum_all()?.backward() {
    // "backward pass found no recorded operations: the tape is empty or was
    //  already drained"
    println!("{err}");
}
# Ok(())
# }
```

`GradMode::Disabled.scope(...)` restores the previous mode when the closure
returns, including on error, so a scoped inference pass cannot leak into the
next training step.

## Common mistakes

- **Looking for `zero_grad`.** There is nothing to clear: each `backward()`
  returns a new `Gradients`, and model parameters never accumulate.
- **Reusing a `Gradients` after a successful `step`.** The commit reassigns
  parameter storage, so the old value covers nothing and the second call is
  refused with *"no parameter in this group received a gradient"*.
- **Calling `backward()` from another thread.** Tapes are thread-local, so a
  reverse walk on a fresh thread drains an empty graph and produces the same
  refusal.
- **Leaving `eval()` on after validation.** Dropout and friends stay in
  inference behaviour for the rest of the epoch; `train()` must be called
  explicitly.
- **Trusting a lenient `step` for a model with branches.** Use `step_strict`
  when a partially-covered step should be a bug, not a skip.
- **Expecting `grads.require(&t)` to conjure a gradient.** It reports the
  missing recipe rather than returning zeros.

Next: [Save and load](./howto_save_load.md) once the loop converges, or
[Data loading](./data_loading.md) to put a real dataset in front of it.
