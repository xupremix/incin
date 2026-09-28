# Losses, optimizers, and schedulers

## Losses

Every loss module's `forward` takes references, and the target argument is
`NoGrad`: the label data isn't something you differentiate with respect to,
and that's stated in the type rather than left as a convention to remember:

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let pred = Tensor::<s![3, 2], B>::ones(())?;
let target = Tensor::<s![3, 2], B, f32, NoGrad>::zeros(())?;

let loss = MSELoss::new().forward(&pred, &target)?;
let loss2 = L1Loss::new().forward(&pred, &target)?;
# Ok::<(), incin::Error>(())
```

`Mean` is one of three reduction modes (`Mean`, `Sum`, `NoneReduction`) that
parameterize a loss's own type: `MSELoss<Sum>` sums instead of averaging,
and `MSELoss<NoneReduction>` returns the per-element loss unreduced, each a
different type rather than a runtime flag.

Beyond MSE/L1/cross-entropy/BCE: `SmoothL1Loss` (Huber, configurable beta,
linear past the kink with the sign slope), `NLLLoss` over log-probabilities
(pair it with the `LogSoftmax` module), and token-level cross-entropy over
`[B, T, C]` logits with `[B, T]` targets, which flattens onto the rank-2
core and restores the `[B, T]` geometry for `NoneReduction` (no
`ignore_index` convention: the rank-2 loss has none, so neither does this
path).

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
let loss = MSELoss::new().forward(&pred, &target)?;
let grads = loss.backward()?;
optim.step(&grads)?;
# Ok::<(), incin::Error>(())
```

`SGD`, `Adam`, `AdamW` and `RMSprop` use `from_module(&model, lr)` for the
normal module path, or `from_group(ParameterGroup::from_module(&model)?, lr)`
when a caller needs to assemble a homogeneous group explicitly. `SGD`
carries momentum (with velocity in its state dict), classic coupled weight
decay, and a fail-closed Nesterov flag that refuses to step at zero
momentum; `RMSprop` carries its squared average the same way. All use
`step(&grads)` (`step_strict` refuses a partially-reached group instead of
skipping it). The learning rate is a public field (`optim.lr`), not hidden
behind a setter.

A step that reaches *no* parameter in the group is an error, not a silent
`Ok(())`. That case is nearly always a bug: a `Gradients` from a different
model, or a `backward` that ran on a different thread from the forward pass,
since the tape is thread-local. Skipping *some* parameters stays legal.

One consequence worth knowing: a committed step reassigns parameter storage,
so the `Gradients` value that produced it no longer matches anything. Collect
fresh gradients for each step rather than reusing one across two.

Resuming works the same way, plus one persisted counter. Adam and AdamW
`state_dict` writes the `m`/`v` moments next to a scalar `step` entry, and
`load_state_dict` restores all three, so bias correction continues with the
same `t` the moments were accumulated under. Dictionaries written before the
counter entry restore moments only and keep the loader's current counter
(`set_step_count` sets one explicitly); a present-but-malformed counter is a
typed error, never a silent default.

### Per-parameter learning rates

`set_param_lr("encoder.layer.0", 1e-5)` pins a rate for a parameter subtree
(longest dotted-prefix match, segment-aware); unmatched parameters keep the
base rate. Pins are absolute, so a scheduler ramp moves the base while pinned
groups stay put; `clear_param_lrs` drops every pin. Overrides are validated
at step time, naming the parameter, and are not part of `state_dict`
(re-apply after loading, like `lr` itself):

```rust,no_run
use incin::prelude::*;
type B = DefaultBackend;

let model = Linear::<s![4, 2], B>::build(())?;
let mut optim = SGD::<B>::from_module(&model, 1e-2)?;

// Freeze the weight, train the bias.
optim.set_param_lr("weight", 0.0);
assert_eq!(optim.lr_for("weight"), 0.0);
assert_eq!(optim.lr_for("bias"), 1e-2);
# Ok::<(), incin::Error>(())
```

## Gradient clipping

`clip_grad_norm` rescales every gradient in a group so their total L2 norm is
at most `max_norm`, and returns the norm *before* rescaling, which is the
number worth logging, since it tells you whether clipping actually engaged.

```rust,no_run
use incin::optim::{ParameterGroup, clip_grad_norm};
use incin::prelude::*;
type B = DefaultBackend;

let model = Linear::<s![4, 2], B>::build(())?;
let mut optim = AdamW::<B>::from_module(&model, 1e-2)?;
let group = ParameterGroup::<B, f32>::from_module(&model)?;

let x = Tensor::<s![3, 4], B>::ones(())?;
let target = Tensor::<s![3, 2], B, f32, NoGrad>::zeros(())?;
let pred = model.forward(x.require_grad())?;
let loss = MSELoss::new().forward(&pred, &target)?;

let mut grads = loss.backward()?;
let before = clip_grad_norm(&group, &mut grads, 1.0)?;
optim.step(&grads)?;
println!("gradient norm before clipping: {before}");
# Ok::<(), incin::Error>(())
```

Clip before `step`, never after: the optimizer reads the gradients it is
given, so clipping afterwards changes nothing that was already applied.

`clip_grad_value` is the other form. It clamps every gradient element
independently into `[-clip_value, clip_value]` and returns nothing, because
there is no single "before" number to report:

```rust,ignore
use incin::optim::clip_grad_value;

clip_grad_value(&group, &mut grads, 1.0)?;
```

Both reject a clip bound that is not finite and greater than zero, rather
than silently doing nothing.

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
    let loss = MSELoss::new().forward(&pred, &target)?;
    let grads = loss.backward()?;
    optim.step(&grads)?;

    sched.step();
    optim.lr = sched.get_lr();
}
# Ok::<(), incin::Error>(())
```

`ConstantLR`, `LinearLR`, `CosineAnnealingLR`, `StepLR`,
`ExponentialLR`, `LinearWarmup` and `CosineWithWarmup` all implement the
same `LRScheduler` trait (`get_lr() -> f64`, `step(&mut self)`), so swapping
one for another is a one-line change. Warmup schedulers ramp from zero over
their warmup steps before following their decay; every optimizer binds with
`optim.step_scheduler(&scheduler)`, which copies the current value into the
base rate (per-parameter pins stay absolute).

## The whole loop

Putting it together (model, loss, optimizer, scheduler, several steps):

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
        let loss = MSELoss::new().forward(&pred, &target)?;
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
loading](./data_loading.md); the loop shape is identical, just with a
`DataLoader` iteration in place of the fixed `x`.
