# Debug shape and dtype errors

Incin refuses instead of guessing, and the refusal is designed to be read:
one summary line you can grep, then indented lines that say what the
operation required, what arrived, and which edit satisfies the rule. This
chapter is a field guide — five recipes for classifying the failure you are
looking at and finding the fix.

## 1. Read a shape error as a diff

A `Dyn` shape rule that fails prints the two operands, the rule that related
them, and the remedy:

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
let lhs = Tensor::<Dyn, B>::zeros(vec![2, 3])?;
let rhs = Tensor::<Dyn, B>::zeros(vec![4, 5])?;
if let Err(err) = lhs.matmul(&rhs) {
    // matmul: axis 'k' mismatch: 3 vs 4, which must be equal
    //   lhs axis 'k' = 3
    //   rhs axis 'k' = 4
    //   rule: the two must be equal
    //   fix: change the lhs to 4, or the rhs to 3
    println!("{err}");
}
# Ok(())
# }
```

Read it bottom-up: `fix` is the edit, `rule` is why it is required, and the
first line is the whole error in one clause. Static shapes cannot reach this
point — the same rule is discharged by type checking before the program
runs.

## 2. Tell a compile-time failure from a run-time one

The same contract is proved twice (issue #93). With a static shape the
compiler refuses the program; with `Dyn` the error comes back typed.

```rust,compile_fail
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
// E0080: evaluation panicked: the last axis of a Q8_0 block must be a
// multiple of 32
let x = Cpu.zeros(shape![48])?;
let q = x.quantize(-1)?;
# Ok(())
# }
```

```rust,compile_fail
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
// E0277: the trait bound `incin::Q8_0: FloatCapable` is not satisfied
let q = Cpu.zeros(shape![64])?.quantize(-1)?;
let r = q.abs()?;
# Ok(())
# }
```

Both failures are *invariant* failures — the message states a rule that
holds for every backend — while a `Dyn` extent is only knowable at run time,
so the equivalent check returns
`Generic Message: quantize: axis 1 has extent 48, ...` instead. The
`compile_fail` fixtures under `crates/incin-core/tests/compile_fail/` pin
these messages so they cannot drift.

## 3. Recognize a capability refusal

If the line starts with `backend '<name>' refused the request:`, the backend
checked its capability rows and said no. Nothing is wrong with your shapes —
the operation, dtype or mode is simply not advertised:

```rust,no_run
use incin::backend_authoring::{ExecutionContext, execute, operations::{NoAttributes, op}};
use incin::prelude::*;
use incin_core::exec::TensorHandle;

type B = DefaultBackend;

# fn main() -> Result<()> {
let lhs = Cpu.zeros(shape![2, 32])?.quantize(-1)?;
let rhs = Cpu.zeros(shape![32, 32])?.quantize(-1)?;

let ctx = ExecutionContext::from_scope(B::default()).with_training(true);
let h1 = TensorHandle::from_storage::<B, Q8_0, incin_core::dist::placement::Local>(lhs.inner());
let h2 = TensorHandle::from_storage::<B, Q8_0, incin_core::dist::placement::Local>(rhs.inner());
if let Err(err) = execute::<op::QuantizedMatMul, B>(&ctx, NoAttributes, &[h1, h2]) {
    // backend 'Cpu' refused the request: training is unsupported for
    // quantized_matmul
    println!("{err}");
}
# Ok(())
# }
```

Two things to check: the reason names the *operation* (`training is
unsupported for {operation}`, `dtype {dtype} is unsupported for {operation}`),
and the fix line points at the row that decided — `docs/capabilities.md`
lists the rows per backend. A refusal is never a panic and never a silent
fallback.

## 4. Recognize a dtype-admission error

Before any capability is consulted, the operation's own descriptor checks the
operands. Those errors are shaped like a contract, with `attribute`, `rule`
and `fix` lines rather than a capability name:

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
let a = Cpu.randn(shape![2, 32])?.quantize(-1)?;
let b = Cpu.randn(shape![32, 32])?.quantize(-1)?;
if let Err(err) = a.matmul(&b) {
    // matmul: invalid dtype: operation requires floating-point input
    // metadata
    //   attribute: dtype
    //   rule: operation requires floating-point input metadata
    //   fix: give every operand the same dtype, converting one with
    //   `.to_dtype(..)` if they genuinely differ
    println!("{err}");
}
# Ok(())
# }
```

The distinction that saves time: *`invalid dtype`* means the operation does
not admit this dtype at all (change the operation or the dtype), while
*`refused the request`* means this backend does not implement it (change the
backend, the mode, or the version).

## 5. Classify a training failure in one print

Errors that reach your loop from the optimizer, the tape or the shape layer
each carry their own prefix. Printing `{err}` once is usually enough to
place the failure:

| Prefix you see | Family | Start here |
|:---|:---|:---|
| `backend '...' refused the request:` | capability row said no | recipe 3 |
| `...: invalid dtype: ...` | operation descriptor said no | recipe 4 |
| `...: axis 'k' mismatch: ...` | shape rule | recipe 1 |
| `prepare parameter: ...: shape or dtype mismatch` | checkpoint vs module | [Save and load](./howto_save_load.md) |
| `load state: ...: state paths differ` | parameter path | [Save and load](./howto_save_load.md) |
| `adamw_step: ...: no parameter in this group received a gradient` | tape/coverage | [Run a training loop](./howto_training_loop.md) |
| `Generic Message: ...` | a typed error with no bespoke `Display` | read the text — it still names the operation |

Two training failures that look numeric but are not:

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

# fn main() -> Result<()> {
// A forward that recorded nothing cannot be walked backwards:
let a = Cpu.randn(shape![4, 1])?.require_grad();
let c = Cpu.randn(shape![1, 4])?.require_grad();
let loss = incin_core::exec::GradMode::Disabled.scope(|| {
    let product = a.matmul(&c)?;
    Ok::<_, incin::Error>(product.sum_all()?)
})?;
if let Err(err) = loss.backward() {
    // backward pass found no recorded operations: the tape is empty or was
    // already drained
    println!("{err}");
}

// A hyperparameter that is not a number is refused at step time:
let model = Linear::<s![4, 2], B>::build(())?;
let mut optim = AdamW::<B>::from_module(&model, f64::NAN)?;
let x = Tensor::<s![3, 4], B>::ones(())?;
let grads = model.forward(x)?.sum_all()?.backward()?;
if let Err(err) = optim.step(&grads) {
    // adamw_step: invalid module or state dictionary: learning rate must be
    // finite and non-negative
    println!("{err}");
}
# Ok(())
# }
```

## Common mistakes

- **Fixing the operands instead of the message.** Read `fix:` first — it
  usually names the single edit that satisfies the rule.
- **Treating `Generic Message:` as a low-level panic.** It is a typed error
  whose variant carries a plain string; the text after it is the whole
  contract.
- **Confusing `invalid dtype` with `refused the request`.** One is admission
  (the operation), the other is capability (the backend). Different fixes.
- **Chasing a shape error that is really a mode error.** An empty tape, a
  detached branch and a stale `Gradients` all look like "no gradient" — the
  messages differ, and each names its own cause.
- **Assuming a static shape can reach a runtime block/divisibility check.**
  If it compiled, the invariant held; look elsewhere for the bug.

Next: [Errors](./errors.md) for the full variant list and the contract they
all follow.
