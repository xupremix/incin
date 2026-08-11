# Coming from PyTorch

| PyTorch | Incin | Note |
|---|---|---|
| `torch.zeros(2, 3)` | `Tensor::<s![2, 3], B>::zeros(())?` | Shape is a type; `()` because a static shape's argument is a unit tuple. |
| `torch.zeros(2, 3, dtype=torch.float64)` | `Tensor::<s![2, 3], B, f64>::zeros(())?` | dtype is the third type parameter, `K`. |
| `torch.tensor([1, 2, 3])` | `tensor![1, 2, 3]?` | Both default to `i64` for bare integer literals. |
| `a + b` | `(a + b)?` or `a.add(&b)?` | Operators (`+ - * /`, any owned/reference combination) work and broadcast (`BroadcastShape`); `.add()`/`.sub()`/`.mul()`/`.div()` require an *exact* shape match instead — use `.broadcast_add()` etc. for the method-call form of the same broadcasting `+` does. Both still return `Result`, so the `?` (or a parenthesized `(a + b)?`) is required either way. |
| `a @ b` | `a.matmul(&b)?` | |
| `x.requires_grad_()` | `x.require_grad()` | Changes the type (`G` becomes `Grad`), not just a runtime flag. |
| `x.detach()` | `x.detach()` | Also a type change, to `NoGrad`. |
| `with torch.no_grad(): ...` | `incin_core::exec::no_grad(\|\| { ... })` | Not yet re-exported through `incin::prelude` — see [Autograd](./autograd.md). |
| `loss.backward()` | `loss.backward()?` | Returns `Gradients` rather than mutating `.grad` on each leaf. |
| `param.grad` | `Backend::get_grad::<K>(param.inner(), grads.as_backend())?` | Explicit lookup by tensor, not an attribute. |
| `nn.Linear(768, 256)` | `Linear::<s![768, 256], B>::build(())?` | In/out features are the shape type, not constructor arguments. |
| `nn.Sequential(a, b, c)` | `seq!(a, b, c)` — see [Sequential](./sequential.md) | Type is `SeqTy!(A, B, C)`, built the same way the value is. |
| `optim.Adam(model.parameters(), lr=1e-3)` | `Adam::<B>::new(model.parameters(), 1e-3)` | |
| `optim.step()` | `optim.step(&grads)?` | Takes the `Gradients` from `backward()` explicitly rather than reading accumulated `.grad` fields. |
| `scheduler.step(); optim.param_groups[0]['lr']` | `sched.step(); optim.lr = sched.get_lr();` | `lr` is a public field you copy the scheduler's value into, not managed for you. |
| `nn.MSELoss()(pred, target)` | `MSELoss::<Mean>::new().forward(&pred, &target)?` | Reduction mode (`Mean`/`Sum`/`NoneReduction`) is the loss's own type parameter. |
| `torch.save(model.state_dict(), "m.pt")` | `save_safetensors::<B, _, _>(&model, "m.safetensors")?` | safetensors, not pickle — see [Saving and loading](./saving_loading.md). |
| `DataLoader(dataset, batch_size=4, collate_fn=f)` | `DataLoader::new(dataset, f, 4)?` | Collate function before batch size in the argument order. |
| `logits.argmax(dim=1)` | `logits.argmax(Some(1))?` | `None` reduces over the flattened tensor instead of one axis. |
| `x.to(device)` | `x.to_device::<D2>(&device_arg)?` | Device is part of the type on the receiving end, `Tensor<S, TransferTo<D2>::Output, ...>`. |
| `x.view(-1, 4)` / `x.reshape(...)` | `x.to_shape::<s![usize, 4]>()?` (checked) or `idx!`-based views | `to_shape` re-validates dims at runtime; a fully static target shape is checked at compile time instead. |

The biggest structural difference isn't any one API — it's that shape and
dtype mismatches, tensors that shouldn't require a gradient, and layers fed
the wrong feature count are, as much as possible, compile errors here rather
than runtime exceptions. Code that "just runs" in PyTorch because Python
doesn't check any of that ahead of time often needs its shapes made
explicit — writing `s![768, 256]` rather than trusting two `768`s a hundred
lines apart to agree — to compile in Incin at all. That's the trade this
library is built around, not a friction to work around.
