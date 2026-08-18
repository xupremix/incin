# Coming from PyTorch

| PyTorch | Incin | Note |
|---|---|---|
| `torch.zeros(2, 3)` | `Cpu.zeros(shape![2, 3])?` | Concrete code uses the target-first constructor. Generic code can use `Tensor::<s![2, 3], B>::zeros(())?`. |
| `torch.zeros(2, 3, dtype=torch.float64)` | `Tensor::<s![2, 3], B, f64>::zeros(())?` | dtype is the third type parameter, `K`. |
| `torch.tensor([1, 2, 3])` | `tensor![1, 2, 3]?` | Both default to `i64` for bare integer literals. |
| `a + b` | `(&a + &b)?` | Tensor operators return `Result<Tensor<...>>`; use `?` or explicit error handling. |
| `a @ b` | `a.matmul(&b)?` | |
| `x.requires_grad_()` | `x.require_grad()` | Changes the type (`G` becomes `Grad`), not just a runtime flag. |
| `x.detach()` | `x.detach()` | Also a type change, to `NoGrad`. |
| `with torch.no_grad(): ...` | `incin_core::exec::GradMode::Disabled.scope(\|\| { ... })` | Scoped gradient policy. |
| `loss.backward()` | `loss.backward()?` | Returns `Gradients` rather than mutating `.grad` on each leaf. |
| `param.grad` | `grads.require(&param)?` | Explicit typed lookup by tensor, not an attribute. `grads.get(&param)?` is the optional form. |
| `nn.Linear(768, 256)` | `Linear::<s![768, 256], B>::build(())?` | In/out features are the shape type, not constructor arguments. |
| `nn.Sequential(a, b, c)` | `seq!(a, b, c)` - see [Sequential](./sequential.md) | Type is `SeqTy!(A, B, C)`, built the same way the value is. |
| `optim.AdamW(model.parameters(), lr=1e-3)` | `AdamW::<B>::from_module(&model, 1e-2)?` | Parameters are collected through the module visitor. |
| `optim.step()` | `optim.step(&grads)?` | Takes the `Gradients` from `backward()` explicitly rather than reading accumulated `.grad` fields. |
| `scheduler.step(); optim.param_groups[0]['lr']` | `sched.step(); optim.lr = sched.get_lr();` | `lr` is a public field you copy the scheduler's value into, not managed for you. |
| `nn.MSELoss()(pred, target)` | `MSELoss::<Mean>::new().forward(&pred, &target)?` | Reduction mode (`Mean`/`Sum`/`NoneReduction`) is the loss's own type parameter. |
| `torch.save(model.state_dict(), "m.pt")` | `save_safetensors::<B, _, _>(&model, "m.safetensors")?` | safetensors, not pickle - see [Saving and loading](./saving_loading.md). |
| `DataLoader(dataset, batch_size=4)` | `DataLoader::builder(dataset).batch_size(4).build()?` | Builder setters are infallible; validation happens in `build`. Scalar samples become `Vec<T>` and tuple samples are collated field-wise. |
| `logits.argmax(dim=1)` | `logits.argmax(axis!(1))?` | Static and runtime signed selectors share one axis argument. |
| `x.to(device)` | `x.to_device::<D2>(&device_arg)?` | Device is part of the type on the receiving end, `Tensor<S, TransferTo<D2>::Output, ...>`. |
| `x.view(-1, 4)` / `x.reshape(...)` | `x.reshape(shape![usize, 4])?` | A static target keeps its compile-time proof. Runtime extents are checked during the reshape. |

The biggest structural difference is not any one API. Shape and
dtype mismatches, tensors that shouldn't require a gradient, and layers fed
the wrong feature count are, as much as possible, compile errors here rather
than runtime exceptions. Code that "just runs" in PyTorch because Python
doesn't check any of that ahead of time often needs its shapes made
explicit, by writing `s![768, 256]` rather than trusting two `768`s a hundred
lines apart to agree - to compile in Incin at all. That's the trade this
library is built around, not a friction to work around.
