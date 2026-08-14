# Saving and loading

Checkpointing is safetensors-based and works for any type implementing the
typed state visitor contract (which `#[module]` generates automatically).

> `save_safetensors`/`load_safetensors` are not currently re-exported
> through the `incin` facade — only through `incin_core::nn::save` directly.
> Add `incin_core` as an explicit dependency to reach them until that's
> fixed; this is tracked as a known gap.

```rust,no_run
use incin::prelude::*;
use incin_core::nn::save::{load_safetensors, save_safetensors};

type B = DefaultBackend;

fn main() -> Result<()> {
    let model = Linear::<s![4, 2], B>::build(())?;
    save_safetensors::<B, _, _>(&model, "model.safetensors")?;

    let mut reloaded = Linear::<s![4, 2], B>::build(())?;
    load_safetensors::<B, _, _>(&mut reloaded, "model.safetensors")?;

    let x = Tensor::<s![1, 4], B>::ones(())?;
    let a = model.forward(x.clone())?;
    let b = reloaded.forward(x)?;
    assert_eq!(a.to_vec1::<f32>()?, b.to_vec1::<f32>()?);
    Ok(())
}
```

`save_safetensors`/`load_safetensors` work on a single layer, a hand-composed
`#[module]` struct, or a `Sequential` chain through typed state visitors.

## ONNX

`incin::experimental::model!("model.onnx", Name)` and `import_model!` expand
a supported `.onnx` graph into typed Rust code at compile time. Support is
deliberately partial and fail-closed: initializers, unknown rank, control
flow, custom domains, and unsupported node types are macro-expansion errors,
not silently-wrong generated code. Treat this as import tooling for a known,
simple graph shape rather than a general ONNX runtime.
