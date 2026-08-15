# Saving and loading

Checkpointing works through the facade's `ModelExt` trait for any type
implementing the typed state visitor contract (which `#[module]` generates
automatically). Safetensors and postcard are supported state formats.

```rust,no_run
use incin::prelude::*;

type B = DefaultBackend;

fn main() -> Result<()> {
    let model = Linear::<s![4, 2], B>::build(())?;
    model.save(Format::Safetensors, std::path::Path::new("model.safetensors"))?;

    let mut reloaded = Linear::<s![4, 2], B>::build(())?;
    reloaded.load(
        Format::Safetensors,
        std::path::Path::new("model.safetensors"),
        &DeviceId::cpu(),
    )?;

    let x = Tensor::<s![1, 4], B>::ones(())?;
    let a = model.forward(x.clone())?;
    let b = reloaded.forward(x)?;
    assert_eq!(a.to_vec1::<f32>()?, b.to_vec1::<f32>()?);
    Ok(())
}
```

`ModelExt::save`/`ModelExt::load` work on a single layer, a hand-composed
`#[module]` struct, or a `Sequential` chain through typed state visitors. The
trait is exported from `incin::prelude`; import it by name if you do not use
the prelude.

## ONNX

`incin::experimental::model!("model.onnx", Name)` and `import_model!` expand
a supported `.onnx` graph into typed Rust code at compile time. Support is
deliberately partial and fail-closed: initializers, unknown rank, control
flow, custom domains, and unsupported node types are macro-expansion errors,
not silently-wrong generated code. Treat this as import tooling for a known,
simple graph shape rather than a general ONNX runtime.
