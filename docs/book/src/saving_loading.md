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

## Loading does not move the model

`load` restores state in place and leaves every parameter on the device it
already lives on. There is no device argument - it used to take one and
ignore it, which read as a relocation the call never performed. Moving a model
between devices is `ToDevice`, which is explicit and hands back the relocated
model.

Loading is transactional. Every path in the file is checked against the
module's own state before anything is written, so a snapshot that is missing a
parameter or carries an unexpected one is refused with both lists named rather
than half-applied. If a leaf fails during preparation the prepared state is
cleared and the model is left as it was. Commit writes candidates into the
existing variable slots, then rolls back all earlier writes if a later commit
fails; a separately held clone therefore observes a successful load instead
of becoming a stale handle.

The on-disk snapshot has one value per state path and intentionally contains
no alias IDs. A backend can declare that cloned parameter handles share one
slot. When it does, all paths for that tied parameter must contain identical
role, shape, dtype, and payload data; conflicting entries are rejected before
any write. This makes ordinary named snapshots portable while keeping tied
weights safe to restore.

## The format version

Both state formats record `STATE_FORMAT_VERSION`, currently `1`. Safetensors
carries it as an `incin.format.version` metadata key; postcard carries it as
the first field of its envelope, ahead of the payload, so a mismatch is
reported as a version problem rather than as a decode failure several fields
in. Reading refuses a file whose version is newer than the build, naming both
numbers so you learn which release would read it.

The version describes the *envelope* - how paths, roles, dtypes and payload
bytes are arranged - and is deliberately independent of the crate version
and of any individual dtype's descriptor version.

A file written before versioning existed carries no key and is refused as
unversioned; re-save it with this release. Foreign safetensors files were
never loadable through `ModelExt::load` and are unaffected: `import_model!`
reads those, and it does not look for the key.

Sharded checkpoints carry their own, separate `CHECKPOINT_MANIFEST_VERSION`
in the manifest, since topology and payload can change independently.

## ONNX

`incin::experimental::model!("model.onnx", Name)` and `import_model!` expand
a supported `.onnx` graph into typed Rust code at compile time. Support is
deliberately partial and fail-closed: initializers, unknown rank, control
flow, custom domains, and unsupported node types are macro-expansion errors,
not silently-wrong generated code. Treat this as import tooling for a known,
simple graph shape rather than a general ONNX runtime.
