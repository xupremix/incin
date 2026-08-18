# Installation

```toml
[dependencies]
incin = "0.1"
```

Or from a checkout, which is what the examples in this repository do:

```toml
[dependencies]
incin = { path = "../incin/crates/incin" }
# or
incin = { git = "https://github.com/xupremix/incin" }
```

The default feature set is `["std", "cpu"]` - a standard-library CPU build
with no extra setup. That's enough for everything in this book except the
[Backends](./backends.md) chapter.

## Stability

There is none yet, and that is deliberate. `0.1.0` is a first published
version, not a stability promise: under Cargo's rules for `0.x` the minor
version is already the breaking slot, and this project intends to use it. Any
`0.x` bump may remove or reshape public API without a deprecation period.

Read [What changed in 0.1.0](./release_notes.md) as the model for how that will
be communicated: every break is listed with the edit it requires, and
`docs/MIGRATION.md` carries the path-by-path table. Pin an exact version
(`incin = "=0.1.0"`) if you need the surface to hold still while you work.

The parts most likely to move are the ones this book already labels preview:
the accelerator backends, distributed planning, compiled execution, and the
automatic `Trainer`. The typed tensor, shape, autograd, and layer surfaces are
where the design has settled most.

## What the build needs

Nothing beyond a Rust toolchain. The minimum supported version is **1.88**,
held by a CI job pinned to exactly that toolchain rather than asserted; 1.87
is refused by the dependency graph.

In particular there is no `protoc`. The ONNX protobuf module is a checked-in
`prost-build` output, regenerated with `cargo xtask onnx` by anyone changing
the schema, so a system protobuf compiler is not a dependency of every crate
that happens to depend on the facade.

## Feature flags you'll actually reach for

| Feature | What it enables |
|---|---|
| `cpu` | The CPU backend, `DefaultBackend`, `DefaultDevice`. On by default. |
| `cuda`, `wgpu`, `metal` | The respective accelerator backend. Read [Backends](./backends.md) before reaching for these - coverage is much narrower than CPU today. |
| `backend-authoring` | Public contracts for implementing custom backends and operations. |
| `train` | The preview automatic `Trainer`. |
| `distributed` | Mesh/placement/collective planning - a design surface, not yet an execution path. |
| `data-hub` | The Hugging Face Hub client at `incin::hub`. Off by default: it brings an async runtime and a second TLS stack into the graph. |

Enable what you need:

```toml
[dependencies]
incin = { path = "../incin/crates/incin", features = ["cpu"] }
```

## Verifying the install

```rust,no_run
use incin::prelude::*;

fn main() -> Result<()> {
    let x = Tensor::<s![2, 3], DefaultBackend>::zeros(())?;
    println!("{:?}", x.dims());
    Ok(())
}
```

If this compiles and prints `[2, 3]`, you're set up correctly.
