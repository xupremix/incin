# Installation

Incin is not yet published to crates.io (the workspace is at version
`0.0.0`). Until it is, depend on it from a checkout or from the repository
directly:

```toml
[dependencies]
incin = { path = "../incin/crates/incin" }
# or
incin = { git = "https://github.com/xupremix/incin" }
```

The default feature set is `["std", "cpu"]` — a standard-library CPU build
with no extra setup. That's enough for everything in this book except the
[Backends](./backends.md) chapter.

## Feature flags you'll actually reach for

| Feature | What it enables |
|---|---|
| `cpu` | The CPU backend, `DefaultBackend`, `DefaultDevice`. On by default. |
| `cuda`, `wgpu`, `metal` | The respective accelerator backend. Read [Backends](./backends.md) before reaching for these — coverage is much narrower than CPU today. |
| `backend-authoring` | Public contracts for implementing custom backends and operations. |
| `train` | The preview automatic `Trainer`. |
| `distributed` | Mesh/placement/collective planning — a design surface, not yet an execution path. |

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
