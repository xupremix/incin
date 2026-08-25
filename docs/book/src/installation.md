# Installation

Incin is on crates.io. Depend on the facade with:

```toml
[dependencies]
incin = "0.1"
```

Working from a checkout of this repository? A path dependency does what the
examples here do:

```toml
[dependencies]
incin = { path = "../incin/crates/incin" }
```

The default feature set is `["std", "cpu"]`, a standard-library CPU build
with no extra setup. That's enough for everything in this book except the
[Backends](./backends.md) chapter.

## Stability

The published `0.1.0` facade API is Incin's compatibility baseline. Changes to that surface will be additive or follow a documented
deprecation path, even though Cargo treats `0.x` minor releases as potentially
breaking by default.

Read [the planned 0.1.0 release notes](./release_notes.md) for the proposed
baseline and migration notes. Future releases will list any required migration
with the edit it requires, and `docs/MIGRATION.md` carries the path-by-path
table. Pin an exact version (`incin = "=0.1.0"`) if you also need dependency
resolution to hold still while you work.

Preview APIs are outside that selected baseline unless the release notes say
otherwise. They include accelerator backends, distributed planning, the
preview-only CPU compiled reference evaluator, and the automatic `Trainer`.
Compiled plan snapshots are not a deployment format or portable ABI.

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
| `cuda`, `wgpu`, `metal` | The respective accelerator backend. Read [Backends](./backends.md) before reaching for these; coverage is much narrower than CPU today. |
| `backend-authoring` | Public contracts for implementing custom backends and operations. |
| `train` | The preview automatic `Trainer`. |
| `distributed` | Mesh/placement/collective planning: a design surface, not yet an execution path. |
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
