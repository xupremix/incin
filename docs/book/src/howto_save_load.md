# Save and load a model

`ModelExt::save` and `ModelExt::load` are the whole persistence surface for a
module: they walk the same parameter tree in both directions and refuse any
file whose paths, shapes, dtypes or versions disagree with the module in
front of them. This chapter is the doing half of [Saving and
loading](./saving_loading.md) — the recipes plus the exact refusal each
mistake produces.

## 1. Round trip a module

```rust,no_run
use incin::prelude::*;
use std::path::Path;

type B = DefaultBackend;

# fn main() -> Result<()> {
let model = Linear::<s![64, 32], B>::build(())?;
model.save(Format::Safetensors, Path::new("model.safetensors"))?;
model.save(Format::Postcard, Path::new("model.postcard"))?;

// A freshly built module of the same type and geometry:
let mut fresh = Linear::<s![64, 32], B>::build(())?;
fresh.load(Format::Safetensors, Path::new("model.safetensors"))?;
fresh.load(Format::Postcard, Path::new("model.postcard"))?;
# Ok(())
# }
```

`Format::Safetensors` is the interoperable weights format; `Format::Postcard`
is the Rust-native envelope that carries everything safetensors does not (see
recipe 4 for one case where you need it). Both are transactional: a `load`
that fails part way leaves the module as it was, never half-updated.

Loading fills in parameters — it does not move, re-shape or re-allocate the
model, so the destination type has to already match. See [Saving and
loading](./saving_loading.md) for what that means for `to_device` and for
`Sequential` composition.

## 2. Diagnose a checkpoint from a different architecture

Two failures, two different messages, both naming what to change.

Different parameter *paths* — a module nested one level deeper (or flatter)
than the one that wrote the file:

```rust,no_run
use incin::prelude::*;
use std::path::Path;

type B = DefaultBackend;

#[module]
pub struct Wrapped {
    fc: Linear<s![4, 2], B>,
}

# fn main() -> Result<()> {
let flat = Linear::<s![4, 2], B>::build(())?;
flat.save(Format::Safetensors, Path::new("flat.safetensors"))?;

let mut wrapped = Wrapped {
    fc: Linear::build(())?,
};
if let Err(err) = wrapped.load(Format::Safetensors, Path::new("flat.safetensors")) {
    // load state: invalid module or state dictionary: state paths differ:
    // missing ["fc.bias", "fc.weight"], unexpected ["bias", "weight"]
    println!("{err}");
}
# Ok(())
# }
```

Same paths, different *geometry* — the checkpoint is the same shape of model
with a different size:

```rust,no_run
use incin::prelude::*;
use std::path::Path;

type B = DefaultBackend;

# fn main() -> Result<()> {
let saved = Linear::<s![4, 2], B>::build(())?;
saved.save(Format::Safetensors, Path::new("wide.safetensors"))?;

let mut target = Linear::<s![8, 2], B>::build(())?;
if let Err(err) = target.load(Format::Safetensors, Path::new("wide.safetensors")) {
    // prepare parameter: invalid module or state dictionary: shape or dtype
    // mismatch at weight
    println!("{err}");
}
# Ok(())
# }
```

The rule in both cases: the *destination* module is the specification. Fix
the code that built the destination, or find the file that matches it.

## 3. Read a file this build is not allowed to read

Every state file carries `incin.format.version` in its header, and a file
claiming a newer version than `STATE_FORMAT_VERSION` is refused before a
single parameter is touched. The header lives in the first bytes of the file
(8-byte little-endian length, then the JSON), so the check is easy to
demonstrate:

```rust,no_run
use incin::prelude::*;
use std::path::Path;

type B = DefaultBackend;

# fn main() -> Result<()> {
let model = Linear::<s![64, 32], B>::build(())?;
let path = Path::new("/tmp/version-demo.safetensors");
model.save(Format::Safetensors, path)?;

let mut raw = std::fs::read(path)?;
let mut len = [0u8; 8];
len.copy_from_slice(&raw[..8]);
let header_len = u64::from_le_bytes(len) as usize;
let header = String::from_utf8_lossy(&raw[8..8 + header_len]).into_owned();
let bumped = header.replace(
    "\"incin.format.version\":\"1\"",
    "\"incin.format.version\":\"99\"",
);

let mut rewritten = (bumped.len() as u64).to_le_bytes().to_vec();
rewritten.extend_from_slice(bumped.as_bytes());
rewritten.extend_from_slice(&raw[8 + header_len..]);
std::fs::write(path, &rewritten)?;

let mut fresh = Linear::<s![64, 32], B>::build(())?;
if let Err(err) = fresh.load(Format::Safetensors, path) {
    // open safetensors stream: malformed safetensors file: safetensors state
    // file declares format version 99, but this build reads at most version
    // 1; upgrade incin to read it
    println!("{err}");
}
# Ok(())
# }
```

`STATE_FORMAT_VERSION` (in the prelude) is the number this build writes and
the maximum it reads. A file with no version key at all is refused the same
way — it was not written by a format-aware writer.

## 4. Shard on block boundaries, not just on ranks

A `Q8_0` tensor stores 32 elements in one 34-byte block, so a shard boundary
that falls inside a block cannot be described at all. `slice_bytes_for_rank`
— the routine behind `load_resharded_checkpoint` — refuses the split instead
of cutting a block:

```rust,no_run
use incin::prelude::*;

# fn main() -> Result<()> {
// 96 x 64 elements of q8_0: 6144 / 32 blocks x 34 bytes = 6528 bytes.
let bytes = vec![0u8; 6528];
let descriptor = DTypeId::Q8_0.descriptor();

// Shard the last axis (extent 64) across two ranks: 32 per rank — whole
// blocks, so this is accepted.
let (local, local_shape) = incin_core::nn::slice_bytes_for_rank(
    &bytes,
    &[96, 64],
    descriptor.clone(),
    1,
    0,
    2,
)?;
println!("rank 0 holds {local:?} bytes for {local_shape:?}");

// Shard axis 0 (extent 96) across two ranks: 48 rows each, and 48 is not a
// multiple of the 32-element block.
if let Err(err) = incin_core::nn::slice_bytes_for_rank(
    &bytes,
    &[96, 64],
    descriptor,
    0,
    0,
    2,
) {
    // Generic Message: Cannot shard dtype q8_0 along axis 0: local extent 48
    // is not a multiple of block size 32; shard boundary would fall mid-block
    println!("{err}");
}
# Ok(())
# }
```

Pick a shard axis whose per-rank extent divides into whole blocks — for a
`[96, 64]` tensor over two ranks, that is the last axis (64 → 32), not axis 0
(96 → 48). The refusal names the axis, the local extent and the block size,
so the fix is arithmetic rather than guesswork. See
[Quantization](./quantization.md) for the format commitment behind this.

## 5. Register a dtype before loading a manifest that mentions it

A sharded checkpoint manifest persists a `DTypeKey` — a `(namespace, name,
version)` triple — for every tensor. Loading one that names a dtype this
process has never registered is refused rather than guessed at:

```rust,no_run
use incin::prelude::*;
use std::path::Path;

# fn main() -> Result<()> {
let dir = Path::new("/tmp/manifest-demo");
std::fs::create_dir_all(dir)?;

let mut manifest = incin_core::nn::GlobalCheckpointManifest::new(2);
manifest.add_tensor("quant.weight", vec![64], DTypeId::Q8_0, "Sharded:0");
let path = dir.join("manifest.json");
incin_core::nn::save_checkpoint_manifest(&manifest, &path)?;

// Rewrite the key's version from 1 to 2 — a dtype this build never wrote.
let wire = std::fs::read_to_string(&path)?;
let key_at = wire.find("\"q8_0\"").expect("manifest carries the q8_0 key");
let tail = &wire[key_at..];
let close = tail.find(']').expect("key array closes");
let seg = &tail[..close];
let version_at = seg.rfind('1').expect("the key's version digit");
let mut tampered = wire.clone();
tampered.replace_range(key_at + version_at..key_at + version_at + 1, "2");
let tampered_path = dir.join("tampered.json");
std::fs::write(&tampered_path, &tampered)?;

if let Err(err) = incin_core::nn::save::load_checkpoint_manifest(&tampered_path) {
    // Generic Message: Failed to parse checkpoint manifest: Deserializing
    // custom DTypeKey (incin, q8_0, 2) requires that dtype to be registered
    // first: call DTypeRegistry::register(descriptor) at startup, before
    // loading anything that mentions it at line 15 column 9
    println!("{err}");
}
# Ok(())
# }
```

The fix is exactly what the message says: `DTypeRegistry::register(...)`
runs at startup, before anything that mentions the dtype is read. The same
rule covers an encoding that disagrees with the registered dtype — the
checkpoint loses, the registry wins.

## Common mistakes

- **Loading into a freshly built module of a different size.** You get
  *"shape or dtype mismatch at weight"*; the destination type is the spec.
- **Nesting a module and expecting old paths to resolve.** Renaming a field
  renames every state path under it: *"state paths differ: missing ...,
  unexpected ..."*.
- **Reading a checkpoint from a newer build.** *"declares format version 99,
  but this build reads at most version 1"* is a build mismatch, not a corrupt
  file.
- **Sharding a `Q8_0` tensor along an axis whose local extent is not a
  multiple of 32.** *"shard boundary would fall mid-block"* — choose the
  shard axis before choosing the world size.
- **Feeding a manifest a `DTypeKey` version you never registered.** Register
  the descriptor first, or expect *"requires that dtype to be registered
  first"*.
- **Expecting a failed `load` to leave partial parameters.** `load` is
  transactional: on error the module is unchanged.

Next: [Quantization](./howto_quantize.md) for block-quantized parameters, or
[Distributed planning](./distributed.md) for the sharding side.
