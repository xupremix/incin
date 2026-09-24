//! Checkpoint round trip and the #93 block-sharding rule: save a tiny
//! module's state, load it into a fresh instance, prove the bytes match,
//! then split a real `Q8_0` checkpoint tensor across ranks - whole blocks
//! succeed, a mid-block boundary is a typed refusal rather than a
//! truncated or silently misread shard.
//!
//! Two facts tie the sections together. The safetensors writer refuses
//! block dtypes (it always has), so `Q8_0` parameters travel through the
//! state formats, not through `Format::Safetensors`; and the sharding
//! workhorse behind `load_resharded_checkpoint` computes byte spans through
//! `StorageEncoding::size_bytes`, which is why the shard boundary must land
//! between blocks.
//!
//! Run with: `cargo run -p incin --example save_load_roundtrip --no-default-features --features incin-backends/cpu,incin/cpu`

#![cfg(feature = "cpu")]
#![allow(missing_docs)]
#![allow(clippy::type_complexity)]

use incin::nn::Linear;
use incin::prelude::*;
use incin::state::collect_state;
use incin_core::backend_authoring::HostInterop;
use incin_core::nn::save::{load_safetensors, save_safetensors, slice_bytes_for_rank};

type Backend = DefaultBackend;

/// A two-layer module small enough that every parameter fits in the
/// printout - but a real `#[module]`, so `save_safetensors`/`load_safetensors`
/// walk the same state tree a production model would.
#[module]
struct TinyMlp {
    up: Linear<s![4, 8], Backend>,
    down: Linear<s![8, 2], Backend>,
}

impl TinyMlp {
    pub fn new() -> incin::Result<Self> {
        Ok(Self {
            up: Linear::build(())?,
            down: Linear::build(())?,
        })
    }

    pub fn forward(
        &self,
        x: Tensor<s![3, 4], Backend>,
    ) -> incin::Result<Tensor<s![3, 2], Backend, f32, Grad>> {
        let hidden = self.up.forward(x)?;
        self.down.forward(hidden.relu()?)
    }
}

fn main() -> incin::Result<()> {
    section("1. Save a module's state, load it back, assert equality");
    roundtrip()?;

    section("2. Sharding a Q8_0 checkpoint tensor: whole blocks only");
    block_sharding()?;

    Ok(())
}

/// Section 1: the checkpoint contract in its plainest form - what was
/// saved is what comes back, checked twice: the parameter bytes themselves
/// and the model's output over a fixed input.
fn roundtrip() -> incin::Result<()> {
    let temp_dir = std::env::temp_dir().join("incin_save_load_roundtrip");
    std::fs::create_dir_all(&temp_dir).map_err(|error| incin::Error::Msg(error.to_string()))?;
    let checkpoint = temp_dir.join("tiny_mlp.safetensors");

    let original = TinyMlp::new()?;
    let probe = Cpu.randn(shape![3, 4])?;
    let baseline = original.forward(probe.clone())?;

    println!("  saved state leaves:");
    let before = collect_state::<Backend, _>(&original)?;
    for (path, value) in before.iter() {
        println!(
            "    {:<28} shape {:?}, dtype {}",
            path.as_str(),
            value.shape(),
            value.dtype().name()
        );
    }

    save_safetensors::<Backend, _, _>(&original, &checkpoint)?;
    println!(
        "  wrote {:?} ({} bytes)",
        checkpoint.file_name().unwrap_or_default(),
        std::fs::metadata(&checkpoint)
            .map_err(|error| incin::Error::Msg(error.to_string()))?
            .len()
    );

    // A fresh instance: same types, different random weights. If the load
    // did not really restore state, the assertions below cannot pass.
    let mut restored = TinyMlp::new()?;
    load_safetensors::<Backend, _, _>(&mut restored, &checkpoint)?;
    println!("  loaded into a freshly built TinyMlp");

    // Equality, claim 1: every parameter's bytes, path for path.
    let after = collect_state::<Backend, _>(&restored)?;
    assert_eq!(
        before.len(),
        after.len(),
        "state tree must not gain or lose leaves"
    );
    for (path, value) in before.iter() {
        let other = after
            .get(path)
            .unwrap_or_else(|| panic!("restored state is missing {path}"));
        assert_eq!(
            value.bytes(),
            other.bytes(),
            "parameter bytes must round-trip exactly for {}",
            path.as_str()
        );
    }
    println!(
        "  all {} parameter tensors byte-identical after the round trip",
        before.len()
    );

    // Equality, claim 2: the model computes the same function.
    let output = restored.forward(probe)?;
    assert_eq!(
        baseline.to_vec1::<f32>()?,
        output.to_vec1::<f32>()?,
        "restored model output must match the baseline bit-for-bit"
    );
    println!("  forward output unchanged: {:?}", output.to_vec1::<f32>()?);
    println!("  both assertions exact - safetensors round-tripped this state without loss");

    let _ = std::fs::remove_file(&checkpoint);
    let _ = std::fs::remove_dir(&temp_dir);
    Ok(())
}

/// Section 2: #93 made the block layout part of the on-disk format and
/// required sharding to become block-aware. `slice_bytes_for_rank` is the
/// workhorse behind `load_resharded_checkpoint`: given raw checkpoint
/// bytes, a global shape, a dtype, an axis, and a rank, it returns that
/// rank's byte slice and local shape. For a block dtype the local extent
/// along the shard axis must cover whole `Q8_0` blocks, because a shard
/// boundary inside a 34-byte block cannot be expressed as a byte range.
fn block_sharding() -> incin::Result<()> {
    // A real Q8_0 tensor: 4 x 64 = 256 values = 8 blocks. The last axis
    // (64) is a whole multiple of 32, so `quantize` accepts it.
    let grid_values: Vec<f32> = (0..4 * 64)
        .map(|index| ((index % 19) as f32 - 9.0) / 9.0)
        .collect();
    let grid = Tensor::<s![4, 64], Backend>::from_slice(&grid_values, ())?;
    let quantized = grid.quantize(-1)?;

    // The exact bytes this tensor would put in a checkpoint: one 34-byte
    // block (f16 scale + 32 i8 quants) per 32 values, flat row-major.
    let bytes = <Backend as HostInterop>::to_bytes::<Q8_0>(quantized.inner())?;
    let q8 = DTypeId::Q8_0.descriptor();
    println!(
        "  real Q8_0 tensor: shape {:?}, {} logical values = {} blocks x {} bytes = {} bytes",
        quantized.dims(),
        4 * 64,
        bytes.len() / q8.encoding().bytes_per_block(),
        q8.encoding().bytes_per_block(),
        bytes.len()
    );

    // Aligned: 4 x 64 split two ways along axis 1 gives each rank 32
    // values per row - exactly one whole block - so the byte range exists.
    match slice_bytes_for_rank(&bytes, &[4, 64], q8, 1, 0, 2) {
        Ok((slice, shape)) => println!(
            "  world 2, axis 1 (local extent 32): rank 0 gets shape {:?}, {} bytes - whole blocks, split works",
            shape,
            slice.len()
        ),
        Err(error) => println!("  aligned split unexpectedly refused: {error}"),
    }

    // Mid-block: the same tensor four ways along axis 1 gives each rank
    // 16 values per row - half a block - and no byte range can name where
    // that half ends. The refusal names the axis, the extent, and the
    // required multiple.
    match slice_bytes_for_rank(&bytes, &[4, 64], q8, 1, 0, 4) {
        Ok(_) => println!("  world 4 split succeeded, which the block rule forbids"),
        Err(error) => println!("  world 4, axis 1 (local extent 16) refused: {error}"),
    }

    // The contrast that shows this is a dtype rule, not a shape rule: the
    // identical split of an f32 tensor is always expressible, because a
    // scalar dtype has no block to fall inside of.
    let f32_bytes = <Backend as HostInterop>::to_bytes::<f32>(grid.inner())?;
    match slice_bytes_for_rank(&f32_bytes, &[4, 64], DTypeId::F32.descriptor(), 1, 0, 4) {
        Ok((slice, shape)) => println!(
            "  same split, f32 dtype: rank 0 gets shape {:?}, {} bytes - scalar storage has no block boundary",
            shape,
            slice.len()
        ),
        Err(error) => println!("  f32 split unexpectedly refused: {error}"),
    }
    Ok(())
}

fn section(title: &str) {
    println!("\n{title}");
    println!("{}", "-".repeat(title.len()));
}
