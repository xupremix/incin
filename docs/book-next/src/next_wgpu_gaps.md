# WGPU: the operation-gap campaign

WGPU is the only accelerator with automated execution evidence on every
commit: the [stable backends chapter](../#/backends) records 146
advertised operations, and the every-PR software-adapter suite (lavapipe on
Ubuntu) executes the documented subset. This chapter is about how that
subset keeps growing — the campaign tracked under #91, told through one
recently landed operation.

## The shape of the work

WGPU work lands in batches, each one a capability-row change plus a
runtime test that would fail if the row lied:

- **Batch A** worked through the pointwise family: 24 runtime-verified
  unary, binary, trig, and scalar-seed operations.
- **Batch B** covered the structural, normalization, loss, and matmul rows
  on the software adapter.
- **Batch C** added `instance_norm`, dropout, BCE, and the SDPA row.
- **Attention end-to-end**, including a training smoke test that runs
  `backward` to finite, non-zero projection gradients.
- **Cross-entropy with backward**, then the indexing family this chapter
  walks through.

The contract for every batch is fail-closed: a capability row is advertised
only with an `Execute` implementation *and* a CPU-twin test behind it —
forward values against the same recipe on CPU, gradients against CPU's
scatter-add. A uniform seed only counts terms; the tests use hand-picked
values with known answers.

## The indexing pattern: host-walks plus one GPU kernel

The gather family (`embedding`, `gather`, `index_select`, `diag`,
`scatter`, `scatter_add`, `one_hot`, and now `bincount`) downloads its
operands, computes on the host, and uploads the result. That sounds like a
compromise, and it is a deliberate one: these operations are
index-bound, not math-bound, and a host walk through one accessor keeps
every dtype row honest. The one exception is the selection kernel —
`masked_fill`/`where_cond` launch real GPU selection over broadcast masks.

`bincount` joined this family most recently, and it shows the whole
pattern in miniature. The capability declaration puts it in the embedding
group for a union reason — one integer index operand in, one `i64`
histogram out, the bin count an attribute:

```text
embedding = [EmbeddingExact, OneHot, Bincount],
```

from `crates/incin-backends/src/capability/declarations.rs`. The output
shape `[bins]` is static (named by the attribute, not the data), which is
what separates this slice from `nonzero`'s data-dependent extent. An index
outside `0..bins` is a typed `InvalidInput` refusal, not a silent skip: a
quietly-low count makes every offset derived from it wrong, so CPU's
`bincount_storage` errors for the same reason. Forward-only, no tape entry
— the same `GradientRule::None` family `one_hot` belongs to.

## The implementation, end to end

The executor walks the same `index_values` accessor `one_hot` uses, so the
integer-dtype widening (`u8`/`u32`/`i64`) is proven once and shared:

```rust,ignore
// crates/incin-backends/src/wgpu/backend/indexing.rs
pub(crate) fn bincount(indices: &WgpuStorage, bins: usize) -> Result<WgpuStorage> {
    if bins == 0 {
        return Err(Error::Backend(BackendError::InvalidInput {
            operation: OperationKind::Bincount,
            reason: "bincount needs at least one bin to count into",
        }));
    }
    let idx_vals = index_values(indices)?;
    let mut counts = alloc::vec![0i64; bins];
    for &raw in &idx_vals {
        match usize::try_from(raw) {
            Ok(address) if address < bins => counts[address] += 1,
            _ => {
                return Err(Error::Backend(BackendError::InvalidInput {
                    operation: OperationKind::Bincount,
                    reason: "bincount index is not a whole number inside the bin range",
                }));
            }
        }
    }
    // ... uploaded as i64 storage with shape [bins]
}
```

The `Execute` implementation is a thin dispatch shell around it — one
input, `bins` from the operation descriptor, errors mapped through the
indexing kernel-error helper. And the test that keeps the capability row
honest runs the operation three ways in
`crates/incin-backends/tests/wgpu_indexing.rs`:

```rust,ignore
// A fixed histogram with no tape entry ...
let (out, recorded) = run_mixed::<op::Bincount, _>(BincountAttributes { bins: 3 }, ...)?;
assert_eq!(recorded, 0, "bincount is forward-only: no tape entry");
assert_eq!(read_histogram(&out), vec![2, 1, 3], "bincount histogram");
// ... rank-2 index operands ignored for shape, spare bins zero-filled,
// u8 indices admitted, out-of-range indices and zero bins refused.
```

Run it yourself against the software adapter — no GPU required:

```sh
cargo test -p incin-backends --features wgpu --test wgpu_indexing
```

## The boundary that remains

WGPU compute is still `f32`-only: storage admits `u8`/`u32`/`i64`/`bool`,
but `native_precision` refuses non-`f32` compute. Every batch lands against
that wall on purpose — widen the storage rows first (so indices and masks
travel), kernelize the math second. The next wall after indexing is the
fused-attention kernel (issue #104), which has no WGPU row yet, and the
`f16`/`bf16` compute story, which the [CUDA chapter](./next_cuda.md)
is writing first.
