# Metal: from host-side to device-side

Metal is the narrowest preview: 117 advertised operations, host-verified on
any OS, with real device execution only on Apple Silicon. This chapter is
about the direction of travel — Batch-A pointwise coverage and the
normalization/layout/linear/SDPA rows (#92) — and the two deliberate
absences that define the boundary.

## What "host-verified" means

The Metal executor (`crates/incin-backends/src/metal/executor.rs`) runs
its math on the host on Linux CI, so a passing suite proves the operation
semantics, the capability rows, and the dtype contracts — but not a Metal
device. The macOS Apple-Silicon hardware job is the device run
(`HARDWARE_METAL_RUNNER` is optional with a `macos-latest` fallback, unlike
the CUDA variables). Read a Metal test result as "the math and the
plumbing are right; the shader ran only on macOS."

The indexing family shows the pattern:
`crates/incin-backends/src/metal/indexing.rs` carries
embedding/gather/index_select as host-walks, the same compromise the
[WGPU campaign](./next_wgpu_gaps.md) makes for index-bound operations.

## Two deliberate absences

- **No spatial group.** Convolution and pooling have no Metal rows even
  though shader and MPS infrastructure exists. An empty capability group
  is a decision, not an oversight: advertising `conv2d` without a device
  proof would repeat the mistake the capability system exists to prevent.
- **No bool-mask or comparison/logical families.** `masked_fill`,
  `where_cond`, and the `cmp_*`/`logical_*` families stay unadvertised
  while the masking contract hardens elsewhere in the tree.

Porting one Batch-A operation from host-walk to an MPS shader is the
canonical first Metal contribution: wire the `Execute` implementation
(the Batch-A wiring is the template), advertise the row, and add the
operation to the host-side suite (`metal_gap_ops.rs`) so Linux CI keeps it
honest before any device ever sees it.

## The boundary that remains

Device proof exists only where the macOS job executes. Everything else
about Metal — the 117-row matrix, the SDPA row, the linear row — is a
checked-in promise waiting for its device run, and this chapter will shrink
as those runs land.
