# Supported feature contract matrix

The supported feature combinations are compiled by
[`tools/feature-matrix.sh`](../tools/feature-matrix.sh). That script is the
single executable source used by `cargo xtask feature-matrix` in both CI and
`tools/ci-local.sh`.

The matrix covers the default and no-default core builds, representative core
opt-ins, backend families and integrations, facade API/preview combinations,
and high-value orthogonal interactions. CUDA, WGPU, Metal, and NCCL rows are
compile contracts; they do not claim that the corresponding hardware or
runtime libraries are present. Runtime hardware coverage remains in the
hardware-specific jobs and focused preview tests.

`cargo-hack --feature-powerset` remains useful for the small `incin-core`,
`incin-macros`, and `incin-diagnostics` feature spaces. It is intentionally not
used as an exhaustive backend or facade contract: those Cartesian products
include combinations that are individually legal Cargo flags but are not
supported product configurations, and their size makes the signal impractical.
