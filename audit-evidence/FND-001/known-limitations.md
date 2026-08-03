# FND-001 known limitations

- The facade change is intentionally semver-breaking at version `0.0.0`; removed accidental wildcard exports are documented in `migration-table.md`.
- `incin::__macro_support` is necessarily public because procedural macro expansions compile in downstream crates. It is doc-hidden and is not a supported end-user namespace.
- CUDA, NCCL, Metal, Metal-MPS, and hardware-backed WGPU behavior are outside this software-only facade gate. No capability or runtime support claim is inferred from compilation.
- The workspace-wide formatting gate had a pre-existing failure at the FND-000 boundary. Its exact FND-001 result is archived; changed Rust files also receive a focused formatting/diff check.
- Internal module-local wildcard re-exports remain inside owning crates where they organize that crate's own API. No cross-crate wildcard remains in the audited `incin` facade/core boundary query.
- `cargo-semver-checks` 0.50.0 cannot produce an API comparison because it enables every `incin` feature and its rustdoc build reaches the pre-existing incomplete Candle/accelerator combination. `semver-incin.txt` preserves the full diagnostic and generated reproducer.
