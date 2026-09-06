# Performance baseline protocol

GOV-004 establishes the series that later budget and regression tasks consume.
Run from a quiet checkout with the normal CPU feature set:

```text
cargo bench -p incin -- --save-baseline main
```

On a machine with a WGPU adapter, record the accelerator series separately so
an unavailable GPU can never be mistaken for a successful zero-cost run:

```text
cargo bench -p incin --features wgpu --bench baselines -- \
  '^(capability/wgpu|gpu/)' --save-baseline wgpu-main
```

On a machine with an NVIDIA device, record the CUDA series the same way:

```text
cargo bench -p incin --features cuda --bench baselines -- \
  '^(capability/cuda|gpu/cuda)' --save-baseline cuda-main
```

On an Apple Silicon device with Metal support, record the Metal series:

```text
cargo bench -p incin --features metal,metal-mps --bench baselines -- \
  '^(capability/metal|gpu/metal)' --save-baseline metal-main
```

The CUDA IDs carry the backend inside the `gpu` group, unlike the WGPU IDs whose
spelling predates them and is frozen. Two accelerators sharing one Criterion ID
would collide the moment a build enabled both features, and the budget key is
`(backend, id)` rather than `id`, so nothing would report the collision.

Criterion's samples remain in `target/criterion/`; the checked-in TOML records
the confidence intervals and enough environment data to interpret them. Runtime
series exclude input construction. Reduction's required tensor-handle clone is
a Criterion setup step and is therefore outside the timed body. Compile sizes
are raw byte sizes from the release artifacts produced by the same commands;
they are not stripped installation sizes.

Not every series comes from one machine. The CPU and WGPU rows were captured on
the host in `[environment]`; the CUDA rows were captured later on the host in
`[environment.cuda_host]`, because the first machine had no NVIDIA device. A
series taken on a second host carries a `host` key naming its environment table,
and a profile or capability block does the same. Recording those rows under the
unqualified `[environment]` would attribute them to a machine that never ran
them, which is the one thing a baseline document must not do.

Rules for comparisons:

- compare the same benchmark ID, feature set, target triple, backend, and host;
- never diff a series or artifact size across two `host` values, including
  compile sizes, which also depend on the linker each host is configured with;
- treat unavailable hardware as `unavailable`, never as a numeric result;
- rerun on an otherwise idle system before declaring a regression;
- use Criterion's saved-baseline comparison rather than comparing rounded
  values in this document;
- add a new benchmark ID if workload semantics change.

CI compiles the benchmarks on every PR (`cargo bench -p incin --no-run` in
the ledger job) so a bitrotted benchmark fails fast, but never executes
them: timing gates need quiet hardware, and live performance gates are
future work (TUN-008/CI-006).

