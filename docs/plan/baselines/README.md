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

Criterion's samples remain in `target/criterion/`; the checked-in TOML records
the confidence intervals and enough environment data to interpret them. Runtime
series exclude input construction. Reduction's required tensor-handle clone is
a Criterion setup step and is therefore outside the timed body. Compile sizes
are raw byte sizes from the release artifacts produced by the same commands;
they are not stripped installation sizes.

Rules for comparisons:

- compare the same benchmark ID, feature set, target triple, and backend;
- treat unavailable hardware as `unavailable`, never as a numeric result;
- rerun on an otherwise idle system before declaring a regression;
- use Criterion's saved-baseline comparison rather than comparing rounded
  values in this document;
- add a new benchmark ID if workload semantics change.

