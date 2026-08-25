# 0.1.0 release notes

This is the user-facing release note for the first version intended for
crates.io. It describes the `0.1.0` baseline and is checked against the
release tag before publication. `CHANGELOG.md` carries the same
ground with more detail, and
`docs/MIGRATION.md` carries the full path-by-path table for anyone moving off a
`0.0.0` snapshot.

Nothing here applies if `0.1.0` is your first version. Skip to
[What's not finished yet](./whats_not_finished.md).

`0.1.0` establishes the selected public compatibility baseline described in
[Stability](./installation.md#stability). Preview APIs remain free to change
when they are explicitly identified as preview.

### Operator expressions compose directly

Tensor `+`, `-`, `*`, `/`, scalar forms, and unary `-` return tensors rather
than `Result`s. This keeps ordinary expressions and chains readable. They are
the documented panic-on-error convenience boundary; use `try_add`, `try_sub`,
`try_mul`, `try_div`, `try_neg`, and scalar named methods when a failure must
be handled. Operator panic text is deterministic and excludes tensor contents
and backend diagnostic text.

### Compiled plans remain preview-only

The `compiled` feature exposes its types only through
`incin::experimental::compiled`. Its CPU path is a reference evaluator for the
admitted descriptor-backed subset, not a stable compiler or deployment target.
`CompiledArtifact` is a preview plan snapshot, never a portable ABI: loading
checks the artifact format and caller-supplied compatibility major/minor values
(patch values may differ), not the running framework version.

## Things to change in your code

### `ModelExt::load` no longer takes a device

```text
// before
model.load(Format::Safetensors, path, &DeviceId::cpu())?;
// after
model.load(Format::Safetensors, path)?;
```

The argument was ignored. Removing it is covered in [Saving and
loading](./saving_loading.md), along with why relocation stays a separate
operation.

### Re-save any state file written by a `0.0.0` snapshot

State files now record a format version, and a file without one is refused as
unversioned rather than read on a guess. Load it with the old build, save it
with this one. Foreign safetensors files are unaffected; they were never
loadable through `ModelExt::load`, and `import_model!` does not look for the
key.

### Collect fresh gradients for every optimizer step

An optimizer step that reaches no parameter at all is now an error rather than
a silent success. The case this catches most often is reusing one `Gradients`
value across two steps: committing a step reassigns parameter storage, so the
second step matches nothing. Skipping *some* parameters is still fine.

### A non-default loss reduction is built with `with_reduction()`

`MSELoss::new()` now works without naming the reduction, the way
`torch.nn.MSELoss()` does. It previously failed to compile: the reduction type
parameter defaults to `Mean`, but a type-parameter default does not drive
inference for an associated function, so every call site had to write
`MSELoss::<Mean>::new()`.

That explicit form still compiles. Only a non-default reduction changes:

```text
use incin::nn::Sum; // Mean, Sum, and NoneReduction live here, not the prelude

// before
let loss = MSELoss::<Sum>::new().forward(&pred, &target)?;
// after
let loss = MSELoss::<Sum>::with_reduction().forward(&pred, &target)?;
```

The same applies to `CrossEntropyLoss`, `L1Loss`, and `BCEWithLogitsLoss`.

### `DummyBackend` is gone

The shape-only stand-in behind the `test-utils` feature has been removed. It
stored a shape instead of data and claimed to execute every operation, so a
test written against it passed whether or not the operation could run and
whatever values it produced. Use a real backend in tests.
`incin::test_utils` still exists and now gates deterministic fault injection
only.

### Backend authors: `AutogradBackend` gained a required method

`set_grad` is required alongside `backward` and `get_grad`. See [Backend
authoring](./backend_authoring.md) for why it is not defaulted.

## Things that got better without asking

### Gradient clipping

`clip_grad_norm` clips a parameter group by total L2 norm and returns the norm
before rescaling. `clip_grad_value` clamps every gradient element independently
into `[-clip_value, clip_value]`. Together they were the training primitives
the framework was missing. See
[Losses, optimizers, and schedulers](./training.md).

### The CPU SIMD kernels became reachable

The AVX2 elementwise kernels were gated on a compile-time check for the AVX2
target feature, which a stock `cargo build` never sets, so every default build
dead-code-eliminated them and ran a scalar loop instead. They are selected by
runtime detection now.

You do not need `RUSTFLAGS="-C target-cpu=native"` to get vectorised
elementwise arithmetic. On an AVX2 machine a 65536-element `f32` add went from
60.7 µs to 6.43 µs.

### Thirteen WGPU activations became reachable

`relu`, `step`, `mish`, `elu`, `gelu`, `abs`, `exp`, `neg`, `sqrt`, `log`,
`tanh`, `sigmoid`, and `swish` had working shaders and `Execute`
implementations, but were never advertised by the capability registry, so
canonical dispatch refused them. They are advertised now and verified
numerically against reference implementations on a software adapter.

A compile-time assertion now runs in both directions: every advertised row must
have an executor, *and* every written executor must be advertised. A kernel
cannot go unreachable that way again.

Metal also gained a column in the generated capability matrix. The
registrations existed and `metal` was a documented feature; the document simply
had no column for it.

### The gradient checks got roughly 25x more sensitive

Nothing about gradients changed; what changed is how well they are checked.
Every finite-difference gradient check in `incin-backends` used a step of
`1e-4`, about a hundredth of the value that minimizes total error for `f32`
storage. Central-difference rounding grows as `1/step`, so that inflated the
noise floor by about the same factor and left the checks asserting at their
own noise level, unable to separate a small real defect from a rounding
artifact.

The step is now the `f32` optimum, which drops the measured worst-case error
on correct gradients from `1.3e-2` to `1.0e-4` and lets the relative ceiling
tighten from `1e-2` to `1e-3`. Measured by injecting a uniform scaling error
into every analytic gradient, the suite previously needed a 5% error before
all affected tests failed; it now saturates at 0.2%.

### Building no longer needs `protoc`

The ONNX protobuf module is checked in rather than generated by a build script,
so a system protobuf compiler is no longer a mandatory dependency of every
crate that depends on the facade. See [Installation](./installation.md).

### A smaller default dependency graph

`onnx-pb` (unreleased since 2020, and the second major of `prost` it pinned
into the tree) is gone. The Hugging Face Hub client moved behind the
`data-hub` feature, which takes an async runtime and a second TLS stack out of
the default graph with it.

## Where to look when this chapter is stale

This page is written by hand and can drift. These do not:

- `docs/capabilities.md`: which operations each backend supports, for which
  dtypes, generated from the registrations.
- `docs/OPERATION_SEMANTICS.md`: the semantic contract for every catalog
  operation.
- The feature table in [Every feature flag](./feature_flags.md), which is
  checked against the Cargo manifests on every run.

If this chapter disagrees with one of those, the generated document is right.
