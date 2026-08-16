# Target-first construction

Incin's application-facing allocation API starts with a target. A target such
as `Cpu` selects the backend and device, while `shape!` supplies static proof
or runtime dimensions. This is part of the normal API and needs no feature
flag.

```rust,no_run
use incin::prelude::*;

let x = Cpu.zeros(shape![2, 3])?;
let batch = 4;
let y = Cpu.zeros(shape![batch, 3])?;
let z = Cpu.zeros([batch, 3])?;
# Ok::<(), incin::Error>(())
```

Static literals and explicit const paths preserve compile-time dimensions:

```rust,no_run
use incin::prelude::*;

const FEATURES: usize = 128;
let weights = Cpu.zeros(shape![const FEATURES, const FEATURES])?;
# Ok::<(), incin::Error>(())
```

The typed constructor remains available when code intentionally fixes the
backend type:

```rust,no_run
use incin::prelude::*;

type Backend = DefaultBackend;
let tensor = Tensor::<s![2, 2], Backend>::ones(())?;
# Ok::<(), incin::Error>(())
```

Runtime reshape inference is a value-level shape specification, separate from
indexing:

```rust,no_run
use incin::prelude::*;

let x = Tensor::<s![2, 3], DefaultBackend>::ones(())?;
let y = x.reshape_infer(shape![6, infer])?;
# Ok::<(), incin::Error>(())
```

Layer construction follows the same target-first boundary through the
canonical builder extension. The older direct `Linear::new` and
`new_on_target` spellings are not part of the public prelude.

```rust,no_run
use incin::prelude::*;

let layer = incin_core::nn::linear::linear(shape![4, 3]).init(&Cpu)?;
assert_eq!(layer.weight.shape_dims(), vec![3, 4]);
# Ok::<(), incin::Error>(())
```

Arithmetic operators return tensors directly and include operation context in
their panic message when a runtime broadcast or backend check fails. Use the
`try_add`, `try_sub`, `try_mul`, `try_div`, and `try_neg` methods when the
failure must remain a recoverable `Result`.

## Axis selectors

Reductions accept one selector value for static axes, named axes, and runtime
signed axes. Structural cursor methods remain an advanced escape hatch for
authoring low-level shape operations. Static reductions and flattening preserve
output shape information at the type level. Transpose and runtime selectors
preserve known rank, while named selectors use the shape facts available after
resolving the axis.

```rust,no_run
use incin::prelude::*;

let x = Cpu.ones(shape![4, 8, 16])?;
let summed = x.sum(axis!(1))?;
let kept = x.sum_keepdim(axis!(-2))?;
let first = x.sum(0isize)?;
let indices = x.argmax(axis!(2))?;
let minima = x.argmin(0isize)?;
let means = x.mean(axis!(-1))?;
let maxima = x.max_keepdim(axis!(0))?;
let minima_by_value = x.min(axis!(1))?;
let argmax = x.argmax(axis!(-1))?;
let transposed = x.transpose(axis!(0), axis!(2))?;
let flattened = x.flatten(axis!(1), axis!(2))?;
let stacked = x.stack(&x, axis!(0))?;
let concatenated = x.concat(&x, axis!(-1))?;
let flattened_runtime = x.flatten(1isize, -1isize)?;
let expanded = x.unsqueeze(axis!(1))?;
let squeezed = expanded.try_squeeze(1isize)?;
# Ok::<(), incin::Error>(())
```

Numeric axes accept arbitrary signed `isize` values, including negative axes,
without a finite lookup table. Pass `axis!(...)` to any reduction or pass a
runtime `isize` directly. Named axes use the same methods when the axis
identity matters more than its numeric position:

```rust,no_run
use incin::prelude::*;

dim!(Batch, Channels, Width);
let x = Tensor::<s![Batch = 4, Channels = 8, Width = 16], DefaultBackend>::ones(())?;
let channels = x.sum(axis!(Channels))?;
let kept = x.mean_keepdim(axis!(Channels))?;
# Ok::<(), incin::Error>(())
```

Runtime selectors validate their normalized position against the tensor rank.
`mean`, `max`, `min`, and their keep-dimension forms follow the same selector
rules.
`flatten` uses the same signed runtime axis convention. Use `concat` and
`stack` with a selector for ordinary code. The structural cursor forms remain
available only for low-level shape implementations. `unsqueeze` and
`try_squeeze` use the same selectors: static selectors retain exact structural
shapes, while signed runtime selectors retain known rank and validate the
selected extent for squeezing.

Axis-preserving operations use the same selectors:

```rust,no_run
use incin::prelude::*;

let x = Cpu.ones(shape![4, 8, 16])?;
let cumulative = x.cumsum(-1)?;
let probabilities = x.softmax(axis!(1))?;
# Ok::<(), incin::Error>(())
```

For generic known-rank runtime shapes, `sum` and `sum_keepdim` retain the rank
arithmetic in the type while leaving extents runtime-valued. The older
`sum_runtime_ranked` spellings remain hidden compatibility helpers.

Named reduction selectors retain known rank. Named dimensions are resolved at
runtime, while the public output uses the strongest shape proof available for
the selected operation. On stable Rust, the blanket recursive lookup cannot
currently produce an exact named post-reduction type without overlapping trait
implementations. The deliberate fallback is `Ranked<R>` or `Ranked<R-1>` for
known-rank inputs, and `Dyn` only when the input rank is dynamic.

Tensor indexing and slicing use `i![...]` with signed indices and ordinary Rust
ranges. Reshape inference is separate from indexing and uses `shape![..., infer]`.
The older `reshape_idx::<idx![... ]>()` spelling remains available for
advanced type-level targets. Axis selection is
also separate, so `axis!(-1)` never changes `i![-1]` indexing rules.

```rust,no_run
use incin::prelude::*;

let x = Cpu.ones(shape![4, 8, 16])?;
let last = x.get(i![-1, .., ..])?;
let window = x.get(i![.., 2..6, ..])?;
# Ok::<(), incin::Error>(())
```

Named dimensions and const dimensions use different syntax. A named axis is
written as `s![Batch, Features]`; a const path must be marked explicitly:

```rust,no_run
use incin::prelude::*;

const BATCH: usize = 32;
const FEATURES: usize = 128;
dim!(Batch, Features);
let x = Cpu.zeros(shape![Batch = const BATCH, Features = const FEATURES])?;
type X = s![Batch = const BATCH, Features = const FEATURES];
let _: Tensor<X, _, f32, NoGrad> = x;
# Ok::<(), incin::Error>(())
```

The explicit `const` marker preserves the existing bare `s![Batch, ...]`
named-dimension grammar and makes the meaning of a path unambiguous.
